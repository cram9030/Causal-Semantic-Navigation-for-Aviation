"""Offline manifest builder: trajectory windows -> tube envelope -> CSJ Streets -> manifest.

This is the Phase 1 deliverable of `docs/INTEGRATION_PLAN.md` §5, implementing
§3.3 step by step for each trajectory window:

1. Grow the window's tube corridor by the sensor's ground FOV radius at that
   window's maximum AGL -> the *visible footprint* (WGS84 polygon).
2. Query CSJ Streets against that footprint's envelope.
3. Clip the returned centerlines to the footprint, in the trajectory's local
   ENU frame, and record each one's off-track offset in meters.
4. Derive the intersections between the clipped centerlines.
5. Record which imagery tiles the footprint covers.

Everything here runs **once, offline, per flight-planning cycle**. Nothing in
this module is on the runtime path: at inference time the "possible roads" node
reads :meth:`csnav.trajectory.manifest.LandmarkManifest.query` instead
(CLAUDE.md, "don't reintroduce a live/global spatial query in the runtime
path").

**The tube radius is never chosen here.** Every entry point takes a
:class:`~csnav.trajectory.tube.TubeModel` (or a
:class:`~csnav.trajectory.config.ConopsConfig` that supplies one) from the
caller, so sweeping radii is a matter of re-running the builder with a
different input - CLAUDE.md core decision 4.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.strtree import STRtree

from csnav.data.arcgis.models import Extent, TileInfo
from csnav.data.arcgis.streets import StreetSegment
from csnav.geometry import shapes
from csnav.geometry.camera import Camera
from csnav.trajectory.coverage import (
    AglProvider,
    TileRef,
    height_as_agl,
    max_agl,
    max_ground_reach,
    tiles_for_footprint,
    visible_footprint,
)
from csnav.trajectory.manifest import (
    LandmarkManifest,
    ManifestBundle,
    ManifestIntersection,
    ManifestLandmark,
)
from csnav.trajectory.trajectory import X0_NODE, Trajectory, TrajectorySet, TrajectoryWindow
from csnav.trajectory.transition import TransitionFamily, TransitionModel
from csnav.trajectory.tube import TubeModel

logger = logging.getLogger(__name__)

#: Field names the CSJ Streets schema has used for roadway width, tried in
#: order. The live schema owns these names, so this is a lookup list rather
#: than a fixed contract - a manifest simply carries ``width=None`` when none
#: of them is present.
WIDTH_FIELD_CANDIDATES = ("WIDTH", "Width", "width", "ROADWIDTH", "RoadWidth", "PAVED_WIDTH", "STREETWIDTH")

#: Field names tried for a human-readable street name, same caveat.
NAME_FIELD_CANDIDATES = ("STREETNAME", "StreetName", "FULLNAME", "FullName", "NAME", "Name", "name")

#: Distance, in meters, within which two computed junction points are treated
#: as the same intersection. Absorbs the sub-meter jitter of clipping several
#: centerlines against the same footprint boundary; not a tube parameter.
DEFAULT_INTERSECTION_SNAP = 2.0


class StreetsSource(Protocol):
    """The subset of :class:`csnav.data.arcgis.streets.CSJStreetsClient` this builder needs.

    Declared as a Protocol so the builder can be driven from a cached GeoJSON
    pull (see :class:`StaticStreetsSource`) as easily as from the live
    FeatureServer - a pinned manifest should be reproducible from an archived
    street snapshot, not only from whatever the weekly refresh currently holds.
    """

    def query(self, bbox: Extent | None = ..., where: str = ..., out_fields: str = ...) -> list[StreetSegment]:
        ...


class StaticStreetsSource:
    """A :class:`StreetsSource` backed by street segments already in memory.

    Lets a manifest be rebuilt offline from an archived CSJ Streets pull (e.g.
    the GeoJSON ``scripts/fetch_csj_streets.py`` writes), which is what pinning
    a manifest to a flight-planning cycle actually requires - the live layer
    refreshes weekly and will not reproduce an older build.
    """

    def __init__(self, segments: list[StreetSegment], source_label: str = "static") -> None:
        self.segments = list(segments)
        self.source_label = source_label

    def query(self, bbox: Extent | None = None, where: str = "1=1", out_fields: str = "*") -> list[StreetSegment]:
        """Return the segments intersecting ``bbox`` (EPSG:4326); all of them if ``bbox`` is ``None``.

        ``where``/``out_fields`` are accepted for interface compatibility and
        ignored - filtering an in-memory snapshot by SQL is out of scope.
        """
        if bbox is None:
            return list(self.segments)
        if bbox.wkid != 4326:
            raise ValueError(f"bbox must be EPSG:4326, got wkid={bbox.wkid}")
        window = Polygon(
            [
                (bbox.xmin, bbox.ymin),
                (bbox.xmax, bbox.ymin),
                (bbox.xmax, bbox.ymax),
                (bbox.xmin, bbox.ymax),
            ]
        )
        return [segment for segment in self.segments if _segment_geometry(segment).intersects(window)]


def _segment_geometry(segment: StreetSegment) -> LineString | MultiLineString:
    """A street segment's centerline as a shapely geometry, in WGS84 (lon, lat)."""
    if len(segment.parts) == 1:
        return LineString(segment.parts[0])
    return MultiLineString([list(part) for part in segment.parts])


def _first_present(attributes: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        value = attributes.get(key)
        if value not in (None, ""):
            return value
    return None


def _segment_width(attributes: dict[str, Any]) -> float | None:
    """Roadway width in meters from the CSJ attributes, or ``None`` if not published.

    CSJ publishes widths in feet; the value is converted to meters here so
    everything downstream of the manifest is metric, matching the ENU frame
    used for buffers and offsets.
    """
    raw = _first_present(attributes, WIDTH_FIELD_CANDIDATES)
    if raw is None:
        return None
    try:
        return float(raw) * 0.3048
    except (TypeError, ValueError):
        return None


def _segment_id(segment: StreetSegment, fallback_index: int) -> str:
    if segment.object_id is not None:
        return str(segment.object_id)
    return f"seg-{fallback_index}"


class ManifestBuilder:
    """Builds pinned landmark manifests for trajectories and trajectory sets.

    ``streets`` is any :class:`StreetsSource` (the live
    :class:`~csnav.data.arcgis.streets.CSJStreetsClient`, or a
    :class:`StaticStreetsSource` over an archived pull). ``tile_info`` and
    ``tile_level`` are optional: supply them to record which imagery tiles each
    window covers - the set the ground-truth builder needs labels for - and
    leave them unset to build road/intersection manifests alone.

    ``agl_provider`` maps a waypoint to its height above ground in meters; the
    default treats waypoint height as AGL, which is only right for plans
    authored in AGL. Pass
    :func:`csnav.trajectory.coverage.agl_from_elevation` over the USGS 3DEP
    client for real terrain.
    """

    def __init__(
        self,
        streets: StreetsSource,
        tile_info: TileInfo | None = None,
        tile_level: int | None = None,
        agl_provider: AglProvider = height_as_agl,
        intersection_snap: float = DEFAULT_INTERSECTION_SNAP,
        streets_where: str = "1=1",
        source_label: str | None = None,
    ) -> None:
        if (tile_info is None) != (tile_level is None):
            raise ValueError("tile_info and tile_level must be supplied together, or neither")
        if intersection_snap <= 0.0:
            raise ValueError(f"intersection_snap must be > 0 meters, got {intersection_snap}")
        self.streets = streets
        self.tile_info = tile_info
        self.tile_level = tile_level
        self.agl_provider = agl_provider
        self.intersection_snap = intersection_snap
        self.streets_where = streets_where
        self.source_label = source_label or getattr(streets, "layer_url", None) or getattr(
            streets, "source_label", None
        )

    # ----- one window ---------------------------------------------------------

    def build_window(
        self,
        trajectory: Trajectory,
        window: TrajectoryWindow,
        tube: TubeModel,
        camera: Camera | None = None,
        segments: list[StreetSegment] | None = None,
    ) -> LandmarkManifest:
        """Build the manifest for one trajectory window.

        ``tube`` supplies the containment radius (meters) - it is never derived
        here. ``camera`` grows the search footprint beyond the tube by how far
        the sensor can see from the window's worst-case height and attitude;
        ``None`` scopes the manifest to the tube itself. ``segments`` lets a
        caller pass street geometry it has already fetched (see
        :meth:`build_trajectory`, which queries once per trajectory rather than
        once per window).
        """
        footprint = visible_footprint(
            trajectory, tube, window=window, camera=camera, agl_provider=self.agl_provider
        )
        envelope = _polygon_extent(footprint)

        if segments is None:
            segments = self.streets.query(bbox=envelope, where=self.streets_where)

        frame = trajectory.local_frame
        footprint_enu = shapes.to_enu(footprint, frame)
        centerline_enu = shapes.to_enu(
            LineString([(wp.lon, wp.lat) for wp in trajectory.segment(window.start_distance, window.end_distance)]),
            frame,
        )

        roads: list[ManifestLandmark] = []
        clipped_enu: list[LineString | MultiLineString] = []
        for index, segment in enumerate(segments):
            geometry_enu = shapes.to_enu(_segment_geometry(segment), frame)
            clipped = geometry_enu.intersection(footprint_enu)
            parts_enu = shapes.line_parts(clipped)
            if not parts_enu:
                continue
            clipped_line = (
                LineString(parts_enu[0]) if len(parts_enu) == 1 else MultiLineString([list(p) for p in parts_enu])
            )
            clipped_enu.append(clipped_line)
            wgs84_parts = shapes.line_parts(shapes.to_wgs84(clipped_line, frame))
            roads.append(
                ManifestLandmark(
                    segment_id=_segment_id(segment, index),
                    parts=wgs84_parts,
                    offset=centerline_enu.distance(clipped_line),
                    width=_segment_width(segment.attributes),
                    name=_first_present(segment.attributes, NAME_FIELD_CANDIDATES),
                    attributes=dict(segment.attributes),
                )
            )

        return LandmarkManifest(
            window=window,
            tube_radius=tube.radius,
            max_agl=max_agl(trajectory, window, self.agl_provider),
            ground_reach=(
                0.0 if camera is None else max_ground_reach(trajectory, window, camera, self.agl_provider)
            ),
            envelope=envelope,
            footprint=footprint,
            candidate_roads=tuple(roads),
            intersections=self._intersections(roads, clipped_enu, trajectory),
            tiles=self._tiles(footprint),
        )

    # ----- one trajectory -----------------------------------------------------

    def build_trajectory(
        self,
        trajectory: Trajectory,
        tube: TubeModel,
        window_length: float,
        camera: Camera | None = None,
        per_window_query: bool = False,
    ) -> tuple[LandmarkManifest, ...]:
        """Build one manifest per window of ``trajectory``.

        By default the street query is issued **once** for the whole
        trajectory's visible envelope and the result reused across its windows,
        then clipped per window. That is one request instead of one per window,
        at the cost of a larger bounding box for a diagonal route. Set
        ``per_window_query=True`` to query each window's own envelope instead -
        preferable for a long or strongly curved trajectory, where the
        whole-route bounding box would pull in far more of the city than the
        corridor touches.
        """
        windows = trajectory.windows(window_length)
        shared: list[StreetSegment] | None = None
        if not per_window_query:
            route_footprint = visible_footprint(
                trajectory, tube, window=None, camera=camera, agl_provider=self.agl_provider
            )
            shared = self.streets.query(bbox=_polygon_extent(route_footprint), where=self.streets_where)
            logger.info(
                "trajectory %s: %d street segments fetched for %d windows",
                trajectory.id,
                len(shared),
                len(windows),
            )
        return tuple(
            self.build_window(trajectory, window, tube, camera=camera, segments=shared)
            for window in windows
        )

    # ----- one transition family ----------------------------------------------

    def build_transition_family(
        self,
        family: TransitionFamily,
        tube: TubeModel,
        window_length: float,
        camera: Camera | None = None,
        per_window_query: bool = False,
    ) -> tuple[LandmarkManifest, ...]:
        """Build manifests over every sampled path in a transition family.

        A transition is not one trajectory but a *family* of generated paths
        (:mod:`csnav.trajectory.transition`) - the region it sweeps is the
        object of interest, and any point in it is a valid state mid-transition.
        Pinning manifests over it means building windows over each sampled
        path's own arc length, at ``tube``'s radius (the CONOPS's transition
        radius, typically wider than a candidate route's). Denser sampling
        (``conops.transition.samples``) gives tighter coverage of that region at
        the cost of more manifests - it is the same fidelity/cost knob it is
        everywhere else in the transition model.

        By default streets are queried **once**, over the whole family's
        reachable footprint (every sampled path's tube, unioned, grown by the
        camera's worst-case reach across every path), and reused across every
        path's windows - one request instead of one per path. Set
        ``per_window_query=True`` to query per window instead, as
        :meth:`build_trajectory` does.

        Returns nothing for an empty family (every initiation screened out) -
        there is no region to cover, not an error.
        """
        if family.is_empty:
            return ()

        shared: list[StreetSegment] | None = None
        if not per_window_query:
            reach = (
                max(
                    max_ground_reach(path.trajectory, None, camera, self.agl_provider)
                    for path in family.paths
                )
                if camera is not None
                else 0.0
            )
            route_footprint = family.reachable_footprint(tube, extra_buffer=reach)
            shared = self.streets.query(bbox=_polygon_extent(route_footprint), where=self.streets_where)
            logger.info(
                "transition %s -> %s: %d street segments fetched for %d sampled paths",
                family.rule.source,
                family.rule.target,
                len(shared),
                len(family.paths),
            )

        manifests: list[LandmarkManifest] = []
        for path in family.paths:
            windows = path.trajectory.windows(window_length)
            manifests.extend(
                self.build_window(path.trajectory, window, tube, camera=camera, segments=shared)
                for window in windows
            )
        return tuple(manifests)

    # ----- a whole trajectory set --------------------------------------------

    def build_set(
        self,
        trajectory_set: TrajectorySet,
        conops: "ConopsConfigLike",
        per_window_query: bool = False,
        include_transitions: bool = True,
    ) -> ManifestBundle:
        """Build and pin manifests for every candidate route in ``T``, and every transition family.

        ``conops`` supplies the swept parameters - the tube radius for each
        trajectory, the window length in meters, the camera, and the
        transition model - via
        :meth:`csnav.trajectory.config.ConopsConfig.tube_for`. Its recorded
        values travel with the bundle so a pinned manifest says what it was
        built under.

        With ``include_transitions`` (the default), every rule in
        ``trajectory_set.transitions`` - other than an entry rule out of
        :data:`~csnav.trajectory.trajectory.X0_NODE`, which carries no geometry
        - is generated via ``conops.transition`` and covered by
        :meth:`build_transition_family`. A rule whose family turns out empty
        (every sampled initiation screened out) is logged and skipped, not
        treated as an error: the manifest simply has nothing to say about a
        hand-off nothing can fly.
        """
        manifests: list[LandmarkManifest] = []
        for trajectory in trajectory_set.trajectories:
            manifests.extend(
                self.build_trajectory(
                    trajectory,
                    conops.tube_for(trajectory),
                    conops.window_length,
                    camera=conops.camera,
                    per_window_query=per_window_query,
                )
            )

        if include_transitions:
            for rule in trajectory_set.transitions:
                if rule.source == X0_NODE:
                    continue
                source = trajectory_set.by_id(rule.source)
                target = trajectory_set.by_id(rule.target)
                family = conops.transition.family(source, target, rule)
                if family.is_empty:
                    logger.warning(
                        "transition %s -> %s admits no feasible path (all %d sampled initiations "
                        "screened out); no manifest built for it",
                        rule.source,
                        rule.target,
                        family.rejected,
                    )
                    continue
                manifests.extend(
                    self.build_transition_family(
                        family,
                        conops.tube_for(family.paths[0].trajectory),
                        conops.window_length,
                        camera=conops.camera,
                        per_window_query=per_window_query,
                    )
                )

        return ManifestBundle(
            trajectory_set_id=trajectory_set.id,
            manifests=tuple(manifests),
            parameters=self._parameters(conops),
            streets_source=self.source_label,
        )

    # ----- helpers ------------------------------------------------------------

    def _parameters(self, conops: "ConopsConfigLike") -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "tube_radius_m": conops.tube_radius,
            "transition_tube_radius_m": conops.transition_tube_radius,
            "window_length_m": conops.window_length,
            "camera": _camera_parameters(conops.camera),
            "per_trajectory_tube_radius_m": dict(conops.per_trajectory_radius),
            "intersection_snap_m": self.intersection_snap,
        }
        if self.tile_info is not None:
            parameters["tile_level"] = self.tile_level
            parameters["tile_scheme_wkid"] = self.tile_info.wkid
        return parameters

    def _tiles(self, footprint: Polygon) -> tuple[TileRef, ...]:
        if self.tile_info is None or self.tile_level is None:
            return ()
        return tiles_for_footprint(footprint, self.tile_info, self.tile_level)

    def _intersections(
        self,
        roads: list[ManifestLandmark],
        clipped_enu: list[LineString | MultiLineString],
        trajectory: Trajectory,
    ) -> tuple[ManifestIntersection, ...]:
        """Junctions between the clipped centerlines, snapped and de-duplicated.

        Computed in the trajectory's local ENU frame so ``intersection_snap``
        is a real distance in meters, then converted back to WGS84. Uses an
        STRtree so this stays near-linear in the number of manifest roads
        rather than testing every pair.
        """
        if len(clipped_enu) < 2:
            return ()

        tree = STRtree(clipped_enu)
        clusters: list[tuple[list[float], list[float], set[str]]] = []
        for index, geometry in enumerate(clipped_enu):
            for other_index in tree.query(geometry):
                if other_index <= index:
                    continue
                meeting = geometry.intersection(clipped_enu[other_index])
                for point in shapes.point_parts(meeting):
                    ids = {roads[index].segment_id, roads[other_index].segment_id}
                    _add_to_cluster(clusters, point.x, point.y, ids, self.intersection_snap)

        frame = trajectory.local_frame
        junctions = []
        for easts, norths, ids in clusters:
            centre = frame.to_wgs84(sum(easts) / len(easts), sum(norths) / len(norths))
            junctions.append(
                ManifestIntersection(lat=centre.lat, lon=centre.lon, segment_ids=tuple(sorted(ids)))
            )
        return tuple(sorted(junctions, key=lambda junction: (junction.lat, junction.lon)))


def _add_to_cluster(
    clusters: list[tuple[list[float], list[float], set[str]]],
    east: float,
    north: float,
    ids: set[str],
    snap: float,
) -> None:
    """Merge an ENU junction point into an existing cluster within ``snap`` meters, or start one."""
    for easts, norths, cluster_ids in clusters:
        centre_east = sum(easts) / len(easts)
        centre_north = sum(norths) / len(norths)
        if (east - centre_east) ** 2 + (north - centre_north) ** 2 <= snap**2:
            easts.append(east)
            norths.append(north)
            cluster_ids.update(ids)
            return
    clusters.append(([east], [north], set(ids)))


def _camera_parameters(camera: Camera | None) -> dict[str, Any] | None:
    """Camera settings recorded with a manifest, so a pinned build says what saw what."""
    if camera is None:
        return None
    return {
        "horizontal_deg": camera.field_of_view.horizontal_deg,
        "vertical_deg": camera.field_of_view.vertical_deg,
        "pose_roll_deg": camera.pose.roll_deg,
        "pose_pitch_deg": camera.pose.pitch_deg,
        "pose_yaw_deg": camera.pose.yaw_deg,
        "attitude_margin_roll_deg": camera.attitude_margin.roll_deg,
        "attitude_margin_pitch_deg": camera.attitude_margin.pitch_deg,
        "maneuver_roll_deg": camera.attitude_margin.maneuver_roll_deg,
        "maneuver_pitch_deg": camera.attitude_margin.maneuver_pitch_deg,
        "maneuver_radius_m": camera.attitude_margin.maneuver_radius,
    }


def _polygon_extent(polygon: Polygon) -> Extent:
    xmin, ymin, xmax, ymax = polygon.bounds
    return Extent(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, wkid=4326)


class ConopsConfigLike(Protocol):
    """Structural type of what :meth:`ManifestBuilder.build_set` needs from a CONOPS config.

    Declared here (rather than importing :class:`csnav.trajectory.config.ConopsConfig`)
    to keep the builder independent of the config file format - a sweep script
    can hand in its own object with these attributes.
    """

    tube_radius: float
    transition_tube_radius: float | None
    window_length: float
    camera: Camera | None
    transition: TransitionModel
    per_trajectory_radius: dict[str, float]

    def tube_for(self, trajectory: Trajectory) -> TubeModel:
        ...
