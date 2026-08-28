"""Interactive Leaflet maps of the trajectory set, its tubes, and the tiles in view.

Built on `folium <https://python-visualization.github.io/folium/>`_ rather than
a bespoke renderer: the output is a self-contained HTML file that opens in a
browser with real San Jose imagery underneath, which is what makes a tube
radius or a tile footprint reviewable ("does that corridor actually cover the
streets we expect?").

Entry points:

* :func:`trajectory_set_map` - all of ``T`` at once: every candidate route with
  its tube, every transition family with the region it can reach, plus ``x_0``.
* :func:`trajectory_map` - one route in detail: its tube corridor, the
  per-window visible footprints, and the imagery tiles those footprints cover.
* :func:`transition_map` - one transition family in detail: every sampled path,
  where each initiates on the source, and the region the family sweeps.
* :func:`manifest_map` - a built manifest drawn over its trajectory: the
  candidate roads and intersections actually pinned for each window.

Everything drawn is WGS84 (EPSG:4326), which is also what Leaflet expects, so
no reprojection happens in this module - the metric work (tube buffering, FOV
growth, tile selection) is already done by :mod:`csnav.trajectory.tube` and
:mod:`csnav.trajectory.coverage` before anything gets here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import Polygon

from csnav.data.arcgis.models import Extent, TileInfo
from csnav.data.arcgis.tiles import web_mercator_tile_info
from csnav.geometry.camera import Camera
from csnav.trajectory.config import ConopsConfig
from csnav.trajectory.coverage import (
    AglProvider,
    TileRef,
    height_as_agl,
    max_agl,
    max_ground_reach,
    merge_tiles,
    tiles_for_footprint,
    visible_footprint,
)
from csnav.trajectory.manifest import LandmarkManifest, ManifestBundle
from csnav.trajectory.trajectory import X0_NODE, Trajectory, TrajectorySet, TransitionRule
from csnav.trajectory.transition import TransitionFamily, TransitionModel
from csnav.trajectory.tube import TubeModel
from csnav.trajectory.waypoints import Waypoint
from csnav.viz.style import (
    INTERSECTION_COLOR,
    LANDMARK_COLOR,
    ROLE_LABELS,
    TILE_COLOR,
    TRANSITION_COLOR,
    color_for,
)

#: San Jose's own cached aerial imagery, as an XYZ-addressable overlay. This is
#: the same cache :class:`csnav.data.arcgis.client.ArcGISTileClient` fetches
#: training/observation tiles from (``/tile/{level}/{row}/{col}``), so what the
#: map shows under a corridor is what the pipeline will actually see.
DPW_IMAGERY_TILE_URL = (
    "https://geo.sanjoseca.gov/server/rest/services/Imagery/DPW_ImageryCached2025/MapServer/tile/{z}/{y}/{x}"
)
DPW_IMAGERY_ATTRIBUTION = "City of San Jose DPW (DPW_ImageryCached2025)"

_ESRI_WORLD_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_ESRI_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics"


def _folium():
    """Import folium, with an actionable message when the viz extra isn't installed."""
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "csnav.viz.map_view needs folium; install the visualization extra with "
            '`pip install -e ".[viz]"`'
        ) from exc
    return folium


def _polygon_latlon(polygon: Polygon) -> list[list[list[float]]]:
    """Shapely WGS84 (lon, lat) polygon -> folium's ``[[lat, lon], ...]`` rings, exterior first."""
    rings = [list(polygon.exterior.coords)] + [list(interior.coords) for interior in polygon.interiors]
    return [[[lat, lon] for lon, lat in ring] for ring in rings]


def _polygons(geometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon)]


def base_map(center: tuple[float, float], zoom: int = 13, include_imagery: bool = True):
    """A folium map centred on ``(lat, lon)`` with the basemaps this project uses.

    ``include_imagery`` adds San Jose's DPW cached imagery as a toggleable
    overlay, off by default in the layer control - it needs network access to
    ``geo.sanjoseca.gov``, so a map saved for offline review still opens
    cleanly without it.
    """
    folium = _folium()
    fmap = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(fmap)
    folium.TileLayer(
        tiles=_ESRI_WORLD_IMAGERY_URL, attr=_ESRI_ATTRIBUTION, name="Esri World Imagery", show=False
    ).add_to(fmap)
    if include_imagery:
        folium.TileLayer(
            tiles=DPW_IMAGERY_TILE_URL,
            attr=DPW_IMAGERY_ATTRIBUTION,
            name="San Jose DPW imagery (2025 cache)",
            max_zoom=21,
            show=False,
        ).add_to(fmap)
    return fmap


def _add_centerline(group, trajectory: Trajectory, color: str, weight: int = 4) -> None:
    folium = _folium()
    folium.PolyLine(
        [[wp.lat, wp.lon] for wp in trajectory.waypoints],
        color=color,
        weight=weight,
        opacity=0.95,
        tooltip=(
            f"{trajectory.id} ({ROLE_LABELS[trajectory.role]}) - "
            f"{trajectory.length:.0f} m, {trajectory.duration:.0f} s"
        ),
    ).add_to(group)


def _add_waypoints(group, trajectory: Trajectory, color: str) -> None:
    folium = _folium()
    for index, waypoint in enumerate(trajectory.waypoints):
        folium.CircleMarker(
            location=[waypoint.lat, waypoint.lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=1.0,
            weight=2,
            tooltip=(
                f"{trajectory.id} wp {index}<br>"
                f"lat {waypoint.lat:.5f}, lon {waypoint.lon:.5f}<br>"
                f"height {waypoint.height:.0f} m (ellipsoidal)<br>"
                f"t = {waypoint.time:.0f} s"
            ),
        ).add_to(group)


def _add_corridor(group, corridor: Polygon, color: str, tooltip: str, fill_opacity: float = 0.15) -> None:
    folium = _folium()
    for part in _polygons(corridor):
        folium.Polygon(
            locations=_polygon_latlon(part),
            color=color,
            weight=1.5,
            opacity=0.9,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            tooltip=tooltip,
        ).add_to(group)


def _add_x0(fmap, x0: Waypoint) -> None:
    folium = _folium()
    folium.Marker(
        location=[x0.lat, x0.lon],
        tooltip=(
            f"x_0 (known start state)<br>lat {x0.lat:.5f}, lon {x0.lon:.5f}<br>"
            f"height {x0.height:.0f} m, t = {x0.time:.0f} s"
        ),
        icon=folium.Icon(color="black", icon="plane", prefix="fa"),
    ).add_to(fmap)


def _add_tiles(group, tiles: Iterable[TileRef], color: str = TILE_COLOR) -> int:
    folium = _folium()
    count = 0
    for tile in tiles:
        bounds = tile.bounds
        folium.Rectangle(
            bounds=[[bounds.ymin, bounds.xmin], [bounds.ymax, bounds.xmax]],
            color=color,
            weight=0.8,
            opacity=0.7,
            fill=True,
            fill_color=color,
            fill_opacity=0.06,
            tooltip=f"imagery tile {tile.key}",
        ).add_to(group)
        count += 1
    return count


def _fit(fmap, bounds) -> None:
    fmap.fit_bounds([[bounds.ymin, bounds.xmin], [bounds.ymax, bounds.xmax]])


def _add_transition_family(
    group,
    family: TransitionFamily,
    source: Trajectory,
    tube,
    color: str,
) -> None:
    """Draw one transition family: its swept reachable region, then each sampled path."""
    folium = _folium()
    footprint = family.reachable_footprint(tube)
    if not footprint.is_empty:
        _add_corridor(
            group,
            footprint,
            color,
            (
                f"{family.rule.source} &#8594; {family.rule.target}<br>"
                f"reachable while transitioning (tube {tube.radius:.0f} m)<br>"
                f"{len(family)} sampled initiations across "
                f"{family.domain[0]:.0f}-{family.domain[1]:.0f} m of the source"
            ),
            fill_opacity=0.10,
        )
    for path in family.paths:
        origin = source.point_at(path.initiate_distance)
        folium.PolyLine(
            [[wp.lat, wp.lon] for wp in path.trajectory.waypoints],
            color=color,
            weight=2,
            opacity=0.85,
            dash_array="6,4",
            tooltip=(
                f"{path.source_id} &#8594; {path.target_id}<br>"
                f"initiates at {path.initiate_distance:.0f} m of arc length "
                f"(t = {origin.time:.0f} s)<br>"
                f"rejoins at waypoint {path.arrival_index} "
                f"({path.arrival_distance:.0f} m along the target)<br>"
                f"turns: {path.departure_turn:.0f}&#176; out, {path.arrival_turn:.0f}&#176; in"
            ),
        ).add_to(group)
        folium.CircleMarker(
            location=[origin.lat, origin.lon],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.9,
            weight=1,
            tooltip=f"transition initiates here ({path.initiate_distance:.0f} m along {path.source_id})",
        ).add_to(group)


def transition_families(
    trajectory_set: TrajectorySet, model: TransitionModel
) -> dict[tuple[str, str], TransitionFamily]:
    """Generate every transition family the set's rules admit, keyed by ``(source, target)``.

    Entry rules out of ``x_0`` are skipped: they mark a route as flyable from
    the start and carry no geometry.
    """
    families: dict[tuple[str, str], TransitionFamily] = {}
    for rule in trajectory_set.transitions:
        if rule.source == X0_NODE:
            continue
        families[rule.key] = model.family(
            trajectory_set.by_id(rule.source), trajectory_set.by_id(rule.target), rule
        )
    return families


def trajectory_set_map(
    trajectory_set: TrajectorySet,
    conops: ConopsConfig,
    agl_provider: AglProvider = height_as_agl,
    show_visible_footprint: bool = True,
    show_transitions: bool = True,
    include_imagery: bool = True,
):
    """Map the whole candidate set ``T``: every route with its tube, every transition family.

    Each route gets a toggleable layer holding its centerline, its waypoints,
    the tube corridor at the radius ``conops`` assigns it, and - when
    ``show_visible_footprint`` is set and a camera is configured - the wider
    footprint that tube can actually see.

    Each transition rule gets its own layer holding the *family* it admits: one
    dashed path per sampled initiation point, a marker where each begins on the
    source, and the shaded union of their tubes. That union is the point of the
    layer: a transition may begin anywhere along the source, so the aircraft may
    be anywhere in that region while transitioning, not only on one of the drawn
    curves.
    """
    folium = _folium()
    centre = trajectory_set.bounds
    fmap = base_map(
        ((centre.ymin + centre.ymax) / 2.0, (centre.xmin + centre.xmax) / 2.0),
        include_imagery=include_imagery,
    )
    order = tuple(t.id for t in trajectory_set.trajectories)

    for trajectory in trajectory_set.trajectories:
        tube = conops.tube_for(trajectory)
        color = color_for(trajectory.id, trajectory.role, order)
        group = folium.FeatureGroup(
            name=f"{trajectory.id} - {ROLE_LABELS[trajectory.role]} (tube {tube.radius:.0f} m)",
            show=True,
        )
        if show_visible_footprint and conops.camera is not None:
            _add_corridor(
                group,
                visible_footprint(trajectory, tube, camera=conops.camera, agl_provider=agl_provider),
                color,
                f"{trajectory.id}: tube {tube.radius:.0f} m + camera ground reach",
                fill_opacity=0.07,
            )
        _add_corridor(group, tube.corridor(trajectory), color, f"{trajectory.id}: tube radius {tube.radius:.0f} m")
        _add_centerline(group, trajectory, color)
        _add_waypoints(group, trajectory, color)
        group.add_to(fmap)

    if show_transitions:
        for (source_id, target_id), family in transition_families(
            trajectory_set, conops.transition
        ).items():
            source = trajectory_set.by_id(source_id)
            tube = conops.tube_for(family.paths[0].trajectory) if family.paths else conops.tube_for(source)
            group = folium.FeatureGroup(
                # Layer-control names go through JSON escaping, so an arrow -
                # entity or ASCII - comes back out as a literal escape sequence.
                name=f"{source_id} to {target_id}: {len(family)} transitions (tube {tube.radius:.0f} m)",
                show=False,
            )
            _add_transition_family(group, family, source, tube, TRANSITION_COLOR)
            group.add_to(fmap)

    _add_x0(fmap, trajectory_set.x0)
    folium.LayerControl(collapsed=False).add_to(fmap)
    _fit(fmap, trajectory_set.bounds)
    return fmap


def transition_map(
    trajectory_set: TrajectorySet,
    conops: ConopsConfig,
    rule: TransitionRule,
    include_imagery: bool = True,
):
    """One transition family in detail, over its source and target routes.

    Shows what the rule actually admits: where along the source each sampled
    hand-off begins, the curve it flies, the waypoint it rejoins at, and the
    region the whole family sweeps - which is where the aircraft may be while
    the transition is under way.
    """
    folium = _folium()
    source = trajectory_set.by_id(rule.source)
    target = trajectory_set.by_id(rule.target)
    family = conops.transition.family(source, target, rule)

    bounds = trajectory_set.bounds
    fmap = base_map(
        ((bounds.ymin + bounds.ymax) / 2.0, (bounds.xmin + bounds.xmax) / 2.0),
        include_imagery=include_imagery,
    )
    order = tuple(t.id for t in trajectory_set.trajectories)

    for trajectory, label in ((source, "source"), (target, "target")):
        color = color_for(trajectory.id, trajectory.role, order)
        tube = conops.tube_for(trajectory)
        group = folium.FeatureGroup(name=f"{label}: {trajectory.id} (tube {tube.radius:.0f} m)", show=True)
        _add_corridor(group, tube.corridor(trajectory), color, f"{trajectory.id}: tube {tube.radius:.0f} m")
        _add_centerline(group, trajectory, color)
        _add_waypoints(group, trajectory, color)
        group.add_to(fmap)

    transition_tube = conops.tube_for(family.paths[0].trajectory) if family.paths else conops.tube_for(source)
    group = folium.FeatureGroup(
        name=(
            f"transition family - {len(family)} paths, {family.rejected} screened out "
            f"(tube {transition_tube.radius:.0f} m)"
        ),
        show=True,
    )
    _add_transition_family(group, family, source, transition_tube, TRANSITION_COLOR)
    group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    _fit(fmap, trajectory_set.bounds)
    return fmap


def trajectory_map(
    trajectory: Trajectory,
    tube: TubeModel,
    window_length: float,
    camera: Camera | None = None,
    tile_info: TileInfo | None = None,
    tile_level: int | None = None,
    agl_provider: AglProvider = height_as_agl,
    include_imagery: bool = True,
    max_tiles: int = 20_000,
):
    """Map one trajectory in detail: tube radius, per-window footprints, and tiles in view.

    Layers, all independently toggleable:

    * the centerline and its waypoints;
    * the tube corridor at ``tube.radius`` meters;
    * one visible footprint per manifest window (``window_length`` meters of arc
      length each), each tooltipped with its window id, arc-length span, and
      the AGL and camera ground reach that sized it;
    * the imagery tiles those footprints cover, when ``tile_info``/``tile_level``
      are given.

    ``tile_level`` addresses the service's own cache levels. Pass the live
    service's :class:`~csnav.data.arcgis.models.TileInfo` where you have it;
    with ``tile_level`` set and ``tile_info`` left out, the standard EPSG:3857
    scheme (:func:`csnav.data.arcgis.tiles.web_mercator_tile_info`) is assumed,
    which is the scheme San Jose's caches are published against.
    """
    folium = _folium()
    if tile_level is not None and tile_info is None:
        tile_info = web_mercator_tile_info()
    color = color_for(trajectory.id, trajectory.role, (trajectory.id,))
    bounds = tube.envelope(trajectory)
    fmap = base_map(
        ((bounds.ymin + bounds.ymax) / 2.0, (bounds.xmin + bounds.xmax) / 2.0),
        include_imagery=include_imagery,
    )

    tube_group = folium.FeatureGroup(name=f"tube corridor ({tube.radius:.0f} m radius)", show=True)
    _add_corridor(tube_group, tube.corridor(trajectory), color, f"tube radius {tube.radius:.0f} m")
    tube_group.add_to(fmap)

    windows = trajectory.windows(window_length)
    footprints: list[Polygon] = []
    window_group = folium.FeatureGroup(
        name=f"visible footprint per window ({len(windows)} x {window_length:.0f} m)", show=True
    )
    for window in windows:
        footprint = visible_footprint(
            trajectory, tube, window=window, camera=camera, agl_provider=agl_provider
        )
        footprints.append(footprint)
        agl = max_agl(trajectory, window, agl_provider)
        extra = 0.0 if camera is None else max_ground_reach(trajectory, window, camera, agl_provider)
        _add_corridor(
            window_group,
            footprint,
            "#009E73",
            (
                f"window {window.window_id}<br>"
                f"arc {window.start_distance:.0f}-{window.end_distance:.0f} m, "
                f"t {window.start_time:.0f}-{window.end_time:.0f} s<br>"
                f"max AGL {agl:.0f} m, camera ground reach {extra:.0f} m<br>"
                f"search radius {tube.radius + extra:.0f} m"
            ),
            fill_opacity=0.08,
        )
    window_group.add_to(fmap)

    if tile_info is not None and tile_level is not None:
        tiles = _tiles_for_windows(footprints, tile_info, tile_level, max_tiles)
        tile_group = folium.FeatureGroup(name=f"imagery tiles in view (level {tile_level}, {len(tiles)})", show=True)
        _add_tiles(tile_group, tiles)
        tile_group.add_to(fmap)

    path_group = folium.FeatureGroup(name=f"{trajectory.id} centerline", show=True)
    _add_centerline(path_group, trajectory, color)
    _add_waypoints(path_group, trajectory, color)
    path_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    # Fit to the widest thing drawn - the FOV-grown footprint, not the bare tube.
    outer_margin = (
        0.0 if camera is None else max_ground_reach(trajectory, None, camera, agl_provider)
    )
    _fit(fmap, tube.envelope(trajectory, extra_buffer=outer_margin))
    return fmap


def _tiles_for_windows(
    footprints: list[Polygon], tile_info: TileInfo, tile_level: int, max_tiles: int
) -> tuple[TileRef, ...]:
    return merge_tiles(
        tiles_for_footprint(footprint, tile_info, tile_level, max_tiles=max_tiles) for footprint in footprints
    )


def manifest_map(
    trajectory: Trajectory,
    manifests: Iterable[LandmarkManifest],
    include_imagery: bool = True,
    show_tiles: bool = True,
):
    """Map a built manifest over its trajectory: candidate roads, intersections, tiles.

    ``manifests`` is that trajectory's window manifests (e.g.
    :meth:`csnav.trajectory.manifest.ManifestBundle.for_trajectory`). This is
    the "did the offline build pick up the right streets?" view - the pinned
    landmark set drawn where it actually sits, rather than a count in a log
    line.
    """
    folium = _folium()
    manifests = list(manifests)
    if not manifests:
        raise ValueError(f"no manifests supplied for trajectory {trajectory.id!r}")

    color = color_for(trajectory.id, trajectory.role, (trajectory.id,))
    bounds = trajectory.bounds
    fmap = base_map(
        ((bounds.ymin + bounds.ymax) / 2.0, (bounds.xmin + bounds.xmax) / 2.0),
        include_imagery=include_imagery,
    )

    footprint_group = folium.FeatureGroup(name="window footprints", show=True)
    roads_group = folium.FeatureGroup(name="candidate roads (manifest)", show=True)
    junction_group = folium.FeatureGroup(name="intersections (manifest)", show=True)
    tile_group = folium.FeatureGroup(name="imagery tiles in view", show=show_tiles)

    tile_count = 0
    for manifest in manifests:
        _add_corridor(
            footprint_group,
            manifest.footprint,
            "#009E73",
            (
                f"window {manifest.window_id}<br>"
                f"tube {manifest.tube_radius:.0f} m, max AGL {manifest.max_agl:.0f} m<br>"
                f"{len(manifest.candidate_roads)} roads, {len(manifest.intersections)} intersections"
            ),
            fill_opacity=0.06,
        )
        for road in manifest.candidate_roads:
            for part in road.parts:
                folium.PolyLine(
                    [[lat, lon] for lon, lat in part],
                    color=LANDMARK_COLOR,
                    weight=3,
                    opacity=0.9,
                    tooltip=(
                        f"{road.name or road.segment_id}<br>"
                        f"segment {road.segment_id}<br>"
                        f"off-track offset {road.offset:.0f} m"
                        + (f"<br>width {road.width:.1f} m" if road.width else "")
                    ),
                ).add_to(roads_group)
        for junction in manifest.intersections:
            folium.CircleMarker(
                location=[junction.lat, junction.lon],
                radius=3,
                color=INTERSECTION_COLOR,
                fill=True,
                fill_opacity=1.0,
                weight=1,
                tooltip=f"intersection of {', '.join(junction.segment_ids)}",
            ).add_to(junction_group)
        tile_count += _add_tiles(tile_group, manifest.tiles)

    footprint_group.add_to(fmap)
    tile_group.layer_name = f"imagery tiles in view ({tile_count})"
    tile_group.add_to(fmap)
    roads_group.add_to(fmap)
    junction_group.add_to(fmap)

    path_group = folium.FeatureGroup(name=f"{trajectory.id} centerline", show=True)
    _add_centerline(path_group, trajectory, color)
    _add_waypoints(path_group, trajectory, color)
    path_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    _fit(fmap, _envelope_of(manifests))
    return fmap


def _envelope_of(manifests: list[LandmarkManifest]) -> Extent:
    """Bounding box covering every supplied manifest's footprint, in EPSG:4326."""
    return Extent(
        xmin=min(manifest.envelope.xmin for manifest in manifests),
        ymin=min(manifest.envelope.ymin for manifest in manifests),
        xmax=max(manifest.envelope.xmax for manifest in manifests),
        ymax=max(manifest.envelope.ymax for manifest in manifests),
        wkid=4326,
    )


def bundle_map(
    trajectory_set: TrajectorySet,
    bundle: ManifestBundle,
    include_imagery: bool = True,
):
    """Map every trajectory's pinned manifest in one figure, one layer group per trajectory."""
    folium = _folium()
    bounds = trajectory_set.bounds
    fmap = base_map(
        ((bounds.ymin + bounds.ymax) / 2.0, (bounds.xmin + bounds.xmax) / 2.0),
        include_imagery=include_imagery,
    )
    order = tuple(t.id for t in trajectory_set.trajectories)

    for trajectory in trajectory_set.trajectories:
        manifests = bundle.for_trajectory(trajectory.id)
        if not manifests:
            continue
        color = color_for(trajectory.id, trajectory.role, order)
        group = folium.FeatureGroup(
            name=f"{trajectory.id} ({len(manifests)} windows, tube {manifests[0].tube_radius:.0f} m)",
            show=True,
        )
        for manifest in manifests:
            _add_corridor(
                group,
                manifest.footprint,
                color,
                f"{manifest.window_id}: {len(manifest.candidate_roads)} roads, {len(manifest.tiles)} tiles",
                fill_opacity=0.07,
            )
        _add_centerline(group, trajectory, color)
        group.add_to(fmap)

    _add_x0(fmap, trajectory_set.x0)
    folium.LayerControl(collapsed=False).add_to(fmap)
    _fit(fmap, trajectory_set.bounds)
    return fmap


def save_map(fmap: Any, path: str | Path) -> Path:
    """Write a folium map to a self-contained HTML file, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(destination))
    return destination
