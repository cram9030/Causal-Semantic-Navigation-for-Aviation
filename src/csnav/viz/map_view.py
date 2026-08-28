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

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from csnav.trajectory.trajectory import (
    X0_NODE,
    Trajectory,
    TrajectorySet,
    TrajectoryWindow,
    TransitionRule,
)
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
    window_shade,
)
from csnav.viz.window_selector import WindowCategory, WindowGroup, WindowLayers, WindowSelector

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


def _add_corridor(
    group,
    corridor: Polygon,
    color: str,
    tooltip: str,
    fill_opacity: float = 0.15,
    dash_array: str | None = None,
) -> None:
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
            dash_array=dash_array,
            tooltip=tooltip,
        ).add_to(group)


def _window_style(index: int) -> dict[str, Any]:
    """Fill/outline style that alternates between consecutive windows.

    Adjacent windows overlap at their shared boundary and each corridor is
    round-capped, so a uniform style makes a run of windows read as one chain of
    blobs. Alternating the fill weight and dashing the odd ones keeps the
    sequence separable even with all of them shown - a per-window colour is not
    available where the colour already means "which trajectory".
    """
    return (
        {"fill_opacity": 0.16, "dash_array": None}
        if index % 2 == 0
        else {"fill_opacity": 0.05, "dash_array": "7,5"}
    )


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


def _add_transition_paths(group, family: TransitionFamily, source: Trajectory, color: str) -> None:
    """Draw each sampled path in a family: a dashed curve plus a marker at its initiation point."""
    folium = _folium()
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


def _add_transition_family(
    group,
    family: TransitionFamily,
    source: Trajectory,
    tube,
    color: str,
) -> None:
    """Draw one transition family: its swept reachable region, then each sampled path."""
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
    _add_transition_paths(group, family, source, color)


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


def _window_label(window: TrajectoryWindow) -> str:
    """Compact row label for the window selector: index and arc-length span."""
    return f"{window.index:04d} \u00b7 {window.start_distance:,.0f}-{window.end_distance:,.0f} m"


#: Pulls the initiation arc length back out of a generated transition path id
#: ("<source>__<target>__sNNNNN.N", from csnav.trajectory.transition.transition_id).
_TRANSITION_PATH_ID = re.compile(r"__s(\d+(?:\.\d+)?)$")


def _transition_window_label(window: TrajectoryWindow) -> str:
    """Row label for a transition-family window selector: which sampled path, and where in it.

    A transition family has no single arc-length origin the way a candidate
    route does - each sampled path starts fresh - so the label leads with the
    path's own initiation point (where along the source it begins) rather than
    a window index that would mean nothing on its own.
    """
    match = _TRANSITION_PATH_ID.search(window.trajectory_id)
    initiation = f"init {float(match.group(1)):,.0f} m" if match else window.trajectory_id
    return f"{initiation} \u00b7 win {window.index:04d} \u00b7 {window.start_distance:,.0f}-{window.end_distance:,.0f} m"


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

    The tube corridor and the centerline sit in folium's own layer control. The
    windows do not: each gets its own layer, managed by a
    :class:`~csnav.viz.window_selector.WindowSelector` panel so a single window
    can be isolated. Drawn all at once they overlap at every shared boundary,
    which is why they also alternate between two shades.

    Each window's tooltip carries its window id, arc-length span, and the AGL
    and camera ground reach that sized it. ``tile_level`` addresses the
    service's own cache levels; pass the live service's
    :class:`~csnav.data.arcgis.models.TileInfo` where you have it, and with
    ``tile_level`` set and ``tile_info`` left out the standard EPSG:3857 scheme
    (:func:`csnav.data.arcgis.tiles.web_mercator_tile_info`) is assumed, which
    is the scheme San Jose's caches are published against.
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
    window_layers: list[WindowLayers] = []

    for window in windows:
        footprint = visible_footprint(
            trajectory, tube, window=window, camera=camera, agl_provider=agl_provider
        )
        footprints.append(footprint)
        agl = max_agl(trajectory, window, agl_provider)
        extra = 0.0 if camera is None else max_ground_reach(trajectory, window, camera, agl_provider)
        detail = (
            f"arc {window.start_distance:.0f}-{window.end_distance:.0f} m, "
            f"max AGL {agl:.0f} m, search radius {tube.radius + extra:.0f} m"
        )

        footprint_group = folium.FeatureGroup(name=window.window_id, control=False, show=True)
        _add_corridor(
            footprint_group,
            footprint,
            window_shade(window.index),
            (
                f"window {window.window_id}<br>"
                f"arc {window.start_distance:.0f}-{window.end_distance:.0f} m, "
                f"t {window.start_time:.0f}-{window.end_time:.0f} s<br>"
                f"max AGL {agl:.0f} m, camera ground reach {extra:.0f} m<br>"
                f"search radius {tube.radius + extra:.0f} m"
            ),
            **_window_style(window.index),
        )
        footprint_group.add_to(fmap)
        layers: dict[str, Any] = {"footprint": footprint_group}

        if tile_info is not None and tile_level is not None:
            tile_group = folium.FeatureGroup(
                name=f"{window.window_id} tiles", control=False, show=True
            )
            _add_tiles(tile_group, tiles_for_footprint(footprint, tile_info, tile_level, max_tiles=max_tiles))
            tile_group.add_to(fmap)
            layers["tiles"] = tile_group

        window_layers.append(
            WindowLayers(window_id=window.window_id, label=_window_label(window), layers=layers, detail=detail)
        )

    categories = [WindowCategory("footprint", f"footprints ({len(windows)})")]
    if tile_info is not None and tile_level is not None:
        distinct = len(_tiles_for_windows(footprints, tile_info, tile_level, max_tiles))
        categories.append(WindowCategory("tiles", f"tiles L{tile_level} ({distinct} distinct)"))

    path_group = folium.FeatureGroup(name=f"{trajectory.id} centerline", show=True)
    _add_centerline(path_group, trajectory, color)
    _add_waypoints(path_group, trajectory, color)
    path_group.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    WindowSelector(
        groups=[WindowGroup(id=trajectory.id, label=trajectory.id, color=color, windows=tuple(window_layers))],
        categories=categories,
        title=f"Windows ({window_length:.0f} m)",
    ).add_to(fmap)

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


def _add_manifest_roads(group, manifest: LandmarkManifest) -> None:
    folium = _folium()
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
                    f"window {manifest.window_id}<br>"
                    f"off-track offset {road.offset:.0f} m"
                    + (f"<br>width {road.width:.1f} m" if road.width else "")
                ),
            ).add_to(group)


def _add_manifest_intersections(group, manifest: LandmarkManifest) -> None:
    folium = _folium()
    for junction in manifest.intersections:
        folium.CircleMarker(
            location=[junction.lat, junction.lon],
            radius=3,
            color=INTERSECTION_COLOR,
            fill=True,
            fill_opacity=1.0,
            weight=1,
            tooltip=f"intersection of {', '.join(junction.segment_ids)} ({manifest.window_id})",
        ).add_to(group)


def _manifest_window_layers(
    fmap,
    manifest: LandmarkManifest,
    color: str,
    categories: Sequence[str],
    shown: Mapping[str, bool],
    label_fn: Any = _window_label,
) -> WindowLayers:
    """Build one window's per-category layers and register them on the map.

    Layers are created with ``control=False``: they belong to the
    :class:`~csnav.viz.window_selector.WindowSelector`, not to folium's flat
    layer control, which is where 36 window layers would be unreadable.

    ``label_fn`` picks the row label: a candidate route's window means
    something by its index alone (:func:`_window_label`); a transition
    family's does not, since every sampled path starts its own arc length at
    zero, so :func:`_transition_window_label` leads with which path it is.
    """
    folium = _folium()
    window = manifest.window
    layers: dict[str, Any] = {}

    for category in categories:
        group = folium.FeatureGroup(
            name=f"{manifest.window_id} {category}", control=False, show=shown.get(category, True)
        )
        if category == "footprint":
            _add_corridor(
                group,
                manifest.footprint,
                color,
                (
                    f"window {manifest.window_id}<br>"
                    f"tube {manifest.tube_radius:.0f} m, max AGL {manifest.max_agl:.0f} m, "
                    f"camera reach {manifest.ground_reach:.0f} m<br>"
                    f"{len(manifest.candidate_roads)} roads, "
                    f"{len(manifest.intersections)} intersections, {len(manifest.tiles)} tiles"
                ),
                **_window_style(window.index),
            )
        elif category == "roads":
            _add_manifest_roads(group, manifest)
        elif category == "intersections":
            _add_manifest_intersections(group, manifest)
        elif category == "tiles":
            _add_tiles(group, manifest.tiles)
        else:
            raise ValueError(f"unknown manifest layer category: {category!r}")
        group.add_to(fmap)
        layers[category] = group

    return WindowLayers(
        window_id=manifest.window_id,
        label=label_fn(window),
        layers=layers,
        detail=(
            f"arc {window.start_distance:.0f}-{window.end_distance:.0f} m, "
            f"{len(manifest.candidate_roads)} roads, {len(manifest.intersections)} intersections"
        ),
    )


def manifest_map(
    trajectory: Trajectory,
    manifests: Iterable[LandmarkManifest],
    include_imagery: bool = True,
    show_tiles: bool = False,
):
    """Map a built manifest over its trajectory: candidate roads, intersections, tiles.

    ``manifests`` is that trajectory's window manifests (e.g.
    :meth:`csnav.trajectory.manifest.ManifestBundle.for_trajectory`). This is
    the "did the offline build pick up the right streets?" view - the pinned
    landmark set drawn where it actually sits, rather than a count in a log
    line.

    Everything is per window rather than pooled by kind, and a
    :class:`~csnav.viz.window_selector.WindowSelector` panel drives it: solo a
    window to see just its footprint and just the landmarks pinned for it. The
    category checkboxes at the top of that panel cut across windows, so
    "footprints only" and "every window's roads" are both one click. Imagery
    tiles are a category too, off by default - one window's tile set is a lot of
    rectangles.
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

    categories = ["footprint", "roads", "intersections", "tiles"]
    shown = {"footprint": True, "roads": True, "intersections": True, "tiles": show_tiles}
    window_layers = [
        _manifest_window_layers(fmap, manifest, window_shade(manifest.window.index), categories, shown)
        for manifest in manifests
    ]

    path_group = folium.FeatureGroup(name=f"{trajectory.id} centerline", show=True)
    _add_centerline(path_group, trajectory, color)
    _add_waypoints(path_group, trajectory, color)
    path_group.add_to(fmap)

    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    WindowSelector(
        groups=[WindowGroup(id=trajectory.id, label=trajectory.id, color=color, windows=tuple(window_layers))],
        categories=[
            WindowCategory("footprint", "footprints"),
            WindowCategory("roads", f"roads ({sum(len(m.candidate_roads) for m in manifests)})"),
            WindowCategory("intersections", f"junctions ({sum(len(m.intersections) for m in manifests)})"),
            WindowCategory("tiles", f"tiles ({len(merge_tiles(m.tiles for m in manifests))})", enabled=show_tiles),
        ],
        title=f"Manifest windows ({len(manifests)})",
    ).add_to(fmap)

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
    show_landmarks: bool = False,
    transition_model: TransitionModel | None = None,
):
    """Map every pinned manifest in one figure, window by window - candidate routes and transitions alike.

    Centerlines (routes) and sampled paths (transitions) stay in folium's own
    layer control, one per route and one per transition rule. The manifest
    windows are managed by a
    :class:`~csnav.viz.window_selector.WindowSelector` panel instead: expand a
    route or a transition rule to get its windows, tick the ones you want, or
    solo one. Windows overlap at every shared boundary, so with all of them
    shown at once the fills alternate between two opacities to keep the
    sequence readable.

    A transition family has no single arc-length origin the way a route does -
    each sampled path starts fresh - so its window rows lead with which path
    they belong to (where it initiates on the source), via
    :func:`_transition_window_label`.

    ``transition_model`` draws each transition rule's sampled paths as dashed
    curves with initiation markers, the same way :func:`trajectory_set_map`
    does; pass the scenario's ``conops.transition`` for that context. Leave it
    out to show only the pinned manifest windows without regenerating the
    family's geometry - the manifest itself never depended on having it.

    ``show_landmarks`` adds each window's candidate roads and intersections as
    further categories, off by default - across a whole bundle that is a lot of
    geometry, and :func:`manifest_map` is the view for inspecting one
    trajectory's landmarks closely.
    """
    folium = _folium()
    bounds = trajectory_set.bounds
    fmap = base_map(
        ((bounds.ymin + bounds.ymax) / 2.0, (bounds.xmin + bounds.xmax) / 2.0),
        include_imagery=include_imagery,
    )
    order = tuple(t.id for t in trajectory_set.trajectories)
    categories = ["footprint"] + (["roads", "intersections"] if show_landmarks else [])
    shown = {"footprint": True, "roads": False, "intersections": False}

    groups: list[WindowGroup] = []
    for trajectory in trajectory_set.trajectories:
        manifests = bundle.for_trajectory(trajectory.id)
        if not manifests:
            continue
        color = color_for(trajectory.id, trajectory.role, order)

        path_group = folium.FeatureGroup(
            name=f"{trajectory.id} ({len(manifests)} windows, tube {manifests[0].tube_radius:.0f} m)",
            show=True,
        )
        _add_centerline(path_group, trajectory, color)
        path_group.add_to(fmap)

        groups.append(
            WindowGroup(
                id=trajectory.id,
                label=trajectory.id,
                color=color,
                windows=tuple(
                    _manifest_window_layers(fmap, manifest, color, categories, shown)
                    for manifest in manifests
                ),
            )
        )

    for rule in trajectory_set.transitions:
        if rule.source == X0_NODE:
            continue
        manifests = bundle.for_transition(rule.source, rule.target)
        if not manifests:
            continue
        path_ids = bundle.transition_path_ids(rule.source, rule.target)
        rule_label = f"{rule.source} to {rule.target}"

        path_group = folium.FeatureGroup(
            name=(
                f"{rule_label} ({len(path_ids)} paths, {len(manifests)} windows, "
                f"tube {manifests[0].tube_radius:.0f} m)"
            ),
            show=False,
        )
        if transition_model is not None:
            source = trajectory_set.by_id(rule.source)
            family = transition_model.family(source, trajectory_set.by_id(rule.target), rule)
            _add_transition_paths(path_group, family, source, TRANSITION_COLOR)
        path_group.add_to(fmap)

        groups.append(
            WindowGroup(
                id=f"{rule.source}__{rule.target}",
                label=rule_label,
                color=TRANSITION_COLOR,
                windows=tuple(
                    _manifest_window_layers(
                        fmap, manifest, TRANSITION_COLOR, categories, shown, label_fn=_transition_window_label
                    )
                    for manifest in manifests
                ),
            )
        )

    selector_categories = [WindowCategory("footprint", "footprints")]
    if show_landmarks:
        selector_categories += [
            WindowCategory("roads", "candidate roads", enabled=False),
            WindowCategory("intersections", "intersections", enabled=False),
        ]

    _add_x0(fmap, trajectory_set.x0)
    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    WindowSelector(
        groups=groups,
        categories=selector_categories,
        title=f"Manifest windows ({sum(len(g.windows) for g in groups)})",
    ).add_to(fmap)
    _fit(fmap, trajectory_set.bounds)
    return fmap


def save_map(fmap: Any, path: str | Path) -> Path:
    """Write a folium map to a self-contained HTML file, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(destination))
    return destination
