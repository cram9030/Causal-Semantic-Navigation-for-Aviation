"""Offline manifest builder against a hand-checkable street layout.

Per CLAUDE.md's testing priorities: "given a trajectory + tube radius, does the
candidate landmark set match a hand-checked expectation for a small test
trajectory?"

The fixture streets are laid out in the trajectory's own ENU frame at exact
north offsets - 0, 150, 199, 260 and 5000 meters from a due-east centerline -
so each street's cross-track distance is known by construction, and which ones
belong in a 200 m tube is decidable by reading that list rather than by
re-running the builder's geometry. (Laying them out in ENU matters: a
two-vertex line between two equal-*latitude* points is not a constant-offset
line, it sags a meter or two toward the equator over a few kilometres.)
"""

from __future__ import annotations

import pytest
from shapely.geometry import Point as ShapelyPoint

from csnav.data.arcgis.streets import StreetSegment
from csnav.data.arcgis.tiles import web_mercator_tile_info
from csnav.geometry.local_frame import LocalFrame
from csnav.trajectory.config import ConopsConfig
from csnav.trajectory.manifest_builder import ManifestBuilder, StaticStreetsSource
from csnav.trajectory.tube import TubeModel

from .conftest import ORIGIN_LAT, ORIGIN_LON

#: North offsets, in meters, of the parallel test streets. With a 200 m tube
#: and no field of view, exactly the first three belong in the manifest.
STREET_OFFSETS = (0.0, 150.0, 199.0, 260.0, 5_000.0)
INSIDE_A_200M_TUBE = (0.0, 150.0, 199.0)

#: East span of the parallel streets, in meters - well past both ends of the
#: ~2 km trajectory, so clipping has something to cut.
STREET_EAST_SPAN = (-4_000.0, 6_000.0)

#: East position of the crossing street, in meters - near the midpoint.
CROSSING_EAST = 1_000.0

_FRAME = LocalFrame(ORIGIN_LAT, ORIGIN_LON)


def _enu_line(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """ENU ``(east, north)`` meters -> a WGS84 ``(lon, lat)`` vertex tuple."""
    return tuple((latlon.lon, latlon.lat) for latlon in (_FRAME.to_wgs84(east, north) for east, north in points))


def _parallel_street(offset: float, object_id: int) -> StreetSegment:
    """A long due-east street exactly ``offset`` meters north of the trajectory."""
    east_min, east_max = STREET_EAST_SPAN
    return StreetSegment(
        object_id=object_id,
        parts=(_enu_line([(east_min, offset), (east_max, offset)]),),
        attributes={"OBJECTID": object_id, "STREETNAME": f"Parallel {offset:.0f}", "WIDTH": 40},
    )


def _crossing_street(object_id: int = 99) -> StreetSegment:
    """A due-north street crossing the trajectory near its midpoint."""
    return StreetSegment(
        object_id=object_id,
        parts=(_enu_line([(CROSSING_EAST, -1_000.0), (CROSSING_EAST, 1_000.0)]),),
        attributes={"OBJECTID": object_id, "STREETNAME": "Crossing"},
    )


@pytest.fixture
def streets() -> StaticStreetsSource:
    segments = [_parallel_street(offset, index + 1) for index, offset in enumerate(STREET_OFFSETS)]
    segments.append(_crossing_street())
    return StaticStreetsSource(segments, source_label="test-fixture")


@pytest.fixture
def builder(streets) -> ManifestBuilder:
    return ManifestBuilder(streets=streets)


def test_manifest_holds_exactly_the_streets_inside_the_tube(builder, due_east):
    """No field of view: the manifest is the tube's own reach, nothing wider."""
    manifests = builder.build_trajectory(due_east, TubeModel(radius=200.0), window_length=10_000.0)
    assert len(manifests) == 1

    names = {road.name for road in manifests[0].candidate_roads}
    assert names == {f"Parallel {offset:.0f}" for offset in INSIDE_A_200M_TUBE} | {"Crossing"}


def test_a_wider_tube_admits_more_streets(builder, due_east):
    """The radius is the only thing that changed - the manifest grows with it."""
    narrow = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    wide = builder.build_trajectory(due_east, TubeModel(radius=300.0), 10_000.0)[0]

    assert len(wide.candidate_roads) == len(narrow.candidate_roads) + 1
    assert "Parallel 260" in {road.name for road in wide.candidate_roads}
    assert narrow.tube_radius == 200.0 and wide.tube_radius == 300.0


def test_the_camera_extends_the_manifest_beyond_the_tube(builder, due_east, camera):
    """The 260 m street is outside a 200 m tube but visible from inside it."""
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0, camera=camera)[0]

    assert "Parallel 260" in {road.name for road in manifest.candidate_roads}
    assert "Parallel 5000" not in {road.name for road in manifest.candidate_roads}
    assert manifest.max_agl == pytest.approx(300.0)
    assert manifest.ground_reach == pytest.approx(camera.ground_reach(300.0))


def test_a_manifest_built_without_a_camera_records_no_ground_reach(builder, due_east):
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    assert manifest.ground_reach == 0.0


def test_camera_settings_travel_with_the_pinned_bundle(streets, trajectory_set, conops):
    """A pinned manifest has to say what saw what, not just how wide the tube was."""
    bundle = ManifestBuilder(streets=streets).build_set(trajectory_set, conops)
    recorded = bundle.parameters["camera"]

    assert recorded["horizontal_deg"] == conops.camera.field_of_view.horizontal_deg
    assert recorded["pose_pitch_deg"] == conops.camera.pose.pitch_deg
    assert recorded["maneuver_radius_m"] == conops.camera.attitude_margin.maneuver_radius


def test_recorded_offsets_match_the_streets_true_cross_track_distance(builder, due_east):
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    by_name = {road.name: road for road in manifest.candidate_roads}
    for offset in INSIDE_A_200M_TUBE:
        assert by_name[f"Parallel {offset:.0f}"].offset == pytest.approx(offset, abs=1.0)


def test_streets_are_clipped_to_the_window_footprint(builder, due_east):
    """A city-long arterial contributes only the stretch this window can see."""
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    centerline = next(road for road in manifest.candidate_roads if road.name == "Parallel 0")
    lons = [lon for part in centerline.parts for lon, _ in part]

    # The source street spans 10 km east-west; the corridor is ~2.4 km.
    assert max(lons) - min(lons) < 0.03
    assert manifest.footprint.buffer(1e-9).contains(centerline.geometry())


def test_streets_outside_every_window_are_absent(builder, due_east):
    manifests = builder.build_trajectory(due_east, TubeModel(radius=200.0), 500.0)
    assert all(
        "Parallel 5000" not in {road.name for road in manifest.candidate_roads} for manifest in manifests
    )


def test_intersections_are_found_where_streets_actually_cross(builder, due_east):
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    crossing_id = str(_crossing_street().object_id)
    junctions = [j for j in manifest.intersections if crossing_id in j.segment_ids]

    # The north-south street crosses each of the three in-tube parallels once.
    assert len(junctions) == 3
    expected_lon = _FRAME.to_wgs84(CROSSING_EAST, 0.0).lon
    for junction in junctions:
        assert junction.lon == pytest.approx(expected_lon, abs=1e-6)


def test_widths_are_converted_from_feet_to_meters(builder, due_east):
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    parallel = next(road for road in manifest.candidate_roads if road.name == "Parallel 0")
    assert parallel.width == pytest.approx(40 * 0.3048)


def test_missing_width_attribute_yields_none(builder, due_east):
    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    crossing = next(road for road in manifest.candidate_roads if road.name == "Crossing")
    assert crossing.width is None


def test_windows_partition_the_manifests_and_keep_stable_ids(builder, due_east):
    manifests = builder.build_trajectory(due_east, TubeModel(radius=200.0), 500.0)
    assert [manifest.window_id for manifest in manifests] == [
        f"due_east:{index:04d}" for index in range(len(manifests))
    ]
    assert all(manifest.tube_radius == 200.0 for manifest in manifests)


def test_per_window_query_gives_the_same_manifest_as_one_shared_query(builder, due_east):
    shared = builder.build_trajectory(due_east, TubeModel(radius=200.0), 700.0)
    per_window = builder.build_trajectory(due_east, TubeModel(radius=200.0), 700.0, per_window_query=True)
    assert [sorted(r.segment_id for r in m.candidate_roads) for m in shared] == [
        sorted(r.segment_id for r in m.candidate_roads) for m in per_window
    ]


def test_tiles_are_recorded_only_when_a_tile_scheme_is_supplied(streets, due_east):
    without = ManifestBuilder(streets=streets).build_trajectory(due_east, TubeModel(200.0), 10_000.0)[0]
    assert without.tiles == ()

    with_tiles = ManifestBuilder(
        streets=streets, tile_info=web_mercator_tile_info(), tile_level=16
    ).build_trajectory(due_east, TubeModel(200.0), 10_000.0)[0]
    assert with_tiles.tiles
    assert all(tile.level == 16 for tile in with_tiles.tiles)


def test_tile_scheme_arguments_must_be_supplied_together(streets):
    with pytest.raises(ValueError, match="must be supplied together"):
        ManifestBuilder(streets=streets, tile_level=16)


def test_build_set_covers_every_route_and_records_its_parameters(streets, trajectory_set, conops):
    builder = ManifestBuilder(streets=streets, tile_info=web_mercator_tile_info(), tile_level=16)
    bundle = builder.build_set(trajectory_set, conops)

    covered = {manifest.window.trajectory_id for manifest in bundle.manifests}
    for trajectory in trajectory_set.trajectories:
        assert trajectory.id in covered
    assert bundle.parameters["tube_radius_m"] == conops.tube_radius
    assert bundle.parameters["transition_tube_radius_m"] == conops.transition_tube_radius
    assert bundle.parameters["window_length_m"] == conops.window_length
    assert bundle.parameters["tile_level"] == 16
    assert bundle.streets_source == "test-fixture"
    assert bundle.pinned_at


def test_build_set_covers_the_candidate_routes_at_the_conops_radius(streets, trajectory_set, conops):
    bundle = ManifestBuilder(streets=streets).build_set(trajectory_set, conops)
    route_manifests = [
        m
        for m in bundle.manifests
        if m.window.trajectory_id in {t.id for t in trajectory_set.trajectories}
    ]
    assert route_manifests
    assert all(manifest.tube_radius == conops.tube_radius for manifest in route_manifests)


def test_build_set_also_covers_every_transition_family_by_default(streets, trajectory_set, conops):
    """This is the point of the change: transitions get manifests too, not just candidates."""
    bundle = ManifestBuilder(streets=streets).build_set(trajectory_set, conops)

    transition_manifests = bundle.for_transition("due_east", "parallel_north")
    assert transition_manifests
    # Every sampled path in the family contributed at least one window.
    family = conops.transition.family(
        trajectory_set.by_id("due_east"),
        trajectory_set.by_id("parallel_north"),
        trajectory_set.transitions[1],
    )
    assert len(bundle.transition_path_ids("due_east", "parallel_north")) == len(family)
    assert all(manifest.tube_radius == conops.transition_tube_radius for manifest in transition_manifests)


def test_build_set_can_skip_transitions(streets, trajectory_set, conops):
    bundle = ManifestBuilder(streets=streets).build_set(trajectory_set, conops, include_transitions=False)
    assert bundle.for_transition("due_east", "parallel_north") == ()


def test_build_set_skips_an_empty_family_without_raising(streets, due_east, orthogonal, model):
    from csnav.trajectory.trajectory import TrajectorySet, TransitionRule
    from csnav.trajectory.transition import TransitionModel

    trajectory_set = TrajectorySet(
        id="degenerate",
        trajectories=(due_east, orthogonal),
        primary_id="due_east",
        x0=due_east.waypoints[0],
        transitions=(TransitionRule(source="due_east", target="orthogonal", max_turn_deg=1.0),),
    )
    conops = ConopsConfig(
        tube_radius=200.0, window_length=1000.0, transition=TransitionModel(samples=5, max_turn_deg=1.0)
    )
    bundle = ManifestBuilder(streets=streets).build_set(trajectory_set, conops)

    assert bundle.for_transition("due_east", "orthogonal") == ()
    assert bundle.for_trajectory("due_east")  # the candidates are still covered


def test_runtime_lookup_does_not_touch_the_street_source(builder, due_east):
    """The 'possible roads' step is a manifest lookup, never a live spatial query."""

    class Exploding:
        def query(self, *args, **kwargs):
            raise AssertionError("the runtime must not query CSJ Streets")

    manifest = builder.build_trajectory(due_east, TubeModel(radius=200.0), 10_000.0)[0]
    builder.streets = Exploding()

    midpoint = _FRAME.to_wgs84(CROSSING_EAST, 0.0)
    fov_disc = ShapelyPoint(midpoint.lon, midpoint.lat).buffer(0.002)
    subset = manifest.query(fov_disc)

    assert 0 < len(subset) <= len(manifest.candidate_roads)
    assert all(road.geometry().intersects(fov_disc) for road in subset)


def test_static_source_rejects_a_non_wgs84_bbox(streets):
    from csnav.data.arcgis.models import Extent

    with pytest.raises(ValueError, match="must be EPSG:4326"):
        streets.query(bbox=Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857))
