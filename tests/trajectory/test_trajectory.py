"""Trajectory geometry: arc length, interpolation, and window discretization.

Arc length and interpolated positions are checked against ``pyproj.Geod`` -
an independent ellipsoidal computation - rather than against a round-trip
through the code under test, matching how ``tests/geometry/test_local_frame.py``
validates the ENU frame these build on.
"""

from __future__ import annotations

import math

import pytest
from pyproj import Geod

from csnav.trajectory.trajectory import (
    X0_NODE,
    Trajectory,
    TrajectoryError,
    TrajectorySet,
    TransitionRule,
)
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint

from .conftest import ORIGIN_LAT, ORIGIN_LON

_GEOD = Geod(ellps="WGS84")


def test_arc_length_at_ground_level_matches_independent_geodesic():
    """At zero height, arc length is the ellipsoidal geodesic distance."""
    ground = Trajectory(
        id="ground",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, height=0.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.0226, height=0.0, time=100.0),
        ),
    )
    _, _, expected = _GEOD.inv(ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 0.0226, ORIGIN_LAT)
    assert ground.length == pytest.approx(expected, abs=0.05)


def test_arc_length_is_the_distance_actually_flown_not_the_ground_track(due_east):
    """A leg flown at altitude is slightly longer than its ground track.

    Arc length is the 3D path length in the ENU frame, so the same angular
    span at 300 m covers ~(R+h)/R more distance than at the surface. The
    difference is small (~9 cm over 2 km) but it is real, and it is the
    quantity a constant-velocity predictor should advance along.
    """
    _, _, ground_track = _GEOD.inv(ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 0.0226, ORIGIN_LAT)
    scale = 1.0 + due_east.waypoints[0].height / 6_371_000.0
    assert due_east.length > ground_track
    assert due_east.length == pytest.approx(ground_track * scale, abs=0.05)


def test_cumulative_distances_start_at_zero_and_are_monotonic(dogleg):
    distances = dogleg.cumulative_distances
    assert distances[0] == 0.0
    assert distances == tuple(sorted(distances))
    assert distances[-1] == pytest.approx(dogleg.length)


def test_point_at_midpoint_is_halfway_along_the_leg(due_east):
    midpoint = due_east.point_at(due_east.length / 2.0)
    _, _, from_start = _GEOD.inv(
        due_east.waypoints[0].lon, due_east.waypoints[0].lat, midpoint.lon, midpoint.lat
    )
    assert from_start == pytest.approx(due_east.length / 2.0, abs=0.05)
    assert midpoint.time == pytest.approx(50.0)


def test_point_at_clamps_outside_the_trajectory(due_east):
    assert due_east.point_at(-500.0).lon == pytest.approx(due_east.waypoints[0].lon, abs=1e-9)
    assert due_east.point_at(due_east.length * 10).lon == pytest.approx(due_east.waypoints[-1].lon, abs=1e-9)


def test_point_at_time_and_distance_at_time_agree(dogleg):
    distance = dogleg.distance_at_time(150.0)
    assert dogleg.point_at_time(150.0) == dogleg.point_at(distance)
    # 150 s is halfway through the second leg, which starts at the corner.
    corner_distance = dogleg.cumulative_distances[1]
    assert distance == pytest.approx(corner_distance + (dogleg.length - corner_distance) / 2.0, abs=0.5)


def test_windows_tile_the_trajectory_without_gaps(due_east):
    windows = due_east.windows(500.0)
    assert windows[0].start_distance == 0.0
    assert windows[-1].end_distance == pytest.approx(due_east.length)
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.end_distance == pytest.approx(later.start_distance)


def test_windows_absorb_a_short_remainder_instead_of_emitting_a_sliver(due_east):
    # ~2 km at a 1200 m window length: two even windows, not 1200 m + a 300 m sliver.
    windows = due_east.windows(1200.0)
    assert len(windows) == 2
    assert windows[0].length == pytest.approx(windows[1].length)


def test_window_ids_are_stable_and_sorted(due_east):
    ids = [window.window_id for window in due_east.windows(500.0)]
    assert ids == sorted(ids)
    assert ids[0] == "due_east:0000"
    assert [window.window_id for window in due_east.windows(500.0)] == ids


def test_window_for_distance_selects_the_containing_window(due_east):
    window = due_east.window_for_distance(1100.0, 500.0)
    assert window.contains_distance(1100.0)
    assert window.index == 2


def test_trajectory_shorter_than_a_window_yields_exactly_one(due_east):
    assert len(due_east.windows(10_000.0)) == 1


def test_segment_keeps_interior_corner_vertices(dogleg):
    corner_distance = dogleg.cumulative_distances[1]
    span = dogleg.segment(corner_distance - 200.0, corner_distance + 200.0)
    assert len(span) == 3
    assert span[1].lon == pytest.approx(dogleg.waypoints[1].lon)
    assert span[1].lat == pytest.approx(dogleg.waypoints[1].lat)


def test_sample_spacing_covers_the_whole_trajectory(due_east):
    samples = due_east.sample(250.0)
    assert samples[0].lon == pytest.approx(due_east.waypoints[0].lon)
    assert samples[-1].lon == pytest.approx(due_east.waypoints[-1].lon)
    assert len(samples) >= due_east.length / 250.0


def test_trajectory_rejects_a_single_waypoint():
    with pytest.raises(TrajectoryError, match="at least 2 waypoints"):
        Trajectory(id="stub", waypoints=(Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON),))


def test_trajectory_rejects_non_monotonic_times():
    with pytest.raises(TrajectoryError, match="non-monotonic"):
        Trajectory(
            id="backwards",
            waypoints=(
                Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, time=10.0),
                Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.01, time=5.0),
            ),
        )


def test_set_graph_has_a_node_per_route_plus_x0(trajectory_set):
    graph = trajectory_set.to_networkx()
    assert set(graph.nodes) == {X0_NODE, "due_east", "parallel_north"}
    assert graph.graph["primary"] == "due_east"


def test_set_graph_edges_carry_the_rule_not_geometry(trajectory_set):
    """Transitions are not known before flight, so the edge holds a rule, not a path."""
    graph = trajectory_set.to_networkx()
    edge = graph.edges["due_east", "parallel_north"]

    assert isinstance(edge["rule"], TransitionRule)
    assert edge["initiate_from_m"] == 0.0
    assert edge["initiate_to_m"] == pytest.approx(trajectory_set.by_id("due_east").length)
    assert edge["is_entry"] is False
    assert graph.edges[X0_NODE, "due_east"]["is_entry"] is True


def test_narrowed_initiation_window_reaches_the_graph(due_east, parallel_north):
    trajectory_set = TrajectorySet(
        id="narrowed",
        trajectories=(due_east, parallel_north),
        primary_id="due_east",
        x0=due_east.waypoints[0],
        transitions=(
            TransitionRule(source="due_east", target="parallel_north", initiate_from=500.0, initiate_to=1500.0),
        ),
    )
    edge = trajectory_set.to_networkx().edges["due_east", "parallel_north"]
    assert (edge["initiate_from_m"], edge["initiate_to_m"]) == (500.0, 1500.0)


def test_entry_ids_come_from_x0_rules(trajectory_set):
    assert trajectory_set.entry_ids() == ("due_east",)


def test_every_route_is_an_entry_when_no_rule_mentions_x0(due_east, parallel_north):
    trajectory_set = TrajectorySet(
        id="no_entry_rules",
        trajectories=(due_east, parallel_north),
        primary_id="due_east",
        x0=due_east.waypoints[0],
        transitions=(TransitionRule(source="due_east", target="parallel_north"),),
    )
    assert set(trajectory_set.entry_ids()) == {"due_east", "parallel_north"}


def test_terminal_routes_are_those_with_no_onward_transition(trajectory_set):
    assert trajectory_set.terminal_ids() == ("parallel_north",)


def test_route_paths_enumerate_composite_routes(due_east, parallel_north, dogleg):
    """A composed route needs no declaration - it is a path through the rules."""
    trajectory_set = TrajectorySet(
        id="chain",
        trajectories=(due_east, parallel_north, dogleg),
        primary_id="due_east",
        x0=due_east.waypoints[0],
        transitions=(
            TransitionRule(source=X0_NODE, target="due_east"),
            TransitionRule(source="due_east", target="parallel_north"),
            TransitionRule(source="due_east", target="dogleg"),
            TransitionRule(source="parallel_north", target="dogleg"),
        ),
    )
    routes = trajectory_set.route_paths()

    assert ("due_east", "parallel_north", "dogleg") in routes
    assert ("due_east", "dogleg") in routes
    assert all(route[0] in trajectory_set.entry_ids() for route in routes)


def test_authored_transition_trajectories_are_refused(due_east):
    """Transition geometry is generated, never authored into the set."""
    corridor = Trajectory(
        id="hand_drawn",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, time=0.0),
            Waypoint(lat=ORIGIN_LAT + 0.01, lon=ORIGIN_LON, time=30.0),
        ),
        role=TrajectoryRole.TRANSITION,
        connects=("due_east", "somewhere"),
    )
    with pytest.raises(TrajectoryError, match="authored transition trajectories"):
        TrajectorySet(
            id="bad",
            trajectories=(due_east, corridor),
            primary_id="due_east",
            x0=due_east.waypoints[0],
        )


def test_set_bounds_cover_every_waypoint(trajectory_set):
    bounds = trajectory_set.bounds
    for trajectory in trajectory_set.trajectories:
        for waypoint in trajectory.waypoints:
            assert bounds.xmin <= waypoint.lon <= bounds.xmax
            assert bounds.ymin <= waypoint.lat <= bounds.ymax
    assert bounds.wkid == 4326


def test_set_rejects_an_unknown_primary(due_east):
    with pytest.raises(TrajectoryError, match="primary trajectory"):
        TrajectorySet(
            id="bad", trajectories=(due_east,), primary_id="missing", x0=due_east.waypoints[0]
        )


def test_set_rejects_a_transition_to_an_unknown_endpoint(due_east):
    with pytest.raises(TrajectoryError, match="not a known trajectory"):
        TrajectorySet(
            id="bad",
            trajectories=(due_east,),
            primary_id="due_east",
            x0=due_east.waypoints[0],
            transitions=(TransitionRule(source="due_east", target="nowhere"),),
        )


def test_a_transition_targeting_x0_is_refused(due_east):
    """A return is a route ending at x_0, not an edge pointing at the start state."""
    with pytest.raises(TrajectoryError, match="model a return as its own"):
        TrajectorySet(
            id="bad",
            trajectories=(due_east,),
            primary_id="due_east",
            x0=due_east.waypoints[0],
            transitions=(TransitionRule(source="due_east", target=X0_NODE),),
        )


def test_geojson_round_trips_through_coordinates(due_east):
    feature = due_east.to_geojson_feature()
    assert feature["geometry"]["type"] == "LineString"
    assert feature["geometry"]["coordinates"][0] == [ORIGIN_LON, ORIGIN_LAT]
    assert feature["properties"]["role"] == "primary"


# ----- direction, projection, speed ------------------------------------------


def test_heading_is_degrees_clockwise_from_north(due_east, dogleg):
    assert due_east.heading_at(0.0) == pytest.approx(90.0, abs=0.5)
    # The dogleg's second leg runs due north; heading wraps, so compare on the circle.
    north = dogleg.heading_at(dogleg.length - 10.0)
    assert min(north, 360.0 - north) == pytest.approx(0.0, abs=0.5)


def test_tangent_is_a_unit_vector_along_the_leg(due_east):
    east, north, up = due_east.tangent_at(500.0)
    assert math.sqrt(east**2 + north**2 + up**2) == pytest.approx(1.0)
    assert east == pytest.approx(1.0, abs=1e-3)


def test_project_recovers_the_arc_length_of_a_point_on_the_track(dogleg):
    for distance in (0.0, 900.0, 2500.0, dogleg.length):
        point = dogleg.point_at(distance)
        assert dogleg.project(point.lat, point.lon) == pytest.approx(distance, abs=1.0)


def test_project_is_the_nearest_point_for_an_off_track_position(due_east):
    """An off-track point projects to the track position abeam it."""
    on_track = due_east.point_at(1200.0)
    offset_lon, offset_lat, _ = _GEOD.fwd(on_track.lon, on_track.lat, 0.0, 400.0)
    assert due_east.project(offset_lat, offset_lon) == pytest.approx(1200.0, abs=1.0)


def test_project_clamps_beyond_either_end(due_east):
    before_lon, before_lat, _ = _GEOD.fwd(
        due_east.waypoints[0].lon, due_east.waypoints[0].lat, 270.0, 1000.0
    )
    assert due_east.project(before_lat, before_lon) == pytest.approx(0.0, abs=1.0)


def test_distance_to_nearest_waypoint_is_zero_at_a_waypoint(dogleg):
    corner = dogleg.cumulative_distances[1]
    assert dogleg.distance_to_nearest_waypoint(corner) == pytest.approx(0.0, abs=1e-6)
    assert dogleg.distance_to_nearest_waypoint(corner - 300.0) == pytest.approx(300.0, abs=1.0)


def test_speed_at_follows_the_flight_plan_schedule(due_east):
    assert due_east.speed_at(500.0) == pytest.approx(due_east.length / 100.0, rel=1e-9)


def test_speed_at_is_zero_when_the_plan_gives_no_duration():
    stalled = Trajectory(
        id="stalled",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, time=5.0),
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.01, time=5.0),
        ),
    )
    assert stalled.speed_at(100.0) == 0.0
