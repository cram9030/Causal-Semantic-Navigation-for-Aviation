"""RNP tube containment and corridor geometry.

The load-bearing property here is that the radius is *exactly* what the caller
passed, in meters on the ground - checked against ``pyproj.Geod`` distances
rather than against the tube's own ENU math - and that nothing in the module
derives or adjusts it (CLAUDE.md core decision 4).
"""

from __future__ import annotations

import pytest
from pyproj import Geod
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import unary_union

from csnav.trajectory.tube import TubeModel, union_corridor
from csnav.trajectory.waypoints import Waypoint

from .conftest import ORIGIN_LAT, ORIGIN_LON

_GEOD = Geod(ellps="WGS84")


def _offset_north(lat: float, lon: float, meters: float) -> tuple[float, float]:
    """Point ``meters`` due north of ``(lat, lon)``, via an independent geodesic step."""
    east_lon, north_lat, _ = _GEOD.fwd(lon, lat, 0.0, meters)
    return north_lat, east_lon


def test_radius_must_be_positive():
    with pytest.raises(ValueError, match="must be > 0 meters"):
        TubeModel(radius=0.0)


def test_cross_track_distance_matches_an_independent_geodesic_offset(due_east, tube):
    lat, lon = _offset_north(ORIGIN_LAT, ORIGIN_LON + 0.0113, 150.0)
    state = Waypoint(lat=lat, lon=lon, height=300.0, time=50.0)
    assert tube.cross_track_distance(state, due_east) == pytest.approx(150.0, abs=0.5)


def test_containment_is_decided_at_the_configured_radius(due_east, tube):
    inside_lat, inside_lon = _offset_north(ORIGIN_LAT, ORIGIN_LON + 0.0113, tube.radius - 5.0)
    outside_lat, outside_lon = _offset_north(ORIGIN_LAT, ORIGIN_LON + 0.0113, tube.radius + 5.0)
    assert tube.contains(Waypoint(lat=inside_lat, lon=inside_lon), due_east)
    assert not tube.contains(Waypoint(lat=outside_lat, lon=outside_lon), due_east)


def test_containment_is_lateral_and_ignores_height(due_east, tube):
    """Height is not part of the containment test - RNP containment is lateral."""
    on_track = Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.0113, height=5_000.0)
    assert tube.contains(on_track, due_east)


def test_a_wider_tube_contains_a_narrower_one(due_east):
    narrow = TubeModel(radius=100.0).corridor(due_east)
    wide = TubeModel(radius=300.0).corridor(due_east)
    assert wide.contains(narrow)
    assert wide.area > narrow.area


def test_corridor_half_width_equals_the_radius_on_the_ground(due_east, tube):
    """The corridor edge sits exactly ``radius`` meters off the centerline."""
    corridor = tube.corridor(due_east)
    midpoint = due_east.point_at(due_east.length / 2.0)
    for offset, expected_inside in ((tube.radius - 10.0, True), (tube.radius + 10.0, False)):
        lat, lon = _offset_north(midpoint.lat, midpoint.lon, offset)
        assert corridor.contains(ShapelyPoint(lon, lat)) is expected_inside


def test_extra_buffer_adds_to_the_radius_without_changing_it(due_east, tube):
    grown = tube.corridor(due_east, extra_buffer=500.0)
    midpoint = due_east.point_at(due_east.length / 2.0)
    lat, lon = _offset_north(midpoint.lat, midpoint.lon, tube.radius + 400.0)
    assert grown.contains(ShapelyPoint(lon, lat))
    assert tube.radius == 200.0  # the model itself is untouched


def test_extra_buffer_rejects_a_negative_value(due_east, tube):
    with pytest.raises(ValueError, match="must be >= 0 meters"):
        tube.corridor(due_east, extra_buffer=-1.0)


def test_window_corridor_stays_inside_the_whole_trajectory_corridor(due_east, tube):
    """A window's corridor is a sub-length of the same tube, so it adds no width.

    Both polygons approximate round caps/joins with finite segments, so a
    sliver of the window polygon can fall a fraction of a square meter outside
    the whole-route polygon; the assertion allows for that and nothing more.
    """
    window = due_east.windows(1000.0)[0]
    window_corridor = tube.corridor(due_east, window=window)
    whole = tube.corridor(due_east)
    assert window_corridor.difference(whole).area < window_corridor.area * 1e-6
    assert window_corridor.area < whole.area


def test_window_corridors_cover_the_full_corridor_together(due_east, tube):
    windows = due_east.windows(500.0)
    covered = unary_union([tube.corridor(due_east, window=window) for window in windows])
    whole = tube.corridor(due_east)
    # Windows meet at shared boundary points, so their union reconstructs the
    # corridor up to the same buffer-discretization slivers.
    assert whole.difference(covered).area < whole.area * 1e-6


def test_envelope_bounds_the_corridor(due_east, tube):
    envelope = tube.envelope(due_east)
    xmin, ymin, xmax, ymax = tube.corridor(due_east).bounds
    assert envelope.wkid == 4326
    assert (envelope.xmin, envelope.ymin, envelope.xmax, envelope.ymax) == pytest.approx(
        (xmin, ymin, xmax, ymax)
    )


def test_union_corridor_merges_several_trajectories(due_east, parallel_north):
    merged = union_corridor([(due_east, TubeModel(200.0)), (parallel_north, TubeModel(200.0))])
    assert merged.area > TubeModel(200.0).corridor(due_east).area
