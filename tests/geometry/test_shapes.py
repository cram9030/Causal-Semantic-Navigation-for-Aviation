"""WGS84 <-> local ENU conversion of whole shapely geometries."""

from __future__ import annotations

import pytest
from pyproj import Geod
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from csnav.geometry import shapes
from csnav.geometry.local_frame import LocalFrame

ORIGIN_LAT, ORIGIN_LON = 37.3382, -121.8863
_GEOD = Geod(ellps="WGS84")


@pytest.fixture
def frame() -> LocalFrame:
    return LocalFrame(ORIGIN_LAT, ORIGIN_LON)


def test_enu_lengths_are_meters_matching_an_independent_geodesic(frame):
    line = LineString([(ORIGIN_LON, ORIGIN_LAT), (ORIGIN_LON + 0.0226, ORIGIN_LAT)])
    _, _, expected = _GEOD.inv(ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 0.0226, ORIGIN_LAT)
    assert shapes.to_enu(line, frame).length == pytest.approx(expected, abs=0.05)


def test_round_trip_preserves_coordinates(frame):
    polygon = Polygon([(ORIGIN_LON, ORIGIN_LAT), (ORIGIN_LON + 0.01, ORIGIN_LAT), (ORIGIN_LON, ORIGIN_LAT + 0.01)])
    restored = shapes.to_wgs84(shapes.to_enu(polygon, frame), frame)
    for (lon, lat), (restored_lon, restored_lat) in zip(polygon.exterior.coords, restored.exterior.coords):
        assert restored_lon == pytest.approx(lon, abs=1e-9)
        assert restored_lat == pytest.approx(lat, abs=1e-9)


def test_round_trip_preserves_polygon_holes(frame):
    outer = [(ORIGIN_LON, ORIGIN_LAT), (ORIGIN_LON + 0.02, ORIGIN_LAT), (ORIGIN_LON + 0.02, ORIGIN_LAT + 0.02)]
    inner = [
        (ORIGIN_LON + 0.005, ORIGIN_LAT + 0.002),
        (ORIGIN_LON + 0.010, ORIGIN_LAT + 0.002),
        (ORIGIN_LON + 0.010, ORIGIN_LAT + 0.006),
    ]
    polygon = Polygon(outer, [inner])
    restored = shapes.to_wgs84(shapes.to_enu(polygon, frame), frame)
    assert len(restored.interiors) == 1
    assert restored.area == pytest.approx(polygon.area, rel=1e-6)


def test_line_parts_flattens_multilinestrings():
    parts = shapes.line_parts(MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]))
    assert parts == (((0.0, 0.0), (1.0, 1.0)), ((2.0, 2.0), (3.0, 3.0)))


def test_line_parts_drops_degenerate_and_non_line_fragments():
    assert shapes.line_parts(Point(0, 0)) == ()
    assert shapes.line_parts(LineString()) == ()


def test_point_parts_collects_points_from_a_collection():
    from shapely.geometry import GeometryCollection

    collection = GeometryCollection([Point(1, 2), LineString([(0, 0), (1, 1)]), Point(3, 4)])
    assert [(p.x, p.y) for p in shapes.point_parts(collection)] == [(1.0, 2.0), (3.0, 4.0)]


def test_polygon_parts_flattens_multipolygons():
    from shapely.geometry import MultiPolygon, box

    multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    assert len(shapes.polygon_parts(multi)) == 2
    assert len(shapes.polygon_parts(box(0, 0, 1, 1))) == 1
