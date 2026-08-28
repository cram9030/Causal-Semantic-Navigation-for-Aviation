import math

import pytest
from pyproj import Geod

from csnav.geometry.local_frame import LatLon, LocalFrame, Point

# Downtown San Jose - used as a representative origin throughout.
ORIGIN_LAT, ORIGIN_LON = 37.3382, -121.8863

# Independent of LocalFrame's own cart+topocentric pipeline: used to
# hand-check expected east/north against a straight ellipsoidal geodesic
# distance/azimuth computation.
_GEOD = Geod(ellps="WGS84")


def test_origin_maps_to_enu_zero():
    frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON)
    point = frame.to_enu(ORIGIN_LAT, ORIGIN_LON)
    assert point.east == pytest.approx(0.0, abs=1e-6)
    assert point.north == pytest.approx(0.0, abs=1e-6)
    assert point.up == pytest.approx(0.0, abs=1e-6)


def test_enu_zero_maps_back_to_origin():
    frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON)
    latlon = frame.to_wgs84(0.0, 0.0)
    assert latlon.lat == pytest.approx(ORIGIN_LAT, abs=1e-9)
    assert latlon.lon == pytest.approx(ORIGIN_LON, abs=1e-9)
    assert latlon.height == pytest.approx(0.0, abs=1e-6)


def test_to_enu_matches_independent_geodesic_distance_and_bearing():
    # A point ~2.7km northeast of the origin - far enough to exercise real
    # geometry, close enough that tangent-plane distortion is negligible.
    lat, lon = 37.3562, -121.8663
    frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON)

    point = frame.to_enu(lat, lon)

    az12, _, distance = _GEOD.inv(ORIGIN_LON, ORIGIN_LAT, lon, lat)
    expected_east = distance * math.sin(math.radians(az12))
    expected_north = distance * math.cos(math.radians(az12))

    assert point.east == pytest.approx(expected_east, abs=0.01)
    assert point.north == pytest.approx(expected_north, abs=0.01)


@pytest.mark.parametrize(
    "lat,lon",
    [
        (ORIGIN_LAT, ORIGIN_LON),
        (37.3562, -121.8663),  # ~2.7km northeast
        (37.30, -121.95),  # further offset, near AOI edge
        (37.36, -121.85),
    ],
)
def test_wgs84_to_enu_roundtrip(lat, lon):
    frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON)
    point = frame.to_enu(lat, lon)
    back = frame.to_wgs84(point.east, point.north, point.up)
    assert back.lat == pytest.approx(lat, abs=1e-8)
    assert back.lon == pytest.approx(lon, abs=1e-8)


def test_height_round_trips_through_up():
    frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON)
    point = frame.to_enu(37.3562, -121.8663, height=650.0)
    back = frame.to_wgs84(point.east, point.north, point.up)
    assert back.height == pytest.approx(650.0, abs=1e-3)


def test_nonzero_origin_height_shifts_up_component():
    lat, lon = 37.3562, -121.8663
    sea_level_frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON, origin_height=0.0)
    elevated_frame = LocalFrame(ORIGIN_LAT, ORIGIN_LON, origin_height=100.0)

    point_sea_level = sea_level_frame.to_enu(lat, lon, height=650.0)
    point_elevated = elevated_frame.to_enu(lat, lon, height=650.0)

    assert point_elevated.up == pytest.approx(point_sea_level.up - 100.0, abs=1e-2)


def test_point_and_latlon_default_to_zero():
    assert Point(east=1.0, north=2.0).up == 0.0
    assert LatLon(lat=1.0, lon=2.0).height == 0.0
