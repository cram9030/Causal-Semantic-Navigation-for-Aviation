import pytest

from csnav.data.arcgis.models import Extent
from csnav.data.arcgis.projections import (
    extent_3857_to_4326,
    extent_4326_to_3857,
    is_web_mercator,
    lonlat_to_3857,
    xy_3857_to_lonlat,
)


def test_lonlat_roundtrip_through_3857():
    lon, lat = -121.8863, 37.3382  # downtown San Jose
    x, y = lonlat_to_3857(lon, lat)
    lon2, lat2 = xy_3857_to_lonlat(x, y)
    assert lon2 == pytest.approx(lon, abs=1e-6)
    assert lat2 == pytest.approx(lat, abs=1e-6)


def test_extent_4326_to_3857_and_back():
    extent = Extent(xmin=-121.95, ymin=37.30, xmax=-121.85, ymax=37.36, wkid=4326)
    mercator = extent_4326_to_3857(extent)
    assert mercator.wkid == 3857
    back = extent_3857_to_4326(mercator)
    assert back.xmin == pytest.approx(extent.xmin, abs=1e-6)
    assert back.ymax == pytest.approx(extent.ymax, abs=1e-6)


def test_extent_4326_to_3857_rejects_wrong_wkid():
    extent = Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857)
    with pytest.raises(ValueError):
        extent_4326_to_3857(extent)


def test_is_web_mercator():
    assert is_web_mercator(3857)
    assert is_web_mercator(102100)
    assert not is_web_mercator(4326)
