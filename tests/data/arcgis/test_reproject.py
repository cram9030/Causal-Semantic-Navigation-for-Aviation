import io

import pytest
import rasterio
from PIL import Image

from csnav.data.arcgis.models import Extent
from csnav.data.arcgis.reproject import reproject_tile_to_4326


def _make_png_bytes(width: int = 16, height: int = 16, color=(10, 20, 30)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# A 1km x 1km tile roughly in the San Jose, CA area (Web Mercator).
SAN_JOSE_BOUNDS_3857 = Extent(
    xmin=-13606000.0, ymin=4479000.0, xmax=-13605000.0, ymax=4480000.0, wkid=3857
)


def test_reproject_tile_to_4326_basic():
    result = reproject_tile_to_4326(_make_png_bytes(), SAN_JOSE_BOUNDS_3857)

    assert result.crs == "EPSG:4326"
    assert result.data.shape[0] == 3  # RGB bands preserved
    assert result.width > 0
    assert result.height > 0
    # a constant-color source tile should still be roughly constant after warping
    assert result.data.mean() == pytest.approx(20.0, abs=15.0)


def test_reproject_tile_accepts_wkid_102100_alias():
    bounds = Extent(**{**SAN_JOSE_BOUNDS_3857.__dict__, "wkid": 102100})
    result = reproject_tile_to_4326(_make_png_bytes(), bounds)
    assert result.crs == "EPSG:4326"


def test_reproject_tile_rejects_non_web_mercator_bounds():
    bounds = Extent(xmin=-122.0, ymin=37.0, xmax=-121.9, ymax=37.1, wkid=4326)
    with pytest.raises(ValueError):
        reproject_tile_to_4326(_make_png_bytes(), bounds)


def test_reproject_tile_to_geotiff_roundtrip(tmp_path):
    result = reproject_tile_to_4326(_make_png_bytes(), SAN_JOSE_BOUNDS_3857)

    out_path = tmp_path / "tile.tif"
    result.to_geotiff(out_path)

    with rasterio.open(out_path) as ds:
        assert ds.crs.to_epsg() == 4326
        assert ds.count == 3
        assert ds.width == result.width
        assert ds.height == result.height
