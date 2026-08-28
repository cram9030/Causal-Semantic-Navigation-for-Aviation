import numpy as np
import pytest
import responses
from rasterio.io import MemoryFile

from csnav.data.arcgis.models import Extent
from csnav.data.lidar import (
    DEFAULT_SERVICE_URL,
    LidarElevationClient,
    LidarElevationError,
)

SERVICE_URL = DEFAULT_SERVICE_URL


def _tiny_tiff_bytes(width: int = 2, height: int = 2) -> bytes:
    data = np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float32")
    with MemoryFile() as memfile:
        with memfile.open(driver="GTiff", width=width, height=height, count=1, dtype="float32") as dst:
            dst.write(data, 1)
        return memfile.read()


@responses.activate
def test_identify_returns_elevation_value():
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"value": "123.45"})

    client = LidarElevationClient()
    assert client.identify(lon=-121.9, lat=37.3) == pytest.approx(123.45)

    url = responses.calls[0].request.url
    assert "geometryType=esriGeometryPoint" in url
    # Regression test: the point's spatial reference must be embedded in the
    # geometry JSON, not passed as a separate bare `sr` param - the latter is
    # silently ignored by the live service (confirmed against a real
    # request), causing the point to be misinterpreted under the service's
    # native Web Mercator SR instead of EPSG:4326.
    assert "spatialReference" in url
    assert "4326" in url
    assert "&sr=4326" not in url and "?sr=4326" not in url


@responses.activate
def test_identify_returns_none_for_nodata():
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"value": "NoData"})

    client = LidarElevationClient()
    assert client.identify(lon=-121.9, lat=37.3) is None


@responses.activate
def test_identify_raises_on_arcgis_error_payload():
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"error": {"code": 400, "message": "boom"}})

    client = LidarElevationClient()
    with pytest.raises(LidarElevationError):
        client.identify(lon=-121.9, lat=37.3)


@responses.activate
def test_get_metadata_parses_extent_and_pixel_info():
    responses.add(
        responses.GET,
        SERVICE_URL,
        json={
            "extent": {
                "xmin": -125.0, "ymin": 24.0, "xmax": -66.0, "ymax": 50.0,
                "spatialReference": {"wkid": 4326},
            },
            "pixelSizeX": 1.0,
            "pixelSizeY": 1.0,
            "pixelType": "F32",
        },
    )

    client = LidarElevationClient()
    meta = client.get_metadata()

    assert meta.extent == Extent(xmin=-125.0, ymin=24.0, xmax=-66.0, ymax=50.0, wkid=4326)
    assert meta.pixel_type == "F32"


@responses.activate
def test_read_window_requests_4326_and_returns_georeferenced_tile():
    tiff_bytes = _tiny_tiff_bytes()
    responses.add(responses.GET, f"{SERVICE_URL}/exportImage", body=tiff_bytes, content_type="image/tiff")

    client = LidarElevationClient()
    bbox = Extent(xmin=-122.0, ymin=37.2, xmax=-121.8, ymax=37.4, wkid=4326)
    tile = client.read_window(bbox, width=2, height=2)

    assert tile.crs == "EPSG:4326"
    assert tile.data.shape == (1, 2, 2)
    assert tile.data[0, 0, 0] == pytest.approx(10.0)
    assert tile.transform.c == pytest.approx(bbox.xmin)
    assert tile.transform.f == pytest.approx(bbox.ymax)

    url = responses.calls[0].request.url
    assert "bboxSR=4326" in url
    assert "imageSR=4326" in url


def test_read_window_rejects_non_4326_bbox():
    client = LidarElevationClient()
    bbox = Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857)
    with pytest.raises(ValueError):
        client.read_window(bbox, width=2, height=2)
