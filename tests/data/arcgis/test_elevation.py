import numpy as np
import pytest
import responses
from rasterio.io import MemoryFile

from csnav.data.arcgis.elevation import (
    LidarElevationClient,
    LidarElevationError,
    load_elevation_tile,
)
from csnav.data.arcgis.models import Extent

SERVICE_URL = "https://example.test/server/rest/services/Imagery/DPW_Elevation2025/ImageServer"


def _tiny_tiff_bytes(width: int = 2, height: int = 2) -> bytes:
    data = np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float32")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", width=width, height=height, count=1, dtype="float32",
        ) as dst:
            dst.write(data, 1)
        return memfile.read()


@responses.activate
def test_identify_returns_elevation_value():
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"value": "123.45"})

    client = LidarElevationClient(SERVICE_URL)
    assert client.identify(lon=-121.9, lat=37.3) == pytest.approx(123.45)

    url = responses.calls[0].request.url
    assert "geometryType=esriGeometryPoint" in url
    assert "sr=4326" in url


@responses.activate
def test_identify_returns_none_for_nodata():
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"value": "NoData"})

    client = LidarElevationClient(SERVICE_URL)
    assert client.identify(lon=-121.9, lat=37.3) is None


@responses.activate
def test_identify_raises_on_arcgis_error_payload():
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"error": {"code": 400, "message": "boom"}})

    client = LidarElevationClient(SERVICE_URL)
    with pytest.raises(LidarElevationError):
        client.identify(lon=-121.9, lat=37.3)


@responses.activate
def test_get_metadata_parses_extent_and_pixel_info():
    responses.add(
        responses.GET,
        SERVICE_URL,
        json={
            "extent": {
                "xmin": -122.0, "ymin": 37.2, "xmax": -121.8, "ymax": 37.4,
                "spatialReference": {"wkid": 4326},
            },
            "pixelSizeX": 1.0,
            "pixelSizeY": 1.0,
            "bandCount": 1,
            "pixelType": "F32",
        },
    )

    client = LidarElevationClient(SERVICE_URL)
    meta = client.get_metadata()

    assert meta.extent == Extent(xmin=-122.0, ymin=37.2, xmax=-121.8, ymax=37.4, wkid=4326)
    assert meta.pixel_type == "F32"
    assert meta.band_count == 1


@responses.activate
def test_export_elevation_requests_4326_and_returns_bytes():
    tiff_bytes = _tiny_tiff_bytes()
    responses.add(responses.GET, f"{SERVICE_URL}/exportImage", body=tiff_bytes, content_type="image/tiff")

    client = LidarElevationClient(SERVICE_URL)
    extent = Extent(xmin=-122.0, ymin=37.2, xmax=-121.8, ymax=37.4, wkid=4326)
    result = client.export_elevation(extent, width=2, height=2)

    assert result == tiff_bytes
    url = responses.calls[0].request.url
    assert "bboxSR=4326" in url
    assert "imageSR=4326" in url


def test_export_elevation_rejects_non_4326_extent():
    client = LidarElevationClient(SERVICE_URL)
    extent = Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857)
    with pytest.raises(ValueError):
        client.export_elevation(extent, width=2, height=2)


def test_load_elevation_tile_builds_transform_from_requested_extent():
    tiff_bytes = _tiny_tiff_bytes()
    extent = Extent(xmin=-122.0, ymin=37.2, xmax=-121.8, ymax=37.4, wkid=4326)

    tile = load_elevation_tile(tiff_bytes, extent, width=2, height=2)

    assert tile.crs == "EPSG:4326"
    assert tile.data.shape == (1, 2, 2)
    assert tile.data[0, 0, 0] == pytest.approx(10.0)
    # top-left pixel origin should match the requested extent's xmin/ymax
    assert tile.transform.c == pytest.approx(extent.xmin)
    assert tile.transform.f == pytest.approx(extent.ymax)


def test_load_elevation_tile_rejects_non_4326_extent():
    with pytest.raises(ValueError):
        load_elevation_tile(b"", Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857), width=1, height=1)
