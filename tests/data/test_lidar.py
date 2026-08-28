import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import responses
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from csnav.data.arcgis.models import Extent
from csnav.data.arcgis.projections import lonlat_to_3857
from csnav.data.lidar import (
    LIDAR_PRODUCT_URLS,
    LidarElevationClient,
    LidarElevationError,
    download_archive,
    extract_archive,
    find_raster_files,
    read_elevation_window,
)

# A small AOI, used to build a synthetic source raster in EPSG:3857 (a CRS
# other than the output EPSG:4326, to prove read_elevation_window actually
# reprojects rather than assuming the source is already 4326).
AOI_4326 = Extent(xmin=-121.90, ymin=37.30, xmax=-121.89, ymax=37.31, wkid=4326)
ELEVATION_VALUE = 42.5


def _tiff_bytes_3857(bbox_4326: Extent, width: int = 20, height: int = 20, value: float = ELEVATION_VALUE) -> bytes:
    xmin, ymin = lonlat_to_3857(bbox_4326.xmin, bbox_4326.ymin)
    xmax, ymax = lonlat_to_3857(bbox_4326.xmax, bbox_4326.ymax)
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    data = np.full((height, width), value, dtype="float32")
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", width=width, height=height, count=1, dtype="float32",
            crs="EPSG:3857", transform=transform,
        ) as dst:
            dst.write(data, 1)
        return memfile.read()


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_download_archive_streams_to_dest_and_skips_if_present(tmp_path):
    dest = tmp_path / "archive.zip"

    @responses.activate
    def _run():
        responses.add(responses.GET, LIDAR_PRODUCT_URLS["5ft"], body=b"zip-bytes", status=200)
        path = download_archive(LIDAR_PRODUCT_URLS["5ft"], dest)
        assert path == dest
        assert dest.read_bytes() == b"zip-bytes"
        assert len(responses.calls) == 1

    _run()

    # second call: already on disk, no network request needed
    @responses.activate
    def _run_again():
        path = download_archive(LIDAR_PRODUCT_URLS["5ft"], dest)
        assert path == dest
        assert len(responses.calls) == 0

    _run_again()


def test_extract_archive_finds_raster_files(tmp_path):
    zip_path = tmp_path / "archive.zip"
    tiff_bytes = _tiff_bytes_3857(AOI_4326)
    zip_path.write_bytes(_zip_bytes({"tiles/tile_0.tif": tiff_bytes, "readme.txt": b"hello"}))

    dest_dir = tmp_path / "extracted"
    rasters = extract_archive(zip_path, dest_dir)

    assert rasters == [dest_dir / "tiles" / "tile_0.tif"]
    assert find_raster_files(dest_dir) == rasters

    # second call is a no-op (marker file present) - still returns the same rasters
    rasters_again = extract_archive(zip_path, dest_dir)
    assert rasters_again == rasters


def test_read_elevation_window_reprojects_to_4326(tmp_path):
    tiff_path = tmp_path / "tile.tif"
    tiff_path.write_bytes(_tiff_bytes_3857(AOI_4326))

    tile = read_elevation_window([tiff_path], AOI_4326)

    assert tile.crs == "EPSG:4326"
    assert tile.data.shape[0] == 1
    # the tile is a constant-value raster, so every pixel (mod resampling
    # edge effects) should come back close to the original value
    assert np.nanmedian(tile.data) == pytest.approx(ELEVATION_VALUE, abs=0.5)


def test_read_elevation_window_raises_when_nothing_intersects(tmp_path):
    tiff_path = tmp_path / "tile.tif"
    tiff_path.write_bytes(_tiff_bytes_3857(AOI_4326))

    far_away = Extent(xmin=10, ymin=10, xmax=11, ymax=11, wkid=4326)
    with pytest.raises(LidarElevationError):
        read_elevation_window([tiff_path], far_away)


def test_read_elevation_window_rejects_non_4326_bbox(tmp_path):
    tiff_path = tmp_path / "tile.tif"
    tiff_path.write_bytes(_tiff_bytes_3857(AOI_4326))

    with pytest.raises(ValueError):
        read_elevation_window([tiff_path], Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857))


@responses.activate
def test_client_ensure_local_downloads_and_extracts(tmp_path):
    tiff_bytes = _tiff_bytes_3857(AOI_4326)
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": tiff_bytes}), status=200,
    )

    client = LidarElevationClient(cache_dir=tmp_path, product="5ft")
    rasters = client.ensure_local()

    assert rasters == [tmp_path / "5ft" / "dem.tif"]
    assert len(responses.calls) == 1

    # a second ensure_local() call, without re-registering the mock, must
    # not need another network request - both the zip and extraction are
    # already cached on disk.
    rasters_again = client.ensure_local()
    assert rasters_again == rasters
    assert len(responses.calls) == 1


@responses.activate
def test_client_read_window_and_identify(tmp_path):
    tiff_bytes = _tiff_bytes_3857(AOI_4326)
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": tiff_bytes}), status=200,
    )

    client = LidarElevationClient(cache_dir=tmp_path, product="5ft")
    tile = client.read_window(AOI_4326)
    assert tile.crs == "EPSG:4326"

    center_lon = (AOI_4326.xmin + AOI_4326.xmax) / 2
    center_lat = (AOI_4326.ymin + AOI_4326.ymax) / 2
    elevation = client.identify(center_lon, center_lat)
    assert elevation == pytest.approx(ELEVATION_VALUE, abs=0.5)

    assert client.identify(lon=20.0, lat=20.0) is None


def test_client_rejects_unknown_product(tmp_path):
    with pytest.raises(ValueError):
        LidarElevationClient(cache_dir=tmp_path, product="10ft")
