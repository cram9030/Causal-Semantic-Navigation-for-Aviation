import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import responses
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_lidar_elevation as fle  # noqa: E402
from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.arcgis.projections import lonlat_to_3857  # noqa: E402
from csnav.data.lidar import LIDAR_PRODUCT_URLS, LidarElevationError  # noqa: E402

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


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    fle.main()


@responses.activate
def test_main_writes_geotiff_for_bbox(tmp_path, monkeypatch, capsys):
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": _tiff_bytes_3857(AOI_4326)}), status=200,
    )

    out_path = tmp_path / "dem.tif"
    argv = [
        "fetch_lidar_elevation.py",
        "--cache-dir", str(tmp_path / "cache"),
        "--bbox", str(AOI_4326.xmin), str(AOI_4326.ymin), str(AOI_4326.xmax), str(AOI_4326.ymax),
        "--output", str(out_path),
    ]
    _run_main(argv, monkeypatch)

    assert out_path.exists()
    with rasterio.open(out_path) as ds:
        assert ds.crs.to_epsg() == 4326
        assert np.nanmedian(ds.read(1)) == pytest.approx(ELEVATION_VALUE, abs=0.5)


@responses.activate
def test_main_identify_prints_value(tmp_path, monkeypatch, capsys):
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": _tiff_bytes_3857(AOI_4326)}), status=200,
    )

    center_lon = (AOI_4326.xmin + AOI_4326.xmax) / 2
    center_lat = (AOI_4326.ymin + AOI_4326.ymax) / 2
    argv = [
        "fetch_lidar_elevation.py",
        "--cache-dir", str(tmp_path / "cache"),
        "--identify", str(center_lon), str(center_lat),
    ]
    _run_main(argv, monkeypatch)

    captured = capsys.readouterr()
    assert float(captured.out.strip()) == 42.5


@responses.activate
def test_main_second_run_does_not_redownload(tmp_path, monkeypatch):
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": _tiff_bytes_3857(AOI_4326)}), status=200,
    )

    cache_dir = tmp_path / "cache"
    argv = [
        "fetch_lidar_elevation.py",
        "--cache-dir", str(cache_dir),
        "--identify", "-121.895", "37.305",
    ]
    _run_main(argv, monkeypatch)
    assert len(responses.calls) == 1

    _run_main(argv, monkeypatch)
    assert len(responses.calls) == 1


@responses.activate
def test_main_prefetch_only_downloads_without_bbox_or_identify(tmp_path, monkeypatch, capsys):
    # Regression test for the "why does --bbox exist if the whole archive
    # always downloads?" confusion - running with neither --bbox nor
    # --identify should still trigger the (one-time) download/extract and
    # just report what was found, rather than erroring.
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": _tiff_bytes_3857(AOI_4326)}), status=200,
    )

    cache_dir = tmp_path / "cache"
    argv = ["fetch_lidar_elevation.py", "--cache-dir", str(cache_dir)]
    _run_main(argv, monkeypatch)

    assert len(responses.calls) == 1
    captured = capsys.readouterr()
    assert "dem.tif" in captured.out


@responses.activate
def test_main_bbox_without_output_errors(tmp_path, monkeypatch):
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes({"dem.tif": _tiff_bytes_3857(AOI_4326)}), status=200,
    )

    argv = [
        "fetch_lidar_elevation.py",
        "--cache-dir", str(tmp_path / "cache"),
        "--bbox", str(AOI_4326.xmin), str(AOI_4326.ymin), str(AOI_4326.xmax), str(AOI_4326.ymax),
    ]
    with pytest.raises(SystemExit):
        _run_main(argv, monkeypatch)


@responses.activate
def test_main_surfaces_gdb_diagnostic_when_no_raster_found(tmp_path, monkeypatch):
    responses.add(
        responses.GET, LIDAR_PRODUCT_URLS["5ft"],
        body=_zip_bytes(
            {
                "5ft_contours.txt": b"some contour export",
                "LiDAR5FT.gdb/a00000001.gdbtable": b"not a real gdbtable",
            }
        ),
        status=200,
    )

    argv = ["fetch_lidar_elevation.py", "--cache-dir", str(tmp_path / "cache")]
    with pytest.raises(LidarElevationError) as exc_info:
        _run_main(argv, monkeypatch)

    message = str(exc_info.value)
    assert "LiDAR5FT.gdb" in message
    assert ".txt" in message
