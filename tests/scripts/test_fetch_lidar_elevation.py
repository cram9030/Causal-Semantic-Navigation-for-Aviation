import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import responses
from rasterio.io import MemoryFile

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_lidar_elevation as fle  # noqa: E402
from csnav.data.lidar import DEFAULT_SERVICE_URL  # noqa: E402

SERVICE_URL = DEFAULT_SERVICE_URL


def _tiny_tiff_bytes() -> bytes:
    data = np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float32")
    with MemoryFile() as memfile:
        with memfile.open(driver="GTiff", width=2, height=2, count=1, dtype="float32") as dst:
            dst.write(data, 1)
        return memfile.read()


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    fle.main()


@responses.activate
def test_main_exports_geotiff(tmp_path, monkeypatch):
    responses.add(responses.GET, f"{SERVICE_URL}/exportImage", body=_tiny_tiff_bytes(), content_type="image/tiff")

    out_path = tmp_path / "dem.tif"
    argv = [
        "fetch_lidar_elevation.py",
        "--bbox", "-122.0", "37.2", "-121.8", "37.4",
        "--width", "2", "--height", "2",
        "--output", str(out_path),
    ]
    _run_main(argv, monkeypatch)

    assert out_path.exists()
    with rasterio.open(out_path) as ds:
        assert ds.crs.to_epsg() == 4326
        assert ds.read(1)[0, 0] == pytest.approx(10.0)


@responses.activate
def test_main_identify_prints_value(capsys, monkeypatch):
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"value": "42.0"})

    argv = ["fetch_lidar_elevation.py", "--identify", "-121.9", "37.3"]
    _run_main(argv, monkeypatch)

    captured = capsys.readouterr()
    assert captured.out.strip() == "42.0"


def test_main_requires_bbox_and_output_or_identify(monkeypatch):
    argv = ["fetch_lidar_elevation.py"]
    with pytest.raises(SystemExit):
        _run_main(argv, monkeypatch)
