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

BASE = "https://example.test/server/rest/services"
SERVICE_URL = f"{BASE}/Imagery/DPW_Elevation2025/ImageServer"


def _args(**overrides):
    defaults = dict(base_url=BASE, name_contains="Elevation", root="", service_url=None)
    defaults.update(overrides)
    return type("Args", (), defaults)()


def _tiny_tiff_bytes() -> bytes:
    data = np.array([[10.0, 11.0], [12.0, 13.0]], dtype="float32")
    with MemoryFile() as memfile:
        with memfile.open(driver="GTiff", width=2, height=2, count=1, dtype="float32") as dst:
            dst.write(data, 1)
        return memfile.read()


@responses.activate
def test_resolve_service_url_uses_catalog_discovery():
    responses.add(
        responses.GET, BASE,
        json={
            "folders": [],
            "services": [{"name": "Imagery/DPW_Elevation2025", "type": "ImageServer"}],
        },
    )

    service_url = fle.resolve_service_url(_args())

    assert service_url == SERVICE_URL


def test_resolve_service_url_skips_discovery_when_explicit():
    args = _args(service_url=SERVICE_URL)
    assert fle.resolve_service_url(args) == SERVICE_URL


@responses.activate
def test_resolve_service_url_raises_when_no_match(capsys):
    responses.add(responses.GET, BASE, json={"folders": [], "services": []})

    with pytest.raises(SystemExit):
        fle.resolve_service_url(_args())


@responses.activate
def test_main_exports_geotiff(tmp_path):
    responses.add(responses.GET, f"{SERVICE_URL}/exportImage", body=_tiny_tiff_bytes(), content_type="image/tiff")

    out_path = tmp_path / "dem.tif"
    argv = [
        "fetch_lidar_elevation.py",
        "--service-url", SERVICE_URL,
        "--bbox", "-122.0", "37.2", "-121.8", "37.4",
        "--width", "2", "--height", "2",
        "--output", str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fle.main()
    finally:
        sys.argv = old_argv

    assert out_path.exists()
    with rasterio.open(out_path) as ds:
        assert ds.crs.to_epsg() == 4326
        assert ds.read(1)[0, 0] == pytest.approx(10.0)


@responses.activate
def test_main_identify_prints_value(capsys):
    responses.add(responses.GET, f"{SERVICE_URL}/identify", json={"value": "42.0"})

    argv = [
        "fetch_lidar_elevation.py",
        "--service-url", SERVICE_URL,
        "--identify", "-121.9", "37.3",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fle.main()
    finally:
        sys.argv = old_argv

    captured = capsys.readouterr()
    assert captured.out.strip() == "42.0"
