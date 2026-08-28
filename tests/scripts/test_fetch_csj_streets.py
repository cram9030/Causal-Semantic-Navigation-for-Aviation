import json
import sys
from pathlib import Path

import responses

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_csj_streets as fcs  # noqa: E402

BASE = "https://example.test/server/rest/services"
SERVICE_URL = f"{BASE}/OPN/OPN_OpenDataService/MapServer"


def _args(**overrides):
    defaults = dict(
        base_url=BASE,
        service_name_contains="OpenDataService",
        layer_name_contains="Streets",
        root="",
        layer_url=None,
    )
    defaults.update(overrides)
    return type("Args", (), defaults)()


@responses.activate
def test_resolve_layer_url_uses_catalog_discovery():
    responses.add(
        responses.GET, BASE,
        json={"folders": [], "services": [{"name": "OPN/OPN_OpenDataService", "type": "MapServer"}]},
    )
    responses.add(
        responses.GET, SERVICE_URL,
        json={"layers": [{"id": 60, "name": "Streets"}]},
    )

    layer_url = fcs.resolve_layer_url(_args())

    assert layer_url == f"{SERVICE_URL}/60"


def test_resolve_layer_url_skips_discovery_when_explicit():
    args = _args(layer_url=f"{SERVICE_URL}/60")
    assert fcs.resolve_layer_url(args) == f"{SERVICE_URL}/60"


@responses.activate
def test_main_writes_geojson_feature_collection(tmp_path):
    layer_url = f"{SERVICE_URL}/60"
    responses.add(
        responses.GET,
        f"{layer_url}/query",
        json={
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[-121.9, 37.3], [-121.8, 37.31]]},
                    "properties": {"OBJECTID": 1, "STREETNAME": "First St"},
                }
            ]
        },
    )

    out_path = tmp_path / "streets.geojson"
    argv = [
        "fetch_csj_streets.py",
        "--layer-url", layer_url,
        "--bbox", "-122.0", "37.2", "-121.8", "37.4",
        "--output", str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    written = json.loads(out_path.read_text())
    assert written["type"] == "FeatureCollection"
    assert len(written["features"]) == 1
    assert written["features"][0]["properties"]["STREETNAME"] == "First St"

    query_url = responses.calls[0].request.url
    assert "geometryType=esriGeometryEnvelope" in query_url
