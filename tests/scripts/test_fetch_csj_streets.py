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


@responses.activate
def test_main_defaults_to_the_pinned_layer_and_street_centerline_filter(tmp_path):
    """A bare invocation (no --layer-url/--where) must hit the pinned DEFAULT_LAYER_URL directly
    - not substring-match discovery, which is exactly the ambiguity that silently drifted onto
    the wrong layer before (no discovery endpoint is mocked here, so a discovery attempt would
    raise a connection error) - filtered by the confirmed FEATURECLASS='StreetCenterline',
    which excludes the sanitary-sewer/storm-water infrastructure lines real ground-truth builds
    found occluding real streets once rasterized.
    """
    responses.add(responses.GET, f"{fcs.DEFAULT_LAYER_URL}/query", json={"features": []})

    out_path = tmp_path / "streets.geojson"
    argv = ["fetch_csj_streets.py", "--output", str(out_path)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    query_url = responses.calls[0].request.url
    assert query_url.startswith(f"{fcs.DEFAULT_LAYER_URL}/query")
    assert "where=FEATURECLASS%3D%27StreetCenterline%27" in query_url


@responses.activate
def test_main_where_flag_overrides_the_default_filter(tmp_path):
    layer_url = f"{SERVICE_URL}/60"
    responses.add(responses.GET, f"{layer_url}/query", json={"features": []})

    out_path = tmp_path / "streets.geojson"
    argv = [
        "fetch_csj_streets.py", "--layer-url", layer_url, "--where", "1=1", "--output", str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    query_url = responses.calls[0].request.url
    assert "where=1%3D1" in query_url


@responses.activate
def test_empty_layer_url_falls_back_to_discovery(tmp_path):
    """--layer-url "" is the documented escape hatch to re-pin after a catalog reorganization."""
    responses.add(
        responses.GET, BASE,
        json={"folders": [], "services": [{"name": "OPN/OPN_OpenDataService", "type": "MapServer"}]},
    )
    responses.add(
        responses.GET, SERVICE_URL,
        json={"layers": [{"id": 60, "name": "Streets"}]},
    )
    responses.add(responses.GET, f"{SERVICE_URL}/60/query", json={"features": []})

    out_path = tmp_path / "streets.geojson"
    argv = [
        "fetch_csj_streets.py", "--base-url", BASE, "--root", "", "--layer-url", "", "--output", str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    assert responses.calls[-1].request.url.startswith(f"{SERVICE_URL}/60/query")


@responses.activate
def test_list_fields_prints_names_types_and_coded_values(capsys):
    layer_url = f"{SERVICE_URL}/60"
    responses.add(
        responses.GET, layer_url,
        json={
            "fields": [
                {"name": "OBJECTID", "type": "esriFieldTypeOID", "alias": "OBJECTID"},
                {
                    "name": "FEATURECLASS", "type": "esriFieldTypeSmallInteger", "alias": "Feature Class",
                    "domain": {
                        "type": "codedValue",
                        "codedValues": [
                            {"code": 1, "name": "Street Centerline"},
                            {"code": 2, "name": "Alley"},
                        ],
                    },
                },
            ]
        },
    )

    argv = ["fetch_csj_streets.py", "--layer-url", layer_url, "--list-fields"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "FEATURECLASS" in out
    assert "Feature Class" in out
    assert "1" in out and "Street Centerline" in out
    assert "2" in out and "Alley" in out
    assert not responses.calls[-1].request.url.endswith("/query")  # no query was ever issued


@responses.activate
def test_list_fields_does_not_require_output(capsys):
    layer_url = f"{SERVICE_URL}/60"
    responses.add(responses.GET, layer_url, json={"fields": []})

    argv = ["fetch_csj_streets.py", "--layer-url", layer_url, "--list-fields"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()  # must not raise for missing --output
    finally:
        sys.argv = old_argv


@responses.activate
def test_list_layers_prints_every_matching_layer_not_just_the_first(capsys):
    """find_layer's first-match-wins discovery is exactly how a real session ended up querying
    the wrong "Streets"-named layer after CSJ's catalog reorganized - --list-layers shows every
    candidate so the right one can be picked deliberately via --layer-url.
    """
    responses.add(
        responses.GET, BASE,
        json={"folders": [], "services": [{"name": "OPN/OPN_OpenDataService", "type": "MapServer"}]},
    )
    responses.add(
        responses.GET, SERVICE_URL,
        json={"layers": [{"id": 60, "name": "Streets"}, {"id": 522, "name": "Street Centerlines"}]},
    )

    argv = [
        "fetch_csj_streets.py", "--base-url", BASE, "--service-name-contains", "OpenDataService",
        "--layer-name-contains", "Street", "--root", "", "--list-layers",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert f"{SERVICE_URL}/60" in out
    assert f"{SERVICE_URL}/522" in out


@responses.activate
def test_distinct_values_prints_every_value_and_issues_no_write(tmp_path, capsys):
    layer_url = f"{SERVICE_URL}/60"
    responses.add(
        responses.GET, f"{layer_url}/query",
        json={"features": [{"attributes": {"DESIGNATION": "Local"}}, {"attributes": {"DESIGNATION": "Alley"}}]},
    )

    argv = ["fetch_csj_streets.py", "--layer-url", layer_url, "--distinct-values", "DESIGNATION"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fcs.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "Local" in out and "Alley" in out
    query_url = responses.calls[0].request.url
    assert "returnDistinctValues=true" in query_url
    assert "outFields=DESIGNATION" in query_url
