import json
import sys
from pathlib import Path

import responses

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_csj_intersections as fci  # noqa: E402


@responses.activate
def test_main_writes_geojson_feature_collection(tmp_path):
    layer_url = fci.DEFAULT_LAYER_URL
    responses.add(
        responses.GET,
        f"{layer_url}/query",
        json={
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [-121.89, 37.343]},
                    "properties": {"OBJECTID": 1, "INTNAME": "N 5th St & E Julian St"},
                }
            ]
        },
    )

    out_path = tmp_path / "intersections.geojson"
    argv = [
        "fetch_csj_intersections.py",
        "--bbox", "-122.0", "37.2", "-121.8", "37.4",
        "--output", str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fci.main()
    finally:
        sys.argv = old_argv

    written = json.loads(out_path.read_text())
    assert written["type"] == "FeatureCollection"
    assert len(written["features"]) == 1
    assert written["features"][0]["properties"]["INTNAME"] == "N 5th St & E Julian St"

    query_url = responses.calls[0].request.url
    assert query_url.startswith(f"{layer_url}/query")
    assert "geometryType=esriGeometryEnvelope" in query_url


@responses.activate
def test_main_defaults_to_the_pinned_layer_and_intersection_only_filter(tmp_path):
    responses.add(responses.GET, f"{fci.DEFAULT_LAYER_URL}/query", json={"features": []})

    out_path = tmp_path / "intersections.geojson"
    argv = ["fetch_csj_intersections.py", "--output", str(out_path)]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fci.main()
    finally:
        sys.argv = old_argv

    query_url = responses.calls[0].request.url
    assert query_url.startswith(f"{fci.DEFAULT_LAYER_URL}/query")
    assert "where=INTTYPE%3D%27Intersection%27" in query_url


@responses.activate
def test_main_list_fields_prints_and_does_not_query(tmp_path, capsys):
    responses.add(
        responses.GET,
        fci.DEFAULT_LAYER_URL,
        json={
            "fields": [
                {
                    "name": "INTERSECTIONTYPE", "type": "esriFieldTypeString", "alias": "DOT Intersection Type",
                    "domain": {"type": "codedValue", "codedValues": [{"code": "4 Leg", "name": "4 Leg"}]},
                }
            ]
        },
    )

    argv = ["fetch_csj_intersections.py", "--list-fields"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fci.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "INTERSECTIONTYPE" in out
    assert "4 Leg" in out
    assert len(responses.calls) == 1  # metadata only - no /query call


def test_main_requires_output_without_list_fields():
    argv = ["fetch_csj_intersections.py"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        try:
            fci.main()
            raised = False
        except SystemExit:
            raised = True
    finally:
        sys.argv = old_argv
    assert raised
