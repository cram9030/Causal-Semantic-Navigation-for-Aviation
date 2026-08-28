import pytest
import responses

from csnav.data.arcgis.models import Extent
from csnav.data.arcgis.streets import CSJStreetsClient, CSJStreetsError, StreetSegment

LAYER_URL = "https://example.test/server/rest/services/OPN/OPN_OpenDataService/MapServer/60"


def _feature(object_id: int, coords, **props):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {"OBJECTID": object_id, **props},
    }


@responses.activate
def test_query_returns_all_segments_single_page():
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={
            "type": "FeatureCollection",
            "features": [
                _feature(1, [[-121.9, 37.3], [-121.8, 37.31]], STREETNAME="First St", WIDTH=40),
                _feature(2, [[-121.85, 37.32], [-121.84, 37.33]], STREETNAME="Second St", WIDTH=30),
            ],
        },
    )

    client = CSJStreetsClient(LAYER_URL)
    segments = client.query()

    assert len(segments) == 2
    assert segments[0].object_id == 1
    assert segments[0].attributes["STREETNAME"] == "First St"
    assert segments[0].parts == (((-121.9, 37.3), (-121.8, 37.31)),)
    assert len(responses.calls) == 1


@responses.activate
def test_query_paginates_until_no_more_results():
    page_size = 2
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={
            "features": [_feature(1, [[0, 0], [1, 1]]), _feature(2, [[1, 1], [2, 2]])],
            "exceededTransferLimit": True,
        },
    )
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={"features": [_feature(3, [[2, 2], [3, 3]])]},
    )

    client = CSJStreetsClient(LAYER_URL, page_size=page_size)
    segments = client.query()

    assert [s.object_id for s in segments] == [1, 2, 3]
    assert len(responses.calls) == 2
    first_params = responses.calls[0].request.url
    second_params = responses.calls[1].request.url
    assert "resultOffset=0" in first_params
    assert "resultOffset=2" in second_params


@responses.activate
def test_query_with_bbox_sends_envelope_params():
    responses.add(responses.GET, f"{LAYER_URL}/query", json={"features": []})

    client = CSJStreetsClient(LAYER_URL)
    bbox = Extent(xmin=-122.0, ymin=37.2, xmax=-121.8, ymax=37.4, wkid=4326)
    client.query(bbox=bbox)

    url = responses.calls[0].request.url
    assert "geometryType=esriGeometryEnvelope" in url
    assert "spatialRel=esriSpatialRelIntersects" in url
    assert "inSR=4326" in url


def test_query_rejects_non_4326_bbox():
    client = CSJStreetsClient(LAYER_URL)
    bbox = Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857)
    with pytest.raises(ValueError):
        client.query(bbox=bbox)


@responses.activate
def test_query_raises_on_arcgis_error_payload():
    responses.add(responses.GET, f"{LAYER_URL}/query", json={"error": {"code": 400, "message": "boom"}})

    client = CSJStreetsClient(LAYER_URL)
    with pytest.raises(CSJStreetsError):
        client.query()


@responses.activate
def test_query_multilinestring_geometry():
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [[[0, 0], [1, 1]], [[2, 2], [3, 3]]],
                    },
                    "properties": {"OBJECTID": 9},
                }
            ]
        },
    )

    client = CSJStreetsClient(LAYER_URL)
    segments = client.query()

    assert len(segments) == 1
    assert segments[0].parts == (((0, 0), (1, 1)), ((2, 2), (3, 3)))


def test_street_segment_round_trips_to_geojson_feature():
    segment = StreetSegment(object_id=1, parts=(((0, 0), (1, 1)),), attributes={"STREETNAME": "First St"})
    feature = segment.to_geojson_feature()
    assert feature["geometry"] == {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
    assert feature["properties"] == {"STREETNAME": "First St"}
