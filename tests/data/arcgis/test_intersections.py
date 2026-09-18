import pytest
import responses

from csnav.data.arcgis.intersections import (
    CSJIntersectionsClient,
    CSJIntersectionsError,
    StreetIntersection,
    intersection_leg_count,
    intersection_name,
    intersection_point,
    intersection_traffic_control,
    intersections_from_geojson,
)
from csnav.data.arcgis.models import Extent

LAYER_URL = "https://example.test/server/rest/services/OPN/OPN_OpenDataService/MapServer/276"


def _feature(object_id: int, lon: float, lat: float, **props):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"OBJECTID": object_id, **props},
    }


@responses.activate
def test_query_returns_all_intersections_single_page():
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={
            "features": [
                _feature(1, -121.89, 37.343, INTNAME="N 5th St & E Julian St", INTERSECTIONTYPE="4 Leg"),
                _feature(2, -121.88, 37.34, INTNAME="Second & Main"),
            ],
        },
    )

    client = CSJIntersectionsClient(LAYER_URL)
    intersections = client.query()

    assert len(intersections) == 2
    assert intersections[0].object_id == 1
    assert intersections[0].lon == -121.89
    assert intersections[0].lat == 37.343
    assert intersections[0].attributes["INTNAME"] == "N 5th St & E Julian St"


@responses.activate
def test_query_paginates_until_no_more_results():
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={"features": [_feature(1, 0, 0), _feature(2, 1, 1)], "exceededTransferLimit": True},
    )
    responses.add(responses.GET, f"{LAYER_URL}/query", json={"features": [_feature(3, 2, 2)]})

    client = CSJIntersectionsClient(LAYER_URL, page_size=2)
    intersections = client.query()

    assert [i.object_id for i in intersections] == [1, 2, 3]
    assert len(responses.calls) == 2


@responses.activate
def test_query_defaults_to_intersection_only_filter():
    responses.add(responses.GET, f"{LAYER_URL}/query", json={"features": []})

    client = CSJIntersectionsClient(LAYER_URL)
    client.query()

    assert "where=INTTYPE%3D%27Intersection%27" in responses.calls[0].request.url


@responses.activate
def test_query_with_bbox_sends_envelope_params():
    responses.add(responses.GET, f"{LAYER_URL}/query", json={"features": []})

    client = CSJIntersectionsClient(LAYER_URL)
    bbox = Extent(xmin=-122.0, ymin=37.2, xmax=-121.8, ymax=37.4, wkid=4326)
    client.query(bbox=bbox)

    url = responses.calls[0].request.url
    assert "geometryType=esriGeometryEnvelope" in url
    assert "spatialRel=esriSpatialRelIntersects" in url


def test_query_rejects_non_4326_bbox():
    client = CSJIntersectionsClient(LAYER_URL)
    bbox = Extent(xmin=0, ymin=0, xmax=1, ymax=1, wkid=3857)
    with pytest.raises(ValueError):
        client.query(bbox=bbox)


@responses.activate
def test_query_raises_on_arcgis_error_payload():
    responses.add(responses.GET, f"{LAYER_URL}/query", json={"error": {"code": 400, "message": "boom"}})

    client = CSJIntersectionsClient(LAYER_URL)
    with pytest.raises(CSJIntersectionsError):
        client.query()


@responses.activate
def test_query_rejects_non_point_geometry():
    responses.add(
        responses.GET,
        f"{LAYER_URL}/query",
        json={
            "features": [
                {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}, "properties": {}}
            ]
        },
    )

    client = CSJIntersectionsClient(LAYER_URL)
    with pytest.raises(CSJIntersectionsError):
        client.query()


def test_street_intersection_round_trips_to_geojson_feature():
    intersection = StreetIntersection(object_id=1, lon=-121.89, lat=37.343, attributes={"INTNAME": "First & Main"})
    feature = intersection.to_geojson_feature()
    assert feature["geometry"] == {"type": "Point", "coordinates": [-121.89, 37.343]}
    assert feature["properties"] == {"INTNAME": "First & Main"}


def test_intersection_point_matches_lon_lat():
    intersection = StreetIntersection(object_id=1, lon=-121.89, lat=37.343, attributes={})
    point = intersection_point(intersection)
    assert (point.x, point.y) == (-121.89, 37.343)


def test_intersections_from_geojson_round_trips_and_skips_non_point():
    data = {
        "type": "FeatureCollection",
        "features": [
            _feature(1, -121.89, 37.343, INTNAME="First & Main"),
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}, "properties": {}},
        ],
    }
    intersections = intersections_from_geojson(data)
    assert len(intersections) == 1
    assert intersections[0].object_id == 1


def test_intersection_name_tries_candidates_in_order():
    assert intersection_name({"INTNAME": "N 5th St & E Julian St"}) == "N 5th St & E Julian St"
    assert intersection_name({}) is None


def test_intersection_leg_count_reads_intersectiontype():
    assert intersection_leg_count({"INTERSECTIONTYPE": "4 Leg"}) == "4 Leg"
    assert intersection_leg_count({}) is None


def test_intersection_traffic_control_reads_trafficcontroltype():
    assert intersection_traffic_control({"TRAFFICCONTROLTYPE": "Signal"}) == "Signal"
    assert intersection_traffic_control({}) is None
