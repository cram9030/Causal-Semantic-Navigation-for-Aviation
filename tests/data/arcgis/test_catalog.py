import pytest
import responses

from csnav.data.arcgis.catalog import ArcGISCatalog, ArcGISCatalogError, extract_year
from csnav.data.arcgis.models import ServiceRef

BASE = "https://example.test/server/rest/services"


@responses.activate
def test_discover_imagery_services_returns_all_vintages_newest_first():
    responses.add(
        responses.GET,
        f"{BASE}/Imagery",
        json={
            "folders": [],
            "services": [
                {"name": "Imagery/DPW_ImageryCached", "type": "MapServer"},
                {"name": "Imagery/DPW_Imagery_2016", "type": "MapServer"},
                {"name": "Imagery/DPW_Imagery_2012", "type": "MapServer"},
                {"name": "Imagery/Parcels", "type": "MapServer"},
            ],
        },
    )

    catalog = ArcGISCatalog(base_url=BASE)
    services = catalog.discover_imagery_services()

    # every DPW_Imagery* vintage is returned, not just the newest/cached one
    assert [s.name for s in services] == [
        "DPW_Imagery_2016",
        "DPW_Imagery_2012",
        "DPW_ImageryCached",
    ]
    assert all(s.folder == "Imagery" for s in services)


@responses.activate
def test_discover_imagery_services_excludes_non_matching_services():
    responses.add(
        responses.GET,
        f"{BASE}/Imagery",
        json={
            "folders": [],
            "services": [
                {"name": "Imagery/DPW_ImageryCached", "type": "MapServer"},
                {"name": "Imagery/Parcels", "type": "MapServer"},
                {"name": "Imagery/DPW_ImageryCached_Preview", "type": "ImageServer"},
            ],
        },
    )

    catalog = ArcGISCatalog(base_url=BASE)
    services = catalog.discover_imagery_services()

    names = {s.name for s in services}
    assert names == {"DPW_ImageryCached", "DPW_ImageryCached_Preview"}
    assert "Parcels" not in names


@responses.activate
def test_walk_recurses_into_subfolders():
    responses.add(
        responses.GET,
        f"{BASE}/Imagery",
        json={
            # ArcGIS returns sub-folder names relative to the current folder,
            # unlike service "name" values which include the full path.
            "folders": ["Historical"],
            "services": [{"name": "Imagery/DPW_ImageryCached", "type": "MapServer"}],
        },
    )
    responses.add(
        responses.GET,
        f"{BASE}/Imagery/Historical",
        json={
            "folders": [],
            "services": [{"name": "Imagery/Historical/DPW_Imagery_2008", "type": "MapServer"}],
        },
    )

    catalog = ArcGISCatalog(base_url=BASE)
    services = catalog.discover_imagery_services()

    full_names = {s.full_name for s in services}
    assert full_names == {"Imagery/DPW_ImageryCached", "Imagery/Historical/DPW_Imagery_2008"}


@responses.activate
def test_catalog_error_on_arcgis_error_payload():
    responses.add(responses.GET, f"{BASE}/Imagery", json={"error": {"code": 400, "message": "boom"}})

    catalog = ArcGISCatalog(base_url=BASE)
    with pytest.raises(ArcGISCatalogError):
        catalog.discover_imagery_services()


def test_extract_year():
    assert extract_year("DPW_Imagery_2012") == 2012
    assert extract_year("DPW_ImageryCached") is None
    assert extract_year("Historical_1998_Flight") == 1998


def test_service_rest_url():
    catalog = ArcGISCatalog(base_url=BASE)
    ref = ServiceRef(folder="Imagery", name="DPW_ImageryCached", service_type="MapServer")
    assert catalog.service_rest_url(ref) == f"{BASE}/Imagery/DPW_ImageryCached/MapServer"

    root_ref = ServiceRef(folder="", name="TopLevel", service_type="MapServer")
    assert catalog.service_rest_url(root_ref) == f"{BASE}/TopLevel/MapServer"


@responses.activate
def test_discover_services_filters_by_type_and_name():
    responses.add(
        responses.GET,
        f"{BASE}",
        json={
            "folders": [],
            "services": [
                {"name": "OPN_OpenDataService", "type": "MapServer"},
                {"name": "DPW_Elevation2025", "type": "ImageServer"},
                {"name": "Hosted/CSJWebMapBase", "type": "FeatureServer"},
            ],
        },
    )

    catalog = ArcGISCatalog(base_url=BASE)
    services = catalog.discover_services(name_contains="Elevation", service_types=("ImageServer",))

    assert [s.name for s in services] == ["DPW_Elevation2025"]


@responses.activate
def test_find_layer_resolves_sublayer_url_from_matching_service():
    responses.add(
        responses.GET,
        f"{BASE}",
        json={"folders": [], "services": [{"name": "OPN/OPN_OpenDataService", "type": "MapServer"}]},
    )
    service_url = f"{BASE}/OPN/OPN_OpenDataService/MapServer"
    responses.add(
        responses.GET,
        service_url,
        json={"layers": [{"id": 12, "name": "Parcels"}, {"id": 60, "name": "Streets"}]},
    )

    catalog = ArcGISCatalog(base_url=BASE)
    layer_url = catalog.find_layer("Streets", service_name_contains="OpenDataService")

    assert layer_url == f"{service_url}/60"


@responses.activate
def test_walk_skips_folder_that_404s_without_aborting():
    responses.add(
        responses.GET,
        f"{BASE}",
        json={
            "folders": ["Internal", "Imagery"],
            "services": [],
        },
    )
    responses.add(responses.GET, f"{BASE}/Internal", status=404)
    responses.add(
        responses.GET,
        f"{BASE}/Imagery",
        json={"folders": [], "services": [{"name": "Imagery/DPW_ImageryCached", "type": "MapServer"}]},
    )

    catalog = ArcGISCatalog(base_url=BASE)
    services = list(catalog.walk())

    assert [s.name for s in services] == ["DPW_ImageryCached"]


@responses.activate
def test_find_layer_skips_service_that_404s_and_checks_the_next_one():
    responses.add(
        responses.GET,
        f"{BASE}",
        json={
            "folders": [],
            "services": [
                {"name": "OPN/Broken", "type": "MapServer"},
                {"name": "OPN/OPN_OpenDataService", "type": "MapServer"},
            ],
        },
    )
    responses.add(responses.GET, f"{BASE}/OPN/Broken/MapServer", status=404)
    responses.add(
        responses.GET,
        f"{BASE}/OPN/OPN_OpenDataService/MapServer",
        json={"layers": [{"id": 60, "name": "Streets"}]},
    )

    catalog = ArcGISCatalog(base_url=BASE)
    layer_url = catalog.find_layer("Streets", service_name_contains="")

    assert layer_url == f"{BASE}/OPN/OPN_OpenDataService/MapServer/60"


@responses.activate
def test_find_layer_raises_when_no_layer_matches():
    responses.add(
        responses.GET,
        f"{BASE}",
        json={"folders": [], "services": [{"name": "OPN/OPN_OpenDataService", "type": "MapServer"}]},
    )
    responses.add(
        responses.GET,
        f"{BASE}/OPN/OPN_OpenDataService/MapServer",
        json={"layers": [{"id": 12, "name": "Parcels"}]},
    )

    catalog = ArcGISCatalog(base_url=BASE)
    with pytest.raises(ArcGISCatalogError):
        catalog.find_layer("Streets", service_name_contains="OpenDataService")
