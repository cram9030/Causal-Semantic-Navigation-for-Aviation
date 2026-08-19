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
