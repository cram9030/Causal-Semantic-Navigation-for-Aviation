from unittest.mock import patch

import pytest
import responses

from csnav.data.arcgis.client import (
    ArcGISClientError,
    ArcGISTileClient,
    TileTransport,
    WMTSLayerInfo,
    parse_wmts_capabilities,
)
from csnav.data.arcgis.models import Extent

SERVICE_URL = "https://example.test/server/rest/services/Imagery/DPW_ImageryCached/MapServer"
WMTS_URL = f"{SERVICE_URL}/WMTS/1.0.0/WMTSCapabilities.xml"

TILED_METADATA = {
    "capabilities": "Map,Query,TilesOnly",
    "tileInfo": {
        "rows": 256,
        "cols": 256,
        "format": "PNG32",
        "origin": {"x": -20037508.342787, "y": 20037508.342787},
        "spatialReference": {"wkid": 102100, "latestWkid": 3857},
        "lods": [
            {"level": 0, "resolution": 156543.03392800014, "scale": 591657527.591555},
            {"level": 1, "resolution": 78271.51696400007, "scale": 295828763.795777},
        ],
    },
    "fullExtent": {
        "xmin": -13611977.0, "ymin": 4442339.0, "xmax": -13563862.0, "ymax": 4489957.0,
        "spatialReference": {"wkid": 102100, "latestWkid": 3857},
    },
}

DYNAMIC_METADATA = {
    "capabilities": "Map,Query,Data",
    "fullExtent": {
        "xmin": -122.0, "ymin": 37.2, "xmax": -121.7, "ymax": 37.4,
        "spatialReference": {"wkid": 4326},
    },
}

WMTS_CAPABILITIES_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<Capabilities xmlns="http://www.opengis.net/wmts/1.0"
              xmlns:ows="http://www.opengis.net/ows/1.1">
  <Contents>
    <Layer>
      <ows:Identifier>DPW_ImageryCached</ows:Identifier>
      <Style isDefault="true">
        <ows:Identifier>default</ows:Identifier>
      </Style>
      <TileMatrixSetLink>
        <TileMatrixSet>default028mm</TileMatrixSet>
      </TileMatrixSetLink>
      <ResourceURL format="image/png" resourceType="tile"
        template="{SERVICE_URL}/WMTS/tile/1.0.0/DPW_ImageryCached/{{Style}}/{{TileMatrixSet}}/{{TileMatrix}}/{{TileRow}}/{{TileCol}}.png"/>
    </Layer>
  </Contents>
</Capabilities>
""".encode()


def _mock_metadata(payload: dict, wmts_available: bool) -> None:
    responses.add(responses.GET, SERVICE_URL, json=payload)
    if wmts_available:
        responses.add(responses.GET, WMTS_URL, body=WMTS_CAPABILITIES_XML, status=200)
    else:
        responses.add(responses.GET, WMTS_URL, status=404)


@responses.activate
def test_get_metadata_parses_tiled_service():
    _mock_metadata(TILED_METADATA, wmts_available=False)
    client = ArcGISTileClient(SERVICE_URL)

    meta = client.get_metadata()

    assert meta.supports_tiles
    assert not meta.supports_wmts
    assert meta.tile_info.rows == 256
    assert meta.tile_info.wkid == 3857
    assert meta.tile_info.lod_for_level(1).resolution == pytest.approx(78271.51696400007)
    assert meta.full_extent.wkid == 3857


@responses.activate
def test_get_metadata_parses_dynamic_service():
    _mock_metadata(DYNAMIC_METADATA, wmts_available=False)
    client = ArcGISTileClient(SERVICE_URL)

    meta = client.get_metadata()

    assert not meta.supports_tiles
    assert meta.supports_export
    assert meta.full_extent.wkid == 4326


@responses.activate
def test_best_transport_prefers_tile_over_wmts():
    # A cached MapServer's own /tile resource uses the exact z/row/col grid
    # our tile math is built from, so it wins over WMTS even when both are
    # available - avoids relying on the WMTS TileMatrix identifiers lining
    # up with tileInfo's levels.
    _mock_metadata(TILED_METADATA, wmts_available=True)
    client = ArcGISTileClient(SERVICE_URL)
    assert client.best_transport() is TileTransport.TILE


@responses.activate
def test_best_transport_falls_back_to_tile_without_wmts():
    _mock_metadata(TILED_METADATA, wmts_available=False)
    client = ArcGISTileClient(SERVICE_URL)
    assert client.best_transport() is TileTransport.TILE


@responses.activate
def test_best_transport_prefers_wmts_over_export_for_dynamic_service():
    # A dynamic (non-cached) MapServer has no tileInfo, so /tile isn't an
    # option, but WMTS can still be enabled on it - prefer that tile-aligned
    # transport over a raw /export bbox request.
    _mock_metadata(DYNAMIC_METADATA, wmts_available=True)
    client = ArcGISTileClient(SERVICE_URL)
    assert client.best_transport() is TileTransport.WMTS


@responses.activate
def test_best_transport_falls_back_to_export():
    _mock_metadata(DYNAMIC_METADATA, wmts_available=False)
    client = ArcGISTileClient(SERVICE_URL)
    assert client.best_transport() is TileTransport.EXPORT


@responses.activate
def test_best_transport_raises_when_nothing_supported():
    payload = {"capabilities": "Query"}
    _mock_metadata(payload, wmts_available=False)
    client = ArcGISTileClient(SERVICE_URL)
    with pytest.raises(ArcGISClientError):
        client.best_transport()


@responses.activate
def test_fetch_tile_builds_expected_url():
    responses.add(responses.GET, f"{SERVICE_URL}/tile/3/10/20", body=b"tile-bytes")
    client = ArcGISTileClient(SERVICE_URL)
    assert client.fetch_tile(3, 10, 20) == b"tile-bytes"


@responses.activate
def test_fetch_export_sends_expected_params():
    responses.add(responses.GET, f"{SERVICE_URL}/export", body=b"export-bytes")
    client = ArcGISTileClient(SERVICE_URL)
    bbox = Extent(xmin=-1.0, ymin=-2.0, xmax=1.0, ymax=2.0, wkid=3857)

    result = client.fetch_export(bbox, width=128, height=64)

    assert result == b"export-bytes"
    req = responses.calls[0].request
    assert "bbox=-1.0%2C-2.0%2C1.0%2C2.0" in req.url
    assert "bboxSR=3857" in req.url
    assert "size=128%2C64" in req.url


@responses.activate
def test_fetch_tile_auto_dispatches_to_tile_transport():
    _mock_metadata(TILED_METADATA, wmts_available=False)
    responses.add(responses.GET, f"{SERVICE_URL}/tile/1/0/0", body=b"tile-bytes")

    client = ArcGISTileClient(SERVICE_URL)
    assert client.fetch_tile_auto(1, 0, 0) == b"tile-bytes"


@responses.activate
def test_fetch_tile_auto_dispatches_to_wmts_for_dynamic_service():
    _mock_metadata(DYNAMIC_METADATA, wmts_available=True)
    responses.add(responses.GET, WMTS_URL, body=WMTS_CAPABILITIES_XML, status=200)
    expected_tile_url = (
        f"{SERVICE_URL}/WMTS/tile/1.0.0/DPW_ImageryCached/default/default028mm/4/1/2.png"
    )
    responses.add(responses.GET, expected_tile_url, body=b"wmts-tile-bytes")

    client = ArcGISTileClient(SERVICE_URL)
    assert client.fetch_tile_auto(4, 1, 2) == b"wmts-tile-bytes"


@responses.activate
def test_fetch_tile_auto_export_requires_tile_info():
    _mock_metadata(DYNAMIC_METADATA, wmts_available=False)
    client = ArcGISTileClient(SERVICE_URL)

    with patch.object(client, "best_transport", return_value=TileTransport.EXPORT):
        with pytest.raises(ArcGISClientError):
            client.fetch_tile_auto(0, 0, 0)


def test_parse_wmts_capabilities():
    info = parse_wmts_capabilities(WMTS_CAPABILITIES_XML)
    assert info.layer_identifier == "DPW_ImageryCached"
    assert info.tile_matrix_set == "default028mm"
    assert info.style == "default"
    assert info.resource_url_template.endswith(
        "{Style}/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.png"
    )


@responses.activate
def test_fetch_wmts_tile_builds_expected_url():
    layer_info = WMTSLayerInfo(
        layer_identifier="DPW_ImageryCached",
        tile_matrix_set="default028mm",
        style="default",
        resource_url_template=(
            f"{SERVICE_URL}/WMTS/tile/1.0.0/DPW_ImageryCached/"
            "{Style}/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.png"
        ),
    )
    expected_url = f"{SERVICE_URL}/WMTS/tile/1.0.0/DPW_ImageryCached/default/default028mm/2/5/7.png"
    responses.add(responses.GET, expected_url, body=b"wmts-tile-bytes")

    client = ArcGISTileClient(SERVICE_URL)
    result = client.fetch_wmts_tile(layer_info, matrix="2", row=5, col=7)

    assert result == b"wmts-tile-bytes"


@responses.activate
def test_fetch_wmts_tile_raises_clear_error_for_unknown_placeholder():
    layer_info = WMTSLayerInfo(
        layer_identifier="DPW_ImageryCached",
        tile_matrix_set="default028mm",
        style="default",
        resource_url_template=f"{SERVICE_URL}/WMTS/tile/1.0.0/{{Layer}}/{{TileMatrix}}/{{TileRow}}/{{TileCol}}.png",
    )
    client = ArcGISTileClient(SERVICE_URL)

    with pytest.raises(ArcGISClientError):
        client.fetch_wmts_tile(layer_info, matrix="2", row=5, col=7)
