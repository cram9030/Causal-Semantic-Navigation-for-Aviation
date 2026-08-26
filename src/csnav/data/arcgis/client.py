"""Client for fetching imagery from a single ArcGIS MapServer/ImageServer.

Supports three transports, in order of preference:

1. **WMTS** - standards-based, used when the service exposes a
   ``WMTS/1.0.0/WMTSCapabilities.xml`` document.
2. **``/tile/{z}/{y}/{x}``** - ArcGIS's native cached-tile resource, used when
   the service is a tiled (cached) MapServer.
3. **``/export``** - dynamic image export, the fallback for non-cached
   services or arbitrary (non tile-aligned) bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from xml.etree import ElementTree as ET

import requests

from .models import Extent, LevelOfDetail, ServiceMetadata, TileInfo


class TileTransport(str, Enum):
    WMTS = "wmts"
    TILE = "tile"
    EXPORT = "export"


class ArcGISClientError(RuntimeError):
    """Raised on malformed responses or when no supported transport is available."""


_WMTS_NAMESPACES = {
    "wmts": "http://www.opengis.net/wmts/1.0",
    "ows": "http://www.opengis.net/ows/1.1",
}


@dataclass(frozen=True)
class WMTSLayerInfo:
    layer_identifier: str
    tile_matrix_set: str
    resource_url_template: str
    style: str = "default"


def _parse_default_style(layer: ET.Element) -> str:
    style = None
    for elem in layer.findall("wmts:Style", _WMTS_NAMESPACES):
        identifier = elem.findtext("ows:Identifier", namespaces=_WMTS_NAMESPACES)
        is_default = elem.attrib.get("isDefault") == "true"
        if style is None or is_default:
            style = identifier or style
        if is_default:
            break
    return style or "default"


def parse_wmts_capabilities(xml_bytes: bytes) -> WMTSLayerInfo:
    root = ET.fromstring(xml_bytes)
    contents = root.find("wmts:Contents", _WMTS_NAMESPACES)
    if contents is None:
        raise ArcGISClientError("WMTS capabilities document has no <Contents>")

    layer = contents.find("wmts:Layer", _WMTS_NAMESPACES)
    if layer is None:
        raise ArcGISClientError("WMTS capabilities document has no <Layer>")

    identifier = layer.findtext("ows:Identifier", namespaces=_WMTS_NAMESPACES) or ""
    tile_matrix_set = (
        layer.findtext("wmts:TileMatrixSetLink/wmts:TileMatrixSet", namespaces=_WMTS_NAMESPACES) or ""
    )
    style = _parse_default_style(layer)

    resource_url = layer.find("wmts:ResourceURL[@resourceType='tile']", _WMTS_NAMESPACES)
    if resource_url is None or "template" not in resource_url.attrib:
        raise ArcGISClientError("WMTS capabilities document has no tile ResourceURL template")

    return WMTSLayerInfo(
        layer_identifier=identifier,
        tile_matrix_set=tile_matrix_set,
        resource_url_template=resource_url.attrib["template"],
        style=style,
    )


class ArcGISTileClient:
    def __init__(
        self,
        service_url: str,
        session: requests.Session | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self._metadata: ServiceMetadata | None = None

    # -- metadata -------------------------------------------------------

    def get_metadata(self, refresh: bool = False, probe_wmts: bool = True) -> ServiceMetadata:
        if self._metadata is not None and not refresh:
            return self._metadata

        resp = self.session.get(self.service_url, params={"f": "json"}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise ArcGISClientError(f"ArcGIS error for {self.service_url}: {data['error']}")

        capabilities = tuple(
            c.strip() for c in str(data.get("capabilities", "")).split(",") if c.strip()
        )

        tile_info = None
        raw_tile_info = data.get("tileInfo")
        if raw_tile_info:
            origin = raw_tile_info.get("origin", {})
            sr = raw_tile_info.get("spatialReference", {})
            wkid = sr.get("latestWkid") or sr.get("wkid")
            lods = tuple(
                LevelOfDetail(level=lod["level"], resolution=lod["resolution"], scale=lod["scale"])
                for lod in raw_tile_info.get("lods", [])
            )
            tile_info = TileInfo(
                rows=raw_tile_info.get("rows", 256),
                cols=raw_tile_info.get("cols", 256),
                image_format=raw_tile_info.get("format", "PNG32"),
                origin_x=origin.get("x", 0.0),
                origin_y=origin.get("y", 0.0),
                wkid=wkid,
                lods=lods,
            )

        full_extent = None
        raw_extent = data.get("fullExtent")
        if raw_extent:
            sr = raw_extent.get("spatialReference", {})
            wkid = sr.get("latestWkid") or sr.get("wkid") or 4326
            full_extent = Extent(
                xmin=raw_extent["xmin"],
                ymin=raw_extent["ymin"],
                xmax=raw_extent["xmax"],
                ymax=raw_extent["ymax"],
                wkid=wkid,
            )

        wmts_url = self._probe_wmts() if probe_wmts else None

        self._metadata = ServiceMetadata(
            service_url=self.service_url,
            capabilities=capabilities,
            tile_info=tile_info,
            full_extent=full_extent,
            wmts_url=wmts_url,
        )
        return self._metadata

    def _probe_wmts(self) -> str | None:
        candidate = f"{self.service_url}/WMTS/1.0.0/WMTSCapabilities.xml"
        try:
            resp = self.session.get(candidate, timeout=self.timeout)
        except requests.RequestException:
            return None
        if resp.status_code == 200 and b"Capabilities" in resp.content[:2000]:
            return candidate
        return None

    def best_transport(self) -> TileTransport:
        meta = self.get_metadata()
        if meta.supports_wmts:
            return TileTransport.WMTS
        if meta.supports_tiles:
            return TileTransport.TILE
        if meta.supports_export:
            return TileTransport.EXPORT
        raise ArcGISClientError(f"{self.service_url} exposes no supported tile transport")

    # -- fetch: ArcGIS-native cached tile --------------------------------

    def fetch_tile(self, level: int, row: int, col: int) -> bytes:
        url = f"{self.service_url}/tile/{level}/{row}/{col}"
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    # -- fetch: dynamic export ------------------------------------------

    def fetch_export(
        self,
        bbox: Extent,
        width: int = 512,
        height: int = 512,
        image_format: str = "png32",
    ) -> bytes:
        params = {
            "f": "image",
            "bbox": f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax}",
            "bboxSR": bbox.wkid,
            "imageSR": bbox.wkid,
            "size": f"{width},{height}",
            "format": image_format,
            "transparent": "false",
        }
        resp = self.session.get(f"{self.service_url}/export", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    # -- fetch: WMTS ------------------------------------------------------

    def get_wmts_layer_info(self) -> WMTSLayerInfo:
        meta = self.get_metadata()
        if not meta.wmts_url:
            raise ArcGISClientError(f"{self.service_url} does not expose WMTS")
        resp = self.session.get(meta.wmts_url, timeout=self.timeout)
        resp.raise_for_status()
        return parse_wmts_capabilities(resp.content)

    def fetch_wmts_tile(self, layer_info: WMTSLayerInfo, matrix: str, row: int, col: int) -> bytes:
        fields = {
            "Style": layer_info.style,
            "TileMatrixSet": layer_info.tile_matrix_set,
            "TileMatrix": matrix,
            "TileRow": row,
            "TileCol": col,
        }
        try:
            url = layer_info.resource_url_template.format(**fields)
        except KeyError as exc:
            raise ArcGISClientError(
                f"WMTS ResourceURL template {layer_info.resource_url_template!r} references "
                f"placeholder {exc}, which is not one of {sorted(fields)}"
            ) from exc
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content

    # -- unified dispatch -------------------------------------------------

    def fetch_tile_auto(self, level: int, row: int, col: int) -> bytes:
        """Fetch a single tile via whichever transport the service supports best."""
        transport = self.best_transport()
        if transport is TileTransport.TILE:
            return self.fetch_tile(level, row, col)
        if transport is TileTransport.WMTS:
            layer_info = self.get_wmts_layer_info()
            meta = self.get_metadata()
            assert meta.tile_info is not None
            return self.fetch_wmts_tile(layer_info, matrix=str(level), row=row, col=col)
        # EXPORT: derive the tile's bounds from tileInfo and export that bbox.
        from .tiles import tile_bounds  # local import to avoid a cycle at module load

        meta = self.get_metadata()
        if meta.tile_info is None:
            raise ArcGISClientError(
                f"{self.service_url} has no tileInfo; fetch_export() with an explicit bbox instead"
            )
        bounds = tile_bounds(meta.tile_info, level, row, col)
        return self.fetch_export(bounds, width=meta.tile_info.cols, height=meta.tile_info.rows)
