"""Client for the CSJ ``Streets`` FeatureServer/MapServer layer.

San Jose publishes street centerlines (with width/lane attributes, refreshed
weekly) as a queryable layer on its ArcGIS Server - see
:class:`csnav.data.arcgis.catalog.ArcGISCatalog.find_layer` for how its REST
URL is resolved without hardcoding which service currently hosts it. This
module only *queries* that layer (``/query``, paginated, output forced to
EPSG:4326 via ``outSR``) - it does not know about trajectories, tube models,
or manifest windows. Per the integration plan (`docs/INTEGRATION_PLAN.md`
§3.3), the offline ``ManifestBuilder`` is the only caller that queries this
client with a bounding box; the runtime "possible roads" lookup hits the
precomputed manifest, never this client, directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .models import Extent

#: Coordinates are always requested/returned in this spatial reference -
#: matches the "WGS84 for all storage/interop" rule in CLAUDE.md.
OUTPUT_WKID = 4326


class CSJStreetsError(RuntimeError):
    """Raised when the Streets layer returns an error payload or bad data."""


@dataclass(frozen=True)
class StreetSegment:
    """One street centerline feature, in EPSG:4326.

    ``parts`` mirrors GeoJSON LineString/MultiLineString geometry: a tuple of
    parts, each part a tuple of ``(lon, lat)`` vertex pairs - kept as plain
    tuples rather than a GeoDataFrame/shapely geometry so this module doesn't
    pull in a geospatial-geometry dependency the rest of the pipeline
    doesn't otherwise need. ``attributes`` is the raw field dict from the
    service (e.g. width/lane fields) - kept generic since the exact field
    names are a property of the live schema, not this client.
    """

    object_id: int | None
    parts: tuple[tuple[tuple[float, float], ...], ...]
    attributes: dict[str, Any]

    def to_geojson_feature(self) -> dict[str, Any]:
        geometry = (
            {"type": "LineString", "coordinates": [list(pt) for pt in self.parts[0]]}
            if len(self.parts) == 1
            else {"type": "MultiLineString", "coordinates": [[list(pt) for pt in part] for part in self.parts]}
        )
        return {"type": "Feature", "geometry": geometry, "properties": dict(self.attributes)}


def _as_part(coords: list[list[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((pt[0], pt[1]) for pt in coords)


def _segment_from_geojson_feature(feature: dict[str, Any]) -> StreetSegment:
    geometry = feature.get("geometry") or {}
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []

    if geom_type == "LineString":
        parts = (_as_part(coordinates),)
    elif geom_type == "MultiLineString":
        parts = tuple(_as_part(part) for part in coordinates)
    else:
        raise CSJStreetsError(f"unsupported street geometry type: {geom_type!r}")

    attributes = dict(feature.get("properties") or {})
    object_id = attributes.get("OBJECTID") or attributes.get("FID") or attributes.get("objectid")
    return StreetSegment(object_id=object_id, parts=parts, attributes=attributes)


class CSJStreetsClient:
    """Query one street-centerline layer's ``/query`` endpoint, paginated.

    ``layer_url`` must point at a specific layer (e.g.
    ``.../OPN_OpenDataService/MapServer/60`` or a hosted
    ``.../FeatureServer/0``), not the parent service - resolve it first with
    :meth:`csnav.data.arcgis.catalog.ArcGISCatalog.find_layer` rather than
    hardcoding it, since San Jose's exact service/layer naming can change
    independently of this client.
    """

    def __init__(
        self,
        layer_url: str,
        session: requests.Session | None = None,
        timeout: float = 60.0,
        page_size: int = 2000,
    ) -> None:
        self.layer_url = layer_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.page_size = page_size

    def get_metadata(self) -> dict[str, Any]:
        """Raw layer metadata (fields, geometryType, name, ...) as returned by ArcGIS."""
        resp = self.session.get(self.layer_url, params={"f": "json"}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise CSJStreetsError(f"ArcGIS error for {self.layer_url}: {data['error']}")
        return data

    def query(
        self,
        bbox: Extent | None = None,
        where: str = "1=1",
        out_fields: str = "*",
    ) -> list[StreetSegment]:
        """Query centerlines, optionally restricted to ``bbox``, in EPSG:4326.

        ``bbox`` must already be in EPSG:4326 (this client never does its own
        metric geometry - callers building a tube envelope should convert it
        back to WGS84, per ``geometry/local_frame.py``, before calling this).
        Paginates via ``resultOffset``/``resultRecordCount`` until the
        service reports no more results, so this is safe to call unbounded
        against a layer with more features than one page can hold.
        """
        if bbox is not None and bbox.wkid != OUTPUT_WKID:
            raise ValueError(f"bbox must be EPSG:{OUTPUT_WKID}, got wkid={bbox.wkid}")

        segments: list[StreetSegment] = []
        offset = 0
        while True:
            params: dict[str, Any] = {
                "f": "geojson",
                "where": where,
                "outFields": out_fields,
                "outSR": OUTPUT_WKID,
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": self.page_size,
            }
            if bbox is not None:
                params.update(
                    {
                        "geometry": f"{bbox.xmin},{bbox.ymin},{bbox.xmax},{bbox.ymax}",
                        "geometryType": "esriGeometryEnvelope",
                        "inSR": OUTPUT_WKID,
                        "spatialRel": "esriSpatialRelIntersects",
                    }
                )

            resp = self.session.get(f"{self.layer_url}/query", params=params, timeout=self.timeout)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as exc:
                raise CSJStreetsError(f"non-JSON response from {self.layer_url}/query") from exc
            if isinstance(data, dict) and data.get("error"):
                raise CSJStreetsError(f"ArcGIS error querying {self.layer_url}: {data['error']}")

            features = data.get("features") or []
            segments.extend(_segment_from_geojson_feature(f) for f in features)

            more = data.get("exceededTransferLimit")
            if more is None:
                more = len(features) >= self.page_size
            if not more or not features:
                break
            offset += len(features)

        return segments


def segments_from_geojson(data: dict[str, Any]) -> list[StreetSegment]:
    """Parse a GeoJSON ``FeatureCollection`` (EPSG:4326) into :class:`StreetSegment` objects.

    The inverse of :meth:`StreetSegment.to_geojson_feature`, so a cached pull
    written by ``scripts/fetch_csj_streets.py`` can be read back and fed to the
    offline manifest builder. Pinning a manifest to a flight-planning cycle
    (integration plan §3.3) needs exactly this: rebuilding from an archived
    snapshot rather than from whatever the weekly refresh currently holds.
    Non-line features are skipped rather than raising, since a mixed
    collection is a property of the export, not an error.
    """
    segments: list[StreetSegment] = []
    for feature in data.get("features") or []:
        geometry_type = (feature.get("geometry") or {}).get("type")
        if geometry_type not in ("LineString", "MultiLineString"):
            continue
        segments.append(_segment_from_geojson_feature(feature))
    return segments
