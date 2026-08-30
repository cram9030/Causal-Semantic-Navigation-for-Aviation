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
from shapely.geometry import LineString, MultiLineString

from .models import Extent

#: Coordinates are always requested/returned in this spatial reference -
#: matches the "WGS84 for all storage/interop" rule in CLAUDE.md.
OUTPUT_WKID = 4326

#: Field names the CSJ Streets schema has used for roadway width, tried in
#: order. The live schema owns these names, so this is a lookup list rather
#: than a fixed contract - callers get ``None`` when none of them is present.
#: Shared by :mod:`csnav.trajectory.manifest_builder` (candidate-road width)
#: and :mod:`csnav.data.ground_truth.rasterize` (buffer width), so both read
#: the CSJ schema the same way.
WIDTH_FIELD_CANDIDATES = ("WIDTH", "Width", "width", "ROADWIDTH", "RoadWidth", "PAVED_WIDTH", "STREETWIDTH")

#: Field names tried for a human-readable street name, same caveat.
NAME_FIELD_CANDIDATES = ("STREETNAME", "StreetName", "FULLNAME", "FullName", "NAME", "Name", "name")


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


def segment_geometry(segment: StreetSegment) -> LineString | MultiLineString:
    """A street segment's centerline as a shapely geometry, in WGS84 (lon, lat)."""
    if len(segment.parts) == 1:
        return LineString(segment.parts[0])
    return MultiLineString([list(part) for part in segment.parts])


def _first_present(attributes: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        value = attributes.get(key)
        if value not in (None, ""):
            return value
    return None


def street_width_m(attributes: dict[str, Any]) -> float | None:
    """Roadway width in meters from the CSJ attributes, or ``None`` if not published.

    CSJ publishes widths in feet; the value is converted to meters here so
    everything downstream (manifest offsets, ground-truth buffers) is metric,
    matching the ENU frame used for that math.
    """
    raw = _first_present(attributes, WIDTH_FIELD_CANDIDATES)
    if raw is None:
        return None
    try:
        return float(raw) * 0.3048
    except (TypeError, ValueError):
        return None


def street_name(attributes: dict[str, Any]) -> str | None:
    """Human-readable street name from the CSJ attributes, or ``None`` if not published."""
    return _first_present(attributes, NAME_FIELD_CANDIDATES)


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
        historic_moment: str | int | None = None,
    ) -> list[StreetSegment]:
        """Query centerlines, optionally restricted to ``bbox``, in EPSG:4326.

        ``bbox`` must already be in EPSG:4326 (this client never does its own
        metric geometry - callers building a tube envelope should convert it
        back to WGS84, per ``geometry/local_frame.py``, before calling this).
        Paginates via ``resultOffset``/``resultRecordCount`` until the
        service reports no more results, so this is safe to call unbounded
        against a layer with more features than one page can hold.

        ``historic_moment`` forwards ArcGIS's ``historicMoment`` query
        parameter (an epoch-millisecond timestamp, or an ISO 8601 string the
        server accepts) - the standard way to read an *archiving-enabled*
        layer as of a past edit moment, which is what pairing ground-truth
        labels with a historic imagery vintage needs (CSJ Streets only
        publishes the current network otherwise). Whether this specific
        layer has archiving enabled is **not confirmed** - inspect
        :meth:`get_metadata`'s ``archivingInfo`` field first, or expect this
        to have no effect (current-moment data returned unchanged) if it
        doesn't. See ``docs/phase2_ground_truth_rasterization.md``.
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
            if historic_moment is not None:
                params["historicMoment"] = historic_moment
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
