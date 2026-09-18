"""Client for the CSJ ``Street Intersections`` layer (point features).

San Jose separately publishes named street-intersection *locations* - not
street geometry - as their own layer, distinct from the `Streets` centerline
layer :mod:`csnav.data.arcgis.streets` queries. Confirmed against the live
schema (``geo.sanjoseca.gov``, ``OPN/OPN_OpenDataService/MapServer/276``,
display name "Street Intersections"):

* ``geometryType`` is ``esriGeometryPoint`` - this layer carries intersection
  *locations*, never a polygon footprint. That directly bears on how
  `csnav.data.ground_truth.rasterize` can use it: it can enrich a derived
  intersection instance with a real name/leg-count/traffic-control-type once
  matched to one of these points, but it cannot supply a physical shape to
  rasterize instead of the width-derived circle already used there - there
  is no polygon geometry here to measure one from, and inventing a
  per-``INTERSECTIONTYPE`` size multiplier (e.g. "a 4-leg intersection is N
  meters bigger") would just be a different unverified guess. See
  `docs/phase2_ground_truth_rasterization.md`'s "A derived intersection's
  radius scales with the roads meeting there" section.
* ``INTTYPE`` is a coded-value domain (``End``, ``Intersection``, ``Muni``,
  ``Non-Intersection``, ``Ramp``, ``Range Exception``) - only
  ``'Intersection'`` rows are genuine street-crossing junctions; the rest are
  reference points for other purposes (a dead end, a municipal boundary
  marker, a ramp gore, ...). ``DEFAULT_WHERE`` filters to it, the same
  "filter at the query" reasoning `csnav.data.arcgis.streets`'s
  ``FEATURECLASS='StreetCenterline'`` filter documents for the Streets
  layer.
* ``INTERSECTIONTYPE`` (a *different* field - leg count: ``2 Leg``,
  ``3 Leg``, ``4 Leg``, ``Light Rail Crossing``) and ``TRAFFICCONTROLTYPE``
  (``Signal``, ``4 Way Stop``, ...) are read through, unfiltered, as
  metadata on a matched instance - see
  :func:`csnav.data.ground_truth.rasterize`'s intersection-matching step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from shapely.geometry import Point

from .models import Extent

#: Coordinates are always requested/returned in this spatial reference -
#: matches the "WGS84 for all storage/interop" rule in CLAUDE.md.
OUTPUT_WKID = 4326

#: The confirmed CSJ Street Intersections layer - see this module's
#: docstring. Pinned by URL (not resolved by name at import time) for the
#: same reason `csnav.data.arcgis.streets.DEFAULT_LAYER_URL`-equivalent
#: pinning exists there: San Jose's catalog has more than one layer whose
#: name contains "Intersection" (traffic-signal-specific layers, a railroad-
#: crossing layer, planning-department duplicates of this same layer under
#: different services), and picking the wrong one by luck of discovery order
#: is exactly the failure mode that pinning avoids.
DEFAULT_LAYER_URL = "https://geo.sanjoseca.gov/server/rest/services/OPN/OPN_OpenDataService/MapServer/276"

#: Only ``INTTYPE='Intersection'`` rows are genuine street-crossing
#: junctions - see this module's docstring.
DEFAULT_WHERE = "INTTYPE='Intersection'"

#: Field names tried for a human-readable intersection name.
NAME_FIELD_CANDIDATES = ("INTNAME",)

#: Field name for the leg-count classification (``2 Leg``/``3 Leg``/
#: ``4 Leg``/``Light Rail Crossing``).
LEG_COUNT_FIELD_CANDIDATES = ("INTERSECTIONTYPE",)

#: Field name for the traffic-control classification (``Signal``,
#: ``4 Way Stop``, ``No Control``, ...).
TRAFFIC_CONTROL_FIELD_CANDIDATES = ("TRAFFICCONTROLTYPE",)


class CSJIntersectionsError(RuntimeError):
    """Raised when the Street Intersections layer returns an error payload or bad data."""


@dataclass(frozen=True)
class StreetIntersection:
    """One named street-intersection location, in EPSG:4326.

    ``attributes`` is the raw field dict from the service - kept generic
    (mirroring `csnav.data.arcgis.streets.StreetSegment.attributes`) rather
    than pulled apart into named fields here, since only a few of its fields
    (name, leg count, traffic control) are actually read by this project;
    :func:`intersection_name`/:func:`intersection_leg_count`/
    :func:`intersection_traffic_control` are the lookups for those.
    """

    object_id: int | None
    lon: float
    lat: float
    attributes: dict[str, Any]

    def to_geojson_feature(self) -> dict[str, Any]:
        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
            "properties": dict(self.attributes),
        }


def intersection_point(intersection: StreetIntersection) -> Point:
    """A street intersection's location as a shapely geometry, in WGS84 ``(lon, lat)``."""
    return Point(intersection.lon, intersection.lat)


def _first_present(attributes: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    for key in candidates:
        value = attributes.get(key)
        if value not in (None, ""):
            return value
    return None


def intersection_name(attributes: dict[str, Any]) -> str | None:
    """Human-readable intersection name from the CSJ attributes, or ``None`` if not published."""
    return _first_present(attributes, NAME_FIELD_CANDIDATES)


def intersection_leg_count(attributes: dict[str, Any]) -> str | None:
    """The leg-count classification (e.g. ``"4 Leg"``) from the CSJ attributes, or ``None``."""
    return _first_present(attributes, LEG_COUNT_FIELD_CANDIDATES)


def intersection_traffic_control(attributes: dict[str, Any]) -> str | None:
    """The traffic-control classification (e.g. ``"Signal"``) from the CSJ attributes, or ``None``."""
    return _first_present(attributes, TRAFFIC_CONTROL_FIELD_CANDIDATES)


def _intersection_from_geojson_feature(feature: dict[str, Any]) -> StreetIntersection:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        raise CSJIntersectionsError(f"unsupported street intersection geometry type: {geometry.get('type')!r}")
    coordinates = geometry.get("coordinates") or [None, None]

    attributes = dict(feature.get("properties") or {})
    object_id = attributes.get("OBJECTID") or attributes.get("FID") or attributes.get("objectid")
    return StreetIntersection(object_id=object_id, lon=coordinates[0], lat=coordinates[1], attributes=attributes)


class CSJIntersectionsClient:
    """Query the CSJ Street Intersections layer's ``/query`` endpoint, paginated.

    Structurally mirrors `csnav.data.arcgis.streets.CSJStreetsClient` (same
    pagination/bbox/error-handling shape against the same kind of ArcGIS
    ``/query`` endpoint) but parses point, not line, geometry - kept as a
    separate class rather than a shared base since the two layers' geometry
    parsing genuinely differs and neither reuses the other's query logic
    elsewhere in this codebase.
    """

    def __init__(
        self,
        layer_url: str = DEFAULT_LAYER_URL,
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
            raise CSJIntersectionsError(f"ArcGIS error for {self.layer_url}: {data['error']}")
        return data

    def query(
        self, bbox: Extent | None = None, where: str = DEFAULT_WHERE, out_fields: str = "*"
    ) -> list[StreetIntersection]:
        """Query intersections, optionally restricted to ``bbox``, in EPSG:4326.

        ``bbox`` must already be in EPSG:4326. Paginates via
        ``resultOffset``/``resultRecordCount`` until the service reports no
        more results, matching `csnav.data.arcgis.streets.CSJStreetsClient.query`.
        """
        if bbox is not None and bbox.wkid != OUTPUT_WKID:
            raise ValueError(f"bbox must be EPSG:{OUTPUT_WKID}, got wkid={bbox.wkid}")

        intersections: list[StreetIntersection] = []
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
                raise CSJIntersectionsError(f"non-JSON response from {self.layer_url}/query") from exc
            if isinstance(data, dict) and data.get("error"):
                raise CSJIntersectionsError(f"ArcGIS error querying {self.layer_url}: {data['error']}")

            features = data.get("features") or []
            intersections.extend(_intersection_from_geojson_feature(f) for f in features)

            more = data.get("exceededTransferLimit")
            if more is None:
                more = len(features) >= self.page_size
            if not more or not features:
                break
            offset += len(features)

        return intersections


def intersections_from_geojson(data: dict[str, Any]) -> list[StreetIntersection]:
    """Parse a GeoJSON ``FeatureCollection`` (EPSG:4326) into :class:`StreetIntersection` objects.

    The inverse of :meth:`StreetIntersection.to_geojson_feature`, so a cached
    pull written by ``scripts/fetch_csj_intersections.py`` can be read back
    and fed to :meth:`csnav.data.ground_truth.rasterize.GroundTruthBuilder.rasterize`.
    Non-point features are skipped rather than raising, mirroring
    `csnav.data.arcgis.streets.segments_from_geojson`.
    """
    intersections: list[StreetIntersection] = []
    for feature in data.get("features") or []:
        if (feature.get("geometry") or {}).get("type") != "Point":
            continue
        intersections.append(_intersection_from_geojson_feature(feature))
    return intersections
