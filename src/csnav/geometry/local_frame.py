"""WGS84 (EPSG:4326) <-> local East-North-Up (ENU) tangent-plane conversions.

Degrees of latitude/longitude are not uniform distances, so every metric
geometry operation downstream of this module - RNP tube containment, street
buffer widths, FOV-to-ground-distance projection - must not do distance/area
math directly on raw WGS84 degrees. Convert to a :class:`LocalFrame` with
`to_enu`, do the metric math there (meters), then convert the result back to
WGS84 with `to_wgs84` before storing it or handing it to the SCM. See
`docs/INTEGRATION_PLAN.md` §2 and §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyproj import Transformer


@dataclass(frozen=True)
class Point:
    """A point in a :class:`LocalFrame`'s ENU tangent plane, in meters, relative to that frame's origin."""

    east: float
    north: float
    up: float = 0.0


@dataclass(frozen=True)
class LatLon:
    """A WGS84 geographic point.

    lat/lon in decimal degrees (EPSG:4326); height in meters above the
    WGS84 ellipsoid.
    """

    lat: float
    lon: float
    height: float = 0.0


class LocalFrame:
    """Local East-North-Up (ENU) tangent plane anchored at a WGS84 origin.

    Built on PROJ's geocentric ("cart") + topocentric pipeline, so it
    accounts for the WGS84 ellipsoid rather than a flat-earth
    approximation. Accuracy is best near the origin and degrades with
    distance from it, so per `docs/INTEGRATION_PLAN.md` §2/§3.2, long
    flights should re-anchor a new `LocalFrame` per trajectory window
    rather than reuse one anchor for the whole flight.
    """

    def __init__(self, origin_lat: float, origin_lon: float, origin_height: float = 0.0) -> None:
        """origin_lat/origin_lon: decimal degrees (WGS84). origin_height: meters above the WGS84 ellipsoid."""
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.origin_height = origin_height
        self._transformer = Transformer.from_pipeline(
            "+proj=pipeline"
            " +step +proj=cart +ellps=WGS84"
            f" +step +proj=topocentric +ellps=WGS84"
            f" +lat_0={origin_lat} +lon_0={origin_lon} +h_0={origin_height}"
        )

    def to_enu(self, lat: float, lon: float, height: float = 0.0) -> Point:
        """Convert a WGS84 point (degrees, +meters height) to this frame's local ENU meters."""
        east, north, up = self._transformer.transform(lon, lat, height)
        return Point(east=east, north=north, up=up)

    def to_wgs84(self, x: float, y: float, up: float = 0.0) -> LatLon:
        """Convert local ENU meters (relative to this frame's origin) back to WGS84 (degrees, +meters height)."""
        lon, lat, height = self._transformer.transform(x, y, up, direction="INVERSE")
        return LatLon(lat=lat, lon=lon, height=height)
