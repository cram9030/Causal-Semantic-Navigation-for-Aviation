"""Frame conversion for shapely geometries: WGS84 <-> local ENU.

:mod:`csnav.geometry.local_frame` converts points; this converts whole
geometries, so that clipping, buffering, and distance work can be done in
meters in a local ENU tangent plane and the result handed back in WGS84 for
storage - the pattern CLAUDE.md requires of every metric geometry operation.

Convention: WGS84 geometries are in ``(lon, lat)`` axis order (GeoJSON /
shapely convention), ENU geometries in ``(east, north)`` meters. Both are 2D -
the vertical component is carried separately (waypoint height, AGL), not in
these geometries.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from csnav.geometry.local_frame import LocalFrame


def to_enu(geometry: BaseGeometry, frame: LocalFrame) -> BaseGeometry:
    """Convert a WGS84 ``(lon, lat)`` geometry into ``frame``'s ENU meters."""

    def project(coords: np.ndarray) -> np.ndarray:
        points = [frame.to_enu(lat, lon) for lon, lat in coords]
        return np.array([[point.east, point.north] for point in points], dtype=float).reshape(len(points), 2)

    return shapely.transform(geometry, project)


def to_wgs84(geometry: BaseGeometry, frame: LocalFrame) -> BaseGeometry:
    """Convert an ENU-meter geometry in ``frame`` back to WGS84 ``(lon, lat)``."""

    def project(coords: np.ndarray) -> np.ndarray:
        points = [frame.to_wgs84(east, north) for east, north in coords]
        return np.array([[point.lon, point.lat] for point in points], dtype=float).reshape(len(points), 2)

    return shapely.transform(geometry, project)


def line_parts(geometry: BaseGeometry) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Flatten a (Multi)LineString into the tuple-of-parts form used across this codebase.

    Matches :attr:`csnav.data.arcgis.streets.StreetSegment.parts` and
    :attr:`csnav.trajectory.manifest.ManifestLandmark.parts`. Non-linear
    fragments a clip can leave behind (a tangent Point, an empty geometry) are
    dropped, and parts with fewer than two vertices are skipped.
    """
    if geometry.is_empty:
        return ()
    if isinstance(geometry, LineString):
        candidates: list[LineString] = [geometry]
    elif isinstance(geometry, MultiLineString):
        candidates = list(geometry.geoms)
    elif hasattr(geometry, "geoms"):
        candidates = [geom for geom in geometry.geoms if isinstance(geom, LineString)]
    else:
        return ()
    return tuple(
        tuple((x, y) for x, y in line.coords) for line in candidates if not line.is_empty and len(line.coords) > 1
    )


def polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    """Flatten a Polygon/MultiPolygon into a tuple of Polygons (empties dropped)."""
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(part for part in geometry.geoms if not part.is_empty)
    return tuple(part for part in getattr(geometry, "geoms", ()) if isinstance(part, Polygon) and not part.is_empty)


def point_parts(geometry: BaseGeometry) -> tuple[Point, ...]:
    """Flatten any geometry into the Point components it contains (empties dropped)."""
    if geometry.is_empty:
        return ()
    if isinstance(geometry, Point):
        return (geometry,)
    if hasattr(geometry, "geoms"):
        points: list[Point] = []
        for part in geometry.geoms:
            points.extend(point_parts(part))
        return tuple(points)
    return ()
