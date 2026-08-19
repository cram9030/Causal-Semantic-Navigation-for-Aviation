"""Coordinate transforms between EPSG:4326 (lon/lat) and EPSG:3857 (Web Mercator).

ArcGIS Server tiled imagery services are almost always cached in Web Mercator
(EPSG:3857 / WKID 102100 or 3857), while downstream training data for this
project is kept in geographic coordinates (EPSG:4326). These helpers centralize
the handful of conversions the catalog, tile-grid math, and reprojection
modules all need.
"""

from __future__ import annotations

from pyproj import Transformer

from .models import Extent

# ArcGIS reports Web Mercator as either wkid 102100 or 3857; both mean the
# same projection (EPSG:3857) for our purposes.
WEB_MERCATOR_WKIDS = frozenset({102100, 102113, 3857})

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    return _TO_3857.transform(lon, lat)


def xy_3857_to_lonlat(x: float, y: float) -> tuple[float, float]:
    return _TO_4326.transform(x, y)


def extent_4326_to_3857(extent: Extent) -> Extent:
    if extent.wkid != 4326:
        raise ValueError(f"expected an EPSG:4326 extent, got wkid={extent.wkid}")
    xmin, ymin = lonlat_to_3857(extent.xmin, extent.ymin)
    xmax, ymax = lonlat_to_3857(extent.xmax, extent.ymax)
    return Extent(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, wkid=3857)


def extent_3857_to_4326(extent: Extent) -> Extent:
    if extent.wkid not in WEB_MERCATOR_WKIDS:
        raise ValueError(f"expected a Web Mercator extent, got wkid={extent.wkid}")
    xmin, ymin = xy_3857_to_lonlat(extent.xmin, extent.ymin)
    xmax, ymax = xy_3857_to_lonlat(extent.xmax, extent.ymax)
    return Extent(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, wkid=4326)


def is_web_mercator(wkid: int) -> bool:
    return wkid in WEB_MERCATOR_WKIDS
