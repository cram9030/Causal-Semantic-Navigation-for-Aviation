"""Tile-grid math for ArcGIS cached map services.

ArcGIS tile schemes are defined by a service-specific ``tileInfo`` (origin +
per-level resolution), not the fixed global XYZ/slippy-map grid, so tile
bounds and the row/col range covering an area of interest must be derived
from that ``tileInfo`` rather than assumed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from .models import Extent, TileInfo


def tile_bounds(tile_info: TileInfo, level: int, row: int, col: int) -> Extent:
    """Bounds of a single tile, in the tile scheme's spatial reference."""
    lod = tile_info.lod_for_level(level)
    tile_width = tile_info.cols * lod.resolution
    tile_height = tile_info.rows * lod.resolution

    xmin = tile_info.origin_x + col * tile_width
    xmax = xmin + tile_width
    ymax = tile_info.origin_y - row * tile_height
    ymin = ymax - tile_height

    return Extent(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, wkid=tile_info.wkid)


def tiles_covering_extent(tile_info: TileInfo, level: int, extent: Extent) -> Iterator[tuple[int, int]]:
    """Yield ``(row, col)`` pairs for every tile intersecting ``extent``.

    ``extent`` must already be in the tile scheme's spatial reference
    (``tile_info.wkid``) - reproject with :mod:`csnav.data.arcgis.projections`
    first if the area of interest was captured in EPSG:4326.
    """
    if extent.wkid != tile_info.wkid:
        raise ValueError(
            f"extent wkid {extent.wkid} does not match tile scheme wkid {tile_info.wkid}; "
            "reproject the extent before calling tiles_covering_extent"
        )

    lod = tile_info.lod_for_level(level)
    tile_width = tile_info.cols * lod.resolution
    tile_height = tile_info.rows * lod.resolution

    col_min = math.floor((extent.xmin - tile_info.origin_x) / tile_width)
    col_max = math.floor((extent.xmax - tile_info.origin_x) / tile_width - 1e-9)
    row_min = math.floor((tile_info.origin_y - extent.ymax) / tile_height)
    row_max = math.floor((tile_info.origin_y - extent.ymin) / tile_height - 1e-9)

    for row in range(max(row_min, 0), row_max + 1):
        for col in range(max(col_min, 0), col_max + 1):
            yield row, col


def best_level_for_resolution(tile_info: TileInfo, target_resolution: float) -> int:
    """Pick the coarsest level detailed enough to meet ``target_resolution``.

    ``resolution`` is map units per pixel, so smaller means more detail.
    This returns the level with the largest resolution that still satisfies
    ``resolution <= target_resolution`` - the least amount of data that meets
    the requested precision - falling back to the finest available level if
    every level is coarser than requested.
    """
    candidates = [lod for lod in tile_info.lods if lod.resolution <= target_resolution]
    if candidates:
        return max(candidates, key=lambda lod: lod.resolution).level
    return min(tile_info.lods, key=lambda lod: lod.resolution).level
