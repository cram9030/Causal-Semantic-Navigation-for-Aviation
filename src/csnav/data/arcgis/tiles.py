"""Tile-grid math for ArcGIS cached map services.

ArcGIS tile schemes are defined by a service-specific ``tileInfo`` (origin +
per-level resolution), not the fixed global XYZ/slippy-map grid, so tile
bounds and the row/col range covering an area of interest must be derived
from that ``tileInfo`` rather than assumed.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from .models import Extent, LevelOfDetail, TileInfo


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


def tile_row_col_range(tile_info: TileInfo, level: int, extent: Extent) -> tuple[int, int, int, int]:
    """Return ``(row_min, row_max, col_min, col_max)`` (inclusive) covering ``extent``.

    This is the shared math behind :func:`tiles_covering_extent`,
    :func:`tile_count_covering_extent`, and :func:`sample_tiles_covering_extent`
    - computing just the bounding range is O(1), so callers that only need a
    count or a sample don't have to enumerate (or hold in memory) every tile,
    which matters at fine zoom levels where a single AOI can cover millions
    of tiles.

    ``extent`` must already be in the tile scheme's spatial reference
    (``tile_info.wkid``) - reproject with :mod:`csnav.data.arcgis.projections`
    first if the area of interest was captured in EPSG:4326.
    """
    if extent.wkid != tile_info.wkid:
        raise ValueError(
            f"extent wkid {extent.wkid} does not match tile scheme wkid {tile_info.wkid}; "
            "reproject the extent before calling tile_row_col_range"
        )

    lod = tile_info.lod_for_level(level)
    tile_width = tile_info.cols * lod.resolution
    tile_height = tile_info.rows * lod.resolution

    col_min = math.floor((extent.xmin - tile_info.origin_x) / tile_width)
    col_max = math.floor((extent.xmax - tile_info.origin_x) / tile_width - 1e-9)
    row_min = math.floor((tile_info.origin_y - extent.ymax) / tile_height)
    row_max = math.floor((tile_info.origin_y - extent.ymin) / tile_height - 1e-9)

    return max(row_min, 0), row_max, max(col_min, 0), col_max


def tiles_covering_extent(tile_info: TileInfo, level: int, extent: Extent) -> Iterator[tuple[int, int]]:
    """Yield ``(row, col)`` pairs for every tile intersecting ``extent``.

    ``extent`` must already be in the tile scheme's spatial reference
    (``tile_info.wkid``) - reproject with :mod:`csnav.data.arcgis.projections`
    first if the area of interest was captured in EPSG:4326.
    """
    row_min, row_max, col_min, col_max = tile_row_col_range(tile_info, level, extent)
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            yield row, col


def tile_count_covering_extent(tile_info: TileInfo, level: int, extent: Extent) -> int:
    """Number of tiles :func:`tiles_covering_extent` would yield, without enumerating them."""
    row_min, row_max, col_min, col_max = tile_row_col_range(tile_info, level, extent)
    return max(row_max - row_min + 1, 0) * max(col_max - col_min + 1, 0)


def sample_tiles_covering_extent(
    tile_info: TileInfo, level: int, extent: Extent, sample_size: int
) -> list[tuple[int, int]]:
    """Up to ``sample_size`` ``(row, col)`` pairs spread evenly across ``extent``.

    Unlike ``list(tiles_covering_extent(...))``, this never materializes the
    full tile grid - it walks a coarse stride over the row/col range - so
    it's safe to call even when that grid would be far too large to hold in
    memory (e.g. checking a fine level's coverage before committing to it).
    """
    row_min, row_max, col_min, col_max = tile_row_col_range(tile_info, level, extent)
    row_count = row_max - row_min + 1
    col_count = col_max - col_min + 1
    if row_count <= 0 or col_count <= 0:
        return []

    total = row_count * col_count
    if total <= sample_size:
        return [(row, col) for row in range(row_min, row_max + 1) for col in range(col_min, col_max + 1)]

    side = max(1, math.isqrt(sample_size))
    row_step = max(1, row_count // side)
    col_step = max(1, col_count // side)

    coords: list[tuple[int, int]] = []
    for row in range(row_min, row_max + 1, row_step):
        for col in range(col_min, col_max + 1, col_step):
            coords.append((row, col))
            if len(coords) >= sample_size:
                return coords
    return coords


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


#: The standard Web Mercator (EPSG:3857) tiling scheme shared by ArcGIS
#: Online, Google/OSM XYZ tiles, and every ArcGIS cache published against
#: Esri's "ArcGIS Online / Bing Maps / Google Maps" scheme - which
#: ``DPW_ImageryCached2025`` is. Level 0 is one 256x256 tile spanning the
#: world; each level halves the resolution.
WEB_MERCATOR_ORIGIN = 20037508.342787
WEB_MERCATOR_LEVEL0_RESOLUTION = 156543.033928
_ESRI_DPI = 96.0
_METERS_PER_INCH = 0.0254


def web_mercator_tile_info(max_level: int = 23, tile_size: int = 256) -> TileInfo:
    """The standard EPSG:3857 tiling scheme, levels ``0..max_level``.

    A convenience for callers that need tile-grid math without a live service
    request - e.g. drawing which imagery tiles a trajectory's field of view
    covers. When a service's own ``tileInfo`` is available (from
    :class:`csnav.data.arcgis.client.ArcGISTileClient`), prefer it: this is
    the *standard* scheme, and a service is free to publish a custom one.
    """
    lods = tuple(
        LevelOfDetail(
            level=level,
            resolution=WEB_MERCATOR_LEVEL0_RESOLUTION / (2**level),
            scale=WEB_MERCATOR_LEVEL0_RESOLUTION / (2**level) * _ESRI_DPI / _METERS_PER_INCH,
        )
        for level in range(max_level + 1)
    )
    return TileInfo(
        rows=tile_size,
        cols=tile_size,
        image_format="PNG",
        origin_x=-WEB_MERCATOR_ORIGIN,
        origin_y=WEB_MERCATOR_ORIGIN,
        wkid=3857,
        lods=lods,
    )
