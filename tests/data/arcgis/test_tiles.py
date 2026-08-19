import pytest

from csnav.data.arcgis.models import Extent, LevelOfDetail, TileInfo
from csnav.data.arcgis.tiles import best_level_for_resolution, tile_bounds, tiles_covering_extent


@pytest.fixture
def tile_info() -> TileInfo:
    # Small synthetic scheme, easy to reason about by hand: 2x2 pixel tiles,
    # origin at (0, 100), level 0 resolution 10 units/px, level 1 resolution 5.
    return TileInfo(
        rows=2,
        cols=2,
        image_format="PNG",
        origin_x=0.0,
        origin_y=100.0,
        wkid=3857,
        lods=(
            LevelOfDetail(level=0, resolution=10.0, scale=1.0),
            LevelOfDetail(level=1, resolution=5.0, scale=0.5),
        ),
    )


def test_tile_bounds_origin_tile(tile_info):
    bounds = tile_bounds(tile_info, level=0, row=0, col=0)
    assert bounds == Extent(xmin=0.0, ymin=80.0, xmax=20.0, ymax=100.0, wkid=3857)


def test_tile_bounds_offset_tile(tile_info):
    bounds = tile_bounds(tile_info, level=0, row=1, col=2)
    assert bounds == Extent(xmin=40.0, ymin=60.0, xmax=60.0, ymax=80.0, wkid=3857)


def test_tile_bounds_unknown_level_raises(tile_info):
    with pytest.raises(KeyError):
        tile_bounds(tile_info, level=5, row=0, col=0)


def test_tiles_covering_extent(tile_info):
    extent = Extent(xmin=5.0, ymin=50.0, xmax=45.0, ymax=95.0, wkid=3857)
    tiles = set(tiles_covering_extent(tile_info, level=0, extent=extent))
    assert tiles == {(r, c) for r in range(0, 3) for c in range(0, 3)}


def test_tiles_covering_extent_rejects_mismatched_wkid(tile_info):
    extent = Extent(xmin=5.0, ymin=50.0, xmax=45.0, ymax=95.0, wkid=4326)
    with pytest.raises(ValueError):
        list(tiles_covering_extent(tile_info, level=0, extent=extent))


def test_best_level_for_resolution_picks_coarsest_sufficient(tile_info):
    assert best_level_for_resolution(tile_info, target_resolution=7.0) == 1
    assert best_level_for_resolution(tile_info, target_resolution=100.0) == 0


def test_best_level_for_resolution_falls_back_to_finest(tile_info):
    assert best_level_for_resolution(tile_info, target_resolution=1.0) == 1
