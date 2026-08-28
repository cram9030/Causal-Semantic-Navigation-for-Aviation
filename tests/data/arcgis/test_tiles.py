import pytest

from csnav.data.arcgis.models import Extent, LevelOfDetail, TileInfo
from csnav.data.arcgis.tiles import (
    best_level_for_resolution,
    sample_tiles_covering_extent,
    tile_bounds,
    tile_count_covering_extent,
    tiles_covering_extent,
)


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


def test_tile_count_covering_extent_matches_enumeration(tile_info):
    extent = Extent(xmin=5.0, ymin=50.0, xmax=45.0, ymax=95.0, wkid=3857)
    count = tile_count_covering_extent(tile_info, level=0, extent=extent)
    assert count == 9
    assert count == len(list(tiles_covering_extent(tile_info, level=0, extent=extent)))


def test_sample_tiles_covering_extent_returns_full_set_when_smaller_than_sample(tile_info):
    extent = Extent(xmin=5.0, ymin=50.0, xmax=45.0, ymax=95.0, wkid=3857)
    sample = sample_tiles_covering_extent(tile_info, level=0, extent=extent, sample_size=25)
    assert set(sample) == set(tiles_covering_extent(tile_info, level=0, extent=extent))


def test_sample_tiles_covering_extent_caps_at_sample_size(tile_info):
    extent = Extent(xmin=5.0, ymin=50.0, xmax=45.0, ymax=95.0, wkid=3857)
    sample = sample_tiles_covering_extent(tile_info, level=0, extent=extent, sample_size=4)
    assert 0 < len(sample) <= 4
    full = set(tiles_covering_extent(tile_info, level=0, extent=extent))
    assert set(sample).issubset(full)


def test_sample_tiles_covering_extent_stays_cheap_for_a_huge_grid():
    # A grid far too large to enumerate (billions of tiles); sampling must
    # never materialize the full grid to produce a bounded-size sample.
    huge_tile_info = TileInfo(
        rows=256, cols=256, image_format="PNG",
        origin_x=-20037508.342787, origin_y=20037508.342787, wkid=3857,
        lods=(LevelOfDetail(level=0, resolution=0.01, scale=1.0),),
    )
    extent = Extent(xmin=-1_000_000.0, ymin=-1_000_000.0, xmax=1_000_000.0, ymax=1_000_000.0, wkid=3857)
    total = tile_count_covering_extent(huge_tile_info, level=0, extent=extent)
    assert total > 10_000_000_000  # confirms this would be infeasible to enumerate

    sample = sample_tiles_covering_extent(huge_tile_info, level=0, extent=extent, sample_size=30)
    assert 0 < len(sample) <= 30


def test_web_mercator_scheme_level0_is_one_world_tile():
    """The standard EPSG:3857 scheme: level 0 is a single 256x256 world tile."""
    from csnav.data.arcgis.tiles import WEB_MERCATOR_ORIGIN, web_mercator_tile_info

    tile_info = web_mercator_tile_info(max_level=5)
    bounds = tile_bounds(tile_info, 0, 0, 0)

    assert tile_info.wkid == 3857
    assert (tile_info.rows, tile_info.cols) == (256, 256)
    assert bounds.xmin == pytest.approx(-WEB_MERCATOR_ORIGIN)
    assert bounds.ymax == pytest.approx(WEB_MERCATOR_ORIGIN)
    assert bounds.xmax - bounds.xmin == pytest.approx(2 * WEB_MERCATOR_ORIGIN)


def test_web_mercator_resolution_halves_each_level():
    from csnav.data.arcgis.tiles import web_mercator_tile_info

    tile_info = web_mercator_tile_info(max_level=20)
    for level in range(1, 21):
        assert tile_info.lod_for_level(level).resolution == pytest.approx(
            tile_info.lod_for_level(level - 1).resolution / 2.0
        )
    assert tile_info.max_level == 20
