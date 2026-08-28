"""Visible footprints and the imagery tiles that cover them."""

from __future__ import annotations

import pytest
from pyproj import Geod
from shapely.geometry import Point as ShapelyPoint, box

from csnav.data.arcgis.tiles import web_mercator_tile_info
from csnav.geometry.fov import FieldOfView
from csnav.trajectory.coverage import (
    agl_from_elevation,
    height_as_agl,
    max_agl,
    merge_tiles,
    tiles_for_footprint,
    visible_footprint,
)
from csnav.trajectory.waypoints import Waypoint

_GEOD = Geod(ellps="WGS84")


def test_height_as_agl_is_the_waypoint_height():
    assert height_as_agl(Waypoint(lat=37.0, lon=-121.0, height=250.0)) == 250.0
    assert height_as_agl(Waypoint(lat=37.0, lon=-121.0, height=-5.0)) == 0.0


def test_agl_from_elevation_subtracts_ground_height():
    provider = agl_from_elevation(lambda lon, lat: 35.0)
    assert provider(Waypoint(lat=37.0, lon=-121.0, height=350.0)) == pytest.approx(315.0)


def test_agl_from_elevation_falls_back_when_the_source_has_no_data():
    provider = agl_from_elevation(lambda lon, lat: None)
    assert provider(Waypoint(lat=37.0, lon=-121.0, height=350.0)) == pytest.approx(350.0)


def test_agl_from_elevation_clamps_below_ground_to_zero():
    provider = agl_from_elevation(lambda lon, lat: 500.0)
    assert provider(Waypoint(lat=37.0, lon=-121.0, height=350.0)) == 0.0


def test_max_agl_uses_the_highest_point_in_the_window(dogleg):
    """The window's search footprint is sized from its highest point, not its mean."""
    window = dogleg.windows(10_000.0)[0]
    assert max_agl(dogleg, window, height_as_agl) == pytest.approx(400.0, abs=1.0)


def test_visible_footprint_without_a_fov_is_just_the_tube(due_east, tube):
    assert visible_footprint(due_east, tube).equals(tube.corridor(due_east))


def test_visible_footprint_extends_the_tube_by_the_fov_ground_radius(due_east, tube):
    fov = FieldOfView(horizontal_deg=60.0, vertical_deg=45.0)
    footprint = visible_footprint(due_east, tube, field_of_view=fov)
    extra = fov.ground_radius(due_east.waypoints[0].height)

    midpoint = due_east.point_at(due_east.length / 2.0)
    inside_lon, inside_lat, _ = _GEOD.fwd(midpoint.lon, midpoint.lat, 0.0, tube.radius + extra - 20.0)
    outside_lon, outside_lat, _ = _GEOD.fwd(midpoint.lon, midpoint.lat, 0.0, tube.radius + extra + 20.0)

    assert footprint.contains(ShapelyPoint(inside_lon, inside_lat))
    assert not footprint.contains(ShapelyPoint(outside_lon, outside_lat))


def test_tiles_for_footprint_cover_the_footprint(due_east, tube):
    footprint = visible_footprint(due_east, tube)
    tile_info = web_mercator_tile_info()
    tiles = tiles_for_footprint(footprint, tile_info, 16)

    assert tiles
    covered = None
    for tile in tiles:
        bounds = tile.bounds
        rectangle = box(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)
        covered = rectangle if covered is None else covered.union(rectangle)
    assert covered.contains(footprint)


def test_tiles_for_footprint_excludes_tiles_the_corridor_only_grazes_by_bbox(due_east, tube):
    """A corridor is a thin shape - bounding-box tiles it never touches are dropped."""
    footprint = visible_footprint(due_east, tube)
    tile_info = web_mercator_tile_info()
    tiles = tiles_for_footprint(footprint, tile_info, 17)
    for tile in tiles:
        bounds = tile.bounds
        assert box(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax).intersects(footprint)


def test_tiles_for_footprint_refuses_an_unmanageably_fine_level(due_east, tube):
    footprint = visible_footprint(due_east, tube)
    with pytest.raises(ValueError, match="over the .* limit"):
        tiles_for_footprint(footprint, web_mercator_tile_info(), 22, max_tiles=100)


def test_tile_bounds_are_reported_in_wgs84(due_east, tube):
    tiles = tiles_for_footprint(visible_footprint(due_east, tube), web_mercator_tile_info(), 16)
    assert all(tile.bounds.wkid == 4326 for tile in tiles)
    assert tiles[0].key == f"{tiles[0].level}/{tiles[0].row}/{tiles[0].col}"


def test_merge_tiles_deduplicates_across_overlapping_windows(due_east, tube):
    tile_info = web_mercator_tile_info()
    per_window = [
        tiles_for_footprint(visible_footprint(due_east, tube, window=window), tile_info, 16)
        for window in due_east.windows(500.0)
    ]
    merged = merge_tiles(per_window)
    total = sum(len(group) for group in per_window)

    assert len(merged) < total  # adjacent windows share boundary tiles
    assert len({(tile.level, tile.row, tile.col) for tile in merged}) == len(merged)
    assert list(merged) == sorted(merged, key=lambda tile: (tile.level, tile.row, tile.col))


def test_tile_geojson_feature_is_a_closed_ring(due_east, tube):
    tile = tiles_for_footprint(visible_footprint(due_east, tube), web_mercator_tile_info(), 16)[0]
    ring = tile.to_geojson_feature()["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert len(ring) == 5
