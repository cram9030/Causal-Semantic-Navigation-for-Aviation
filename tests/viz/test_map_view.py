"""Interactive map rendering.

folium output is HTML, so these check the rendered document: that every
trajectory becomes a toggleable layer, that the tube radius and window ids
reach the tooltips a reviewer reads, and that the tile layer holds the tiles
the coverage code says are in view.
"""

from __future__ import annotations

import pytest

from csnav.data.arcgis.tiles import web_mercator_tile_info
from csnav.trajectory.coverage import merge_tiles, tiles_for_footprint, visible_footprint
from csnav.trajectory.manifest_builder import ManifestBuilder, StaticStreetsSource
from csnav.viz.map_view import (
    base_map,
    bundle_map,
    manifest_map,
    save_map,
    trajectory_map,
    trajectory_set_map,
)

from tests.trajectory.test_manifest_builder import STREET_OFFSETS, _crossing_street, _parallel_street


@pytest.fixture
def bundle(trajectory_set, conops):
    segments = [_parallel_street(offset, index + 1) for index, offset in enumerate(STREET_OFFSETS)]
    segments.append(_crossing_street())
    builder = ManifestBuilder(
        streets=StaticStreetsSource(segments, source_label="test-fixture"),
        tile_info=web_mercator_tile_info(),
        tile_level=16,
    )
    return builder.build_set(trajectory_set, conops)


def test_base_map_offers_the_san_jose_imagery_layer():
    html = base_map((37.3382, -121.8863)).get_root().render()
    assert "geo.sanjoseca.gov" in html and "DPW_ImageryCached2025" in html
    assert "OpenStreetMap" in html


def test_base_map_can_omit_the_imagery_layer_for_offline_review():
    html = base_map((37.3382, -121.8863), include_imagery=False).get_root().render()
    assert "geo.sanjoseca.gov" not in html


def test_set_map_gives_every_trajectory_its_own_layer(trajectory_set, conops):
    html = trajectory_set_map(trajectory_set, conops).get_root().render()
    for trajectory in trajectory_set.trajectories:
        assert trajectory.id in html
    assert "x_0 (known start state)" in html


def test_set_map_labels_each_layer_with_its_configured_radius(trajectory_set, conops):
    html = trajectory_set_map(trajectory_set, conops).get_root().render()
    assert f"tube {conops.tube_radius:.0f} m" in html
    assert f"tube {conops.transition_tube_radius:.0f} m" in html


def test_trajectory_map_tooltips_carry_window_ids_and_geometry(due_east, tube, conops):
    fmap = trajectory_map(due_east, tube, conops.window_length, field_of_view=conops.field_of_view)
    html = fmap.get_root().render()

    for window in due_east.windows(conops.window_length):
        assert window.window_id in html
    assert f"tube corridor ({tube.radius:.0f} m radius)" in html
    assert "FOV ground radius" in html


def test_trajectory_map_tile_layer_matches_the_coverage_calculation(due_east, tube, conops):
    tile_info = web_mercator_tile_info()
    expected = merge_tiles(
        tiles_for_footprint(
            visible_footprint(due_east, tube, window=window, field_of_view=conops.field_of_view),
            tile_info,
            16,
        )
        for window in due_east.windows(conops.window_length)
    )

    html = trajectory_map(
        due_east, tube, conops.window_length, field_of_view=conops.field_of_view, tile_level=16
    ).get_root().render()

    assert f"imagery tiles in view (level 16, {len(expected)})" in html
    assert f"imagery tile {expected[0].key}" in html


def test_trajectory_map_without_a_tile_level_draws_no_tile_layer(due_east, tube, conops):
    html = trajectory_map(due_east, tube, conops.window_length).get_root().render()
    assert "imagery tiles in view" not in html


def test_manifest_map_draws_the_pinned_landmarks(due_east, bundle):
    manifests = bundle.for_trajectory("due_east")
    html = manifest_map(due_east, manifests).get_root().render()

    assert "candidate roads (manifest)" in html
    assert "intersections (manifest)" in html
    assert "off-track offset" in html
    assert manifests[0].window_id in html


def test_manifest_map_refuses_an_empty_manifest_list(due_east):
    with pytest.raises(ValueError, match="no manifests supplied"):
        manifest_map(due_east, [])


def test_bundle_map_groups_layers_by_trajectory(trajectory_set, bundle):
    html = bundle_map(trajectory_set, bundle).get_root().render()
    for trajectory in trajectory_set.trajectories:
        manifests = bundle.for_trajectory(trajectory.id)
        assert f"{trajectory.id} ({len(manifests)} windows" in html


def test_save_map_writes_a_self_contained_html_file(trajectory_set, conops, tmp_path):
    destination = save_map(trajectory_set_map(trajectory_set, conops), tmp_path / "nested" / "map.html")
    html = destination.read_text(encoding="utf-8")

    assert destination.exists()
    assert html.startswith("<!DOCTYPE html>")
    assert "L.map(" in html
