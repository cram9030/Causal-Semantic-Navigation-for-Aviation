"""Interactive ground-truth review map rendering.

Mirrors ``tests/viz/test_map_view.py``'s approach: check the rendered HTML
document rather than folium internals - that tile footprints, road/
intersection polygons vectorized back out of the label rasters, and their
layer names (with counts) all make it into the page.
"""

from __future__ import annotations

import re

import pytest

from csnav.data.arcgis.streets import StreetSegment
from csnav.data.ground_truth.rasterize import GroundTruthBuilder
from csnav.viz.ground_truth_view import ground_truth_review_map, save_ground_truth_map

from tests.data.ground_truth.conftest import TILE_HEIGHT_PX, TILE_WIDTH_PX


@pytest.fixture
def label(tile, transform, crossing_streets):
    return GroundTruthBuilder().rasterize(crossing_streets, tile, TILE_WIDTH_PX, TILE_HEIGHT_PX, transform)


@pytest.fixture
def label_with_no_intersections(tile, transform):
    """A tile with exactly one road and no crossings - the intersections layer's FeatureCollection is empty.

    Regression test for GeoJsonTooltip's own assertion that its fields exist
    among the data's properties keys: that check fails outright against an
    *empty* FeatureCollection (there is nothing to check the fields
    against), which crashed ``.render()``/``.save()`` for any label set with
    zero rasterized intersections before this was guarded.
    """
    single_road = [
        StreetSegment(
            object_id=1,
            parts=(((tile.bounds.xmin, 37.3382), (tile.bounds.xmax, 37.3382)),),
            attributes={"WIDTH": 30.0},
        )
    ]
    return GroundTruthBuilder().rasterize(single_road, tile, TILE_WIDTH_PX, TILE_HEIGHT_PX, transform)


def test_ground_truth_review_map_rejects_empty_input():
    with pytest.raises(ValueError):
        ground_truth_review_map([])


def test_ground_truth_review_map_includes_tile_and_layer_counts(label):
    """Layer-control counts are vectorized polygon *parts*, not logical segments:

    an intersection is drawn on top of the roads meeting there, which can
    split a road's own buffer into more than one connected piece where the
    intersection blob interrupts it - so the map's "roads (N)" legitimately
    counts higher than ``len(road segments)``. This only checks both counts
    are present and positive, not tied to that vectorization detail.
    """
    fmap = ground_truth_review_map([label])
    html = fmap.get_root().render()

    assert "tile" in html
    road_match = re.search(r"roads \((\d+)\)", html)
    intersection_match = re.search(r"intersections \((\d+)\)", html)
    assert road_match and int(road_match.group(1)) > 0
    assert intersection_match and int(intersection_match.group(1)) > 0


def test_save_ground_truth_map_writes_file(tmp_path, label):
    fmap = ground_truth_review_map([label])
    destination = save_ground_truth_map(fmap, tmp_path / "nested" / "map.html")
    assert destination.exists()
    assert destination.read_text().startswith("<!DOCTYPE html>") or "<html" in destination.read_text()


def test_save_ground_truth_map_renders_with_an_empty_layer(tmp_path, label_with_no_intersections):
    fmap = ground_truth_review_map([label_with_no_intersections])
    destination = save_ground_truth_map(fmap, tmp_path / "map.html")
    html = destination.read_text()
    assert "intersections (0)" in html
    road_match = re.search(r"roads \((\d+)\)", html)
    assert road_match and int(road_match.group(1)) > 0


def test_ground_truth_review_map_accepts_a_true_one_shot_generator(label):
    """Regression test: this must work with a generator, not just a list/Sequence.

    ``scripts/visualize_ground_truth.py`` passes a generator that loads one
    label from disk at a time specifically so the whole label set is never
    resident in memory at once - a function that secretly needed ``len()``
    or a second pass over ``labels`` would silently defeat that.
    """

    def one_shot():
        yield label

    fmap = ground_truth_review_map(one_shot())
    html = fmap.get_root().render()
    assert "tile" in html
