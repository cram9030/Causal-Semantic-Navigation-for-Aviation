"""Static QA gallery rendering: per-tile PNGs and the embedded page manifest."""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio

from csnav.data.ground_truth.rasterize import GroundTruthBuilder
from csnav.viz.ground_truth_gallery import build_gallery, render_tile_images, write_gallery

from tests.data.ground_truth.conftest import TILE_HEIGHT_PX, TILE_WIDTH_PX


@pytest.fixture
def label(tile, transform, crossing_streets):
    return GroundTruthBuilder().rasterize(crossing_streets, tile, TILE_WIDTH_PX, TILE_HEIGHT_PX, transform)


@pytest.fixture
def imagery_path(tmp_path, tile, transform):
    path = tmp_path / f"{tile.level}_{tile.row}_{tile.col}.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=TILE_HEIGHT_PX, width=TILE_WIDTH_PX, count=3, dtype="uint8",
        crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.full((3, TILE_HEIGHT_PX, TILE_WIDTH_PX), 128, dtype="uint8"))
    return path


def test_render_tile_images_writes_expected_files(tmp_path, label, imagery_path):
    output_dir = tmp_path / "gallery"
    gallery_tile = render_tile_images(label, imagery_path, output_dir)

    assert (output_dir / gallery_tile.imagery).exists()
    assert (output_dir / gallery_tile.label_png).exists()
    assert (output_dir / gallery_tile.thumb).exists()
    assert gallery_tile.stem == label.stem
    assert gallery_tile.total_segments == len(label.segments)
    assert gallery_tile.road_count == sum(1 for s in label.segments if s.class_id == 1)
    assert gallery_tile.intersection_count == sum(1 for s in label.segments if s.class_id == 2)
    assert gallery_tile.default_width_count == sum(1 for s in label.segments if s.default_width_used)


def test_render_tile_images_lists_each_road_segment_for_manual_investigation(tmp_path, label, imagery_path):
    """A reviewer needs the OBJECTID/name/width of every road in a tile to go investigate it
    against CSJ's own data - not just an aggregate count.
    """
    gallery_tile = render_tile_images(label, imagery_path, tmp_path / "gallery")

    road_segments = [s for s in label.segments if s.class_id == 1]
    assert len(gallery_tile.segments) == len(road_segments)
    by_objectid = {entry["objectid"]: entry for entry in gallery_tile.segments}
    for segment in road_segments:
        entry = by_objectid[segment.segment_id]
        assert entry["name"] == (segment.name or "")
        assert entry["default_width"] == segment.default_width_used


def test_label_png_is_transparent_where_background(tmp_path, label, imagery_path):
    from PIL import Image

    output_dir = tmp_path / "gallery"
    gallery_tile = render_tile_images(label, imagery_path, output_dir)
    label_image = np.array(Image.open(output_dir / gallery_tile.label_png))

    background = label.semantic == 0
    assert np.all(label_image[..., 3][background] == 0)
    assert np.all(label_image[..., 3][~background] == 255)


def test_build_gallery_writes_index_html_with_embedded_tiles(tmp_path, label, imagery_path):
    output_dir = tmp_path / "gallery"
    index_path = build_gallery([(label, imagery_path)], output_dir)

    assert index_path.name == "index.html"
    html = index_path.read_text(encoding="utf-8")
    assert "var TILES" in html
    assert label.stem in html


def test_render_tile_images_skips_existing_by_default(tmp_path, label, imagery_path, monkeypatch):
    output_dir = tmp_path / "gallery"
    render_tile_images(label, imagery_path, output_dir)

    calls = []
    monkeypatch.setattr(
        "csnav.viz.ground_truth_gallery._read_imagery_rgb",
        lambda path: calls.append(path) or np.zeros((4, 4, 3), dtype=np.uint8),
    )
    gallery_tile = render_tile_images(label, imagery_path, output_dir)

    assert calls == []  # imagery was never re-read - the existing PNGs were left alone
    assert gallery_tile.stem == label.stem


def test_render_tile_images_overwrite_forces_a_re_render(tmp_path, label, imagery_path, monkeypatch):
    output_dir = tmp_path / "gallery"
    render_tile_images(label, imagery_path, output_dir)

    calls = []
    shape = label.semantic.shape

    def fake_read(path):
        calls.append(path)
        return np.zeros((*shape, 3), dtype=np.uint8)

    monkeypatch.setattr("csnav.viz.ground_truth_gallery._read_imagery_rgb", fake_read)
    render_tile_images(label, imagery_path, output_dir, overwrite=True)

    assert calls == [imagery_path]


def test_build_gallery_accepts_a_true_one_shot_generator(tmp_path, label, imagery_path):
    """Regression test: must work with a generator, not just a list/Sequence.

    ``scripts/visualize_ground_truth.py`` passes a generator that loads one
    label from disk at a time so the whole label set is never resident in
    memory at once.
    """

    def one_shot():
        yield label, imagery_path

    output_dir = tmp_path / "gallery"
    index_path = build_gallery(one_shot(), output_dir)
    assert index_path.exists()
    assert label.stem in index_path.read_text(encoding="utf-8")


def test_write_gallery_embeds_json_escaped_against_script_injection(tmp_path):
    from csnav.viz.ground_truth_gallery import GalleryTile

    malicious = GalleryTile(
        stem="</script><script>alert(1)</script>",
        road_count=1, intersection_count=0, default_width_count=0, total_segments=1,
        thumb="thumbs/x.png", imagery="images/x_imagery.png", label_png="images/x_label.png",
    )
    index_path = write_gallery([malicious], tmp_path / "gallery")
    html = index_path.read_text(encoding="utf-8")
    assert "</script><script>alert(1)" not in html
    assert "\\u003c/script\\u003e" in html


def _tile(stem: str) -> "GalleryTile":
    from csnav.viz.ground_truth_gallery import GalleryTile

    return GalleryTile(
        stem=stem, road_count=1, intersection_count=0, default_width_count=0, total_segments=1,
        thumb=f"thumbs/{stem}.png", imagery=f"images/{stem}_imagery.png", label_png=f"images/{stem}_label.png",
    )


def test_write_gallery_merges_with_a_smaller_later_selection(tmp_path):
    """Regression test for a real incident: a full gallery run followed by a smaller
    --limit/--sample re-run into the same --gallery-dir must not drop the tiles the first
    run already rendered - their PNGs are still sitting on disk, just no longer reachable
    from the page, which looked exactly like a rendering bug rather than the silent
    index.html truncation it actually was.
    """
    output_dir = tmp_path / "gallery"
    write_gallery([_tile("a"), _tile("b"), _tile("c")], output_dir)

    index_path = write_gallery([_tile("a")], output_dir)

    stems = {t["stem"] for t in json.loads(index_path.read_text(encoding="utf-8").split("var TILES = ", 1)[1].split(";\n", 1)[0])}
    assert stems == {"a", "b", "c"}


def test_write_gallery_a_fresh_render_overrides_the_same_stems_data(tmp_path):
    """A tile re-rendered on a later call (e.g. --overwrite after a data fix) should win over
    its own earlier, stale entry - merging must not just keep whichever came first.
    """
    from csnav.viz.ground_truth_gallery import GalleryTile

    output_dir = tmp_path / "gallery"
    write_gallery([_tile("a")], output_dir)

    refreshed = GalleryTile(
        stem="a", road_count=9, intersection_count=0, default_width_count=0, total_segments=9,
        thumb="thumbs/a.png", imagery="images/a_imagery.png", label_png="images/a_label.png",
    )
    index_path = write_gallery([refreshed], output_dir)

    tiles = json.loads(index_path.read_text(encoding="utf-8").split("var TILES = ", 1)[1].split(";\n", 1)[0])
    by_stem = {t["stem"]: t for t in tiles}
    assert by_stem["a"]["road_count"] == 9


def test_build_gallery_extends_rather_than_replaces_a_prior_smaller_gallery(tmp_path, label, imagery_path):
    """End-to-end version of the merge regression, through build_gallery/render_tile_images
    rather than write_gallery directly.
    """
    output_dir = tmp_path / "gallery"
    build_gallery([(label, imagery_path)], output_dir)

    # A second, empty-selection call (e.g. a --limit that matched nothing this time) must
    # still leave the first call's tile visible on the page.
    index_path = build_gallery([], output_dir)

    html = index_path.read_text(encoding="utf-8")
    assert label.stem in html
