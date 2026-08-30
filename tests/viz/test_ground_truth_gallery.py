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
