"""End-to-end run of scripts/visualize_ground_truth.py: map + gallery from a label set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_ground_truth as bgt  # noqa: E402
import visualize_ground_truth as vgt  # noqa: E402


def _run(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


BOUNDS = (-121.9008, 37.3300, -121.8992, 37.3316)
WIDTH, HEIGHT = 64, 64
TILE_COUNT = 5


@pytest.fixture
def imagery_dir(tmp_path):
    directory = tmp_path / "imagery"
    directory.mkdir()
    step = 0.0016
    for col in range(TILE_COUNT):
        xmin = BOUNDS[0] + col * step
        transform = from_bounds(xmin, BOUNDS[1], xmin + step, BOUNDS[1] + step, WIDTH, HEIGHT)
        with rasterio.open(
            directory / f"18_100_{200 + col}.tif", "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=3,
            dtype="uint8", crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(np.full((3, HEIGHT, WIDTH), 100, dtype="uint8"))
    return directory


@pytest.fixture
def labels_dir(tmp_path, imagery_dir, monkeypatch):
    step = 0.0016
    xmax = BOUNDS[0] + TILE_COUNT * step
    streets_path = tmp_path / "streets.geojson"
    streets_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[BOUNDS[0], 37.3308], [xmax, 37.3308]]},
                "properties": {"OBJECTID": 1, "WIDTH": 40.0},
            },
        ],
    }))
    output_dir = tmp_path / "labels"
    _run(
        bgt,
        ["--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_path), "--output-dir", str(output_dir)],
        monkeypatch,
    )
    return output_dir


def test_visualize_ground_truth_writes_map_and_gallery(tmp_path, imagery_dir, labels_dir, monkeypatch):
    map_path = tmp_path / "map.html"
    gallery_dir = tmp_path / "gallery"
    _run(
        vgt,
        [
            "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
            "--map", str(map_path), "--gallery-dir", str(gallery_dir),
        ],
        monkeypatch,
    )

    assert map_path.exists()
    assert (gallery_dir / "index.html").exists()


def test_visualize_ground_truth_requires_at_least_one_output(labels_dir, imagery_dir, monkeypatch):
    with pytest.raises(SystemExit):
        _run(vgt, ["--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir)], monkeypatch)


def test_visualize_ground_truth_requires_labels(tmp_path, imagery_dir, monkeypatch):
    empty_labels = tmp_path / "empty_labels"
    empty_labels.mkdir()
    with pytest.raises(SystemExit):
        _run(
            vgt,
            [
                "--labels-dir", str(empty_labels), "--imagery-dir", str(imagery_dir),
                "--map", str(tmp_path / "map.html"),
            ],
            monkeypatch,
        )


def test_visualize_ground_truth_limit_restricts_gallery_tile_count(tmp_path, imagery_dir, labels_dir, monkeypatch):
    gallery_dir = tmp_path / "gallery"
    _run(
        vgt,
        [
            "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
            "--gallery-dir", str(gallery_dir), "--limit", "2",
        ],
        monkeypatch,
    )
    html = (gallery_dir / "index.html").read_text(encoding="utf-8")
    assert html.count('"stem":') == 2


def test_visualize_ground_truth_gallery_resumes_by_default(tmp_path, imagery_dir, labels_dir, monkeypatch):
    """A second invocation shouldn't touch PNGs the first one already wrote."""
    gallery_dir = tmp_path / "gallery"
    _run(
        vgt,
        ["--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir), "--gallery-dir", str(gallery_dir)],
        monkeypatch,
    )
    an_image = next((gallery_dir / "images").glob("*_imagery.png"))
    first_mtime = an_image.stat().st_mtime_ns

    _run(
        vgt,
        ["--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir), "--gallery-dir", str(gallery_dir)],
        monkeypatch,
    )
    assert an_image.stat().st_mtime_ns == first_mtime

    _run(
        vgt,
        [
            "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
            "--gallery-dir", str(gallery_dir), "--overwrite",
        ],
        monkeypatch,
    )
    assert an_image.stat().st_mtime_ns > first_mtime


def test_visualize_ground_truth_sample_restricts_gallery_tile_count(tmp_path, imagery_dir, labels_dir, monkeypatch):
    gallery_dir = tmp_path / "gallery"
    _run(
        vgt,
        [
            "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
            "--gallery-dir", str(gallery_dir), "--sample", "3",
        ],
        monkeypatch,
    )
    html = (gallery_dir / "index.html").read_text(encoding="utf-8")
    assert html.count('"stem":') == 3


def test_visualize_ground_truth_limit_and_sample_are_mutually_exclusive(labels_dir, imagery_dir, monkeypatch):
    with pytest.raises(SystemExit):
        _run(
            vgt,
            [
                "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
                "--map", "map.html", "--limit", "1", "--sample", "1",
            ],
            monkeypatch,
        )


def test_visualize_ground_truth_map_reflects_the_selected_subset(tmp_path, imagery_dir, labels_dir, monkeypatch):
    """Fewer tiles selected should mean fewer tile features drawn - proof the selection actually applies."""
    import re

    map_path = tmp_path / "map.html"
    _run(
        vgt,
        [
            "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
            "--map", str(map_path), "--limit", "2",
        ],
        monkeypatch,
    )
    html = map_path.read_text(encoding="utf-8")
    match = re.search(r"tiles \((\d+)\)", html)
    assert match and int(match.group(1)) == 2


def test_sidecar_paths_lists_every_label_sorted(labels_dir):
    paths = vgt._sidecar_paths(labels_dir)
    assert len(paths) == TILE_COUNT
    assert paths == sorted(paths)


def test_select_with_limit_takes_the_first_n():
    paths = [Path(f"{i}.json") for i in range(10)]
    assert vgt._select(paths, limit=3, sample=None) == paths[:3]


def test_select_with_sample_spreads_evenly_and_is_capped_at_available():
    paths = [Path(f"{i}.json") for i in range(10)]
    sampled = vgt._select(paths, limit=None, sample=4)
    assert len(sampled) == 4
    assert len(set(sampled)) == 4  # no duplicates
    assert sampled == sorted(sampled, key=paths.index)  # stays in the original (sorted) order
    assert vgt._select(paths, limit=None, sample=100) == paths  # can't sample more than exist


def test_select_with_neither_returns_everything():
    paths = [Path(f"{i}.json") for i in range(4)]
    assert vgt._select(paths, limit=None, sample=None) == paths
