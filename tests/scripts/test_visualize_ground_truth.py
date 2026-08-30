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


@pytest.fixture
def imagery_dir(tmp_path):
    directory = tmp_path / "imagery"
    directory.mkdir()
    transform = from_bounds(*BOUNDS, WIDTH, HEIGHT)
    with rasterio.open(
        directory / "18_100_200.tif", "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=3,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.full((3, HEIGHT, WIDTH), 100, dtype="uint8"))
    return directory


@pytest.fixture
def labels_dir(tmp_path, imagery_dir, monkeypatch):
    streets_path = tmp_path / "streets.geojson"
    streets_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[BOUNDS[0], 37.3308], [BOUNDS[2], 37.3308]]},
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
