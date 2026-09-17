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


def test_visualize_ground_truth_warns_when_reusing_a_gallery_dir_without_overwrite(
    tmp_path, imagery_dir, labels_dir, monkeypatch, caplog
):
    """A real incident: labels were rebuilt (a streets re-fetch, a rasterization fix) but the
    gallery was re-run against the same --gallery-dir without --overwrite, so it silently kept
    showing images rendered from the *old* labels - indistinguishable, from the browser, from a
    genuine rendering bug. This must be surfaced as a warning, not left silent.
    """
    gallery_dir = tmp_path / "gallery"

    with caplog.at_level("WARNING"):
        _run(
            vgt,
            ["--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir), "--gallery-dir", str(gallery_dir)],
            monkeypatch,
        )
    assert "already has rendered images" not in caplog.text  # nothing to be stale yet, first run

    caplog.clear()
    with caplog.at_level("WARNING"):
        _run(
            vgt,
            ["--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir), "--gallery-dir", str(gallery_dir)],
            monkeypatch,
        )
    assert "already has rendered images" in caplog.text
    assert "--overwrite" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        _run(
            vgt,
            [
                "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
                "--gallery-dir", str(gallery_dir), "--overwrite",
            ],
            monkeypatch,
        )
    assert "already has rendered images" not in caplog.text  # --overwrite means it's not stale


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


def test_sidecar_paths_sorts_numerically_not_as_strings(tmp_path):
    """A mix of zoom levels/unpadded row-col numbers must sort by (level, row, col) as
    integers - plain string sorting would put "9_1_1" after "10_1_1" and "19_..." before
    "21_...", exactly the ordering bug that let a --limit selection land entirely among stale,
    unrelated tiles in a real incident (see docs/phase2_ground_truth_rasterization.md).
    """
    for stem in ["21_1_1", "9_1_1", "19_2_2", "9_10_1"]:
        (tmp_path / f"{stem}.json").write_text("{}")
    paths = vgt._sidecar_paths(tmp_path)
    assert [path.stem for path in paths] == ["9_1_1", "9_10_1", "19_2_2", "21_1_1"]


def test_paths_with_imagery_excludes_labels_with_no_matching_tile(tmp_path):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    (imagery_dir / "18_1_1.tif").write_bytes(b"")
    paths = [Path(tmp_path / "18_1_1.json"), Path(tmp_path / "18_1_2.json")]
    assert vgt._paths_with_imagery(paths, imagery_dir) == [paths[0]]


def test_gallery_limit_skips_orphaned_labels_with_no_imagery(tmp_path, imagery_dir, labels_dir, monkeypatch):
    """Real incident regression test: a labels_dir that outgrew its imagery_dir left a pile of
    orphaned label files (an earlier/different imagery pull) sorting before the tiles that
    still have matching imagery. --limit must draw only from tiles that can actually render,
    not fail with "no label had matching imagery" just because the orphans sort first.
    """
    import shutil

    # An orphaned label at a much lower tile level than the real fixture tiles (level 18) -
    # sorts first, and has no matching file under imagery_dir.
    real_json = next(labels_dir.glob("18_100_200.json"))
    real_tif = real_json.with_suffix(".tif")
    shutil.copy(real_json, labels_dir / "1_1_1.json")
    shutil.copy(real_tif, labels_dir / "1_1_1.tif")

    gallery_dir = tmp_path / "gallery"
    _run(
        vgt,
        [
            "--labels-dir", str(labels_dir), "--imagery-dir", str(imagery_dir),
            "--gallery-dir", str(gallery_dir), "--limit", "1",
        ],
        monkeypatch,
    )
    html = (gallery_dir / "index.html").read_text(encoding="utf-8")
    assert '"stem": "1_1_1"' not in html
    assert html.count('"stem":') == 1


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
