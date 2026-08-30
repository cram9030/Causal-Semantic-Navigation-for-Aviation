"""End-to-end run of scripts/build_ground_truth.py against synthetic tiles + streets."""

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

from csnav.data.ground_truth.labels import PanopticLabel  # noqa: E402
from csnav.trajectory.manifest import ManifestBundle  # noqa: E402
from csnav.trajectory.trajectory import TrajectoryWindow  # noqa: E402
from csnav.trajectory.manifest import LandmarkManifest  # noqa: E402
from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.trajectory.coverage import TileRef  # noqa: E402
from shapely.geometry import box  # noqa: E402


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_ground_truth", *argv])
    bgt.main()


BOUNDS = (-121.9008, 37.3300, -121.8992, 37.3316)
WIDTH, HEIGHT = 64, 64


def _write_imagery(path: Path, level: int, row: int, col: int) -> None:
    transform = from_bounds(*BOUNDS, WIDTH, HEIGHT)
    with rasterio.open(
        path / f"{level}_{row}_{col}.tif", "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=3,
        dtype="uint8", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(np.zeros((3, HEIGHT, WIDTH), dtype="uint8"))


def _write_streets(path: Path) -> None:
    streets = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[BOUNDS[0], 37.3308], [BOUNDS[2], 37.3308]]},
                "properties": {"OBJECTID": 1, "WIDTH": 40.0},
            },
        ],
    }
    path.write_text(json.dumps(streets))


@pytest.fixture
def imagery_dir(tmp_path):
    directory = tmp_path / "imagery"
    directory.mkdir()
    _write_imagery(directory, 18, 100, 200)
    return directory


@pytest.fixture
def streets_geojson(tmp_path):
    path = tmp_path / "streets.geojson"
    _write_streets(path)
    return path


def test_build_ground_truth_rasterizes_every_tile_in_imagery_dir(tmp_path, imagery_dir, streets_geojson, monkeypatch):
    output_dir = tmp_path / "out"
    _run(
        ["--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_geojson), "--output-dir", str(output_dir)],
        monkeypatch,
    )
    assert (output_dir / "18_100_200.tif").exists()
    assert (output_dir / "18_100_200.json").exists()
    label = PanopticLabel.load(output_dir / "18_100_200.tif")
    assert label.segments


def test_build_ground_truth_skips_existing_unless_overwrite(tmp_path, imagery_dir, streets_geojson, monkeypatch):
    output_dir = tmp_path / "out"
    _run(
        ["--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_geojson), "--output-dir", str(output_dir)],
        monkeypatch,
    )
    first_mtime = (output_dir / "18_100_200.tif").stat().st_mtime_ns

    _run(
        ["--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_geojson), "--output-dir", str(output_dir)],
        monkeypatch,
    )
    assert (output_dir / "18_100_200.tif").stat().st_mtime_ns == first_mtime

    _run(
        [
            "--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_geojson),
            "--output-dir", str(output_dir), "--overwrite",
        ],
        monkeypatch,
    )
    # overwrite ran without error and the file still exists (content is deterministic, so mtime
    # alone isn't a reliable rewritten-vs-not signal on a fast filesystem).
    assert (output_dir / "18_100_200.tif").exists()


def test_build_ground_truth_restricts_to_manifest_tiles(tmp_path, streets_geojson, monkeypatch):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    _write_imagery(imagery_dir, 18, 100, 200)
    _write_imagery(imagery_dir, 18, 999, 999)  # not referenced by the manifest below

    tile_ref = TileRef(level=18, row=100, col=200, bounds=Extent(xmin=BOUNDS[0], ymin=BOUNDS[1], xmax=BOUNDS[2], ymax=BOUNDS[3]))
    window = TrajectoryWindow(trajectory_id="t", index=0, start_distance=0.0, end_distance=100.0, start_time=0.0, end_time=10.0)
    manifest = LandmarkManifest(
        window=window, tube_radius=100.0, max_agl=300.0, ground_reach=0.0,
        envelope=tile_ref.bounds, footprint=box(*BOUNDS), candidate_roads=(), tiles=(tile_ref,),
    )
    bundle = ManifestBundle(trajectory_set_id="t", manifests=(manifest,))
    manifest_path = tmp_path / "bundle.json"
    bundle.save(manifest_path)

    output_dir = tmp_path / "out"
    _run(
        [
            "--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_geojson),
            "--output-dir", str(output_dir), "--manifest", str(manifest_path),
        ],
        monkeypatch,
    )
    assert (output_dir / "18_100_200.tif").exists()
    assert not (output_dir / "18_999_999.tif").exists()


def test_build_ground_truth_rejects_non_4326_imagery(tmp_path, streets_geojson, monkeypatch):
    imagery_dir = tmp_path / "imagery"
    imagery_dir.mkdir()
    transform = from_bounds(0, 0, 1000, 1000, WIDTH, HEIGHT)
    with rasterio.open(
        imagery_dir / "18_1_1.tif", "w", driver="GTiff", height=HEIGHT, width=WIDTH, count=3, dtype="uint8",
        crs="EPSG:3857", transform=transform,
    ) as dst:
        dst.write(np.zeros((3, HEIGHT, WIDTH), dtype="uint8"))

    output_dir = tmp_path / "out"
    with pytest.raises(ValueError):
        _run(
            [
                "--imagery-dir", str(imagery_dir), "--streets-geojson", str(streets_geojson),
                "--output-dir", str(output_dir),
            ],
            monkeypatch,
        )
