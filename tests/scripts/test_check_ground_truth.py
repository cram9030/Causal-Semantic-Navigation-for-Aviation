"""End-to-end run of scripts/check_ground_truth.py against a small label set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_ground_truth as cgt  # noqa: E402

from csnav.data.arcgis.models import Extent  # noqa: E402
from csnav.data.ground_truth.labels import PanopticClass, PanopticLabel, SegmentInfo  # noqa: E402
from csnav.trajectory.coverage import TileRef  # noqa: E402


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_ground_truth", *argv])
    cgt.main()


def _good_label(level, row, col):
    bounds = Extent(xmin=0.0, ymin=0.0, xmax=0.001, ymax=0.001)
    tile = TileRef(level=level, row=row, col=col, bounds=bounds)
    transform = from_bounds(bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax, 8, 8)
    semantic = np.zeros((8, 8), dtype=np.uint32)
    instance = np.zeros((8, 8), dtype=np.uint32)
    semantic[2:4, :] = int(PanopticClass.ROAD)
    instance[2:4, :] = 1
    return PanopticLabel(
        tile=tile, semantic=semantic, instance=instance, transform=transform, crs="EPSG:4326",
        segments=(SegmentInfo(instance_id=1, class_id=int(PanopticClass.ROAD), segment_id="1"),),
    )


def test_check_ground_truth_exits_zero_for_clean_labels(tmp_path, monkeypatch):
    _good_label(18, 1, 1).save(tmp_path)
    report_path = tmp_path / "report.json"
    _run(["--labels-dir", str(tmp_path), "--report", str(report_path)], monkeypatch)

    report = json.loads(report_path.read_text())
    assert report["ok"] is True
    assert report["error_count"] == 0


def test_check_ground_truth_exits_nonzero_on_error(tmp_path, monkeypatch):
    label = _good_label(18, 2, 2)
    label.instance[0, 0] = 42  # orphan instance id
    label.save(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        _run(["--labels-dir", str(tmp_path)], monkeypatch)
    assert excinfo.value.code != 0


def test_check_ground_truth_warns_without_failing_on_empty_directory(tmp_path, monkeypatch, caplog):
    _run(["--labels-dir", str(tmp_path)], monkeypatch)  # no labels present, not an error
