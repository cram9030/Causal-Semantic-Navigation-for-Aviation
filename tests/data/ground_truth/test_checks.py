from __future__ import annotations

import numpy as np

from csnav.data.ground_truth.checks import DEFAULT_WIDTH_WARN_FRACTION, check_label, check_label_directory
from csnav.data.ground_truth.labels import PanopticClass, PanopticLabel, SegmentInfo


def _consistent_label(tile, transform):
    semantic = np.zeros((8, 8), dtype=np.uint32)
    instance = np.zeros((8, 8), dtype=np.uint32)
    semantic[2:4, 2:6] = int(PanopticClass.ROAD)
    instance[2:4, 2:6] = 1
    return PanopticLabel(
        tile=tile, semantic=semantic, instance=instance, transform=transform, crs="EPSG:4326",
        segments=(SegmentInfo(instance_id=1, class_id=int(PanopticClass.ROAD), segment_id="1"),),
    )


def test_check_label_passes_for_a_consistent_label(tile, transform):
    report = check_label(_consistent_label(tile, transform))
    assert report.ok
    assert report.issues == ()
    assert report.road_pixel_fraction == 8 / 64


def test_check_label_flags_orphan_instance_pixel(tile, transform):
    label = _consistent_label(tile, transform)
    label.instance[0, 0] = 99  # no segments_info entry, and semantic is BACKGROUND there
    report = check_label(label)
    assert not report.ok
    messages = [issue.message for issue in report.issues]
    assert any("no segments_info entry" in message for message in messages)
    assert any("background pixel" in message for message in messages)


def test_check_label_flags_foreground_pixel_missing_instance(tile, transform):
    label = _consistent_label(tile, transform)
    label.instance[2, 2] = 0  # still ROAD in semantic, but instance id zeroed out
    report = check_label(label)
    assert not report.ok
    assert any("non-background pixel" in issue.message for issue in report.issues)


def test_check_label_flags_unused_segment_as_warning(tile, transform):
    label = _consistent_label(tile, transform)
    label.segments = label.segments + (SegmentInfo(instance_id=2, class_id=int(PanopticClass.ROAD), segment_id="2"),)
    report = check_label(label)
    assert report.ok  # warning only, not an error
    assert any(issue.severity == "warning" and "never rasterized" in issue.message for issue in report.issues)


def test_check_label_warns_on_high_default_width_fraction(tile, transform):
    segments = (
        SegmentInfo(instance_id=1, class_id=int(PanopticClass.ROAD), segment_id="1", default_width_used=True),
    )
    semantic = np.full((8, 8), int(PanopticClass.ROAD), dtype=np.uint32)
    instance = np.full((8, 8), 1, dtype=np.uint32)
    label = PanopticLabel(
        tile=tile, semantic=semantic, instance=instance, transform=transform, crs="EPSG:4326", segments=segments
    )
    report = check_label(label)
    assert report.default_width_fraction == 1.0
    assert report.default_width_fraction > DEFAULT_WIDTH_WARN_FRACTION
    assert any("default width" in issue.message for issue in report.issues)


def test_check_label_directory_reports_missing_raster(tmp_path):
    (tmp_path / "18_1_1.json").write_text("{}")
    result = check_label_directory(tmp_path)
    assert len(result.tiles) == 1
    assert not result.ok
    assert result.error_count == 1


def test_check_label_directory_over_saved_labels(tmp_path, tile, transform):
    _consistent_label(tile, transform).save(tmp_path)
    result = check_label_directory(tmp_path)
    assert result.ok
    assert len(result.tiles) == 1
    assert result.to_dict()["tile_count"] == 1
