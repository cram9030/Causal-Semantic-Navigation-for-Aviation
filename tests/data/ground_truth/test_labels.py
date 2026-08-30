from __future__ import annotations

import numpy as np
import pytest

from csnav.data.ground_truth.labels import LABEL_SCHEMA_VERSION, PanopticClass, PanopticLabel, SegmentInfo


def _label(tile, transform):
    height, width = 8, 8
    semantic = np.zeros((height, width), dtype=np.uint32)
    instance = np.zeros((height, width), dtype=np.uint32)
    semantic[2:4, 2:6] = int(PanopticClass.ROAD)
    instance[2:4, 2:6] = 1
    semantic[3, 3] = int(PanopticClass.INTERSECTION)
    instance[3, 3] = 2
    segments = (
        SegmentInfo(instance_id=1, class_id=int(PanopticClass.ROAD), segment_id="1", name="First St", width_m=12.0),
        SegmentInfo(instance_id=2, class_id=int(PanopticClass.INTERSECTION), intersection_segment_ids=("1", "2")),
    )
    return PanopticLabel(
        tile=tile, semantic=semantic, instance=instance, transform=transform, crs="EPSG:4326",
        segments=segments, streets_source="test.geojson", imagery_source="test.tif",
    )


def test_shape_mismatch_raises(tile, transform):
    with pytest.raises(ValueError):
        PanopticLabel(
            tile=tile,
            semantic=np.zeros((8, 8), dtype=np.uint32),
            instance=np.zeros((4, 4), dtype=np.uint32),
            transform=transform,
            crs="EPSG:4326",
            segments=(),
        )


def test_save_and_load_round_trips(tmp_path, tile, transform):
    label = _label(tile, transform)
    raster_path, sidecar_path = label.save(tmp_path)

    assert raster_path.name == f"{tile.level}_{tile.row}_{tile.col}.tif"
    assert sidecar_path.name == f"{tile.level}_{tile.row}_{tile.col}.json"

    loaded = PanopticLabel.load(raster_path, sidecar_path)
    assert np.array_equal(loaded.semantic, label.semantic)
    assert np.array_equal(loaded.instance, label.instance)
    assert loaded.segments == label.segments
    assert loaded.tile == label.tile
    assert loaded.streets_source == label.streets_source
    assert loaded.imagery_source == label.imagery_source


def test_load_defaults_sidecar_path_from_raster_path(tmp_path, tile, transform):
    label = _label(tile, transform)
    raster_path, _ = label.save(tmp_path)

    loaded = PanopticLabel.load(raster_path)
    assert loaded.segments == label.segments


def test_load_rejects_mismatched_schema_version(tmp_path, tile, transform):
    label = _label(tile, transform)
    raster_path, sidecar_path = label.save(tmp_path)

    import json

    payload = json.loads(sidecar_path.read_text())
    payload["schema_version"] = LABEL_SCHEMA_VERSION + 1
    sidecar_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        PanopticLabel.load(raster_path, sidecar_path)


def test_segment_info_round_trips_through_dict():
    segment = SegmentInfo(
        instance_id=1, class_id=int(PanopticClass.ROAD), segment_id="42", name="Main St",
        width_m=5.5, default_width_used=True,
    )
    assert SegmentInfo.from_dict(segment.to_dict()) == segment
