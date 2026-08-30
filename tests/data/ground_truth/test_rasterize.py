from __future__ import annotations

import numpy as np
import pytest

from csnav.data.arcgis.streets import StreetSegment
from csnav.data.ground_truth.labels import PanopticClass
from csnav.data.ground_truth.rasterize import GroundTruthBuilder


def test_rasterize_two_crossing_streets_produces_road_and_intersection(tile, transform, crossing_streets):
    builder = GroundTruthBuilder()
    label = builder.rasterize(crossing_streets, tile, 64, 64, transform)

    assert label.semantic.shape == (64, 64)
    assert label.instance.shape == (64, 64)
    assert set(np.unique(label.semantic).tolist()) == {
        int(PanopticClass.BACKGROUND), int(PanopticClass.ROAD), int(PanopticClass.INTERSECTION)
    }

    # Every non-background pixel has a nonzero instance id and vice versa.
    background = label.semantic == int(PanopticClass.BACKGROUND)
    assert not np.any((label.instance != 0) & background)
    assert not np.any((label.instance == 0) & ~background)

    road_segments = [s for s in label.segments if s.class_id == int(PanopticClass.ROAD)]
    intersection_segments = [s for s in label.segments if s.class_id == int(PanopticClass.INTERSECTION)]
    assert {s.segment_id for s in road_segments} == {"1", "2"}
    assert len(intersection_segments) == 1
    assert set(intersection_segments[0].intersection_segment_ids) == {"1", "2"}

    # Segment 1 published a width (40 ft); segment 2 did not and falls back to the default.
    by_id = {s.segment_id: s for s in road_segments}
    assert by_id["1"].default_width_used is False
    assert by_id["1"].width_m == pytest.approx(40.0 * 0.3048)
    assert by_id["1"].name == "First St"
    assert by_id["2"].default_width_used is True
    assert by_id["2"].width_m == pytest.approx(builder.default_width_m)


def test_rasterize_no_streets_in_tile_is_all_background(tile, transform):
    builder = GroundTruthBuilder()
    far_away_street = StreetSegment(
        object_id=99, parts=(((-70.0, 10.0), (-70.001, 10.001)),), attributes={}
    )
    label = builder.rasterize([far_away_street], tile, 64, 64, transform)

    assert not label.segments
    assert np.all(label.semantic == int(PanopticClass.BACKGROUND))
    assert np.all(label.instance == 0)


def test_rasterize_instance_ids_are_unique_and_dense(tile, transform, crossing_streets):
    builder = GroundTruthBuilder()
    label = builder.rasterize(crossing_streets, tile, 64, 64, transform)

    instance_ids = [s.instance_id for s in label.segments]
    assert len(instance_ids) == len(set(instance_ids))
    assert sorted(instance_ids) == list(range(1, len(instance_ids) + 1))


def test_rasterize_wider_road_covers_more_pixels(tile, transform):
    builder = GroundTruthBuilder()
    narrow = [
        StreetSegment(object_id=1, parts=(((tile.bounds.xmin, 37.3382), (tile.bounds.xmax, 37.3382)),), attributes={"WIDTH": 10.0})
    ]
    wide = [
        StreetSegment(object_id=1, parts=(((tile.bounds.xmin, 37.3382), (tile.bounds.xmax, 37.3382)),), attributes={"WIDTH": 100.0})
    ]
    narrow_label = builder.rasterize(narrow, tile, 64, 64, transform)
    wide_label = builder.rasterize(wide, tile, 64, 64, transform)

    narrow_pixels = int(np.count_nonzero(narrow_label.semantic == int(PanopticClass.ROAD)))
    wide_pixels = int(np.count_nonzero(wide_label.semantic == int(PanopticClass.ROAD)))
    assert wide_pixels > narrow_pixels


def test_builder_rejects_non_positive_parameters():
    with pytest.raises(ValueError):
        GroundTruthBuilder(default_width_m=0.0)
    with pytest.raises(ValueError):
        GroundTruthBuilder(intersection_radius_m=-1.0)
    with pytest.raises(ValueError):
        GroundTruthBuilder(intersection_snap_m=0.0)
