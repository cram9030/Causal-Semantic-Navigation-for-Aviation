from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString

from csnav.data.arcgis.intersections import StreetIntersection
from csnav.data.arcgis.streets import StreetSegment
from csnav.data.ground_truth.labels import PanopticClass
from csnav.data.ground_truth.rasterize import GroundTruthBuilder, _chain_runs, _intersection_radius
from tests.data.ground_truth.conftest import ORIGIN_LAT, ORIGIN_LON, TILE_BOUNDS


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

    # Each road instance here is a single, unmerged CSJ segment - segment_ids
    # is still populated (mirrors segment_id) so a caller never has to
    # special-case "was this instance merged from more than one OBJECTID".
    assert {s.segment_ids for s in road_segments} == {("1",), ("2",)}

    # Segment 1 published a width (40 ft); segment 2 did not and falls back to the default.
    by_id = {s.segment_id: s for s in road_segments}
    assert by_id["1"].default_width_used is False
    assert by_id["1"].width_m == pytest.approx(40.0 * 0.3048)
    assert by_id["1"].name == "First St"
    assert by_id["2"].default_width_used is True
    assert by_id["2"].width_m == pytest.approx(builder.default_width_m)

    # The raw CSJ attributes ride along too - a reviewer diagnosing a width
    # discrepancy needs to see exactly what CSJ published, not just this
    # module's WIDTH_FIELD_CANDIDATES-based interpretation of it.
    assert by_id["1"].attributes == {"WIDTH": 40.0, "STREETNAME": "First St"}
    assert by_id["2"].attributes == {}


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


# ----- intersection radius scales with the roads meeting there (fix 1) -----


def test_intersection_radius_floors_at_the_configured_default():
    assert _intersection_radius(("1", "2"), {"1": 4.0, "2": 4.0}, floor=3.0) == 3.0


def test_intersection_radius_grows_to_half_the_widest_intersecting_road():
    assert _intersection_radius(("1", "2"), {"1": 20.0, "2": 4.0}, floor=3.0) == 10.0


def test_intersection_radius_ignores_ids_it_has_no_width_for():
    # An intersection_segment_ids entry with no matching road width (should
    # not happen, but this is the defensive default) never grows the radius.
    assert _intersection_radius(("unknown",), {}, floor=3.0) == 3.0


def test_intersection_pixel_footprint_grows_with_the_widest_crossing_road(tile, transform):
    def crossing(east_west_width_ft: float) -> list[StreetSegment]:
        return [
            StreetSegment(
                object_id=1,
                parts=(((TILE_BOUNDS.xmin, ORIGIN_LAT), (TILE_BOUNDS.xmax, ORIGIN_LAT)),),
                attributes={"WIDTH": east_west_width_ft},
            ),
            StreetSegment(
                object_id=2,
                parts=(((ORIGIN_LON, TILE_BOUNDS.ymin), (ORIGIN_LON, TILE_BOUNDS.ymax)),),
                attributes={"WIDTH": 10.0},
            ),
        ]

    builder = GroundTruthBuilder()
    narrow_label = builder.rasterize(crossing(10.0), tile, 64, 64, transform)
    wide_label = builder.rasterize(crossing(150.0), tile, 64, 64, transform)

    narrow_px = int(np.count_nonzero(narrow_label.semantic == int(PanopticClass.INTERSECTION)))
    wide_px = int(np.count_nonzero(wide_label.semantic == int(PanopticClass.INTERSECTION)))
    assert wide_px > narrow_px


# ----- contiguous same-street segments rasterize as one continuous ribbon (fix 2) -----


def test_contiguous_same_street_segments_merge_into_one_road_instance(tile, transform):
    # CSJ splits one physical street into a fresh OBJECTID at every cross-
    # street - these two meet exactly at the tile's own origin (a "mid-block"
    # split, not at a tile edge), same master street/width/name throughout.
    west_half = StreetSegment(
        object_id=5,
        parts=(((TILE_BOUNDS.xmin, ORIGIN_LAT), (ORIGIN_LON, ORIGIN_LAT)),),
        attributes={"WIDTH": 40.0, "STREETNAME": "First St", "STREETMASTERID": 100},
    )
    east_half = StreetSegment(
        object_id=2,
        parts=(((ORIGIN_LON, ORIGIN_LAT), (TILE_BOUNDS.xmax, ORIGIN_LAT)),),
        attributes={"WIDTH": 40.0, "STREETNAME": "First St", "STREETMASTERID": 100},
    )
    builder = GroundTruthBuilder()
    label = builder.rasterize([west_half, east_half], tile, 64, 64, transform)

    road_segments = [s for s in label.segments if s.class_id == int(PanopticClass.ROAD)]
    assert len(road_segments) == 1
    assert road_segments[0].segment_ids == ("2", "5")
    assert road_segments[0].segment_id == "2"  # the lower of the two OBJECTIDs

    # The two halves meeting is an artifact of the CSJ split, not a real
    # junction with a different street - merging them must not also stamp a
    # spurious intersection instance on top of their own shared point.
    assert not [s for s in label.segments if s.class_id == int(PanopticClass.INTERSECTION)]

    # One continuous ribbon: every column with any road pixel has the same
    # pixel height as every other - a rounded cap left over from buffering
    # each half separately would taper the height near the split (column 32).
    road_col_counts = np.count_nonzero(label.semantic == int(PanopticClass.ROAD), axis=0)
    nonzero = road_col_counts[road_col_counts > 0]
    assert nonzero.min() == nonzero.max()


def test_segments_with_different_identity_do_not_merge(tile, transform):
    # Same geometry as the merge test above, but the two halves disagree on
    # STREETMASTERID - merging them would silently blend two different CSJ
    # rows into one instance, so this must stay conservative and leave them
    # as two separate ROAD instances instead.
    west_half = StreetSegment(
        object_id=5,
        parts=(((TILE_BOUNDS.xmin, ORIGIN_LAT), (ORIGIN_LON, ORIGIN_LAT)),),
        attributes={"WIDTH": 40.0, "STREETNAME": "First St", "STREETMASTERID": 100},
    )
    east_half = StreetSegment(
        object_id=2,
        parts=(((ORIGIN_LON, ORIGIN_LAT), (TILE_BOUNDS.xmax, ORIGIN_LAT)),),
        attributes={"WIDTH": 40.0, "STREETNAME": "First St", "STREETMASTERID": 200},
    )
    builder = GroundTruthBuilder()
    label = builder.rasterize([west_half, east_half], tile, 64, 64, transform)

    road_segments = [s for s in label.segments if s.class_id == int(PanopticClass.ROAD)]
    assert {s.segment_ids for s in road_segments} == {("2",), ("5",)}


def test_segments_missing_a_master_id_are_not_merged(tile, transform):
    west_half = StreetSegment(
        object_id=5,
        parts=(((TILE_BOUNDS.xmin, ORIGIN_LAT), (ORIGIN_LON, ORIGIN_LAT)),),
        attributes={"WIDTH": 40.0, "STREETNAME": "First St"},
    )
    east_half = StreetSegment(
        object_id=2,
        parts=(((ORIGIN_LON, ORIGIN_LAT), (TILE_BOUNDS.xmax, ORIGIN_LAT)),),
        attributes={"WIDTH": 40.0, "STREETNAME": "First St"},
    )
    builder = GroundTruthBuilder()
    label = builder.rasterize([west_half, east_half], tile, 64, 64, transform)

    road_segments = [s for s in label.segments if s.class_id == int(PanopticClass.ROAD)]
    assert {s.segment_ids for s in road_segments} == {("2",), ("5",)}


def test_road_width_is_flush_at_the_tile_edge_not_rounded_off(tile, transform):
    # A road's true extent reaches well beyond the tile on both sides -
    # candidates handed to rasterize() are never pre-clipped to the tile
    # (scripts/build_ground_truth.py passes whatever an envelope spatial
    # index returned), so this is the normal case, not a contrived one.
    far = 0.01  # degrees - well beyond the ~180 m tile on either side
    street = [
        StreetSegment(
            object_id=1,
            parts=(((TILE_BOUNDS.xmin - far, ORIGIN_LAT), (TILE_BOUNDS.xmax + far, ORIGIN_LAT)),),
            attributes={"WIDTH": 40.0},
        )
    ]
    builder = GroundTruthBuilder()
    label = builder.rasterize(street, tile, 64, 64, transform)

    road_col_counts = np.count_nonzero(label.semantic == int(PanopticClass.ROAD), axis=0)
    # Every column has road pixels (the road spans the whole tile), and every
    # one of them has the same height - a round cap clipped mid-tile (the
    # previous clip-then-buffer order) would instead taper off near the tile
    # edges.
    assert np.all(road_col_counts > 0)
    assert road_col_counts.min() == road_col_counts.max()


# ----- CSJ Street Intersections metadata enrichment -----


def test_matched_street_intersection_enriches_the_instance(tile, transform, crossing_streets):
    real_intersection = StreetIntersection(
        object_id=1, lon=ORIGIN_LON, lat=ORIGIN_LAT,
        attributes={"INTNAME": "First St & Second St", "INTERSECTIONTYPE": "4 Leg", "TRAFFICCONTROLTYPE": "Signal"},
    )
    builder = GroundTruthBuilder()
    label = builder.rasterize(crossing_streets, tile, 64, 64, transform, street_intersections=[real_intersection])

    intersection_segments = [s for s in label.segments if s.class_id == int(PanopticClass.INTERSECTION)]
    assert len(intersection_segments) == 1
    assert intersection_segments[0].name == "First St & Second St"
    assert intersection_segments[0].attributes["INTERSECTIONTYPE"] == "4 Leg"


def test_street_intersection_outside_snap_tolerance_does_not_match(tile, transform, crossing_streets):
    far_intersection = StreetIntersection(
        object_id=1, lon=TILE_BOUNDS.xmax, lat=TILE_BOUNDS.ymax, attributes={"INTNAME": "Nowhere Near It"}
    )
    builder = GroundTruthBuilder()
    label = builder.rasterize(crossing_streets, tile, 64, 64, transform, street_intersections=[far_intersection])

    intersection_segments = [s for s in label.segments if s.class_id == int(PanopticClass.INTERSECTION)]
    assert len(intersection_segments) == 1
    assert intersection_segments[0].name is None
    assert intersection_segments[0].attributes == {}


def test_no_street_intersections_supplied_leaves_instances_unenriched(tile, transform, crossing_streets):
    builder = GroundTruthBuilder()
    label = builder.rasterize(crossing_streets, tile, 64, 64, transform)

    intersection_segments = [s for s in label.segments if s.class_id == int(PanopticClass.INTERSECTION)]
    assert intersection_segments[0].name is None
    assert intersection_segments[0].attributes == {}


# ----- _chain_runs (endpoint-adjacency grouping) -----


def _endpoints_set(line: LineString) -> set[tuple[float, float]]:
    coords = list(line.coords)
    return {coords[0], coords[-1]}


def test_chain_runs_merges_two_lines_sharing_an_endpoint():
    a = LineString([(0.0, 0.0), (10.0, 0.0)])
    b = LineString([(10.0, 0.0), (20.0, 0.0)])
    runs = _chain_runs([(0, a), (1, b)], snap=0.5)

    assert len(runs) == 1
    indices, merged = runs[0]
    assert sorted(indices) == [0, 1]
    # _chain_runs doesn't promise a winding direction - only that the merged
    # line spans the two originals' full extent with no interior gap.
    assert _endpoints_set(merged) == {(0.0, 0.0), (20.0, 0.0)}
    assert merged.length == pytest.approx(20.0)
    assert (10.0, 0.0) in list(merged.coords)


def test_chain_runs_handles_a_reversed_neighbor():
    a = LineString([(0.0, 0.0), (10.0, 0.0)])
    b = LineString([(20.0, 0.0), (10.0, 0.0)])  # digitized in the opposite direction
    runs = _chain_runs([(0, a), (1, b)], snap=0.5)

    assert len(runs) == 1
    _, merged = runs[0]
    assert _endpoints_set(merged) == {(0.0, 0.0), (20.0, 0.0)}
    assert merged.length == pytest.approx(20.0)


def test_chain_runs_leaves_disjoint_lines_unmerged():
    a = LineString([(0.0, 0.0), (10.0, 0.0)])
    b = LineString([(1000.0, 1000.0), (1010.0, 1000.0)])
    runs = _chain_runs([(0, a), (1, b)], snap=0.5)

    assert len(runs) == 2
    assert {tuple(indices) for indices, _ in runs} == {(0,), (1,)}
