"""Manifest serialization and lookup.

A pinned manifest is the artifact a flight-planning cycle is committed to
(integration plan §3.3), so it has to survive a save/load round trip byte for
byte in meaning: same windows, same geometry, same tube radius, same tiles.
"""

from __future__ import annotations

import json

import pytest
from shapely.geometry import Point as ShapelyPoint

from csnav.data.arcgis.tiles import web_mercator_tile_info
from csnav.trajectory.manifest import MANIFEST_SCHEMA_VERSION, ManifestBundle
from csnav.trajectory.manifest_builder import ManifestBuilder, StaticStreetsSource

from .test_manifest_builder import _crossing_street, _parallel_street, STREET_OFFSETS


@pytest.fixture
def bundle(trajectory_set, conops):
    segments = [_parallel_street(offset, index + 1) for index, offset in enumerate(STREET_OFFSETS)]
    segments.append(_crossing_street())
    builder = ManifestBuilder(
        streets=StaticStreetsSource(segments, source_label="test-fixture"),
        tile_info=web_mercator_tile_info(),
        tile_level=16,
    )
    return builder.build_set(trajectory_set, conops)


def test_bundle_round_trips_through_json(bundle, tmp_path):
    path = bundle.save(tmp_path / "nested" / "bundle.json")
    restored = ManifestBundle.load(path)

    assert restored.trajectory_set_id == bundle.trajectory_set_id
    assert restored.pinned_at == bundle.pinned_at
    assert restored.streets_source == bundle.streets_source
    assert restored.parameters == bundle.parameters
    assert len(restored.manifests) == len(bundle.manifests)


def test_manifest_geometry_survives_the_round_trip(bundle, tmp_path):
    restored = ManifestBundle.load(bundle.save(tmp_path / "bundle.json"))
    for original, copy in zip(bundle.manifests, restored.manifests):
        assert copy.window_id == original.window_id
        assert copy.tube_radius == original.tube_radius
        assert copy.max_agl == pytest.approx(original.max_agl)
        assert copy.ground_reach == pytest.approx(original.ground_reach)
        assert copy.footprint.equals_exact(original.footprint, 1e-12)
        assert [road.segment_id for road in copy.candidate_roads] == [
            road.segment_id for road in original.candidate_roads
        ]
        assert [road.parts for road in copy.candidate_roads] == [
            road.parts for road in original.candidate_roads
        ]
        assert copy.intersections == original.intersections
        assert copy.tiles == original.tiles


def test_saved_bundle_is_plain_reviewable_json(bundle, tmp_path):
    path = bundle.save(tmp_path / "bundle.json")
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert raw["manifests"][0]["candidate_roads"][0]["type"] == "Feature"
    assert raw["manifests"][0]["footprint"]["type"] in ("Polygon", "MultiPolygon")


def test_loading_a_mismatched_schema_version_is_refused(bundle, tmp_path):
    path = bundle.save(tmp_path / "bundle.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = MANIFEST_SCHEMA_VERSION + 1
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version"):
        ManifestBundle.load(path)


def test_by_window_id_is_the_runtime_entry_point(bundle):
    manifest = bundle.by_window_id("due_east:0000")
    assert manifest.window.trajectory_id == "due_east"
    assert manifest.window.index == 0

    with pytest.raises(KeyError, match="no manifest for window"):
        bundle.by_window_id("due_east:9999")


def test_for_trajectory_returns_windows_in_order(bundle):
    manifests = bundle.for_trajectory("due_east")
    assert [manifest.window.index for manifest in manifests] == list(range(len(manifests)))


def test_for_transition_covers_the_generated_family_not_just_candidates(bundle, trajectory_set, conops):
    """The bug this fixes: transitions were entirely absent from a built bundle."""
    transition_manifests = bundle.for_transition("due_east", "parallel_north")
    assert transition_manifests

    family = conops.transition.family(
        trajectory_set.by_id("due_east"), trajectory_set.by_id("parallel_north"), trajectory_set.transitions[1]
    )
    path_ids = bundle.transition_path_ids("due_east", "parallel_north")
    assert len(path_ids) == len(family)
    assert set(path_ids) == {path.id for path in family.paths}


def test_for_transition_does_not_pick_up_candidate_route_manifests(bundle):
    """Route ids never look like '<source>__<target>__s...', but the boundary is still worth pinning."""
    assert bundle.for_transition("due_east", "does_not_exist") == ()
    route_manifests = set(bundle.for_trajectory("due_east"))
    transition_manifests = set(bundle.for_transition("due_east", "parallel_north"))
    assert route_manifests.isdisjoint(transition_manifests)


def test_transition_path_ids_are_generated_transition_trajectory_ids(bundle):
    for path_id in bundle.transition_path_ids("due_east", "parallel_north"):
        assert path_id.startswith("due_east__parallel_north__s")


def test_all_tiles_deduplicates_across_the_whole_set(bundle):
    merged = bundle.all_tiles()
    per_manifest_total = sum(len(manifest.tiles) for manifest in bundle.manifests)

    assert len(merged) < per_manifest_total
    assert len({tile.key for tile in merged}) == len(merged)


def test_query_with_no_footprint_returns_the_whole_manifest(bundle):
    manifest = bundle.by_window_id("due_east:0000")
    assert manifest.query() == manifest.candidate_roads
    assert manifest.query_intersections() == manifest.intersections


def test_query_filters_to_the_supplied_footprint(bundle):
    manifest = bundle.by_window_id("due_east:0000")
    centroid = manifest.footprint.centroid
    disc = ShapelyPoint(centroid.x, centroid.y).buffer(0.0005)

    subset = manifest.query(disc)
    assert 0 < len(subset) < len(manifest.candidate_roads)
    assert all(road.geometry().intersects(disc) for road in subset)


def test_query_intersections_filters_by_point_containment(bundle):
    manifest = next(m for m in bundle.manifests if m.intersections)
    junction = manifest.intersections[0]
    disc = ShapelyPoint(junction.lon, junction.lat).buffer(1e-5)

    assert manifest.query_intersections(disc) == (junction,)
