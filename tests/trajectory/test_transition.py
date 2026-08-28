"""Generated transition families.

The properties that carry the formalization: a transition may initiate anywhere
along its source, it arrives at the first target waypoint *ahead* of where that
initiation point projects, and the curve leaves and joins tangentially. The
fixtures are parallel and orthogonal straight legs, so the arrival waypoint and
the turn angles are decidable by inspection.
"""

from __future__ import annotations

import math

import pytest
from pyproj import Geod

from csnav.trajectory.trajectory import TransitionError, TransitionRule
from csnav.trajectory.transition import MIN_TRANSITION_CHORD, TransitionModel, transition_id
from csnav.trajectory.tube import TubeModel
from csnav.trajectory.waypoints import TrajectoryRole

_GEOD = Geod(ellps="WGS84")


# ----- the rule ---------------------------------------------------------------


def test_rule_rejects_a_self_loop():
    with pytest.raises(TransitionError, match="self-loop"):
        TransitionRule(source="t_p", target="t_p")


def test_rule_rejects_an_inverted_initiation_window():
    with pytest.raises(TransitionError, match="initiate_to"):
        TransitionRule(source="a", target="b", initiate_from=900.0, initiate_to=100.0)


def test_rule_domain_defaults_to_the_whole_source(due_east):
    assert TransitionRule(source="due_east", target="x").domain(due_east) == (0.0, due_east.length)


def test_rule_domain_is_clipped_to_the_source(due_east):
    rule = TransitionRule(source="due_east", target="x", initiate_from=-100.0, initiate_to=99_999.0)
    assert rule.domain(due_east) == (0.0, due_east.length)


# ----- the arrival rule -------------------------------------------------------


def test_arrival_is_the_first_target_waypoint_ahead_of_the_projection(
    model, due_east, parallel_north
):
    """The source and target run parallel, so projection is just the along-track position."""
    # 500 m along a 2 km leg projects before the target's midpoint waypoint (index 1).
    assert model.arrival_index(due_east, parallel_north, 500.0) == 1
    # 1500 m is past it, so the next one ahead is the final waypoint.
    assert model.arrival_index(due_east, parallel_north, 1500.0) == 2


def test_no_arrival_once_the_projection_passes_the_targets_end(model, due_east, parallel_north):
    assert model.arrival_index(due_east, parallel_north, due_east.length) is None
    assert model.path(due_east, parallel_north, due_east.length) is None


def test_arrival_never_goes_backwards_along_the_target(model, due_east, parallel_north):
    """Using the projection, not raw proximity, is what guarantees forward progress."""
    previous = -1
    for step in range(11):
        index = model.arrival_index(due_east, parallel_north, due_east.length * step / 12)
        assert index is not None and index >= previous
        previous = index


# ----- the generated path -----------------------------------------------------


def test_path_starts_at_the_initiation_point_and_ends_at_the_arrival_waypoint(
    model, due_east, parallel_north
):
    path = model.path(due_east, parallel_north, 800.0)
    origin = due_east.point_at(800.0)
    arrival = parallel_north.waypoints[path.arrival_index]

    first, last = path.trajectory.waypoints[0], path.trajectory.waypoints[-1]
    assert _GEOD.inv(first.lon, first.lat, origin.lon, origin.lat)[2] < 1.0
    assert _GEOD.inv(last.lon, last.lat, arrival.lon, arrival.lat)[2] < 1.0


def test_path_leaves_along_the_sources_heading_and_joins_along_the_targets(
    due_east, parallel_north
):
    """Hermite endpoint tangents are the point: the curve is heading-continuous at both ends.

    A sampled curve only approaches its own endpoint tangents as the sampling
    gets finer, so this checks that the error shrinks with resolution and lands
    within a degree - which is the real claim, and is not satisfied by a curve
    that simply heads off toward the arrival point.
    """
    errors = []
    for resolution in (25, 100, 400):
        path = TransitionModel(resolution=resolution).path(due_east, parallel_north, 800.0)
        generated = path.trajectory
        departure = abs(generated.heading_at(0.0) - due_east.heading_at(800.0))
        arrival = abs(
            generated.heading_at(generated.length)
            - parallel_north.heading_at(path.arrival_distance - 1.0)
        )
        errors.append(max(departure, arrival))

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 1.0


def test_generated_path_is_a_transition_trajectory_naming_what_it_connects(
    model, due_east, parallel_north
):
    path = model.path(due_east, parallel_north, 800.0)
    assert path.trajectory.role is TrajectoryRole.TRANSITION
    assert path.trajectory.connects == ("due_east", "parallel_north")
    assert path.trajectory.metadata["initiate_distance_m"] == 800.0
    assert path.id == transition_id("due_east", "parallel_north", 800.0)


def test_generated_ids_are_stable_across_regeneration(model, due_east, parallel_north):
    first = model.family(due_east, parallel_north, TransitionRule("due_east", "parallel_north"))
    second = model.family(due_east, parallel_north, TransitionRule("due_east", "parallel_north"))
    assert [p.id for p in first.paths] == [p.id for p in second.paths]


def test_path_time_continues_the_sources_clock(model, due_east, parallel_north):
    path = model.path(due_east, parallel_north, 800.0)
    generated = path.trajectory

    assert generated.waypoints[0].time == pytest.approx(due_east.point_at(800.0).time)
    assert generated.duration > 0.0
    # Travelled at the source's own ground speed.
    assert generated.duration == pytest.approx(generated.length / due_east.speed_at(800.0), rel=0.01)


def test_resolution_controls_the_curves_vertex_count(due_east, parallel_north):
    path = TransitionModel(resolution=40).path(due_east, parallel_north, 800.0)
    assert len(path.trajectory.waypoints) == 41


def test_a_larger_tangent_gain_bulges_the_curve_further(due_east, parallel_north):
    tight = TransitionModel(tangent_gain=0.3).path(due_east, parallel_north, 800.0)
    loose = TransitionModel(tangent_gain=1.5).path(due_east, parallel_north, 800.0)
    assert loose.trajectory.length > tight.trajectory.length


def test_gain_is_scale_free_so_shape_survives_a_longer_transition(due_east, parallel_north):
    """Tangents scale with the chord, so a short and a long transition bulge alike."""
    ratios = []
    for distance in (200.0, 1400.0):
        path = TransitionModel(tangent_gain=1.0).path(due_east, parallel_north, distance)
        origin = due_east.point_at(distance)
        arrival = path.trajectory.waypoints[-1]
        chord = _GEOD.inv(origin.lon, origin.lat, arrival.lon, arrival.lat)[2]
        ratios.append(path.trajectory.length / chord)
    assert ratios[0] == pytest.approx(ratios[1], rel=0.15)


def test_a_transition_onto_where_you_already_are_is_not_generated(model, due_east, returning):
    """Initiating a return at the very start projects onto the return's own end - at x_0.

    There is nothing to fly, so no path is generated. This is a real case, not
    a rounding artefact: every outbound route in a scenario starts where its
    return route finishes.
    """
    assert model.path(due_east, returning, 0.0) is None


def test_a_return_is_generated_from_further_along_the_source(model, due_east, returning):
    path = model.path(due_east, returning, due_east.length * 0.75)
    assert path is not None
    assert path.trajectory.length > MIN_TRANSITION_CHORD


def test_a_return_demands_a_large_departure_turn(model, due_east, returning):
    """Diverting onto a return is a turn-around; the screen cannot tell that from a mistake."""
    path = model.path(due_east, returning, due_east.length * 0.6)
    assert path.departure_turn > 90.0


# ----- the feasibility screen -------------------------------------------------


def test_turn_screen_drops_a_near_orthogonal_divert(due_east, orthogonal):
    """The screen's purpose: a target that would need a ~90 deg turn to reach."""
    rule = TransitionRule(source="due_east", target="orthogonal")
    permissive = TransitionModel(samples=9, max_turn_deg=180.0).family(due_east, orthogonal, rule)
    strict = TransitionModel(samples=9, max_turn_deg=45.0).family(due_east, orthogonal, rule)

    assert permissive.paths
    assert len(strict) < len(permissive)
    assert strict.rejected > 0
    assert all(max(p.departure_turn, p.arrival_turn) <= 45.0 for p in strict.paths)


def test_a_per_edge_limit_overrides_the_models_default(due_east, orthogonal):
    strict_model = TransitionModel(samples=9, max_turn_deg=30.0)
    relaxed_rule = TransitionRule(source="due_east", target="orthogonal", max_turn_deg=180.0)

    assert len(strict_model.family(due_east, orthogonal, relaxed_rule)) > len(
        strict_model.family(due_east, orthogonal, TransitionRule("due_east", "orthogonal"))
    )


def test_a_per_edge_gain_overrides_the_models_default(due_east, parallel_north):
    model = TransitionModel(tangent_gain=0.3, samples=1)
    rule = TransitionRule(source="due_east", target="parallel_north", tangent_gain=1.5, initiate_from=800.0)
    assert model.family(due_east, parallel_north, rule).paths[0].trajectory.length > model.path(
        due_east, parallel_north, 800.0
    ).trajectory.length


def test_the_screen_is_off_by_default(due_east, orthogonal):
    """180 deg admits everything, so nothing is dropped unless a scenario asks."""
    assert TransitionModel().max_turn_deg == 180.0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"tangent_gain": 0.0}, "tangent_gain"),
        ({"samples": 0}, "samples"),
        ({"resolution": 1}, "resolution"),
        ({"max_turn_deg": 0.0}, "max_turn_deg"),
        ({"speed": -1.0}, "speed"),
    ],
)
def test_model_rejects_nonsense_parameters(kwargs, message):
    with pytest.raises(TransitionError, match=message):
        TransitionModel(**kwargs)


# ----- the family -------------------------------------------------------------


def test_family_samples_across_the_whole_initiation_domain(due_east, parallel_north):
    rule = TransitionRule(source="due_east", target="parallel_north")
    family = TransitionModel(samples=7, max_turn_deg=180.0).family(due_east, parallel_north, rule)

    assert family.domain == (0.0, due_east.length)
    assert len(family) + family.rejected == 7
    initiations = [path.initiate_distance for path in family.paths]
    assert initiations == sorted(initiations)


def test_family_respects_a_narrowed_domain(due_east, parallel_north):
    rule = TransitionRule(
        source="due_east", target="parallel_north", initiate_from=400.0, initiate_to=900.0
    )
    family = TransitionModel(samples=5).family(due_east, parallel_north, rule)

    assert family.domain == (400.0, 900.0)
    assert all(400.0 <= path.initiate_distance <= 900.0 for path in family.paths)


def test_family_counts_what_it_screened_out_rather_than_hiding_it(due_east, orthogonal):
    rule = TransitionRule(source="due_east", target="orthogonal")
    family = TransitionModel(samples=9, max_turn_deg=20.0).family(due_east, orthogonal, rule)
    assert family.rejected == 9 - len(family)


def test_an_empty_family_reports_itself_as_empty(due_east, orthogonal):
    rule = TransitionRule(source="due_east", target="orthogonal")
    family = TransitionModel(samples=5, max_turn_deg=1.0).family(due_east, orthogonal, rule)
    assert family.is_empty
    assert family.turn_range == (0.0, 0.0)
    assert family.reachable_footprint(TubeModel(200.0)).is_empty


def test_reachable_footprint_covers_every_sampled_path(due_east, parallel_north):
    """The family sweeps a region, and any point in it is a valid state mid-transition."""
    rule = TransitionRule(source="due_east", target="parallel_north")
    family = TransitionModel(samples=6, max_turn_deg=180.0).family(due_east, parallel_north, rule)
    tube = TubeModel(200.0)

    footprint = family.reachable_footprint(tube)
    for path in family.paths:
        corridor = tube.corridor(path.trajectory)
        assert corridor.difference(footprint).area < corridor.area * 1e-6
    assert footprint.area > max(tube.corridor(p.trajectory).area for p in family.paths)


def test_with_samples_changes_only_the_sampling_density(model):
    denser = model.with_samples(40)
    assert denser.samples == 40
    assert denser.tangent_gain == model.tangent_gain
    assert denser.max_turn_deg == model.max_turn_deg
    assert model.samples != 40


def test_single_sample_family_takes_the_domain_start(due_east, parallel_north):
    rule = TransitionRule(source="due_east", target="parallel_north", initiate_from=700.0)
    family = TransitionModel(samples=1).family(due_east, parallel_north, rule)
    assert [path.initiate_distance for path in family.paths] == [700.0]


def test_generated_paths_stay_between_the_two_routes(due_east, parallel_north):
    """No sampled path wanders outside the corridor spanned by source and target."""
    rule = TransitionRule(source="due_east", target="parallel_north")
    family = TransitionModel(samples=6, max_turn_deg=180.0).family(due_east, parallel_north, rule)

    south = min(wp.lat for wp in due_east.waypoints)
    north = max(wp.lat for wp in parallel_north.waypoints)
    for path in family.paths:
        for waypoint in path.trajectory.waypoints:
            assert south - 1e-3 <= waypoint.lat <= north + 1e-3


def test_transition_id_encodes_its_provenance():
    assert transition_id("t_p", "t_alt", 1234.5) == "t_p__t_alt__s01234.5"
    assert math.isclose(float(transition_id("a", "b", 12.0).split("s")[-1]), 12.0)
