"""Scenario config loading, and the tube-radius sweep entry point.

The property that matters most here: the tube radius has to come from config
and can never be defaulted (CLAUDE.md core decision 4), and overriding it must
be a one-call operation so a sweep needs no code change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from csnav.geometry.camera import Camera
from csnav.geometry.fov import FieldOfView
from csnav.trajectory.config import (
    ConopsConfig,
    ScenarioConfigError,
    load_scenario,
    scenario_from_dict,
)
from csnav.trajectory.transition import TransitionModel
from csnav.trajectory.trajectory import X0_NODE
from csnav.trajectory.waypoints import TrajectoryRole

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_SCENARIO = REPO_ROOT / "configs" / "scenarios" / "san_jose_downtown.yaml"

MINIMAL = {
    "id": "minimal",
    "trajectory_set": {
        "id": "minimal_set",
        "x0": {"lat": 37.3382, "lon": -121.8863, "height_m": 300.0, "time_s": 0.0},
        "trajectories": [
            {
                "id": "t_p",
                "role": "primary",
                "waypoints": [
                    {"lat": 37.3382, "lon": -121.8863, "height_m": 300.0, "time_s": 0.0},
                    {"lat": 37.3382, "lon": -121.8637, "height_m": 300.0, "time_s": 100.0},
                ],
            }
        ],
    },
    "conops": {"tube_radius_m": 250.0, "window_length_m": 800.0},
}


def _config(**conops_overrides) -> dict:
    config = {**MINIMAL, "conops": {**MINIMAL["conops"], **conops_overrides}}
    return config


def test_pilot_scenario_loads_and_is_internally_consistent():
    scenario = load_scenario(PILOT_SCENARIO)
    trajectory_set = scenario.trajectory_set

    assert trajectory_set.primary.role is TrajectoryRole.PRIMARY
    assert trajectory_set.candidates and trajectory_set.transitions
    assert scenario.conops.tube_radius > 0.0
    assert scenario.source_path == PILOT_SCENARIO

    graph = trajectory_set.to_networkx()
    assert X0_NODE in graph
    # Every transition rule points at routes the set actually holds.
    for source, target, data in graph.edges(data=True):
        assert target in {t.id for t in trajectory_set.trajectories}
        if not data["is_entry"]:
            assert trajectory_set.by_id(source)


def test_pilot_scenario_permits_the_composite_return_routes():
    """Fly out, divert to an alternate, then return - a path through the rules, not a declaration."""
    routes = load_scenario(PILOT_SCENARIO).trajectory_set.route_paths()
    assert ("t_p", "t_alt_north", "t_return_via_north") in routes
    assert ("t_p", "t_alt_east", "t_return_via_east") in routes
    assert ("t_p", "t_return_via_p") in routes


def test_pilot_scenario_generates_a_family_for_every_transition_rule():
    scenario = load_scenario(PILOT_SCENARIO)
    trajectory_set, conops = scenario.trajectory_set, scenario.conops

    for rule in trajectory_set.transitions:
        if rule.source == X0_NODE:
            continue
        family = conops.transition.family(
            trajectory_set.by_id(rule.source), trajectory_set.by_id(rule.target), rule
        )
        assert family.paths, f"{rule.source} -> {rule.target} admits nothing"


def test_pilot_scenario_relaxes_the_turn_screen_only_on_the_reversals():
    """A return is a turn-around, so the conops-level screen would reject every one."""
    trajectory_set = load_scenario(PILOT_SCENARIO).trajectory_set
    for rule in trajectory_set.transitions:
        if rule.target.startswith("t_return"):
            assert rule.max_turn_deg == 180.0
        else:
            assert rule.max_turn_deg is None


def test_pilot_scenario_flies_a_nadir_camera_with_no_attitude_allowance():
    """The first proof of concept: the mechanism exists, switched off."""
    camera = load_scenario(PILOT_SCENARIO).conops.camera
    assert camera.pose.is_nadir
    assert camera.attitude_margin.is_zero


def test_pilot_scenario_heights_sit_inside_the_operating_envelope():
    """200-4000 ft AGL is the project's envelope; San Jose ground is ~35 m."""
    scenario = load_scenario(PILOT_SCENARIO)
    for trajectory in scenario.trajectory_set.trajectories:
        for waypoint in trajectory.waypoints:
            agl = waypoint.height - 35.0
            assert 61.0 <= agl <= 1219.0


def test_tube_radius_has_no_default():
    config = _config()
    del config["conops"]["tube_radius_m"]
    with pytest.raises(ScenarioConfigError, match="tube_radius_m is required"):
        scenario_from_dict(config)


@pytest.mark.parametrize("radius", [0.0, -10.0])
def test_non_positive_tube_radius_is_refused(radius):
    with pytest.raises(ScenarioConfigError, match="tube_radius_m must be > 0"):
        scenario_from_dict(_config(tube_radius_m=radius))


def test_waypoints_accept_the_compact_list_form():
    config = {
        **MINIMAL,
        "trajectory_set": {
            **MINIMAL["trajectory_set"],
            "trajectories": [
                {
                    "id": "t_p",
                    "role": "primary",
                    "waypoints": [[37.3382, -121.8863, 300.0, 0.0], [37.3382, -121.8637, 300.0, 100.0]],
                }
            ],
        },
    }
    trajectory = scenario_from_dict(config).trajectory_set.primary
    assert trajectory.waypoints[1].height == 300.0
    assert trajectory.waypoints[1].time == 100.0


def test_primary_is_inferred_when_exactly_one_trajectory_declares_it():
    config = {**MINIMAL}
    config["trajectory_set"] = {k: v for k, v in MINIMAL["trajectory_set"].items()}
    assert "primary" not in config["trajectory_set"]
    assert scenario_from_dict(config).trajectory_set.primary_id == "t_p"


def test_missing_x0_is_refused():
    config = {**MINIMAL, "trajectory_set": {k: v for k, v in MINIMAL["trajectory_set"].items() if k != "x0"}}
    with pytest.raises(ScenarioConfigError, match="x0"):
        scenario_from_dict(config)


def test_unknown_role_names_the_valid_ones():
    config = {**MINIMAL}
    config["trajectory_set"] = {
        **MINIMAL["trajectory_set"],
        "trajectories": [{**MINIMAL["trajectory_set"]["trajectories"][0], "role": "backup"}],
        "primary": "t_p",
    }
    with pytest.raises(ScenarioConfigError, match="primary, alternate"):
        scenario_from_dict(config)


def test_authoring_a_transition_trajectory_is_refused_with_a_pointer():
    config = {**MINIMAL}
    config["trajectory_set"] = {
        **MINIMAL["trajectory_set"],
        "trajectories": [
            MINIMAL["trajectory_set"]["trajectories"][0],
            {**MINIMAL["trajectory_set"]["trajectories"][0], "id": "corridor", "role": "transition"},
        ],
        "primary": "t_p",
    }
    with pytest.raises(ScenarioConfigError, match="they are generated from a transitions: rule"):
        scenario_from_dict(config)


def test_transition_rules_are_parsed_with_their_overrides():
    config = {**MINIMAL}
    config["trajectory_set"] = {
        **MINIMAL["trajectory_set"],
        "primary": "t_p",
        "transitions": [
            {"source": "x0", "target": "t_p"},
            {
                "source": "t_p",
                "target": "t_p_2",
                "initiate_from_m": 200.0,
                "initiate_to_m": 900.0,
                "max_turn_deg": 45.0,
                "tangent_gain": 0.5,
            },
        ],
        "trajectories": [
            MINIMAL["trajectory_set"]["trajectories"][0],
            {**MINIMAL["trajectory_set"]["trajectories"][0], "id": "t_p_2", "role": "alternate"},
        ],
    }
    rules = scenario_from_dict(config).trajectory_set.transitions
    assert rules[0].source == X0_NODE
    assert (rules[1].initiate_from, rules[1].initiate_to) == (200.0, 900.0)
    assert (rules[1].max_turn_deg, rules[1].tangent_gain) == (45.0, 0.5)


def test_transition_model_defaults_apply_when_the_section_is_absent():
    conops = scenario_from_dict(_config()).conops
    assert conops.transition == TransitionModel()


def test_transition_model_is_parsed_from_config():
    conops = scenario_from_dict(
        _config(transition={"tangent_gain": 0.4, "max_turn_deg": 60.0, "samples": 3, "resolution": 8})
    ).conops
    assert conops.transition == TransitionModel(
        tangent_gain=0.4, max_turn_deg=60.0, samples=3, resolution=8
    )


def test_with_transition_samples_changes_only_the_sampling_density():
    conops = scenario_from_dict(_config()).conops
    denser = conops.with_transition_samples(40)
    assert denser.transition.samples == 40
    assert denser.tube_radius == conops.tube_radius
    assert conops.transition.samples != 40


def _generated_transition(due_east, parallel_north):
    """A generated transition trajectory - what the transition tube radius applies to."""
    return TransitionModel().path(due_east, parallel_north, 800.0).trajectory


def test_transition_radius_defaults_to_the_primary_radius(due_east, parallel_north):
    generated = _generated_transition(due_east, parallel_north)
    conops = ConopsConfig(tube_radius=250.0, window_length=1000.0)
    assert conops.radius_for(generated) == 250.0
    assert conops.tube_for(generated).radius == 250.0


def test_transition_radius_is_used_when_configured(due_east, parallel_north):
    generated = _generated_transition(due_east, parallel_north)
    conops = ConopsConfig(tube_radius=250.0, window_length=1000.0, transition_tube_radius=400.0)
    assert conops.radius_for(generated) == 400.0
    assert conops.radius_for(due_east) == 250.0


def test_per_trajectory_override_wins_over_both(due_east, parallel_north):
    generated = _generated_transition(due_east, parallel_north)
    conops = ConopsConfig(
        tube_radius=250.0,
        window_length=1000.0,
        transition_tube_radius=400.0,
        per_trajectory_radius={"due_east": 111.0, generated.id: 222.0},
    )
    assert conops.radius_for(due_east) == 111.0
    assert conops.radius_for(generated) == 222.0


def test_with_tube_radius_sweeps_the_whole_set(due_east, parallel_north):
    generated = _generated_transition(due_east, parallel_north)
    conops = ConopsConfig(
        tube_radius=250.0,
        window_length=1000.0,
        transition_tube_radius=400.0,
        per_trajectory_radius={"due_east": 111.0},
    )
    swept = conops.with_tube_radius(500.0, label="sweep_500")

    assert swept.radius_for(due_east) == 500.0
    assert swept.radius_for(generated) == 500.0
    assert swept.label == "sweep_500"
    assert swept.window_length == conops.window_length
    assert conops.tube_radius == 250.0  # the original is untouched


def test_sweeping_keeps_transitions_tracking_the_primary_when_they_already_did(
    due_east, parallel_north
):
    generated = _generated_transition(due_east, parallel_north)
    conops = ConopsConfig(tube_radius=250.0, window_length=1000.0)
    swept = conops.with_tube_radius(500.0)
    assert swept.transition_tube_radius is None
    assert swept.radius_for(generated) == 500.0


def test_scenario_with_tube_radius_returns_a_new_scenario():
    scenario = load_scenario(PILOT_SCENARIO)
    swept = scenario.with_tube_radius(600.0)

    assert swept.conops.tube_radius == 600.0
    assert scenario.conops.tube_radius != 600.0
    assert swept.trajectory_set is scenario.trajectory_set


def test_camera_is_parsed_from_config_including_pose_and_margin():
    scenario = scenario_from_dict(
        _config(
            camera={
                "field_of_view": {"horizontal_deg": 70.0, "vertical_deg": 50.0},
                "pose": {"pitch_deg": 12.0, "lever_arm_m": [1.0, 0.0, 0.5]},
                "attitude_margin": {"roll_deg": 8.0, "maneuver_roll_deg": 25.0, "maneuver_radius_m": 300.0},
            }
        )
    )
    camera = scenario.conops.camera
    assert camera.field_of_view == FieldOfView(horizontal_deg=70.0, vertical_deg=50.0)
    assert camera.pose.pitch_deg == 12.0
    assert camera.pose.lever_arm == (1.0, 0.0, 0.5)
    assert camera.attitude_margin.maneuver_radius == 300.0
    assert scenario.conops.field_of_view is camera.field_of_view


def test_camera_pose_and_margin_default_to_nadir_and_zero():
    conops = scenario_from_dict(_config(camera={"field_of_view": {"horizontal_deg": 60.0}})).conops
    assert conops.camera == Camera(field_of_view=FieldOfView(horizontal_deg=60.0))
    assert conops.camera.pose.is_nadir
    assert conops.camera.attitude_margin.is_zero


def test_a_camera_without_a_field_of_view_is_refused():
    with pytest.raises(ScenarioConfigError, match="needs a field_of_view"):
        scenario_from_dict(_config(camera={"pose": {"pitch_deg": 5.0}}))


def test_no_camera_section_means_no_camera():
    assert scenario_from_dict(_config()).conops.camera is None


def test_conops_to_dict_records_what_a_manifest_was_built_under():
    scenario = load_scenario(PILOT_SCENARIO)
    recorded = scenario.conops.to_dict()
    assert recorded["tube_radius_m"] == scenario.conops.tube_radius
    assert recorded["window_length_m"] == scenario.conops.window_length
    assert recorded["camera"]["field_of_view"]["horizontal_deg"] == (
        scenario.conops.field_of_view.horizontal_deg
    )
    assert recorded["camera"]["pose"]["pitch_deg"] == scenario.conops.camera.pose.pitch_deg
    assert recorded["transition"]["max_turn_deg"] == scenario.conops.transition.max_turn_deg


def test_load_scenario_rejects_a_non_mapping_file(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match="does not contain a YAML mapping"):
        load_scenario(path)
