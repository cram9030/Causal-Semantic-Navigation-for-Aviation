"""Scenario config loading, and the tube-radius sweep entry point.

The property that matters most here: the tube radius has to come from config
and can never be defaulted (CLAUDE.md core decision 4), and overriding it must
be a one-call operation so a sweep needs no code change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from csnav.geometry.fov import FieldOfView
from csnav.trajectory.config import (
    ConopsConfig,
    ScenarioConfigError,
    load_scenario,
    scenario_from_dict,
)
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
    assert trajectory_set.candidates and trajectory_set.corridors
    assert scenario.conops.tube_radius > 0.0
    assert scenario.source_path == PILOT_SCENARIO

    graph = trajectory_set.to_networkx()
    assert X0_NODE in graph
    # Every corridor named by a transition exists as a trajectory in the set.
    for _, _, data in graph.edges(data=True):
        if data["via"]:
            assert trajectory_set.by_id(data["via"]).role is TrajectoryRole.TRANSITION


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
    with pytest.raises(ScenarioConfigError, match="primary, alternate, transition"):
        scenario_from_dict(config)


def test_transition_radius_defaults_to_the_primary_radius(due_east, corridor):
    conops = ConopsConfig(tube_radius=250.0, window_length=1000.0)
    assert conops.radius_for(corridor) == 250.0
    assert conops.tube_for(corridor).radius == 250.0


def test_transition_radius_is_used_when_configured(corridor, due_east):
    conops = ConopsConfig(tube_radius=250.0, window_length=1000.0, transition_tube_radius=400.0)
    assert conops.radius_for(corridor) == 400.0
    assert conops.radius_for(due_east) == 250.0


def test_per_trajectory_override_wins_over_both(due_east, corridor):
    conops = ConopsConfig(
        tube_radius=250.0,
        window_length=1000.0,
        transition_tube_radius=400.0,
        per_trajectory_radius={"due_east": 111.0, "x_east_to_north": 222.0},
    )
    assert conops.radius_for(due_east) == 111.0
    assert conops.radius_for(corridor) == 222.0


def test_with_tube_radius_sweeps_the_whole_set(due_east, corridor):
    conops = ConopsConfig(
        tube_radius=250.0,
        window_length=1000.0,
        transition_tube_radius=400.0,
        per_trajectory_radius={"due_east": 111.0},
    )
    swept = conops.with_tube_radius(500.0, label="sweep_500")

    assert swept.radius_for(due_east) == 500.0
    assert swept.radius_for(corridor) == 500.0
    assert swept.label == "sweep_500"
    assert swept.window_length == conops.window_length
    assert conops.tube_radius == 250.0  # the original is untouched


def test_sweeping_keeps_transitions_tracking_the_primary_when_they_already_did(corridor):
    conops = ConopsConfig(tube_radius=250.0, window_length=1000.0)
    swept = conops.with_tube_radius(500.0)
    assert swept.transition_tube_radius is None
    assert swept.radius_for(corridor) == 500.0


def test_scenario_with_tube_radius_returns_a_new_scenario():
    scenario = load_scenario(PILOT_SCENARIO)
    swept = scenario.with_tube_radius(600.0)

    assert swept.conops.tube_radius == 600.0
    assert scenario.conops.tube_radius != 600.0
    assert swept.trajectory_set is scenario.trajectory_set


def test_field_of_view_is_parsed_from_config():
    scenario = scenario_from_dict(_config(field_of_view={"horizontal_deg": 70.0, "vertical_deg": 50.0}))
    assert scenario.conops.field_of_view == FieldOfView(horizontal_deg=70.0, vertical_deg=50.0)


def test_conops_to_dict_records_what_a_manifest_was_built_under():
    scenario = load_scenario(PILOT_SCENARIO)
    recorded = scenario.conops.to_dict()
    assert recorded["tube_radius_m"] == scenario.conops.tube_radius
    assert recorded["window_length_m"] == scenario.conops.window_length
    assert recorded["field_of_view"]["horizontal_deg"] == scenario.conops.field_of_view.horizontal_deg


def test_load_scenario_rejects_a_non_mapping_file(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ScenarioConfigError, match="does not contain a YAML mapping"):
        load_scenario(path)
