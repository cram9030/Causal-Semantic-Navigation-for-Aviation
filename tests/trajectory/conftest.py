"""Shared fixtures for the Phase 1 trajectory tests.

The trajectories here are deliberately axis-aligned and short: a due-east leg
of a known length makes arc length, cross-track distance, and window edges
hand-checkable against an independent geodesic computation, rather than
against the code under test.
"""

from __future__ import annotations

import pytest

from csnav.trajectory.config import ConopsConfig
from csnav.geometry.fov import FieldOfView
from csnav.trajectory.trajectory import Trajectory, TrajectorySet, Transition
from csnav.trajectory.tube import TubeModel
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint

# Downtown San Jose, matching the origin used in tests/geometry/test_local_frame.py.
ORIGIN_LAT, ORIGIN_LON = 37.3382, -121.8863


@pytest.fixture
def due_east() -> Trajectory:
    """A ~2 km due-east leg at constant height, flown in 100 s."""
    return Trajectory(
        id="due_east",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, height=300.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.0226, height=300.0, time=100.0),
        ),
        role=TrajectoryRole.PRIMARY,
    )


@pytest.fixture
def dogleg() -> Trajectory:
    """Two legs with a corner, for tests that need real interior geometry."""
    return Trajectory(
        id="dogleg",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, height=300.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.0226, height=300.0, time=100.0),
            Waypoint(lat=ORIGIN_LAT + 0.0180, lon=ORIGIN_LON + 0.0226, height=400.0, time=200.0),
        ),
    )


@pytest.fixture
def corridor(due_east) -> Trajectory:
    """A transition corridor connecting the two candidates in :func:`trajectory_set`."""
    return Trajectory(
        id="x_east_to_north",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.0113, height=300.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT + 0.0050, lon=ORIGIN_LON + 0.0113, height=320.0, time=30.0),
        ),
        role=TrajectoryRole.TRANSITION,
        connects=("due_east", "due_north"),
    )


@pytest.fixture
def due_north() -> Trajectory:
    """A ~2 km due-north alternate candidate."""
    return Trajectory(
        id="due_north",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT + 0.0050, lon=ORIGIN_LON + 0.0113, height=320.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT + 0.0230, lon=ORIGIN_LON + 0.0113, height=320.0, time=100.0),
        ),
        role=TrajectoryRole.ALTERNATE,
    )


@pytest.fixture
def trajectory_set(due_east, due_north, corridor) -> TrajectorySet:
    return TrajectorySet(
        id="test_set",
        trajectories=(due_east, due_north, corridor),
        primary_id="due_east",
        x0=Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, height=300.0, time=0.0),
        transitions=(
            Transition(source="x0", target="due_east"),
            Transition(source="due_east", target="due_north", via="x_east_to_north"),
        ),
    )


@pytest.fixture
def conops() -> ConopsConfig:
    return ConopsConfig(
        tube_radius=200.0,
        window_length=1000.0,
        field_of_view=FieldOfView(horizontal_deg=60.0, vertical_deg=45.0),
        transition_tube_radius=350.0,
        tile_level=17,
        label="test",
    )


@pytest.fixture
def tube() -> TubeModel:
    return TubeModel(radius=200.0)
