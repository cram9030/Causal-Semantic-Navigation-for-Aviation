"""Shared fixtures for the Phase 1 trajectory tests.

The trajectories here are deliberately axis-aligned and short: a due-east leg
of a known length makes arc length, cross-track distance, and window edges
hand-checkable against an independent geodesic computation, rather than
against the code under test.
"""

from __future__ import annotations

import pytest

from csnav.geometry.camera import Camera
from csnav.geometry.fov import FieldOfView
from csnav.trajectory.config import ConopsConfig
from csnav.trajectory.trajectory import Trajectory, TrajectorySet, TransitionRule
from csnav.trajectory.transition import TransitionModel
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
def parallel_north() -> Trajectory:
    """A ~2 km due-east alternate, ~1.1 km north of :func:`due_east` and slightly ahead.

    Parallel and offset so a transition onto it is a gentle S-curve with
    hand-predictable geometry: the arrival waypoint is always the one ahead of
    where the source point projects. It stops ~200 m short of
    :func:`due_east`'s end, so the very end of the source projects past it and
    has nothing left to rejoin - the case that yields no transition at all.
    """
    return Trajectory(
        id="parallel_north",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT + 0.0100, lon=ORIGIN_LON, height=320.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT + 0.0100, lon=ORIGIN_LON + 0.0113, height=320.0, time=50.0),
            Waypoint(lat=ORIGIN_LAT + 0.0100, lon=ORIGIN_LON + 0.0200, height=320.0, time=90.0),
        ),
        role=TrajectoryRole.ALTERNATE,
    )


@pytest.fixture
def returning() -> Trajectory:
    """A route from :func:`due_east`'s far end back to its start - a return leg."""
    return Trajectory(
        id="returning",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON + 0.0226, height=300.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT - 0.0060, lon=ORIGIN_LON + 0.0113, height=320.0, time=60.0),
            Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, height=300.0, time=120.0),
        ),
        role=TrajectoryRole.ALTERNATE,
    )


@pytest.fixture
def orthogonal() -> Trajectory:
    """A due-north route crossing :func:`due_east` - the near-orthogonal alternate case."""
    return Trajectory(
        id="orthogonal",
        waypoints=(
            Waypoint(lat=ORIGIN_LAT - 0.0090, lon=ORIGIN_LON + 0.0113, height=320.0, time=0.0),
            Waypoint(lat=ORIGIN_LAT + 0.0090, lon=ORIGIN_LON + 0.0113, height=320.0, time=100.0),
        ),
        role=TrajectoryRole.ALTERNATE,
    )


@pytest.fixture
def trajectory_set(due_east, parallel_north) -> TrajectorySet:
    return TrajectorySet(
        id="test_set",
        trajectories=(due_east, parallel_north),
        primary_id="due_east",
        x0=Waypoint(lat=ORIGIN_LAT, lon=ORIGIN_LON, height=300.0, time=0.0),
        transitions=(
            TransitionRule(source="x0", target="due_east"),
            TransitionRule(source="due_east", target="parallel_north"),
        ),
    )


@pytest.fixture
def camera() -> Camera:
    """A nadir camera - the first prototype's mounting, with no attitude allowance."""
    return Camera(field_of_view=FieldOfView(horizontal_deg=60.0, vertical_deg=45.0))


@pytest.fixture
def conops(camera) -> ConopsConfig:
    return ConopsConfig(
        tube_radius=200.0,
        window_length=1000.0,
        camera=camera,
        transition=TransitionModel(samples=5, resolution=12),
        transition_tube_radius=350.0,
        tile_level=17,
        label="test",
    )


@pytest.fixture
def tube() -> TubeModel:
    return TubeModel(radius=200.0)


@pytest.fixture
def model() -> TransitionModel:
    return TransitionModel(samples=5, resolution=12)
