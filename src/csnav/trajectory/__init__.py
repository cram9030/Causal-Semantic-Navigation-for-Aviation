"""Phase 1: the candidate trajectory set ``T``, its RNP tube, and precomputed manifests.

Implements `docs/INTEGRATION_PLAN.md` §5's Phase 1 - "Define ``T``, ``t_p``,
``x_0``, and the RNP-style tube model" and "implement the offline manifest
builder: trajectory windows -> tube envelope -> CSJ Streets query -> per-window
landmark manifest".
"""

from csnav.trajectory.config import ConopsConfig, Scenario, ScenarioConfigError, load_scenario
from csnav.trajectory.coverage import (
    TileRef,
    agl_from_elevation,
    height_as_agl,
    max_agl,
    max_ground_reach,
    merge_tiles,
    tiles_for_footprint,
    visible_footprint,
)
from csnav.trajectory.manifest import (
    LandmarkManifest,
    ManifestBundle,
    ManifestIntersection,
    ManifestLandmark,
)
from csnav.trajectory.manifest_builder import ManifestBuilder, StaticStreetsSource
from csnav.trajectory.trajectory import (
    X0_NODE,
    Trajectory,
    TrajectoryError,
    TrajectorySet,
    TrajectoryWindow,
    TransitionError,
    TransitionRule,
)
from csnav.trajectory.transition import TransitionFamily, TransitionModel, TransitionPath, transition_id
from csnav.trajectory.tube import TubeModel, union_corridor
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint

__all__ = [
    "ConopsConfig",
    "LandmarkManifest",
    "ManifestBuilder",
    "ManifestBundle",
    "ManifestIntersection",
    "ManifestLandmark",
    "Scenario",
    "ScenarioConfigError",
    "StaticStreetsSource",
    "TileRef",
    "Trajectory",
    "TrajectoryError",
    "TrajectoryRole",
    "TrajectorySet",
    "TrajectoryWindow",
    "TransitionError",
    "TransitionFamily",
    "TransitionModel",
    "TransitionPath",
    "TransitionRule",
    "TubeModel",
    "Waypoint",
    "X0_NODE",
    "agl_from_elevation",
    "height_as_agl",
    "load_scenario",
    "max_agl",
    "max_ground_reach",
    "merge_tiles",
    "tiles_for_footprint",
    "transition_id",
    "union_corridor",
    "visible_footprint",
]
