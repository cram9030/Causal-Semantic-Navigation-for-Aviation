"""Versioned scenario configuration: the trajectory set ``T`` plus its CONOPS parameters.

CLAUDE.md requires that tube radius, trajectory sets, and AOI bounds live in
versioned config files rather than hardcoded constants, because several of them
- the tube radius above all - are meant to be swept across experiments. This
module is that boundary: a YAML scenario file in ``configs/scenarios/`` is
loaded into a :class:`Scenario`, and every downstream component
(:class:`~csnav.trajectory.tube.TubeModel`,
:class:`~csnav.trajectory.manifest_builder.ManifestBuilder`, the visualization
tools) takes its parameters from there.

:meth:`ConopsConfig.with_tube_radius` is the sweep entry point: it returns a
new config at a different radius, so re-running a study is a parameter change,
not a code change (integration plan §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from csnav.geometry.fov import FieldOfView
from csnav.trajectory.trajectory import Trajectory, TrajectorySet, Transition
from csnav.trajectory.tube import TubeModel
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint


class ScenarioConfigError(ValueError):
    """Raised when a scenario config file is missing required fields or malformed."""


@dataclass(frozen=True)
class ConopsConfig:
    """The swept experimental parameters for one concept-of-operations case.

    Units: ``tube_radius``, ``transition_tube_radius`` and ``window_length`` in
    meters; ``field_of_view`` in degrees. ``per_trajectory_radius`` maps a
    trajectory id to its own radius in meters, overriding the defaults.

    ``transition_tube_radius`` of ``None`` means transition corridors share the
    primary tube radius - integration plan §8 lists "whether the
    transition-corridor tube should share the primary trajectory's radius or be
    its own case" as still open, so both are expressible and the choice is
    recorded in the config rather than assumed in code.
    """

    tube_radius: float
    window_length: float
    field_of_view: FieldOfView | None = None
    transition_tube_radius: float | None = None
    per_trajectory_radius: dict[str, float] = field(default_factory=dict)
    tile_level: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.tube_radius <= 0.0:
            raise ScenarioConfigError(f"tube_radius_m must be > 0, got {self.tube_radius}")
        if self.window_length <= 0.0:
            raise ScenarioConfigError(f"window_length_m must be > 0, got {self.window_length}")
        if self.transition_tube_radius is not None and self.transition_tube_radius <= 0.0:
            raise ScenarioConfigError(
                f"transition_tube_radius_m must be > 0 when set, got {self.transition_tube_radius}"
            )
        for trajectory_id, radius in self.per_trajectory_radius.items():
            if radius <= 0.0:
                raise ScenarioConfigError(f"tube radius for {trajectory_id!r} must be > 0, got {radius}")

    def radius_for(self, trajectory: Trajectory) -> float:
        """Tube radius in meters for one trajectory: explicit override, else role default."""
        if trajectory.id in self.per_trajectory_radius:
            return self.per_trajectory_radius[trajectory.id]
        if trajectory.role is TrajectoryRole.TRANSITION and self.transition_tube_radius is not None:
            return self.transition_tube_radius
        return self.tube_radius

    def tube_for(self, trajectory: Trajectory) -> TubeModel:
        """The :class:`~csnav.trajectory.tube.TubeModel` this CONOPS assigns to a trajectory."""
        return TubeModel(radius=self.radius_for(trajectory), label=self.label)

    def with_tube_radius(self, radius: float, label: str | None = None) -> "ConopsConfig":
        """Copy of this config at a different tube radius (meters) - the sweep entry point.

        Per-trajectory overrides are cleared, so a sweep sets one radius across
        the whole set rather than silently keeping a pinned override for some
        trajectories. The transition radius is kept only if it was explicitly
        set relative to nothing (i.e. it stays ``None`` when it was ``None``,
        so transitions keep tracking the primary radius).
        """
        return replace(
            self,
            tube_radius=radius,
            per_trajectory_radius={},
            transition_tube_radius=None if self.transition_tube_radius is None else radius,
            label=label if label is not None else self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tube_radius_m": self.tube_radius,
            "window_length_m": self.window_length,
            "transition_tube_radius_m": self.transition_tube_radius,
            "per_trajectory_tube_radius_m": dict(self.per_trajectory_radius),
            "field_of_view": (
                {
                    "horizontal_deg": self.field_of_view.horizontal_deg,
                    "vertical_deg": self.field_of_view.vertical_deg,
                }
                if self.field_of_view
                else None
            ),
            "tile_level": self.tile_level,
            "label": self.label,
        }


@dataclass(frozen=True)
class Scenario:
    """One versioned experiment case: a trajectory set plus the CONOPS it is flown under."""

    id: str
    trajectory_set: TrajectorySet
    conops: ConopsConfig
    description: str | None = None
    source_path: Path | None = None

    def with_tube_radius(self, radius: float, label: str | None = None) -> "Scenario":
        """Copy of this scenario at a different tube radius (meters)."""
        return replace(self, conops=self.conops.with_tube_radius(radius, label=label))


def _waypoint_from_config(raw: Any, index: int, trajectory_id: str) -> Waypoint:
    """Parse one waypoint: either a mapping or a ``[lat, lon, height_m, time_s]`` list."""
    if isinstance(raw, Mapping):
        try:
            return Waypoint(
                lat=float(raw["lat"]),
                lon=float(raw["lon"]),
                height=float(raw.get("height_m", 0.0)),
                time=float(raw.get("time_s", index)),
            )
        except KeyError as exc:
            raise ScenarioConfigError(
                f"waypoint {index} of trajectory {trajectory_id!r} is missing {exc.args[0]!r}"
            ) from exc
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        values = [float(value) for value in raw]
        return Waypoint(
            lat=values[0],
            lon=values[1],
            height=values[2] if len(values) > 2 else 0.0,
            time=values[3] if len(values) > 3 else float(index),
        )
    raise ScenarioConfigError(
        f"waypoint {index} of trajectory {trajectory_id!r} must be a mapping or "
        f"[lat, lon, height_m, time_s] list, got {raw!r}"
    )


def _trajectory_from_config(raw: Mapping[str, Any]) -> Trajectory:
    try:
        trajectory_id = str(raw["id"])
    except KeyError as exc:
        raise ScenarioConfigError("every trajectory needs an 'id'") from exc

    role_name = str(raw.get("role", TrajectoryRole.ALTERNATE.value)).lower()
    try:
        role = TrajectoryRole(role_name)
    except ValueError as exc:
        roles = ", ".join(item.value for item in TrajectoryRole)
        raise ScenarioConfigError(f"trajectory {trajectory_id!r} has unknown role {role_name!r} (expected: {roles})") from exc

    waypoints_raw = raw.get("waypoints") or []
    waypoints = tuple(
        _waypoint_from_config(item, index, trajectory_id) for index, item in enumerate(waypoints_raw)
    )
    connects_raw = raw.get("connects")
    connects = (str(connects_raw[0]), str(connects_raw[1])) if connects_raw else None
    metadata = dict(raw.get("metadata") or {})
    return Trajectory(id=trajectory_id, waypoints=waypoints, role=role, connects=connects, metadata=metadata)


def _conops_from_config(raw: Mapping[str, Any]) -> ConopsConfig:
    if "tube_radius_m" not in raw:
        raise ScenarioConfigError(
            "conops.tube_radius_m is required - the tube radius is a swept input and has no default"
        )
    fov_raw = raw.get("field_of_view")
    field_of_view = (
        FieldOfView(
            horizontal_deg=float(fov_raw["horizontal_deg"]),
            vertical_deg=None if fov_raw.get("vertical_deg") is None else float(fov_raw["vertical_deg"]),
        )
        if fov_raw
        else None
    )
    transition_raw = raw.get("transition_tube_radius_m")
    return ConopsConfig(
        tube_radius=float(raw["tube_radius_m"]),
        window_length=float(raw.get("window_length_m", 1000.0)),
        field_of_view=field_of_view,
        transition_tube_radius=None if transition_raw is None else float(transition_raw),
        per_trajectory_radius={
            str(key): float(value) for key, value in (raw.get("per_trajectory_tube_radius_m") or {}).items()
        },
        tile_level=None if raw.get("tile_level") is None else int(raw["tile_level"]),
        label=raw.get("label"),
    )


def scenario_from_dict(raw: Mapping[str, Any], source_path: Path | None = None) -> Scenario:
    """Build a :class:`Scenario` from an already-parsed config mapping."""
    if "trajectory_set" not in raw:
        raise ScenarioConfigError("scenario config needs a 'trajectory_set' section")
    if "conops" not in raw:
        raise ScenarioConfigError("scenario config needs a 'conops' section")

    set_raw = raw["trajectory_set"]
    trajectories = tuple(_trajectory_from_config(item) for item in set_raw.get("trajectories") or [])
    if not trajectories:
        raise ScenarioConfigError("trajectory_set.trajectories is empty")

    x0_raw = set_raw.get("x0")
    if x0_raw is None:
        raise ScenarioConfigError("trajectory_set.x0 (the known start state) is required")
    x0 = _waypoint_from_config(x0_raw, 0, "x0")

    primary_id = set_raw.get("primary")
    if primary_id is None:
        primaries = [t.id for t in trajectories if t.role is TrajectoryRole.PRIMARY]
        if len(primaries) != 1:
            raise ScenarioConfigError(
                "trajectory_set.primary is required unless exactly one trajectory has role: primary"
            )
        primary_id = primaries[0]

    transitions = tuple(
        Transition(
            source=str(item["source"]),
            target=str(item["target"]),
            via=None if item.get("via") is None else str(item["via"]),
        )
        for item in set_raw.get("transitions") or []
    )

    trajectory_set = TrajectorySet(
        id=str(set_raw.get("id", raw.get("id", "trajectory_set"))),
        trajectories=trajectories,
        primary_id=str(primary_id),
        x0=x0,
        transitions=transitions,
    )
    return Scenario(
        id=str(raw.get("id", trajectory_set.id)),
        trajectory_set=trajectory_set,
        conops=_conops_from_config(raw["conops"]),
        description=raw.get("description"),
        source_path=source_path,
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a versioned scenario YAML (see ``configs/scenarios/``).

    Raises :class:`ScenarioConfigError` rather than defaulting when a required
    field is missing - notably ``conops.tube_radius_m``, which must always be
    stated because it is the parameter the experiments sweep.
    """
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ScenarioConfigError(f"{source} does not contain a YAML mapping")
    return scenario_from_dict(raw, source_path=source)
