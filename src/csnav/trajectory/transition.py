"""Transitions between candidate trajectories, generated rather than authored.

A transition is **not known before flight**. What is known is that one *could*
begin at any point along the trajectory currently being flown, and that from
there the aircraft would rejoin the target route. So the flight plan does not
carry transition geometry; it carries a :class:`TransitionRule` saying which
route may hand off to which, and :class:`TransitionModel` generates the family
of paths that rule admits.

The formalization, per `docs/INTEGRATION_PLAN.md` §3.2's "transition corridor
between two trajectories, including a return path to ``x_0``":

* **Initiation** is any arc-length position ``s`` on the source trajectory -
  not restricted to its waypoints. A rule may narrow that domain
  (``initiate_from``/``initiate_to``) for cases where a target only becomes
  valid past a certain point.
* **Arrival** is the first waypoint on the target that lies *ahead* of where
  the initiation point projects onto the target's ground track. Using the
  projection rather than raw proximity guarantees the transition makes forward
  progress instead of doubling back to a waypoint already passed.
* **Geometry** is a cubic Hermite spline between those two points, matching the
  source's direction of travel at ``s`` and the target's at the arrival
  waypoint. Endpoint tangent magnitudes are ``tangent_gain x`` the straight-line
  distance between them, which keeps the curve's shape scale-free. *This is a
  placeholder for a real dynamics model* - it is smooth and heading-continuous,
  but it knows nothing about turn rate, bank limits, or airspeed.
* **Feasibility** is screened crudely, by the heading change the transition
  demands at each end: ``max_turn_deg`` drops initiations needing a sharper turn
  than that at either end. It is what keeps a near-orthogonal alternate from
  generating transitions no aircraft could fly.

  Note what this screen cannot do: it measures heading change, so it cannot
  tell a *reversal it should allow* from one it should not. Diverting onto a
  return route is a deliberate turn-around - departure turns of 150-180 deg are
  the normal case, not a defect - while a target waypoint that has effectively
  been passed produces the same number. In practice that means return edges
  need a permissive limit (see the pilot scenario), and a screen that actually
  separates the two would have to test the generated curve's curvature against
  a turn radius rather than test an angle.

**The whole family is the object of interest**, not any one path: because
initiation is continuous in ``s``, the region between two trajectories that the
aircraft could legitimately occupy is the union of the family's tubes
(:meth:`TransitionFamily.reachable_footprint`), and any point inside it is a
valid state. The discrete sample this module returns is a finite stand-in for
that continuum, dense enough to draw and to build manifests over.

**Frames.** Splines are built in the *source* trajectory's local ENU frame, in
meters, and converted back to WGS84 before being returned as a
:class:`~csnav.trajectory.trajectory.Trajectory`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from shapely.geometry import Polygon
from shapely.ops import unary_union

from csnav.geometry.local_frame import LocalFrame
from csnav.trajectory.trajectory import Trajectory, TransitionError, TransitionRule
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint

#: Fallback ground speed, in m/s, used to time a generated transition when the
#: source trajectory's own schedule gives no usable speed at the initiation
#: point (e.g. two waypoints sharing a timestamp).
FALLBACK_SPEED = 45.0

#: How far ahead of the projection, in meters of the target's arc length, a
#: waypoint must sit to count as something to rejoin at. Guards the boundary
#: case where the projection lands on the target's final waypoint: the
#: 2D-to-3D arc-length mapping and shapely's own projection agree only to
#: within float noise there, and a rejoin a centimetre ahead is not a rejoin.
ARRIVAL_TOLERANCE = 1.0

#: Shortest transition, in meters, that means anything. Below this the arrival
#: waypoint is where the aircraft already is - which happens for real, not just
#: from rounding: initiating a return at the very start of an outbound route
#: projects onto the return's own final waypoint, back at ``x_0``. There is no
#: transition to fly, so none is generated.
MIN_TRANSITION_CHORD = 1.0


@dataclass(frozen=True)
class TransitionPath:
    """One realized transition: where it began, where it rejoins, and the curve between.

    ``initiate_distance`` is arc length in meters along the source;
    ``arrival_index`` indexes the target's waypoints and ``arrival_distance`` is
    that waypoint's arc length along the target. ``departure_turn`` and
    ``arrival_turn`` are the heading changes in degrees the path demands at each
    end - the quantities the feasibility screen tests. ``trajectory`` is the
    sampled curve as an ordinary :class:`~csnav.trajectory.trajectory.Trajectory`
    with :attr:`~csnav.trajectory.waypoints.TrajectoryRole.TRANSITION` role, so
    the tube model and every downstream consumer treat it like any other path.
    """

    source_id: str
    target_id: str
    initiate_distance: float
    arrival_index: int
    arrival_distance: float
    departure_turn: float
    arrival_turn: float
    trajectory: Trajectory

    @property
    def id(self) -> str:
        return self.trajectory.id


@dataclass(frozen=True)
class TransitionFamily:
    """Every path one :class:`TransitionRule` admits, sampled across its initiation domain.

    The sampled paths are a finite stand-in for a continuum: initiation is
    continuous in arc length, so the aircraft may be anywhere in the region the
    family sweeps, not only on one of these curves. Use
    :meth:`reachable_footprint` when what matters is that region.
    """

    rule: TransitionRule
    paths: tuple[TransitionPath, ...]
    domain: tuple[float, float]
    rejected: int = 0

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def is_empty(self) -> bool:
        """Whether the rule admitted no feasible path at all (every sample was screened out)."""
        return not self.paths

    @property
    def turn_range(self) -> tuple[float, float]:
        """Smallest and largest heading change, in degrees, across the family (0, 0 when empty)."""
        if not self.paths:
            return 0.0, 0.0
        turns = [max(path.departure_turn, path.arrival_turn) for path in self.paths]
        return min(turns), max(turns)

    def reachable_footprint(self, tube, extra_buffer: float = 0.0) -> Polygon:
        """Ground region the family can occupy, as a WGS84 ``(lon, lat)`` geometry.

        The union of every sampled path's tube corridor, grown by
        ``extra_buffer`` meters. Because the sample stands in for a continuum of
        initiation points, this is the set of positions that are valid *during*
        a transition - the thing the containment assumption has to cover while
        the aircraft is between two trajectories.
        """
        if not self.paths:
            return Polygon()
        return unary_union(
            [tube.corridor(path.trajectory, extra_buffer=extra_buffer) for path in self.paths]
        )


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return (1.0, 0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _horizontal_angle(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Angle in degrees between the horizontal components of two ENU vectors."""
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    norm = math.hypot(ax, ay) * math.hypot(bx, by)
    if norm == 0.0:
        return 0.0
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / norm))
    return math.degrees(math.acos(cosine))


def _hermite(
    p0: tuple[float, float, float],
    m0: tuple[float, float, float],
    p1: tuple[float, float, float],
    m1: tuple[float, float, float],
    u: float,
) -> tuple[float, float, float]:
    """Cubic Hermite interpolation at parameter ``u`` in ``[0, 1]``."""
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return tuple(
        h00 * p0[axis] + h10 * m0[axis] + h01 * p1[axis] + h11 * m1[axis] for axis in range(3)
    )


def _in_frame(waypoint: Waypoint, frame: LocalFrame) -> tuple[float, float, float]:
    point = frame.to_enu(waypoint.lat, waypoint.lon, waypoint.height)
    return (point.east, point.north, point.up)


@dataclass(frozen=True)
class TransitionModel:
    """Generates the family of paths a :class:`TransitionRule` admits.

    ``tangent_gain`` scales the Hermite endpoint tangents by the straight-line
    distance between the endpoints (so a short transition and a long one have
    the same shape; 1.0 is a smooth, moderately bulged curve, smaller is
    tighter). ``max_turn_deg`` is the crude feasibility screen described in the
    module docstring, applied to the heading change at both ends; it defaults to
    180 deg - admitting everything - so that nothing is dropped unless a
    scenario asks for it.

    ``samples`` is how many initiation points are taken across a rule's domain -
    the resolution of the finite stand-in for a continuous family - and
    ``resolution`` is how many vertices each generated curve gets.

    ``speed`` is a fallback ground speed in m/s used to time a generated
    transition when the source's own schedule gives none; the source's local
    speed is preferred wherever it is available.
    """

    tangent_gain: float = 1.0
    max_turn_deg: float | None = 180.0
    samples: int = 12
    resolution: int = 24
    speed: float = FALLBACK_SPEED

    def __post_init__(self) -> None:
        if self.tangent_gain <= 0.0:
            raise TransitionError(f"tangent_gain must be > 0, got {self.tangent_gain}")
        if self.samples < 1:
            raise TransitionError(f"samples must be >= 1, got {self.samples}")
        if self.resolution < 2:
            raise TransitionError(f"resolution must be >= 2 spline vertices, got {self.resolution}")
        if self.max_turn_deg is not None and not 0.0 < self.max_turn_deg <= 180.0:
            raise TransitionError(f"max_turn_deg must be in (0, 180], got {self.max_turn_deg}")
        if self.speed <= 0.0:
            raise TransitionError(f"speed must be > 0 m/s, got {self.speed}")

    # ----- the arrival rule ---------------------------------------------------

    def arrival_index(self, source: Trajectory, target: Trajectory, initiate_distance: float) -> int | None:
        """Index of the target waypoint a transition from ``initiate_distance`` rejoins at.

        The first target waypoint ahead of where the initiation point projects
        onto the target's ground track, by at least :data:`ARRIVAL_TOLERANCE`.
        ``None`` when the projection already lies at or past the target's final
        waypoint - there is nothing left to rejoin, so no transition exists from
        there.
        """
        origin = source.point_at(initiate_distance)
        projected = target.project(origin.lat, origin.lon)
        for index, distance in enumerate(target.cumulative_distances):
            if distance > projected + ARRIVAL_TOLERANCE:
                return index
        return None

    # ----- one path -----------------------------------------------------------

    def path(
        self,
        source: Trajectory,
        target: Trajectory,
        initiate_distance: float,
        rule: TransitionRule | None = None,
    ) -> TransitionPath | None:
        """The transition initiating at ``initiate_distance`` meters along ``source``.

        Returns ``None`` when no transition exists from there: either the
        projection has run past the end of the target, or the required heading
        change exceeds the feasibility screen.
        """
        arrival_index = self.arrival_index(source, target, initiate_distance)
        if arrival_index is None:
            return None

        frame = source.local_frame
        origin = source.point_at(initiate_distance)
        arrival = target.waypoints[arrival_index]

        p0 = _in_frame(origin, frame)
        p1 = _in_frame(arrival, frame)
        chord = math.dist(p0, p1)
        if chord < MIN_TRANSITION_CHORD:
            return None

        departure_tangent = source.tangent_at(initiate_distance)
        arrival_tangent = self._target_tangent(target, arrival_index, frame)
        chord_direction = _unit((p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]))

        departure_turn = _horizontal_angle(departure_tangent, chord_direction)
        arrival_turn = _horizontal_angle(chord_direction, arrival_tangent)
        limit = self.max_turn_deg if rule is None or rule.max_turn_deg is None else rule.max_turn_deg
        if limit is not None and max(departure_turn, arrival_turn) > limit:
            return None

        gain = self.tangent_gain if rule is None or rule.tangent_gain is None else rule.tangent_gain
        scale = gain * chord
        m0 = tuple(component * scale for component in departure_tangent)
        m1 = tuple(component * scale for component in arrival_tangent)

        curve = [
            _hermite(p0, m0, p1, m1, step / self.resolution) for step in range(self.resolution + 1)
        ]
        trajectory = self._as_trajectory(curve, frame, source, target, initiate_distance, origin.time)

        return TransitionPath(
            source_id=source.id,
            target_id=target.id,
            initiate_distance=initiate_distance,
            arrival_index=arrival_index,
            arrival_distance=target.cumulative_distances[arrival_index],
            departure_turn=departure_turn,
            arrival_turn=arrival_turn,
            trajectory=trajectory,
        )

    def _target_tangent(
        self, target: Trajectory, arrival_index: int, frame: LocalFrame
    ) -> tuple[float, float, float]:
        """Target's direction of travel at its arrival waypoint, expressed in ``frame``.

        Computed by differencing the bracketing waypoints *after* converting
        both into the source's frame, rather than by rotating a direction
        between frames - the two tangent planes are not quite parallel, and
        differencing in one frame sidesteps that entirely.
        """
        following = min(arrival_index + 1, len(target.waypoints) - 1)
        preceding = arrival_index if following > arrival_index else max(arrival_index - 1, 0)
        start = _in_frame(target.waypoints[preceding], frame)
        end = _in_frame(target.waypoints[following], frame)
        return _unit((end[0] - start[0], end[1] - start[1], end[2] - start[2]))

    def _as_trajectory(
        self,
        curve: list[tuple[float, float, float]],
        frame: LocalFrame,
        source: Trajectory,
        target: Trajectory,
        initiate_distance: float,
        start_time: float,
    ) -> Trajectory:
        """Turn an ENU curve into a WGS84 transition trajectory with a consistent time schedule.

        Times run forward from the source's flight-plan time at the initiation
        point, advancing at the source's own local ground speed (or this model's
        fallback where the plan gives none), so a generated transition's clock
        is continuous with the route it left.
        """
        speed = source.speed_at(initiate_distance) or self.speed
        waypoints: list[Waypoint] = []
        elapsed = 0.0
        for index, point in enumerate(curve):
            if index > 0:
                elapsed += math.dist(curve[index - 1], point) / speed
            position = frame.to_wgs84(point[0], point[1], point[2])
            waypoints.append(
                Waypoint(
                    lat=position.lat,
                    lon=position.lon,
                    height=position.height,
                    time=start_time + elapsed,
                )
            )
        return Trajectory(
            id=transition_id(source.id, target.id, initiate_distance),
            waypoints=tuple(waypoints),
            role=TrajectoryRole.TRANSITION,
            connects=(source.id, target.id),
            metadata={"initiate_distance_m": initiate_distance, "generated": True},
        )

    # ----- the family ---------------------------------------------------------

    def family(self, source: Trajectory, target: Trajectory, rule: TransitionRule) -> TransitionFamily:
        """Every path ``rule`` admits, sampled evenly across its initiation domain.

        Samples that yield no path - projection past the target's end, or a
        heading change over the screen - are counted in
        :attr:`TransitionFamily.rejected` rather than silently dropped, so an
        edge that generates nothing is visible as such instead of just absent.
        """
        start, end = rule.domain(source)
        span = end - start
        offsets = (
            [start]
            if self.samples == 1 or span == 0.0
            else [start + span * step / (self.samples - 1) for step in range(self.samples)]
        )

        paths: list[TransitionPath] = []
        rejected = 0
        for offset in offsets:
            path = self.path(source, target, offset, rule=rule)
            if path is None:
                rejected += 1
            else:
                paths.append(path)
        return TransitionFamily(rule=rule, paths=tuple(paths), domain=(start, end), rejected=rejected)

    def with_samples(self, samples: int) -> "TransitionModel":
        """Copy of this model at a different sampling density - denser for manifests, sparser to draw."""
        return replace(self, samples=samples)


def transition_id(source_id: str, target_id: str, initiate_distance: float) -> str:
    """Stable id for a generated transition path.

    Encodes what produced it - source, target, and initiation arc length in
    meters - so a generated trajectory is traceable back to its rule without a
    lookup table, and regenerating the same family produces the same ids.
    """
    return f"{source_id}__{target_id}__s{initiate_distance:07.1f}"
