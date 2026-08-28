"""``T``, ``t_p``, ``x_0`` - the candidate trajectory set and its window structure.

Implements the trajectory-side classes of `docs/INTEGRATION_PLAN.md` §7's UML
diagram (``Waypoint`` / ``Trajectory`` / ``TrajectorySet``) plus the
:class:`TrajectoryWindow` discretization §3.3 builds landmark manifests over.

**Frames.** Waypoints are stored in WGS84 (EPSG:4326). Every metric quantity
here - arc length, interpolation along a leg - is computed in a local ENU
tangent plane (:mod:`csnav.geometry.local_frame`) anchored at the trajectory's
first waypoint, then converted back to WGS84 before being returned. Per
CLAUDE.md, no distance math is done on raw lat/lon degrees.

**Arc length.** Progress along a trajectory is tracked as an explicit
arc-length value in meters (integration plan §3.4 / CLAUDE.md core decision 5),
measured as the 3D ENU path length along the waypoint polyline - it is what
decides which trajectory window's precomputed manifest applies, and it is the
state the deterministic ``Predict x(t)`` mechanism advances in Phase 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cached_property

import networkx as nx

from csnav.data.arcgis.models import Extent
from csnav.geometry.local_frame import LocalFrame, Point
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint

#: Graph node id standing for the known start state ``x_0``, so that a
#: "return to ``x_0``" transition corridor (integration plan §3.2) has
#: somewhere to point in :meth:`TrajectorySet.to_networkx`.
X0_NODE = "x0"


class TrajectoryError(ValueError):
    """Raised when a trajectory or trajectory set is structurally invalid."""


@dataclass(frozen=True)
class TrajectoryWindow:
    """One arc-length window of a trajectory - the unit a manifest is built for.

    Units: ``start_distance``/``end_distance`` in meters of arc length from the
    trajectory's first waypoint; ``start_time``/``end_time`` in seconds from
    flight-plan start. A window is half-open in arc length
    (``[start_distance, end_distance)``) except the last, which includes the
    trajectory's end.

    ``window_id`` is the key a precomputed manifest is stored and looked up
    under at runtime (integration plan §3.3), so it must be stable across
    rebuilds of the same trajectory + window length.
    """

    trajectory_id: str
    index: int
    start_distance: float
    end_distance: float
    start_time: float
    end_time: float

    @property
    def window_id(self) -> str:
        """Stable lookup key for this window's precomputed manifest."""
        return f"{self.trajectory_id}:{self.index:04d}"

    @property
    def length(self) -> float:
        """Window extent along the trajectory, in meters of arc length."""
        return self.end_distance - self.start_distance

    def contains_distance(self, distance: float) -> bool:
        """Whether an arc-length position (meters) falls in this window."""
        return self.start_distance <= distance <= self.end_distance


def _interpolate(a: float, b: float, fraction: float) -> float:
    return a + (b - a) * fraction


@dataclass(frozen=True)
class Trajectory:
    """A single candidate trajectory: an ordered polyline of 4D waypoints.

    ``id`` is the trajectory's stable identifier (used in window ids and
    manifest keys). ``role`` distinguishes ``t_p`` from other candidates and
    from transition corridors. ``connects`` is set only for
    :attr:`TrajectoryRole.TRANSITION` paths, as the ``(source_id, target_id)``
    pair of trajectory ids (or :data:`X0_NODE`) the corridor joins.
    """

    id: str
    waypoints: tuple[Waypoint, ...]
    role: TrajectoryRole = TrajectoryRole.ALTERNATE
    connects: tuple[str, str] | None = None
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise TrajectoryError(f"trajectory {self.id!r} needs at least 2 waypoints, got {len(self.waypoints)}")
        times = [wp.time for wp in self.waypoints]
        if any(b < a for a, b in zip(times, times[1:])):
            raise TrajectoryError(f"trajectory {self.id!r} has non-monotonic waypoint times: {times}")
        if self.role is TrajectoryRole.TRANSITION and self.connects is None:
            raise TrajectoryError(f"transition trajectory {self.id!r} must declare which pair it connects")

    # ----- local metric frame -------------------------------------------------

    @cached_property
    def local_frame(self) -> LocalFrame:
        """ENU tangent plane anchored at this trajectory's first waypoint (ground level).

        Anchored per trajectory rather than per flight, per integration plan
        §2/§3.2: tangent-plane distortion grows with distance from the origin.
        The origin height is 0 m so a point's ENU ``up`` equals its height
        above the WGS84 ellipsoid.
        """
        first = self.waypoints[0]
        return LocalFrame(origin_lat=first.lat, origin_lon=first.lon, origin_height=0.0)

    @cached_property
    def enu_vertices(self) -> tuple[Point, ...]:
        """Waypoints in this trajectory's local ENU frame, in meters."""
        frame = self.local_frame
        return tuple(frame.to_enu(wp.lat, wp.lon, wp.height) for wp in self.waypoints)

    @cached_property
    def cumulative_distances(self) -> tuple[float, ...]:
        """Arc length (meters, 3D ENU path length) at each waypoint, starting at 0.0."""
        distances = [0.0]
        for previous, current in zip(self.enu_vertices, self.enu_vertices[1:]):
            step = math.dist(
                (previous.east, previous.north, previous.up),
                (current.east, current.north, current.up),
            )
            distances.append(distances[-1] + step)
        return tuple(distances)

    @property
    def length(self) -> float:
        """Total arc length of the trajectory, in meters."""
        return self.cumulative_distances[-1]

    @property
    def duration(self) -> float:
        """Elapsed flight-plan time from first to last waypoint, in seconds."""
        return self.waypoints[-1].time - self.waypoints[0].time

    # ----- sampling along the path -------------------------------------------

    def _bracket(self, distance: float) -> tuple[int, float]:
        """Index of the leg containing ``distance`` (meters) and the fraction along it."""
        distances = self.cumulative_distances
        clamped = min(max(distance, 0.0), distances[-1])
        for index in range(len(distances) - 1):
            start, end = distances[index], distances[index + 1]
            if clamped <= end:
                span = end - start
                return index, 0.0 if span == 0.0 else (clamped - start) / span
        return len(distances) - 2, 1.0

    def point_at(self, distance: float) -> Waypoint:
        """Waypoint at ``distance`` meters of arc length along the trajectory, in WGS84.

        Interpolates linearly in the local ENU frame (so the interpolated point
        is metrically on the leg, not on a lat/lon-degree chord) and converts
        back to WGS84; ``time`` is interpolated linearly over the same leg, in
        seconds. ``distance`` is clamped to ``[0, length]`` - a trajectory does
        not extend past its final waypoint.
        """
        index, fraction = self._bracket(distance)
        start_enu, end_enu = self.enu_vertices[index], self.enu_vertices[index + 1]
        position = self.local_frame.to_wgs84(
            _interpolate(start_enu.east, end_enu.east, fraction),
            _interpolate(start_enu.north, end_enu.north, fraction),
            _interpolate(start_enu.up, end_enu.up, fraction),
        )
        start_wp, end_wp = self.waypoints[index], self.waypoints[index + 1]
        return Waypoint(
            lat=position.lat,
            lon=position.lon,
            height=position.height,
            time=_interpolate(start_wp.time, end_wp.time, fraction),
        )

    def distance_at_time(self, time: float) -> float:
        """Arc-length position (meters) at flight-plan ``time`` (seconds).

        Interpolates against the waypoint time schedule, clamped to the
        trajectory's own time span.
        """
        times = [wp.time for wp in self.waypoints]
        distances = self.cumulative_distances
        clamped = min(max(time, times[0]), times[-1])
        for index in range(len(times) - 1):
            start, end = times[index], times[index + 1]
            if clamped <= end:
                span = end - start
                fraction = 0.0 if span == 0.0 else (clamped - start) / span
                return _interpolate(distances[index], distances[index + 1], fraction)
        return distances[-1]

    def point_at_time(self, time: float) -> Waypoint:
        """Waypoint at flight-plan ``time`` (seconds), in WGS84."""
        return self.point_at(self.distance_at_time(time))

    def sample(self, spacing: float) -> tuple[Waypoint, ...]:
        """Waypoints resampled every ``spacing`` meters of arc length, endpoints included.

        Used for drawing smooth tube corridors and for densifying a leg before
        buffering it, where a two-vertex leg would otherwise under-represent
        the corridor's curvature on the ellipsoid.
        """
        if spacing <= 0.0:
            raise ValueError(f"spacing must be > 0 meters, got {spacing}")
        steps = max(1, math.ceil(self.length / spacing))
        return tuple(self.point_at(self.length * step / steps) for step in range(steps + 1))

    def segment(self, start_distance: float, end_distance: float) -> tuple[Waypoint, ...]:
        """Waypoints covering the arc-length span ``[start_distance, end_distance]`` (meters).

        Returns the interpolated span endpoints plus every original waypoint
        strictly between them, so a window's polyline keeps the trajectory's
        real corner geometry instead of being straightened into a chord.
        """
        if end_distance < start_distance:
            raise ValueError(f"end_distance {end_distance} is before start_distance {start_distance}")
        distances = self.cumulative_distances
        interior = [
            self.waypoints[index]
            for index, distance in enumerate(distances)
            if start_distance < distance < end_distance
        ]
        return (self.point_at(start_distance), *interior, self.point_at(end_distance))

    def windows(self, window_length: float) -> tuple[TrajectoryWindow, ...]:
        """Discretize into arc-length windows of at most ``window_length`` meters.

        The final window absorbs any remainder shorter than half a window,
        rather than emitting a sliver window whose manifest would be nearly
        identical to its neighbour's. A trajectory shorter than
        ``window_length`` yields exactly one window covering all of it.
        """
        if window_length <= 0.0:
            raise ValueError(f"window_length must be > 0 meters, got {window_length}")

        total = self.length
        count = max(1, round(total / window_length))
        edges = [total * index / count for index in range(count + 1)]

        first_time = self.waypoints[0].time
        return tuple(
            TrajectoryWindow(
                trajectory_id=self.id,
                index=index,
                start_distance=edges[index],
                end_distance=edges[index + 1],
                start_time=self.point_at(edges[index]).time if total > 0 else first_time,
                end_time=self.point_at(edges[index + 1]).time if total > 0 else first_time,
            )
            for index in range(count)
        )

    def window_for_distance(self, distance: float, window_length: float) -> TrajectoryWindow:
        """The window whose manifest applies at ``distance`` meters of arc length."""
        for window in self.windows(window_length):
            if window.contains_distance(distance):
                return window
        raise TrajectoryError(f"arc length {distance} m is outside trajectory {self.id!r} (0..{self.length} m)")

    # ----- interop ------------------------------------------------------------

    @property
    def bounds(self) -> Extent:
        """Bounding box of the waypoints, in EPSG:4326 (no tube buffer applied)."""
        lons = [wp.lon for wp in self.waypoints]
        lats = [wp.lat for wp in self.waypoints]
        return Extent(xmin=min(lons), ymin=min(lats), xmax=max(lons), ymax=max(lats), wkid=4326)

    def to_geojson_feature(self) -> dict:
        """GeoJSON ``LineString`` feature of the centerline, in EPSG:4326."""
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[wp.lon, wp.lat] for wp in self.waypoints]},
            "properties": {
                "id": self.id,
                "role": self.role.value,
                "connects": list(self.connects) if self.connects else None,
                "length_m": self.length,
                "duration_s": self.duration,
                **self.metadata,
            },
        }


@dataclass(frozen=True)
class Transition:
    """A directed transition between two candidate trajectories (or back to ``x_0``).

    ``source``/``target`` are trajectory ids or :data:`X0_NODE`. ``via`` names
    the :class:`TrajectoryRole.TRANSITION` trajectory that gives the corridor
    its geometry, or is ``None`` for a direct hand-off with no distinct
    corridor path of its own.
    """

    source: str
    target: str
    via: str | None = None


@dataclass(frozen=True)
class TrajectorySet:
    """``T`` - the candidate trajectory set, its primary ``t_p``, and the start state ``x_0``.

    Because ``T`` is known before flight, this is what makes the offline
    manifest precomputation of integration plan §3.3 possible at all. The set
    also carries the transition corridors between candidates, so the reachable
    state space it bounds includes trajectory changes and the return path to
    ``x_0``, not just the primary route.
    """

    id: str
    trajectories: tuple[Trajectory, ...]
    primary_id: str
    x0: Waypoint
    transitions: tuple[Transition, ...] = ()

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise TrajectoryError(f"trajectory set {self.id!r} is empty")
        ids = [trajectory.id for trajectory in self.trajectories]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise TrajectoryError(f"duplicate trajectory ids in {self.id!r}: {sorted(duplicates)}")
        if self.primary_id not in ids:
            raise TrajectoryError(f"primary trajectory {self.primary_id!r} is not in {sorted(ids)}")
        known = set(ids) | {X0_NODE}
        for transition in self.transitions:
            for endpoint in (transition.source, transition.target):
                if endpoint not in known:
                    raise TrajectoryError(f"transition endpoint {endpoint!r} is not a known trajectory or {X0_NODE!r}")
            if transition.via is not None and transition.via not in ids:
                raise TrajectoryError(f"transition corridor {transition.via!r} is not a known trajectory")

    def by_id(self, trajectory_id: str) -> Trajectory:
        """Look up one trajectory by id."""
        for trajectory in self.trajectories:
            if trajectory.id == trajectory_id:
                return trajectory
        raise KeyError(f"no trajectory {trajectory_id!r} in set {self.id!r}")

    @property
    def primary(self) -> Trajectory:
        """``t_p`` - the planned trajectory."""
        return self.by_id(self.primary_id)

    @property
    def candidates(self) -> tuple[Trajectory, ...]:
        """The candidate trajectories in ``T`` (everything that isn't a transition corridor)."""
        return tuple(t for t in self.trajectories if t.role is not TrajectoryRole.TRANSITION)

    @property
    def corridors(self) -> tuple[Trajectory, ...]:
        """The transition-corridor paths between candidates."""
        return tuple(t for t in self.trajectories if t.role is TrajectoryRole.TRANSITION)

    def to_networkx(self) -> nx.DiGraph:
        """The trajectory set as a directed graph: candidates as nodes, transitions as edges.

        Nodes are candidate-trajectory ids plus :data:`X0_NODE` for the known
        start state, each carrying ``role``, ``length_m``, ``duration_s`` and
        the trajectory object itself. Edges carry ``via`` (the corridor
        trajectory id, if any) and its ``length_m``.

        ``networkx.DiGraph`` per CLAUDE.md's graph convention - the same type
        the slice DAG spec uses in Phase 3, so no translation layer is needed.
        """
        graph = nx.DiGraph(id=self.id, primary=self.primary_id)
        graph.add_node(
            X0_NODE,
            role="start_state",
            lat=self.x0.lat,
            lon=self.x0.lon,
            height=self.x0.height,
            time=self.x0.time,
        )
        for trajectory in self.candidates:
            graph.add_node(
                trajectory.id,
                role=trajectory.role.value,
                length_m=trajectory.length,
                duration_s=trajectory.duration,
                trajectory=trajectory,
            )
        for transition in self.transitions:
            corridor = self.by_id(transition.via) if transition.via else None
            graph.add_edge(
                transition.source,
                transition.target,
                via=transition.via,
                length_m=corridor.length if corridor else 0.0,
                corridor=corridor,
            )
        return graph

    @property
    def bounds(self) -> Extent:
        """Bounding box of every waypoint in the set, in EPSG:4326 (no tube buffer applied)."""
        extents = [trajectory.bounds for trajectory in self.trajectories]
        return Extent(
            xmin=min(e.xmin for e in extents),
            ymin=min(e.ymin for e in extents),
            xmax=max(e.xmax for e in extents),
            ymax=max(e.ymax for e in extents),
            wkid=4326,
        )

    def to_geojson(self) -> dict:
        """The whole set as a GeoJSON ``FeatureCollection`` of centerlines, in EPSG:4326."""
        return {
            "type": "FeatureCollection",
            "features": [trajectory.to_geojson_feature() for trajectory in self.trajectories],
        }

    def with_trajectories(self, trajectories: tuple[Trajectory, ...]) -> "TrajectorySet":
        """Copy of this set with a different trajectory tuple (same primary/``x_0``/transitions)."""
        return replace(self, trajectories=trajectories)
