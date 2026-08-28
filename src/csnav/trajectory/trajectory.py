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

from shapely.geometry import LineString, Point as ShapelyPoint

from csnav.data.arcgis.models import Extent
from csnav.geometry.local_frame import LocalFrame, Point
from csnav.trajectory.waypoints import TrajectoryRole, Waypoint

#: Graph node id standing for the known start state ``x_0``, so that a
#: "return to ``x_0``" transition corridor (integration plan §3.2) has
#: somewhere to point in :meth:`TrajectorySet.to_networkx`.
X0_NODE = "x0"


class TrajectoryError(ValueError):
    """Raised when a trajectory or trajectory set is structurally invalid."""


class TransitionError(ValueError):
    """Raised when a transition rule is malformed or cannot produce a path."""


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

    def speed_at(self, distance: float) -> float:
        """Ground speed implied by the flight plan at ``distance`` meters of arc length, in m/s.

        Taken from the leg's own length and scheduled duration, so it follows
        whatever the plan says rather than assuming a constant cruise. Returns
        0.0 for a leg with no scheduled duration; callers that need a positive
        speed (the transition model, for instance) supply their own fallback.
        """
        index, _ = self._bracket(distance)
        duration = self.waypoints[index + 1].time - self.waypoints[index].time
        if duration <= 0.0:
            return 0.0
        leg = self.cumulative_distances[index + 1] - self.cumulative_distances[index]
        return leg / duration

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

    # ----- direction, projection, and waypoint proximity ----------------------

    @cached_property
    def ground_track_enu(self) -> LineString:
        """The waypoint polyline projected to 2D in this trajectory's ENU frame, in meters.

        Height is dropped: this is the ground track, which is what lateral
        containment, cross-track distance, and cross-trajectory projection are
        all defined against.
        """
        return LineString([(point.east, point.north) for point in self.enu_vertices])

    @cached_property
    def ground_cumulative_distances(self) -> tuple[float, ...]:
        """Horizontal arc length (meters) at each waypoint - the 2D counterpart of :attr:`cumulative_distances`."""
        distances = [0.0]
        for previous, current in zip(self.enu_vertices, self.enu_vertices[1:]):
            distances.append(
                distances[-1]
                + math.dist((previous.east, previous.north), (current.east, current.north))
            )
        return tuple(distances)

    def project(self, lat: float, lon: float) -> float:
        """Arc length (meters) of the point on this trajectory's ground track nearest ``(lat, lon)``.

        The input is WGS84 degrees; the answer is in this trajectory's own
        arc-length parameterization, i.e. directly comparable with
        :meth:`point_at` and :meth:`windows`. Used to decide where an aircraft
        currently on one trajectory would rejoin another - the arrival rule for
        a transition (see :mod:`csnav.trajectory.transition`).

        The nearest point is found on the *ground* track (2D), then mapped back
        to the 3D arc length, so a climbing leg's arc length is not distorted by
        the projection being horizontal.
        """
        position = self.local_frame.to_enu(lat, lon)
        ground_distance = self.ground_track_enu.project(ShapelyPoint(position.east, position.north))
        return self._ground_to_arc_length(ground_distance)

    def _ground_to_arc_length(self, ground_distance: float) -> float:
        """Map a horizontal arc length (meters) to this trajectory's 3D arc length."""
        ground = self.ground_cumulative_distances
        clamped = min(max(ground_distance, 0.0), ground[-1])
        for index in range(len(ground) - 1):
            start, end = ground[index], ground[index + 1]
            if clamped <= end:
                span = end - start
                fraction = 0.0 if span == 0.0 else (clamped - start) / span
                return _interpolate(
                    self.cumulative_distances[index], self.cumulative_distances[index + 1], fraction
                )
        return self.length

    def tangent_at(self, distance: float) -> tuple[float, float, float]:
        """Unit direction of travel at ``distance`` meters of arc length, in local ENU.

        Returns ``(east, north, up)`` components of a unit vector in this
        trajectory's own frame. At a waypoint the *outgoing* leg's direction is
        returned, so the tangent is well defined at a corner.
        """
        index, _ = self._bracket(distance)
        start, end = self.enu_vertices[index], self.enu_vertices[index + 1]
        vector = (end.east - start.east, end.north - start.north, end.up - start.up)
        norm = math.dist((0.0, 0.0, 0.0), vector)
        if norm == 0.0:
            return (1.0, 0.0, 0.0)
        return (vector[0] / norm, vector[1] / norm, vector[2] / norm)

    def heading_at(self, distance: float) -> float:
        """Ground-track heading at ``distance`` meters of arc length, in degrees clockwise from north."""
        east, north, _ = self.tangent_at(distance)
        return math.degrees(math.atan2(east, north)) % 360.0

    def distance_to_nearest_waypoint(self, distance: float) -> float:
        """Arc-length distance, in meters, from ``distance`` to the nearest waypoint.

        Feeds the attitude margin: excursions are largest around waypoints,
        where the turns are (see
        :class:`csnav.geometry.camera.AttitudeMargin`).
        """
        return min(abs(distance - waypoint) for waypoint in self.cumulative_distances)

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
class TransitionRule:
    """A permitted hand-off from one trajectory to another - the edge of the trajectory graph.

    ``source`` and ``target`` are trajectory ids (or
    :data:`X0_NODE` for the initial entry into a
    route, which has no generated geometry - the aircraft simply starts there).

    ``initiate_from``/``initiate_to`` bound where along the source a transition
    may begin, in meters of arc length; ``None`` means the source's full extent.
    Narrow them for a target that only becomes reachable past some point -
    a near-orthogonal alternate valid only after a given waypoint, say.

    ``max_turn_deg`` and ``tangent_gain`` override the
    :class:`csnav.trajectory.transition.TransitionModel`'s defaults for this edge alone.
    """

    source: str
    target: str
    initiate_from: float | None = None
    initiate_to: float | None = None
    max_turn_deg: float | None = None
    tangent_gain: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise TransitionError(f"transition {self.source!r} -> {self.target!r} is a self-loop")
        if (
            self.initiate_from is not None
            and self.initiate_to is not None
            and self.initiate_to < self.initiate_from
        ):
            raise TransitionError(
                f"transition {self.source!r} -> {self.target!r} has initiate_to "
                f"({self.initiate_to}) before initiate_from ({self.initiate_from})"
            )

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.target

    def domain(self, source: Trajectory) -> tuple[float, float]:
        """Arc-length window on ``source`` where this transition may initiate, in meters."""
        start = 0.0 if self.initiate_from is None else max(0.0, self.initiate_from)
        end = source.length if self.initiate_to is None else min(source.length, self.initiate_to)
        if end < start:
            raise TransitionError(
                f"transition {self.source!r} -> {self.target!r} has an empty initiation domain on a "
                f"{source.length:.0f} m trajectory"
            )
        return start, end


@dataclass(frozen=True)
class TrajectorySet:
    """``T`` - the candidate trajectory set, its primary ``t_p``, and the start state ``x_0``.

    Because ``T`` is known before flight, this is what makes the offline
    manifest precomputation of integration plan §3.3 possible at all.

    Transitions between candidates are **rules, not routes**: a
    :class:`TransitionRule` says a hand-off is permitted and bounds where it may
    begin, and the geometry is generated on demand by
    :class:`csnav.trajectory.transition.TransitionModel`. A transition is not
    known before flight - only that one could start anywhere along the route
    being flown - so authored corridor geometry would be claiming knowledge the
    flight plan does not have. ``trajectories`` therefore holds only candidate
    routes; a generated transition carries
    :attr:`~csnav.trajectory.waypoints.TrajectoryRole.TRANSITION` and lives in a
    :class:`~csnav.trajectory.transition.TransitionFamily`.

    Because the rules form a graph, a *route* through the flight - "fly ``t_p``,
    divert to ``t_alt_north``, then take the northern return" - is a path
    through it, and :meth:`route_paths` enumerates them. Composite routes like
    that need no separate declaration.
    """

    id: str
    trajectories: tuple[Trajectory, ...]
    primary_id: str
    x0: Waypoint
    transitions: tuple[TransitionRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise TrajectoryError(f"trajectory set {self.id!r} is empty")
        ids = [trajectory.id for trajectory in self.trajectories]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise TrajectoryError(f"duplicate trajectory ids in {self.id!r}: {sorted(duplicates)}")
        if self.primary_id not in ids:
            raise TrajectoryError(f"primary trajectory {self.primary_id!r} is not in {sorted(ids)}")

        authored = [t.id for t in self.trajectories if t.role is TrajectoryRole.TRANSITION]
        if authored:
            raise TrajectoryError(
                f"trajectory set {self.id!r} contains authored transition trajectories {sorted(authored)}; "
                "transitions are not known before flight - declare a TransitionRule instead and let "
                "TransitionModel generate the family"
            )

        known = set(ids) | {X0_NODE}
        for rule in self.transitions:
            for endpoint in (rule.source, rule.target):
                if endpoint not in known:
                    raise TrajectoryError(f"transition endpoint {endpoint!r} is not a known trajectory or {X0_NODE!r}")
            if rule.target == X0_NODE:
                raise TrajectoryError(
                    f"transition {rule.source!r} -> {X0_NODE!r} has no target route; model a return as its own "
                    "candidate trajectory ending at x_0 and transition into that"
                )

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
        """The candidate trajectories in ``T``. Every member is a candidate; see the class docstring."""
        return self.trajectories

    def rules_from(self, trajectory_id: str) -> tuple[TransitionRule, ...]:
        """Transition rules whose source is ``trajectory_id`` (or :data:`X0_NODE`)."""
        return tuple(rule for rule in self.transitions if rule.source == trajectory_id)

    def entry_ids(self) -> tuple[str, ...]:
        """Trajectories that may be flown straight from ``x_0``.

        The targets of rules sourced at :data:`X0_NODE`; if no rule mentions
        ``x_0`` at all, every candidate is treated as an entry, since the set
        would otherwise have no way in.
        """
        entries = tuple(rule.target for rule in self.transitions if rule.source == X0_NODE)
        return entries or tuple(trajectory.id for trajectory in self.trajectories)

    def to_networkx(self) -> nx.DiGraph:
        """The trajectory set as a directed graph: candidates as nodes, transition rules as edges.

        Nodes are candidate-trajectory ids plus :data:`X0_NODE` for the known
        start state, each carrying ``role``, ``length_m``, ``duration_s`` and
        the trajectory object. Edges carry the :class:`TransitionRule` itself
        and its declared initiation window, not any generated geometry -
        geometry belongs to a :class:`~csnav.trajectory.transition.TransitionFamily`,
        which depends on the transition model in force.

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
        for trajectory in self.trajectories:
            graph.add_node(
                trajectory.id,
                role=trajectory.role.value,
                length_m=trajectory.length,
                duration_s=trajectory.duration,
                trajectory=trajectory,
            )
        for rule in self.transitions:
            source = None if rule.source == X0_NODE else self.by_id(rule.source)
            start, end = rule.domain(source) if source is not None else (0.0, 0.0)
            graph.add_edge(
                rule.source,
                rule.target,
                rule=rule,
                initiate_from_m=start,
                initiate_to_m=end,
                is_entry=rule.source == X0_NODE,
            )
        return graph

    def terminal_ids(self) -> tuple[str, ...]:
        """Trajectories with no onward transition - where a flight ends."""
        graph = self.to_networkx()
        return tuple(
            node
            for node in graph.nodes
            if node != X0_NODE and graph.out_degree(node) == 0
        )

    def route_paths(self) -> tuple[tuple[str, ...], ...]:
        """Every simple route through the set, as tuples of trajectory ids from ``x_0`` onward.

        A route is a path in the transition graph: ``t_p`` flown to its end is
        one, ``t_p -> t_alt_north -> the northern return`` is another. These
        compositions are what the set actually permits, and none of them has to
        be declared separately - they fall out of the rules. Note that each is a
        *family* of real flights, since where each hand-off begins is continuous
        (see :mod:`csnav.trajectory.transition`).
        """
        graph = self.to_networkx()
        routes: list[tuple[str, ...]] = []
        terminals = set(self.terminal_ids())
        for entry in self.entry_ids():
            if entry not in graph:
                continue
            reachable_terminals = terminals or {entry}
            for terminal in sorted(reachable_terminals):
                for path in nx.all_simple_paths(graph, entry, terminal) if entry != terminal else [[entry]]:
                    routes.append(tuple(path))
        return tuple(sorted(set(routes)))

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
