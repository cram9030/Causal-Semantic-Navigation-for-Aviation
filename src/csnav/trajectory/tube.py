"""RNP-style containment tube around a trajectory.

Integration plan §3.2: the aircraft is assumed to stay within a tube of a
given radius around whichever trajectory it is currently flying (or within a
transition corridor), which is what bounds the reachable state space before
flight and makes offline manifest precomputation possible.

**The radius is an input, never a constant here.** :class:`TubeModel` takes it
in its constructor with no default, and nothing in this module derives or
adjusts it from landmark geometry - per CLAUDE.md core decision 4 it is a
swept experimental parameter supplied from trajectory/CONOPS configuration.

**Frames.** All buffering and cross-track math happens in the trajectory's
local ENU tangent plane, in meters; every polygon/extent returned is
converted back to WGS84 (EPSG:4326) before it leaves this module.

**Containment is lateral.** Following RNP practice, :meth:`TubeModel.contains`
tests horizontal cross-track distance from the trajectory's ground track. A
vertical containment bound is deliberately not modelled at this stage (§8
tracks the tube parameterization as still open).
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import LineString, Point as ShapelyPoint, Polygon
from shapely.ops import unary_union

from csnav.data.arcgis.models import Extent
from csnav.geometry import shapes
from csnav.trajectory.trajectory import Trajectory, TrajectoryWindow
from csnav.trajectory.waypoints import Waypoint

#: Default arc-length spacing, in meters, used to densify a centerline before
#: buffering it. This is a polygon-fidelity knob (how finely a curved corridor
#: boundary is sampled), not a tube parameter - it never affects the radius.
DEFAULT_DENSIFY_SPACING = 100.0

#: Number of segments shapely uses per quarter circle when rounding buffer
#: caps/joins. Also fidelity only.
DEFAULT_QUAD_SEGMENTS = 16


def _ground_track_enu(trajectory: Trajectory, waypoints: tuple[Waypoint, ...]) -> LineString:
    """2D ENU ground track (meters, height dropped) of ``waypoints`` in ``trajectory``'s frame."""
    frame = trajectory.local_frame
    coords = [(point.east, point.north) for point in (frame.to_enu(wp.lat, wp.lon) for wp in waypoints)]
    return LineString(coords)


def polygon_bounds(polygon: Polygon) -> Extent:
    """Bounding box of a WGS84 (lon, lat) polygon, as an EPSG:4326 :class:`Extent`."""
    xmin, ymin, xmax, ymax = polygon.bounds
    return Extent(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, wkid=4326)


@dataclass(frozen=True)
class TubeModel:
    """A constant-radius RNP-style containment tube.

    ``radius`` is the lateral containment radius in **meters**, passed in from
    trajectory/CONOPS configuration - there is intentionally no default, so a
    radius can never be silently inherited from this module. ``label``
    optionally records which CONOPS case the radius came from, purely for
    provenance in manifests and plots.
    """

    radius: float
    label: str | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"tube radius must be > 0 meters, got {self.radius}")

    # ----- containment --------------------------------------------------------

    def cross_track_distance(self, state: Waypoint, trajectory: Trajectory) -> float:
        """Horizontal distance, in meters, from ``state`` to ``trajectory``'s ground track.

        Computed in ``trajectory``'s local ENU frame (CLAUDE.md: never on raw
        lat/lon degrees). ``state``'s height is ignored - this is a lateral
        measure, matching :meth:`contains`.
        """
        track = _ground_track_enu(trajectory, trajectory.waypoints)
        position = trajectory.local_frame.to_enu(state.lat, state.lon)
        return track.distance(ShapelyPoint(position.east, position.north))

    def contains(self, state: Waypoint, trajectory: Trajectory) -> bool:
        """Whether ``state`` lies within this tube's lateral radius of ``trajectory``."""
        return self.cross_track_distance(state, trajectory) <= self.radius

    # ----- corridor geometry --------------------------------------------------

    def corridor(
        self,
        trajectory: Trajectory,
        window: TrajectoryWindow | None = None,
        extra_buffer: float = 0.0,
        densify_spacing: float = DEFAULT_DENSIFY_SPACING,
    ) -> Polygon:
        """The tube's ground footprint as a WGS84 (lon, lat) polygon.

        ``window`` restricts the footprint to one trajectory window's
        arc-length span (integration plan §3.3 builds one manifest per
        window); ``None`` covers the whole trajectory. ``extra_buffer`` is an
        additional radial allowance in meters - the manifest builder passes
        the FOV ground radius here so the footprint covers everything
        *visible* from inside the tube, not just the tube itself. It is added
        to, never folded into, :attr:`radius`.
        """
        if extra_buffer < 0.0:
            raise ValueError(f"extra_buffer must be >= 0 meters, got {extra_buffer}")

        if window is None:
            waypoints = trajectory.sample(densify_spacing)
        else:
            span = trajectory.segment(window.start_distance, window.end_distance)
            waypoints = _densify(trajectory, span, window, densify_spacing)

        track = _ground_track_enu(trajectory, waypoints)
        buffered = track.buffer(
            self.radius + extra_buffer,
            quad_segs=DEFAULT_QUAD_SEGMENTS,
            cap_style="round",
            join_style="round",
        )
        return shapes.to_wgs84(buffered, trajectory.local_frame)

    def envelope(
        self,
        trajectory: Trajectory,
        window: TrajectoryWindow | None = None,
        extra_buffer: float = 0.0,
    ) -> Extent:
        """Bounding box of :meth:`corridor`, in EPSG:4326.

        This is the "maximal spatial envelope" of integration plan §3.3 step 1
        - the box the offline CSJ Streets query is issued against.
        """
        return polygon_bounds(self.corridor(trajectory, window=window, extra_buffer=extra_buffer))


def _densify(
    trajectory: Trajectory,
    span: tuple[Waypoint, ...],
    window: TrajectoryWindow,
    spacing: float,
) -> tuple[Waypoint, ...]:
    """Resample a window's span every ``spacing`` meters of arc length, endpoints kept."""
    if spacing <= 0.0:
        raise ValueError(f"densify spacing must be > 0 meters, got {spacing}")
    length = window.end_distance - window.start_distance
    if length <= spacing:
        return span
    steps = max(1, int(length // spacing))
    return tuple(
        trajectory.point_at(window.start_distance + length * step / steps) for step in range(steps + 1)
    )


def union_corridor(
    pairs: list[tuple[Trajectory, TubeModel]],
    extra_buffer: float = 0.0,
) -> Polygon:
    """Union of several trajectories' tube footprints, as one WGS84 (lon, lat) geometry.

    Used to scope work that spans the whole trajectory set - e.g. which
    imagery tiles the ground-truth builder needs labels for across all of
    ``T`` - rather than one trajectory at a time. ``extra_buffer`` is in
    meters and applies to every corridor.
    """
    return unary_union([tube.corridor(trajectory, extra_buffer=extra_buffer) for trajectory, tube in pairs])
