"""Waypoint and trajectory-role primitives for the candidate trajectory set ``T``.

Per `docs/INTEGRATION_PLAN.md` §3.2, ``T = {t_1, ..., t_n}`` is a set of
candidate trajectories known before flight, each a sequence of 4D waypoints.
This module holds the smallest pieces of that structure; :mod:`csnav.trajectory.trajectory`
assembles them into :class:`~csnav.trajectory.trajectory.Trajectory` and
:class:`~csnav.trajectory.trajectory.TrajectorySet`.

All positions here are WGS84 (EPSG:4326) - metric operations on them must go
through :mod:`csnav.geometry.local_frame` first (CLAUDE.md, coordinates rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from csnav.geometry.local_frame import LatLon


class TrajectoryRole(str, Enum):
    """What a trajectory is *for* within a :class:`TrajectorySet`.

    ``PRIMARY`` is ``t_p`` (the planned trajectory), ``ALTERNATE`` is any other
    candidate in ``T``, and ``TRANSITION`` is a corridor path flown between two
    candidates (or back to ``x_0``) rather than a candidate in its own right.
    The distinction matters because §8 of the integration plan leaves open
    whether a transition corridor shares the primary trajectory's tube radius
    or gets its own - see :class:`csnav.trajectory.config.ConopsConfig`.
    """

    PRIMARY = "primary"
    ALTERNATE = "alternate"
    TRANSITION = "transition"


@dataclass(frozen=True)
class Waypoint:
    """One 4D waypoint on a trajectory, in WGS84.

    Units: ``lat``/``lon`` in decimal degrees (EPSG:4326); ``height`` in meters
    above the WGS84 ellipsoid (*not* AGL - AGL is derived at use time by
    subtracting ground elevation, see `docs/INTEGRATION_PLAN.md` §2);
    ``time`` in seconds from the start of the flight plan.
    """

    lat: float
    lon: float
    height: float = 0.0
    time: float = 0.0

    @property
    def position(self) -> LatLon:
        """This waypoint's position as a :class:`~csnav.geometry.local_frame.LatLon` (degrees, meters)."""
        return LatLon(lat=self.lat, lon=self.lon, height=self.height)

    def with_time(self, time: float) -> "Waypoint":
        """Copy of this waypoint at a different time (seconds from flight-plan start)."""
        return Waypoint(lat=self.lat, lon=self.lon, height=self.height, time=time)
