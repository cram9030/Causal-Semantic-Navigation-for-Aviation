"""Sensor pose, attitude margin, and the ground reach of a camera.

:mod:`csnav.geometry.fov` answers "how wide is the cone"; this module answers
"where does that cone land on the ground", which needs two more things the
integration plan's FOV/occlusion work (§2, §3.4) depends on:

* **Sensor pose** - where the camera is bolted relative to the body frame. The
  first prototype flies a nadir (straight-down) camera, i.e. an all-zero
  :class:`SensorPose`, but the field and the math have to exist so that a
  forward- or side-looking mounting is a config change rather than a rewrite.
* **Attitude margin** - a real aircraft is not level. Bank and pitch excursions
  swing the footprint well off nadir, and they are largest around waypoints,
  where the turns are. :class:`AttitudeMargin` carries that allowance,
  including a larger value that applies only within a configured distance of a
  waypoint. It defaults to zero, so the first proof of concept behaves exactly
  as if it weren't there.

**Frames.** Body frame is the aircraft convention: ``X`` forward, ``Y`` right,
``Z`` down. Sensor frame is the camera convention: ``Z`` along the boresight,
``X`` toward image-right, ``Y`` toward image-down. With an all-zero
:class:`SensorPose` the boresight points along body ``+Z`` (straight down) and
image-up points forward. Everything returned is in meters on flat ground
directly below the aircraft; no terrain relief and no occlusion (integration
plan §2 wants those eventually - they belong with the 3DEP surface, not here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from csnav.geometry.fov import FieldOfView

#: Rotation taking sensor axes to body axes for a nadir mounting: image-right
#: becomes body-right, image-down becomes body-aft, boresight becomes body-down.
_SENSOR_TO_BODY_NADIR = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

#: A ray whose downward component is at or below this is treated as pointing at
#: or above the horizon, where the flat-ground intersection diverges.
_HORIZON_EPSILON = 1e-6


def _rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Body-frame rotation ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``, angles in degrees.

    Right-handed about the body axes (X forward, Y right, Z down), which is the
    usual aircraft convention: positive pitch is nose-up, positive roll is
    right-wing-down.

    Note where that leaves a downward-looking boresight, because the sign is
    easy to get backwards. Nose-up pitch rotates body ``+Z`` (down) toward
    ``+X``, so the camera looks **further ahead** - as you would expect.
    Right-wing-down roll rotates ``+Z`` toward ``-Y``: the belly turns away from
    the dropped wing, so the camera looks **to the left**. Both follow from the
    right-hand rule, and both are what a real aircraft does.
    """
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


class HorizonError(ValueError):
    """Raised when a field of view reaches the horizon, so its ground footprint is unbounded."""


@dataclass(frozen=True)
class SensorPose:
    """Camera mounting relative to the body frame.

    Units: degrees. All-zero is nadir - the boresight points straight down and
    image-up points forward, which is what the first prototype flies.
    ``pitch`` is nose-up positive, which swings the boresight forward;
    ``roll`` is right-wing-down positive, which swings the boresight to the
    left (the belly turns away from the dropped wing - see :func:`_rotation`);
    ``yaw`` spins the image about the boresight. ``lever_arm`` is the sensor's offset from the body origin in
    meters as ``(forward, right, down)`` - carried so a future AGL correction
    can use it, and ignored by the flat-ground footprint math here, where a
    metre of lever arm is negligible next to hundreds of metres of altitude.
    """

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    lever_arm: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def is_nadir(self) -> bool:
        """Whether this is the straight-down mounting (no rotation off the body Z axis)."""
        return self.roll_deg == 0.0 and self.pitch_deg == 0.0

    def rotation(self) -> np.ndarray:
        """Sensor-to-body rotation matrix for this mounting."""
        return _rotation(self.roll_deg, self.pitch_deg, self.yaw_deg) @ _SENSOR_TO_BODY_NADIR


@dataclass(frozen=True)
class AttitudeMargin:
    """Bank/pitch excursion allowed for when sizing a ground footprint.

    Units: degrees, and meters for ``maneuver_radius``. The footprint is
    computed at the worst combination of ``+/-roll`` and ``+/-pitch``, so this
    is a bound on how far the footprint can swing rather than an estimate of
    where it will be.

    ``maneuver_roll``/``maneuver_pitch`` replace the cruise values within
    ``maneuver_radius`` meters of arc length of a waypoint, where the turns
    happen and the excursions are largest. All four default to zero: the first
    proof of concept flies the footprint as if the aircraft were level, and
    turning the allowance on later is a config change.
    """

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    maneuver_roll_deg: float = 0.0
    maneuver_pitch_deg: float = 0.0
    maneuver_radius: float = 0.0

    def __post_init__(self) -> None:
        for name in ("roll_deg", "pitch_deg", "maneuver_roll_deg", "maneuver_pitch_deg"):
            value = getattr(self, name)
            if not -90.0 < value < 90.0:
                raise ValueError(f"{name} must be in (-90, 90) degrees, got {value}")
        if self.maneuver_radius < 0.0:
            raise ValueError(f"maneuver_radius must be >= 0 meters, got {self.maneuver_radius}")

    @property
    def is_zero(self) -> bool:
        """Whether this margin allows no excursion at all (the default)."""
        return (self.roll_deg, self.pitch_deg, self.maneuver_roll_deg, self.maneuver_pitch_deg) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def at(self, distance_to_waypoint: float) -> tuple[float, float]:
        """Roll/pitch excursion in degrees at ``distance_to_waypoint`` meters from the nearest waypoint.

        Inside ``maneuver_radius`` the maneuver values apply (or the cruise
        values, whichever is larger, so a smaller maneuver figure can never
        shrink the allowance); outside it, the cruise values do.
        """
        if distance_to_waypoint <= self.maneuver_radius:
            return (
                max(self.roll_deg, self.maneuver_roll_deg),
                max(self.pitch_deg, self.maneuver_pitch_deg),
            )
        return self.roll_deg, self.pitch_deg


#: The attitude perturbations evaluated when bounding a footprint: every
#: sign combination of the margin's roll and pitch, plus level.
_MARGIN_SIGNS = ((0, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True)
class Camera:
    """A field of view, where it points, and how far off nadir it may swing.

    This is what the trajectory/manifest layers hold instead of a bare
    :class:`~csnav.geometry.fov.FieldOfView`: the ground area a window has to
    account for depends on the mounting and on the attitude allowance, not only
    on the cone angle. With a nadir :class:`SensorPose` and a zero
    :class:`AttitudeMargin`, :meth:`ground_reach` reduces exactly to
    :meth:`csnav.geometry.fov.FieldOfView.ground_radius`.
    """

    field_of_view: FieldOfView
    pose: SensorPose = field(default_factory=SensorPose)
    attitude_margin: AttitudeMargin = field(default_factory=AttitudeMargin)

    def _corner_rays(self) -> np.ndarray:
        """Unit-ish direction vectors of the FOV's four corners, in the sensor frame."""
        half_x = math.tan(math.radians(self.field_of_view.horizontal_deg) / 2.0)
        half_y = math.tan(math.radians(self.field_of_view.effective_vertical_deg) / 2.0)
        return np.array(
            [
                [sx * half_x, sy * half_y, 1.0]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )

    def ground_corners(
        self, agl: float, roll_deg: float = 0.0, pitch_deg: float = 0.0
    ) -> tuple[tuple[float, float], ...]:
        """Ground positions of the FOV corners, as ``(forward, right)`` meters from the nadir point.

        ``agl`` is height above ground in meters. ``roll_deg``/``pitch_deg`` are
        the aircraft's attitude at that instant (not the mounting - that is
        :attr:`pose`). Assumes flat ground below the aircraft.

        Raises :class:`HorizonError` if any corner ray points at or above the
        horizon, where a flat-ground footprint is unbounded - that is a real
        configuration error (too much pitch for the cone angle), not something
        to clamp silently.
        """
        if agl < 0.0:
            raise ValueError(f"agl must be >= 0 meters, got {agl}")

        rotation = _rotation(roll_deg, pitch_deg, 0.0) @ self.pose.rotation()
        directions = self._corner_rays() @ rotation.T

        down = directions[:, 2]
        if float(down.min()) <= _HORIZON_EPSILON:
            raise HorizonError(
                f"field of view reaches the horizon at pose {self.pose} with attitude "
                f"roll={roll_deg}, pitch={pitch_deg}: the flat-ground footprint is unbounded"
            )

        scale = agl / down
        return tuple((float(directions[i, 0] * scale[i]), float(directions[i, 1] * scale[i])) for i in range(4))

    def ground_reach(self, agl: float, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> float:
        """Farthest ground distance from the nadir point the camera can see, in meters.

        A heading-free circular bound: it is the maximum over the FOV corners of
        their horizontal distance from the point directly below the aircraft, so
        it holds whatever the aircraft's heading is. That is what the manifest
        and tile coverage need, since heading inside the tube is not constrained
        a priori.
        """
        corners = self.ground_corners(agl, roll_deg=roll_deg, pitch_deg=pitch_deg)
        return max(math.hypot(forward, right) for forward, right in corners)

    def bounded_ground_reach(self, agl: float, distance_to_waypoint: float = math.inf) -> float:
        """:meth:`ground_reach` maximized over the attitude margin at that point on the route.

        ``distance_to_waypoint`` is arc-length distance in meters to the nearest
        waypoint, which selects the margin's cruise or maneuver values. The
        worst of every ``+/-roll``, ``+/-pitch`` combination is taken, so the
        result bounds the footprint rather than estimating it.
        """
        roll_margin, pitch_margin = self.attitude_margin.at(distance_to_waypoint)
        return max(
            self.ground_reach(agl, roll_deg=roll_margin * sign_roll, pitch_deg=pitch_margin * sign_pitch)
            for sign_roll, sign_pitch in _MARGIN_SIGNS
        )
