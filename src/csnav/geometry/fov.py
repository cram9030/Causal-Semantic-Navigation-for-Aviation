"""Sensor field-of-view -> ground-footprint geometry.

Lives beside :mod:`csnav.geometry.local_frame` because, like the ENU
conversions, this is metric geometry: it turns an angular field of view and a
height above ground into a distance on the ground, in meters. Used by the
offline manifest builder to decide how far outside the RNP tube a road could
still be visible from (integration plan §3.3, "roads/intersections that could
possibly be visible from any state within the tube"), and later by the
runtime "possible roads" step to intersect the precomputed manifest with the
FOV at a specific slice (§3.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldOfView:
    """Angular field of view of a nadir-pointing camera.

    Units: ``horizontal_deg``/``vertical_deg`` are full (not half) angles in
    degrees, across and along the sensor respectively. ``vertical_deg`` of
    ``None`` means a square field of view equal to ``horizontal_deg``.
    """

    horizontal_deg: float
    vertical_deg: float | None = None

    def __post_init__(self) -> None:
        for name, value in (("horizontal_deg", self.horizontal_deg), ("vertical_deg", self.vertical_deg)):
            if value is not None and not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be in (0, 180) degrees, got {value}")

    @property
    def effective_vertical_deg(self) -> float:
        """``vertical_deg``, or ``horizontal_deg`` when the field of view is square. Degrees."""
        return self.horizontal_deg if self.vertical_deg is None else self.vertical_deg

    def ground_half_extent(self, agl: float) -> tuple[float, float]:
        """Half-width and half-height of the nadir ground footprint, in meters.

        ``agl`` is height above *ground* level in meters (not above the WGS84
        ellipsoid - subtract terrain elevation first). Assumes flat ground
        directly below the aircraft and a straight-down camera; terrain relief
        and off-nadir pointing are not modelled here.
        """
        if agl < 0.0:
            raise ValueError(f"agl must be >= 0 meters, got {agl}")
        half_width = agl * math.tan(math.radians(self.horizontal_deg) / 2.0)
        half_height = agl * math.tan(math.radians(self.effective_vertical_deg) / 2.0)
        return half_width, half_height

    def ground_radius(self, agl: float) -> float:
        """Radius, in meters, of a ground circle containing the footprint at any heading.

        ``agl`` is height above ground level in meters. This is the
        half-diagonal of the rectangular nadir footprint, so it bounds the
        footprint for *every* aircraft heading - which is what the manifest
        builder needs, since heading within the tube is not constrained
        a priori.
        """
        half_width, half_height = self.ground_half_extent(agl)
        return math.hypot(half_width, half_height)
