"""FOV -> ground footprint geometry.

Checked against closed-form trigonometry rather than a round trip: a 90 deg
field of view puts the footprint edge exactly one AGL out, which is the case
you can verify without a calculator.
"""

from __future__ import annotations

import math

import pytest

from csnav.geometry.fov import FieldOfView


def test_ninety_degree_fov_reaches_one_agl_to_each_side():
    fov = FieldOfView(horizontal_deg=90.0, vertical_deg=90.0)
    half_width, half_height = fov.ground_half_extent(1000.0)
    assert half_width == pytest.approx(1000.0)
    assert half_height == pytest.approx(1000.0)


def test_ground_radius_is_the_footprint_half_diagonal():
    fov = FieldOfView(horizontal_deg=60.0, vertical_deg=45.0)
    half_width, half_height = fov.ground_half_extent(800.0)
    assert fov.ground_radius(800.0) == pytest.approx(math.hypot(half_width, half_height))
    # The circumscribing circle must reach past either side on its own.
    assert fov.ground_radius(800.0) > max(half_width, half_height)


def test_ground_radius_scales_linearly_with_agl():
    fov = FieldOfView(horizontal_deg=60.0)
    assert fov.ground_radius(2000.0) == pytest.approx(2 * fov.ground_radius(1000.0))
    assert fov.ground_radius(0.0) == 0.0


def test_square_fov_when_vertical_is_unset():
    assert FieldOfView(horizontal_deg=50.0).effective_vertical_deg == 50.0
    half_width, half_height = FieldOfView(horizontal_deg=50.0).ground_half_extent(500.0)
    assert half_width == pytest.approx(half_height)


@pytest.mark.parametrize("angle", [0.0, -10.0, 180.0, 200.0])
def test_rejects_angles_outside_the_open_zero_to_180_range(angle):
    with pytest.raises(ValueError, match="must be in .0, 180. degrees"):
        FieldOfView(horizontal_deg=angle)


def test_rejects_negative_agl():
    with pytest.raises(ValueError, match="agl must be >= 0"):
        FieldOfView(horizontal_deg=60.0).ground_radius(-1.0)
