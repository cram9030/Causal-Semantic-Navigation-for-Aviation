"""Sensor pose, attitude margin, and camera ground reach.

Anchored on cases that can be checked without the code: a 90 deg nadir field of
view reaches exactly one AGL to each side, a nadir camera with no attitude
allowance must agree exactly with the plain FOV radius, and a pitched camera's
boresight lands at ``agl * tan(pitch)``.
"""

from __future__ import annotations

import math

import pytest

from csnav.geometry.camera import AttitudeMargin, Camera, HorizonError, SensorPose
from csnav.geometry.fov import FieldOfView

FOV = FieldOfView(horizontal_deg=60.0, vertical_deg=45.0)


# ----- sensor pose ------------------------------------------------------------


def test_default_pose_is_nadir():
    pose = SensorPose()
    assert pose.is_nadir
    assert not SensorPose(pitch_deg=15.0).is_nadir


def test_nadir_camera_reduces_to_the_plain_fov_radius():
    """The prototype flies nadir, so the camera model must not change its numbers."""
    camera = Camera(FOV)
    for agl in (100.0, 800.0, 3000.0):
        assert camera.ground_reach(agl) == pytest.approx(FOV.ground_radius(agl))


def test_nadir_corners_are_the_fov_half_extents():
    half_width, half_height = FOV.ground_half_extent(1000.0)
    corners = Camera(FOV).ground_corners(1000.0)

    assert {round(abs(forward), 6) for forward, _ in corners} == {round(half_height, 6)}
    assert {round(abs(right), 6) for _, right in corners} == {round(half_width, 6)}


def test_image_up_points_forward_for_a_nadir_mounting():
    """Both along-track corners exist fore and aft, so the footprint is not rotated 90 deg."""
    corners = Camera(FOV).ground_corners(1000.0)
    forwards = [forward for forward, _ in corners]
    assert max(forwards) > 0 and min(forwards) < 0


def test_nose_up_pitch_moves_the_whole_footprint_forward():
    """Pitching up swings a belly camera's view further ahead."""
    level = Camera(FOV).ground_corners(1000.0)
    pitched = Camera(FOV, SensorPose(pitch_deg=20.0)).ground_corners(1000.0)

    assert min(f for f, _ in pitched) > min(f for f, _ in level)
    assert max(f for f, _ in pitched) > max(f for f, _ in level)


def test_pitch_swings_the_near_edge_out_by_roughly_agl_times_tan():
    """The near edge of a nadir footprint sits at -agl*tan(v/2); pitch adds agl*tan(pitch)."""
    pitch = 20.0
    _, half_height = FOV.ground_half_extent(1000.0)
    near_edge = min(f for f, _ in Camera(FOV, SensorPose(pitch_deg=pitch)).ground_corners(1000.0))
    assert near_edge == pytest.approx(1000.0 * math.tan(math.radians(pitch)) - half_height, rel=0.15)


def test_right_wing_down_roll_moves_the_footprint_left():
    """The belly turns away from the dropped wing, so a downward camera looks left."""
    corners = Camera(FOV, SensorPose(roll_deg=20.0)).ground_corners(1000.0)
    assert sum(right for _, right in corners) / 4.0 < 0.0


def test_an_off_nadir_mounting_reaches_further_than_a_nadir_one():
    assert Camera(FOV, SensorPose(pitch_deg=25.0)).ground_reach(1000.0) > Camera(FOV).ground_reach(1000.0)


def test_a_field_of_view_that_reaches_the_horizon_is_refused():
    with pytest.raises(HorizonError, match="reaches the horizon"):
        Camera(FieldOfView(90.0), SensorPose(pitch_deg=50.0)).ground_reach(1000.0)


def test_negative_agl_is_refused():
    with pytest.raises(ValueError, match="agl must be >= 0"):
        Camera(FOV).ground_reach(-1.0)


def test_reach_scales_linearly_with_agl():
    camera = Camera(FOV, SensorPose(pitch_deg=15.0))
    assert camera.ground_reach(2000.0) == pytest.approx(2 * camera.ground_reach(1000.0))


# ----- attitude margin --------------------------------------------------------


def test_default_margin_is_zero_and_changes_nothing():
    """The first proof of concept must behave as if the margin were not there."""
    camera = Camera(FOV)
    assert camera.attitude_margin.is_zero
    assert camera.bounded_ground_reach(1000.0) == pytest.approx(camera.ground_reach(1000.0))


def test_margin_widens_the_reach():
    level = Camera(FOV).bounded_ground_reach(1000.0)
    banked = Camera(FOV, attitude_margin=AttitudeMargin(roll_deg=15.0)).bounded_ground_reach(1000.0)
    assert banked > level


def test_roll_sign_does_not_matter_to_the_margin():
    """A margin bounds the excursion either way, so its reach is symmetric in roll."""
    camera = Camera(FOV, attitude_margin=AttitudeMargin(roll_deg=25.0))
    assert camera.ground_reach(1000.0, roll_deg=25.0) == pytest.approx(
        camera.ground_reach(1000.0, roll_deg=-25.0)
    )
    assert camera.bounded_ground_reach(1000.0) == pytest.approx(camera.ground_reach(1000.0, roll_deg=25.0))


def test_margin_takes_the_worst_sign_combination():
    """A bound, not an estimate: +/-roll and +/-pitch are all evaluated."""
    camera = Camera(FOV, SensorPose(pitch_deg=10.0), AttitudeMargin(pitch_deg=15.0))
    worst = max(
        camera.ground_reach(1000.0, pitch_deg=15.0), camera.ground_reach(1000.0, pitch_deg=-15.0)
    )
    assert camera.bounded_ground_reach(1000.0) == pytest.approx(worst)


def test_maneuver_values_apply_only_near_a_waypoint():
    margin = AttitudeMargin(roll_deg=5.0, maneuver_roll_deg=30.0, maneuver_radius=200.0)
    assert margin.at(50.0) == (30.0, 0.0)
    assert margin.at(500.0) == (5.0, 0.0)


def test_maneuver_values_never_shrink_the_cruise_allowance():
    margin = AttitudeMargin(roll_deg=20.0, maneuver_roll_deg=5.0, maneuver_radius=200.0)
    assert margin.at(10.0) == (20.0, 0.0)


def test_reach_grows_near_a_waypoint_when_a_maneuver_margin_is_set():
    """Approaching and leaving a waypoint is where the vehicle's pose actually swings."""
    camera = Camera(
        FOV, attitude_margin=AttitudeMargin(roll_deg=5.0, maneuver_roll_deg=30.0, maneuver_radius=250.0)
    )
    near = camera.bounded_ground_reach(1000.0, distance_to_waypoint=50.0)
    far = camera.bounded_ground_reach(1000.0, distance_to_waypoint=2000.0)
    assert near > far


def test_reach_without_a_waypoint_distance_uses_the_cruise_margin():
    camera = Camera(FOV, attitude_margin=AttitudeMargin(roll_deg=5.0, maneuver_roll_deg=30.0, maneuver_radius=250.0))
    assert camera.bounded_ground_reach(1000.0) == pytest.approx(
        camera.bounded_ground_reach(1000.0, distance_to_waypoint=9999.0)
    )


@pytest.mark.parametrize("field", ["roll_deg", "pitch_deg", "maneuver_roll_deg", "maneuver_pitch_deg"])
def test_margin_rejects_angles_at_or_past_vertical(field):
    with pytest.raises(ValueError, match="must be in .-90, 90. degrees"):
        AttitudeMargin(**{field: 90.0})


def test_margin_rejects_a_negative_maneuver_radius():
    with pytest.raises(ValueError, match="maneuver_radius must be >= 0"):
        AttitudeMargin(maneuver_radius=-1.0)
