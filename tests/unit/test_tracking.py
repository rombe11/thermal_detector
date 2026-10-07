from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from thermal_board.geometry.planar_homography import translation_homography
from thermal_board.tracking.corner_smoothing import ExponentialCornerSmoother
from thermal_board.tracking.drift_monitor import DriftMonitor
from thermal_board.tracking.phase_correlation_motion import PhaseCorrelationMotionEstimator

if TYPE_CHECKING:
    from tests.support import RenderedScene

BOARD_CENTRE = np.array([100.0, 80.0])


def similarity(angle_deg: float, scale: float) -> np.ndarray:
    cosine, sine = np.cos(np.radians(angle_deg)) * scale, np.sin(np.radians(angle_deg)) * scale
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def shifted(image: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, transform, image.shape[::-1], borderMode=cv2.BORDER_REFLECT)


def test_when_first_frame_arrives_then_no_motion_is_reported(small_scene: RenderedScene) -> None:
    shift = PhaseCorrelationMotionEstimator().estimate(small_scene.normalized_image)
    assert (shift.shift_x_px, shift.shift_y_px, shift.confidence) == (0.0, 0.0, 0.0)


def test_when_frame_shifts_then_the_shift_is_measured(small_scene: RenderedScene) -> None:
    estimator = PhaseCorrelationMotionEstimator()
    estimator.estimate(small_scene.normalized_image)
    shift = estimator.estimate(shifted(small_scene.normalized_image, 2.0, -1.0))
    assert (shift.shift_x_px, shift.shift_y_px) == pytest.approx((2.0, -1.0), abs=0.3)


def test_when_board_translates_then_drift_is_relative_to_the_first_frame() -> None:
    monitor = DriftMonitor(BOARD_CENTRE, vibration_window_frames=10)
    monitor.update(np.eye(3))
    drift = monitor.update(translation_homography(1.5, -0.5))
    assert drift.translation_px == pytest.approx((1.5, -0.5))


def test_when_board_rotates_and_expands_then_rotation_and_ppm_are_reported() -> None:
    monitor = DriftMonitor(BOARD_CENTRE, vibration_window_frames=10)
    monitor.update(np.eye(3))
    drift = monitor.update(similarity(0.5, 1.0002))
    assert drift.rotation_deg == pytest.approx(0.5)
    assert drift.apparent_scale_ppm == pytest.approx(200.0, rel=1e-3)


def test_when_board_jitters_then_vibration_is_rms_of_recent_motion() -> None:
    monitor = DriftMonitor(BOARD_CENTRE, vibration_window_frames=2)
    for shift in (0.0, 3.0, 0.0, 4.0):
        drift = monitor.update(translation_homography(shift, 0.0))
    assert drift.vibration_rms_px == pytest.approx(np.sqrt((3.0**2 + 4.0**2) / 2.0))


def test_when_cell_is_invalid_then_its_smoothed_corners_are_held() -> None:
    smoother = ExponentialCornerSmoother(0.25)
    smoother.update(np.zeros((2, 4, 2)), np.array([True, True]))
    smoothed = smoother.update(np.ones((2, 4, 2)), np.array([True, False]))
    np.testing.assert_allclose(smoothed[0], 0.25)
    np.testing.assert_allclose(smoothed[1], 0.0)
