from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from thermal_board.geometry.board_model import BoardModel
from thermal_board.geometry.board_pose import BoardPose, board_to_image_homography
from thermal_board.geometry.pinhole_camera import PinholeCameraModel, build_intrinsic_matrix
from thermal_board.geometry.planar_homography import (
    HomographyEstimationError,
    fit_homography,
    pixels_per_millimetre,
    project_points,
)

if TYPE_CHECKING:
    from thermal_board.config.system_settings import SystemConfiguration

PERSPECTIVE = np.array([[1.2, 0.05, 30.0], [-0.03, 1.1, 12.0], [1e-4, -5e-5, 1.0]])
BOARD_CENTRE = np.array([100.0, 80.0])


def random_points(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.0, 500.0, (count, 2))


def test_when_board_model_is_built_then_first_cell_starts_after_the_border(
    small_configuration: SystemConfiguration,
) -> None:
    model = BoardModel(small_configuration.board)
    border = small_configuration.board.border_x_mm
    np.testing.assert_allclose(model.cell_corners_mm[0, 0], [border, border])
    np.testing.assert_allclose(model.cell_corners_mm[0, 2], [border + 50.0, border + 50.0])


def test_when_cells_are_enumerated_then_hot_and_cold_cells_alternate(
    small_configuration: SystemConfiguration,
) -> None:
    model = BoardModel(small_configuration.board)
    assert model.hot_cell_mask[:3].tolist() == [True, False, True]
    assert bool(model.hot_cell_mask[12]) is False
    assert int(model.hot_cell_mask.sum()) == model.cell_count // 2


def test_when_points_are_projected_and_back_projected_then_they_are_unchanged() -> None:
    points = random_points(12).reshape(3, 4, 2)
    recovered = project_points(np.linalg.inv(PERSPECTIVE), project_points(PERSPECTIVE, points))
    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_when_correspondences_contain_outliers_then_ransac_rejects_only_them() -> None:
    source = random_points(60)
    target = project_points(PERSPECTIVE, source)
    target[:5] += 40.0
    fit = fit_homography(source, target, ransac_threshold_px=1.0)
    assert fit.inlier_mask.tolist() == [False] * 5 + [True] * 55
    np.testing.assert_allclose(fit.matrix, PERSPECTIVE, rtol=1e-5, atol=1e-7)


def test_when_fewer_than_four_points_are_given_then_estimation_fails() -> None:
    with pytest.raises(HomographyEstimationError):
        fit_homography(random_points(3), random_points(3, seed=1))


def test_when_lens_is_distorted_then_distortion_round_trips(
    configuration: SystemConfiguration,
) -> None:
    lens = dataclasses.replace(configuration.camera, distortion_coefficients=(-0.2, 0.05, 0, 0, 0))
    camera = PinholeCameraModel(lens)
    points = random_points(20) + 300.0
    distorted = camera.distort(points)
    assert np.abs(distorted - points).max() > 1e-3
    np.testing.assert_allclose(camera.undistort(distorted), points, atol=1e-4)


def test_when_board_faces_camera_then_scale_is_focal_length_over_distance(
    configuration: SystemConfiguration,
) -> None:
    intrinsics = build_intrinsic_matrix(configuration.camera)
    pose = BoardPose(0.0, 0.0, 0.0, 20000.0)
    homography = board_to_image_homography(pose, intrinsics, BOARD_CENTRE)
    np.testing.assert_allclose(project_points(homography, BOARD_CENTRE), intrinsics[:2, 2])
    expected = configuration.camera.focal_length_px / 20000.0
    assert pixels_per_millimetre(homography, BOARD_CENTRE) == pytest.approx(expected)


def test_when_board_expands_then_it_grows_about_its_centre(
    configuration: SystemConfiguration,
) -> None:
    intrinsics, pose = build_intrinsic_matrix(configuration.camera), BoardPose(3.0, -2.0, 1.0, 2e4)
    nominal = board_to_image_homography(pose, intrinsics, BOARD_CENTRE)
    expanded = board_to_image_homography(pose, intrinsics, BOARD_CENTRE, board_scale=1.001)
    ratio = pixels_per_millimetre(expanded, BOARD_CENTRE) / pixels_per_millimetre(
        nominal, BOARD_CENTRE
    )
    assert ratio == pytest.approx(1.001, rel=1e-6)
    np.testing.assert_allclose(
        project_points(expanded, BOARD_CENTRE), project_points(nominal, BOARD_CENTRE)
    )
