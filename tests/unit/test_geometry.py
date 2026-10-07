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


def test_when_cells_are_numbered_then_numbers_run_right_to_left_from_the_top_row(
    small_configuration: SystemConfiguration,
) -> None:
    rows_and_columns = BoardModel(small_configuration.board).cell_rows_and_columns
    assert rows_and_columns[0].tolist() == [0, 11]
    assert rows_and_columns[11].tolist() == [0, 0]
    assert rows_and_columns[12].tolist() == [1, 11]
    assert rows_and_columns[-1].tolist() == [9, 0]


def test_when_numbering_order_is_configured_then_cell_zero_follows_it(
    small_configuration: SystemConfiguration,
) -> None:
    board = dataclasses.replace(
        small_configuration.board, numbering_rows="bottom_to_top", numbering_columns="left_to_right"
    )
    model = BoardModel(board)
    assert model.cell_rows_and_columns[:2].tolist() == [[9, 0], [9, 1]]
    assert int(model.cell_number_grid[9, 0]) == 0


def test_when_cell_zero_is_located_then_it_sits_in_the_top_right_corner(
    small_configuration: SystemConfiguration,
) -> None:
    board, model = small_configuration.board, BoardModel(small_configuration.board)
    top_right = [board.board_width_mm - board.border_x_mm, board.border_y_mm]
    np.testing.assert_allclose(model.cell_corners_mm[0, 1], top_right)
    np.testing.assert_allclose(model.cell_centers_mm[0] - model.cell_centers_mm[1], [75.0, 0.0])


def test_when_cells_are_enumerated_then_hot_and_cold_cells_alternate(
    small_configuration: SystemConfiguration,
) -> None:
    hot = BoardModel(small_configuration.board).hot_cell_mask
    assert bool(hot[0]) != bool(hot[1]) != bool(hot[2])
    assert bool(hot[0]) != bool(hot[12])
    assert int(hot.sum()) == hot.size // 2


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
