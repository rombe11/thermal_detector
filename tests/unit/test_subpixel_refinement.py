from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.special import erf

from thermal_board.compute.array_backend import cpu_backend
from thermal_board.refinement.edge_moment_locator import locate_edge_crossings
from thermal_board.refinement.line_fitting import fit_lines, intersect_consecutive_lines
from thermal_board.refinement.subpixel_corner_refiner import SubpixelCornerRefiner

if TYPE_CHECKING:
    from tests.support import RenderedScene
    from thermal_board.refinement.subpixel_corner_refiner import RefinedCorners

PROFILE_OFFSETS = np.arange(-11, 12) * 0.25
MOMENT_HALF_WIDTH = 1.75
SQUARE = np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]])


def edge_profiles(edge_positions: np.ndarray, low: float, high: float) -> np.ndarray:
    blurred_step = erf((PROFILE_OFFSETS[:, None] - edge_positions[None, :]) / (np.sqrt(2.0) * 0.9))
    return low + (high - low) * 0.5 * (1.0 + blurred_step)


def refine(scene: RenderedScene, prediction: np.ndarray, passes: int = 1) -> RefinedCorners:
    settings = dataclasses.replace(scene.configuration.refinement, refinement_passes=passes)
    half_width = 0.8 * 25.0 / scene.configuration.camera.ground_sample_distance_mm / 2.0
    return SubpixelCornerRefiner(settings, cpu_backend()).refine(
        scene.normalized_image, prediction, half_width
    )


def rms_corner_error(scene: RenderedScene, refined: RefinedCorners) -> float:
    error = np.linalg.norm(refined.corners_px - scene.true_corners_px, axis=-1)
    return float(np.sqrt(np.mean(error**2)))


@pytest.mark.parametrize(("low", "high"), [(0.1, 0.9), (0.9, 0.1)])
def test_when_edge_is_off_centre_then_recentring_locates_it_without_shrinkage(
    low: float, high: float
) -> None:
    truth = np.array([-0.8, -0.3, 0.0, 0.37, 0.9])
    crossings = locate_edge_crossings(
        edge_profiles(truth, low, high), PROFILE_OFFSETS, MOMENT_HALF_WIDTH, 6, np
    )
    np.testing.assert_allclose(crossings.offsets_px, truth, atol=0.01)
    assert crossings.inside_window.all()


def test_when_window_is_not_recentred_then_off_centre_edges_shrink_towards_the_centre() -> None:
    truth = np.array([-0.6, 0.6])
    crossings = locate_edge_crossings(
        edge_profiles(truth, 0.1, 0.9), PROFILE_OFFSETS, MOMENT_HALF_WIDTH, 1, np
    )
    gains = crossings.offsets_px / truth
    assert np.all((gains > 0.5) & (gains < 1.0))


def test_when_profile_is_flat_then_it_has_no_edge_contrast() -> None:
    crossings = locate_edge_crossings(
        np.full((23, 2), 0.4), PROFILE_OFFSETS, MOMENT_HALF_WIDTH, 3, np
    )
    np.testing.assert_allclose(crossings.step_contrast, 0.0)


def test_when_one_point_has_zero_weight_then_the_line_ignores_it() -> None:
    x_values, y_values = np.array([[0.0, 1.0, 2.0, 3.0]]), np.array([[1.0, 1.0, 9.0, 1.0]])
    lines = fit_lines(x_values, y_values, np.array([[1.0, 1.0, 0.0, 1.0]]), np)
    a, b, c = lines.coefficients[0]
    assert (abs(a), -c / b) == pytest.approx((0.0, 1.0), abs=1e-9)
    assert lines.residual_rms_px[0] == pytest.approx(0.0, abs=1e-9)


def test_when_square_edges_intersect_then_its_corners_are_recovered() -> None:
    lines = np.array([[[0.0, 1.0, 0.0], [1.0, 0.0, -10.0], [0.0, 1.0, -10.0], [1.0, 0.0, 0.0]]])
    np.testing.assert_allclose(intersect_consecutive_lines(lines, np), SQUARE, atol=1e-12)


def test_when_prediction_is_noisy_then_true_corners_are_recovered(
    small_scene: RenderedScene,
) -> None:
    noise = np.random.default_rng(4).uniform(-0.3, 0.3, small_scene.true_corners_px.shape)
    refined = refine(small_scene, small_scene.true_corners_px + noise)
    assert refined.valid.all()
    assert rms_corner_error(small_scene, refined) < 0.08


def test_when_prediction_is_coherently_offset_then_refinement_converges(
    small_scene: RenderedScene,
) -> None:
    refined = refine(small_scene, small_scene.true_corners_px + np.array([0.6, -0.5]), passes=2)
    assert rms_corner_error(small_scene, refined) < 0.08


def test_when_image_has_no_edges_then_every_cell_is_invalid(small_scene: RenderedScene) -> None:
    blank = dataclasses.replace(
        small_scene, normalized_image=np.full_like(small_scene.normalized_image, 0.5)
    )
    assert not refine(blank, small_scene.true_corners_px).valid.any()
