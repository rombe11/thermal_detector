from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from thermal_board.detection.cell_blob_detector import CellBlobs, detect_cell_blobs
from thermal_board.detection.grid_registration import (
    GridRegistrar,
    GridRegistrationError,
    keep_unique_blob_assignments,
    match_cells_to_blobs,
)
from thermal_board.detection.thermal_level_estimation import ThermalLevels, estimate_thermal_levels
from thermal_board.geometry.planar_homography import project_points
from thermal_board.preprocessing.thermal_normalizer import ThermalNormalizer

if TYPE_CHECKING:
    from tests.support import RenderedScene
    from thermal_board.config.system_settings import SystemConfiguration

UNIT_LEVELS = ThermalLevels(cold=0.0, substrate=0.5, hot=1.0)


def scene_blobs(scene: RenderedScene) -> CellBlobs:
    levels = estimate_thermal_levels(scene.normalized_image)
    return detect_cell_blobs(scene.normalized_image, levels, scene.configuration.detection)


def ideal_blobs(scene: RenderedScene) -> CellBlobs:
    centres = project_points(scene.homography, scene.model.cell_centers_mm)
    return CellBlobs(centres, scene.model.hot_cell_mask.copy(), np.full(len(centres), 70.0))


def test_when_frame_is_normalized_then_it_spans_the_unit_interval(
    small_scene: RenderedScene,
) -> None:
    assert small_scene.normalized_image.dtype == np.float32
    assert float(small_scene.normalized_image.min()) == 0.0
    assert float(small_scene.normalized_image.max()) == 1.0


def test_when_frame_is_flat_then_normalization_stays_finite(
    small_configuration: SystemConfiguration,
) -> None:
    flat = np.full((32, 32), 7000, np.uint16)
    assert np.isfinite(ThermalNormalizer(small_configuration.detection).normalize(flat)).all()


def test_when_three_populations_exist_then_their_levels_are_recovered() -> None:
    generator = np.random.default_rng(3)
    levels = generator.choice([0.1, 0.45, 0.9], size=(120, 160), p=[0.2, 0.6, 0.2])
    image = (levels + generator.normal(0.0, 0.01, levels.shape)).astype(np.float32)
    estimated = estimate_thermal_levels(image)
    np.testing.assert_allclose(
        [estimated.cold, estimated.substrate, estimated.hot], [0.1, 0.45, 0.9], atol=0.02
    )


def test_when_board_is_imaged_then_every_cell_is_detected_with_its_polarity(
    small_scene: RenderedScene,
) -> None:
    blobs = scene_blobs(small_scene)
    assert len(blobs) == small_scene.model.cell_count
    assert int(blobs.is_hot.sum()) == int(small_scene.model.hot_cell_mask.sum())


def test_when_speckles_and_oversized_regions_appear_then_they_are_rejected(
    small_configuration: SystemConfiguration,
) -> None:
    image = np.full((80, 80), 0.5, np.float32)
    image[5:13, 5:13], image[30:38, 5:13], image[5:13, 30:38] = 1.0, 1.0, 1.0
    image[50:75, 50:75], image[45, 20] = 1.0, 1.0
    blobs = detect_cell_blobs(image, UNIT_LEVELS, small_configuration.detection)
    np.testing.assert_array_equal(blobs.areas, [64.0, 64.0, 64.0])


def test_when_blobs_are_registered_then_homography_matches_ground_truth(
    small_scene: RenderedScene,
) -> None:
    registrar = GridRegistrar(small_scene.model, small_scene.configuration.detection)
    registration = registrar.register(scene_blobs(small_scene))
    centres = small_scene.model.cell_centers_mm
    error = project_points(registration.homography, centres) - project_points(
        small_scene.homography, centres
    )
    assert registration.match_ratio == 1.0
    assert np.abs(error).max() < 0.05


def test_when_blob_has_wrong_polarity_or_position_then_it_is_not_matched(
    small_scene: RenderedScene,
) -> None:
    blobs = ideal_blobs(small_scene)
    blobs.is_hot[0] = not blobs.is_hot[0]
    blobs.centroids[1] += 10.0
    matches = match_cells_to_blobs(small_scene.model, small_scene.homography, blobs, 4.0)
    assert {0, 1}.isdisjoint(matches.cell_indices.tolist())
    assert len(matches.cell_indices) == small_scene.model.cell_count - 2


def test_when_two_cells_claim_one_blob_then_the_closest_cell_wins() -> None:
    unique = keep_unique_blob_assignments(
        np.array([4, 7, 9]), np.array([2, 2, 5]), np.array([0.8, 0.2, 0.5])
    )
    assert unique.cell_indices.tolist() == [7, 9]


def test_when_too_few_cells_match_then_registration_fails(small_scene: RenderedScene) -> None:
    settings = dataclasses.replace(small_scene.configuration.detection, minimum_match_ratio=0.9)
    partial = ideal_blobs(small_scene).subset(np.arange(small_scene.model.cell_count // 2))
    with pytest.raises(GridRegistrationError, match="matched"):
        GridRegistrar(small_scene.model, settings).register(partial)
