from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from thermal_board.geometry.planar_homography import project_points
from thermal_board.simulation.synthetic_sequence import (
    SequencePlan,
    SyntheticScene,
    SyntheticSequenceSource,
)
from thermal_board.simulation.thermal_board_renderer import (
    CheckerboardRadianceField,
    ThermalBoardRenderer,
)

if TYPE_CHECKING:
    from tests.support import RenderedScene
    from thermal_board.config.system_settings import SystemConfiguration


def sharp_radiance(
    configuration: SystemConfiguration, points: list[list[float]], offsets: np.ndarray | None = None
) -> list[float]:
    field = CheckerboardRadianceField(
        configuration.board, configuration.simulation, blur_sigma_mm=0.0
    )
    return field.radiance(np.array(points), offsets).tolist()


def test_when_radiance_is_sampled_then_cells_gap_and_background_have_their_levels(
    small_configuration: SystemConfiguration,
) -> None:
    levels = small_configuration.simulation
    sampled = sharp_radiance(small_configuration, [[75, 75], [150, 75], [112.5, 75], [-5, 75]])
    expected = [
        levels.hot_level,
        levels.cold_level,
        levels.substrate_level,
        levels.background_level,
    ]
    assert sampled == expected


def test_when_one_cell_is_offset_then_only_that_cell_moves(
    small_configuration: SystemConfiguration,
) -> None:
    levels = small_configuration.simulation
    offsets = np.zeros((10, 12, 2))
    offsets[0, 0] = (5.0, 0.0)
    sampled = sharp_radiance(
        small_configuration, [[52.0, 75.0], [102.0, 75.0], [150.0, 75.0]], offsets
    )
    assert sampled == [levels.substrate_level, levels.hot_level, levels.cold_level]


def test_when_board_is_rendered_then_cell_centres_carry_cell_temperatures(
    small_scene: RenderedScene,
) -> None:
    centres = np.rint(
        project_points(small_scene.homography, small_scene.model.cell_centers_mm)
    ).astype(int)
    values = small_scene.raw_image[centres[:, 1], centres[:, 0]].astype(float)
    levels, hot = small_scene.configuration.simulation, small_scene.model.hot_cell_mask
    assert np.abs(values[hot] - levels.hot_level).max() < 80
    assert np.abs(values[~hot] - levels.cold_level).max() < 80
    assert small_scene.raw_image.dtype == np.uint16


def test_when_flat_scene_is_digitized_then_noise_matches_configuration(
    small_configuration: SystemConfiguration,
) -> None:
    board, simulation, camera = (
        small_configuration.board,
        small_configuration.simulation,
        small_configuration.camera,
    )
    residual = (
        ThermalBoardRenderer(board, simulation, camera).digitize(np.full((200, 200), 5000.0))
        - 5000.0
    )
    assert residual.std() == pytest.approx(simulation.noise_sigma, rel=0.1)


def test_when_drift_is_planned_then_ground_truth_moves_by_the_drift(
    small_configuration: SystemConfiguration,
) -> None:
    plan = SequencePlan(3, 0.0, drift_per_frame_px=(1.0, -0.5))
    source = SyntheticSequenceSource(small_configuration, plan)
    centre = np.array([487.5, 412.5])
    nominal = project_points(SyntheticScene(small_configuration).homography(), centre)
    np.testing.assert_allclose(
        project_points(source.ground_truth_homography(2), centre) - nominal, [2.0, -1.0], atol=1e-6
    )
