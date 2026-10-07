from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from thermal_board.config.system_settings import AccuracySettings
from thermal_board.measurement.accuracy_report import (
    CoverageQuality,
    DeviationStatistics,
    MetricDeviations,
    count_out_of_tolerance,
    precision_target_met,
)
from thermal_board.measurement.cell_metrology import fit_metric_registration, measure_cells

if TYPE_CHECKING:
    from tests.support import RenderedScene
    from thermal_board.measurement.cell_metrology import CellMeasurements

ACCURACY = AccuracySettings(target_precision_mm=0.5, tolerance_mm=1.0)


def measure(scene: RenderedScene, corners_mm: np.ndarray) -> CellMeasurements:
    return measure_cells(scene.model, corners_mm, np.ones(len(corners_mm), dtype=bool))


def corner_statistics(rms_mm: float, max_mm: float) -> MetricDeviations:
    perfect = DeviationStatistics(0.0, 0.0, 0.0)
    return MetricDeviations(DeviationStatistics(0.0, rms_mm, max_mm), perfect, perfect, perfect)


def test_when_corners_are_nominal_then_every_deviation_is_zero(small_scene: RenderedScene) -> None:
    cells = measure(small_scene, small_scene.model.cell_corners_mm.copy())
    np.testing.assert_allclose(cells.widths_mm, 50.0)
    np.testing.assert_allclose(cells.center_deviation_mm, 0.0, atol=1e-12)
    np.testing.assert_allclose(cells.corner_deviation_mm, 0.0, atol=1e-12)


def test_when_a_cell_is_displaced_then_its_offset_is_reported(small_scene: RenderedScene) -> None:
    corners = small_scene.model.cell_corners_mm.copy()
    corners[5] += (0.4, -0.3)
    cells = measure(small_scene, corners)
    np.testing.assert_allclose(cells.center_deviation_mm[5], [0.4, -0.3])
    np.testing.assert_allclose(cells.corner_deviation_mm[5], 0.5)
    np.testing.assert_allclose(cells.center_deviation_mm[6], 0.0, atol=1e-12)


def test_when_image_corners_are_exact_then_board_millimetres_are_recovered(
    small_scene: RenderedScene,
) -> None:
    valid = np.ones(small_scene.model.cell_count, dtype=bool)
    metric = fit_metric_registration(small_scene.model, small_scene.true_corners_px, valid, 0.5)
    np.testing.assert_allclose(
        metric.board_corners_mm, small_scene.model.cell_corners_mm, atol=1e-6
    )


def test_when_a_corner_is_an_outlier_then_its_cell_is_invalidated(
    small_scene: RenderedScene,
) -> None:
    corners = small_scene.true_corners_px.copy()
    corners[7, 2] += (3.0, 3.0)
    metric = fit_metric_registration(
        small_scene.model, corners, np.ones(len(corners), dtype=bool), 0.5
    )
    assert np.flatnonzero(~metric.valid).tolist() == [7]


@pytest.mark.parametrize(
    ("rms_mm", "max_mm", "out_of_tolerance", "expected"),
    [(0.1, 0.4, 0, True), (0.6, 0.8, 0, False), (0.1, 1.2, 0, False), (0.1, 0.4, 2, False)],
)
def test_when_deviations_are_summarised_then_verdict_follows_rms_max_and_outliers(
    rms_mm: float, max_mm: float, out_of_tolerance: int, expected: bool
) -> None:
    coverage = CoverageQuality(1.0, 1.0, out_of_tolerance)
    assert precision_target_met(corner_statistics(rms_mm, max_mm), coverage, ACCURACY) is expected


def test_when_cells_exceed_tolerance_then_they_are_counted(small_scene: RenderedScene) -> None:
    corners = small_scene.model.cell_corners_mm.copy()
    corners[[3, 4], 0] += 1.5
    assert count_out_of_tolerance(measure(small_scene, corners), ACCURACY) == 2
