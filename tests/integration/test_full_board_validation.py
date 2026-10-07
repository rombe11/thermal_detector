from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from thermal_board.acquisition.recorded_sources import ImageSequenceSource
from thermal_board.acquisition.thermal_frame import ThermalFrame
from thermal_board.compute.array_backend import select_compute_backend
from thermal_board.geometry.planar_homography import project_points
from thermal_board.pipeline.board_measurement_pipeline import TRACKING_MODE
from thermal_board.pipeline.pipeline_factory import build_measurement_pipeline
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource
from thermal_board.simulation.thermal_board_renderer import ThermalBoardRenderer

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import RenderedScene
    from thermal_board.config.system_settings import BoardGeometry, SystemConfiguration
    from thermal_board.pipeline.board_measurement_pipeline import (
        BoardMeasurementPipeline,
        FrameMeasurement,
    )

pytestmark = pytest.mark.integration

TOLERANCE_MM = 1.0
TARGET_RMS_MM = 0.5


def full_pipeline(configuration: SystemConfiguration) -> BoardMeasurementPipeline:
    return build_measurement_pipeline(
        configuration, select_compute_backend(configuration.compute.prefer_gpu)
    )


def measure_rendering(
    scene: RenderedScene, board: BoardGeometry, offsets_mm: np.ndarray | None = None
) -> FrameMeasurement:
    configuration = scene.configuration
    renderer = ThermalBoardRenderer(board, configuration.simulation, configuration.camera, seed=7)
    pixels = renderer.render(scene.homography, offsets_mm)
    return full_pipeline(configuration).process(ThermalFrame(0, 0.0, pixels))


def sparse_cell_offsets(board: BoardGeometry, amplitude_mm: float) -> np.ndarray:
    generator = np.random.default_rng(11)
    offsets = generator.uniform(-amplitude_mm, amplitude_mm, (board.rows, board.columns, 2))
    offsets[generator.random(offsets.shape[:2]) > 0.05] = 0.0
    return offsets


def board_centre_shift(
    source: SyntheticSequenceSource, index: int, centre: np.ndarray
) -> np.ndarray:
    current = project_points(source.ground_truth_homography(index), centre)
    return current - project_points(source.ground_truth_homography(0), centre)


def write_frame(directory: Path, frame: ThermalFrame) -> Path:
    path = directory / f"frame_{frame.index:03d}.png"
    cv2.imwrite(str(path), frame.pixels)
    return path


@pytest.fixture(scope="module")
def full_measurement(full_scene: RenderedScene) -> FrameMeasurement:
    return full_pipeline(full_scene.configuration).process(
        ThermalFrame(0, 0.0, full_scene.raw_image)
    )


def test_when_full_board_is_imaged_then_all_6144_cells_are_measured(
    full_measurement: FrameMeasurement,
) -> None:
    coverage = full_measurement.report.coverage
    assert full_measurement.cells.valid.size == 6144
    assert (coverage.detected_cell_ratio, coverage.measured_cell_ratio) == (1.0, 1.0)
    assert coverage.cells_out_of_tolerance == 0


def test_when_full_board_is_measured_then_every_corner_is_within_one_millimetre(
    full_measurement: FrameMeasurement,
) -> None:
    corner = full_measurement.report.deviations.corner_position
    assert corner.rms_mm < TARGET_RMS_MM
    assert corner.max_abs_mm < TOLERANCE_MM
    assert full_measurement.report.meets_precision_target


def test_when_full_board_is_measured_then_image_corners_match_ground_truth(
    full_scene: RenderedScene, full_measurement: FrameMeasurement
) -> None:
    error_px = np.linalg.norm(
        full_measurement.localization.refined.corners_px - full_scene.true_corners_px, axis=-1
    )
    assert float(np.sqrt(np.mean(error_px**2))) < 0.1
    assert float(error_px.max()) < 0.3


def test_when_cells_are_displaced_then_their_displacements_are_recovered(
    full_scene: RenderedScene,
) -> None:
    board = full_scene.configuration.board
    offsets = sparse_cell_offsets(board, amplitude_mm=3.0).reshape(-1, 2)
    measurement = measure_rendering(
        full_scene, board, offsets.reshape(board.rows, board.columns, 2)
    )
    residual = measurement.cells.center_deviation_mm - offsets
    assert int(np.any(offsets != 0.0, axis=1).sum()) > 250
    assert float(np.abs(residual).max()) < TOLERANCE_MM / 2
    assert float(np.sqrt(np.mean(residual**2))) < 0.15


def test_when_cells_are_one_millimetre_larger_then_measured_sizes_grow_by_one_millimetre(
    full_scene: RenderedScene, full_measurement: FrameMeasurement
) -> None:
    board = full_scene.configuration.board
    larger = dataclasses.replace(board, cell_width_mm=51.0, cell_height_mm=51.0, gap_mm=24.0)
    nominal = full_measurement.report.deviations
    measured = measure_rendering(full_scene, larger).report.deviations
    width_growth = measured.width_error.mean_mm - nominal.width_error.mean_mm
    height_growth = measured.height_error.mean_mm - nominal.height_error.mean_mm
    assert (width_growth, height_growth) == pytest.approx((1.0, 1.0), abs=0.1)


def test_when_board_vibrates_and_drifts_then_tracking_follows_ground_truth(
    configuration: SystemConfiguration,
) -> None:
    plan = SequencePlan(5, vibration_amplitude_px=0.6, drift_per_frame_px=(0.25, -0.15))
    source = SyntheticSequenceSource(configuration, plan, seed=5)
    pipeline = full_pipeline(configuration)
    measurements = [pipeline.process(frame) for frame in source.frames()]
    centre = pipeline.components.model.board_center_mm
    for measurement in measurements:
        expected = board_centre_shift(source, measurement.frame_index, centre)
        np.testing.assert_allclose(measurement.report.drift.translation_px, expected, atol=0.05)
        assert measurement.report.meets_precision_target
    assert all(measurement.localization.mode == TRACKING_MODE for measurement in measurements[1:])


def test_when_board_expands_then_apparent_scale_is_reported_in_ppm(
    configuration: SystemConfiguration,
) -> None:
    plan = SequencePlan(3, vibration_amplitude_px=0.0, expansion_ppm_per_frame=250.0)
    pipeline = full_pipeline(configuration)
    reports = [
        pipeline.process(frame).report
        for frame in SyntheticSequenceSource(configuration, plan).frames()
    ]
    assert reports[0].drift.apparent_scale_ppm == pytest.approx(0.0, abs=1e-6)
    assert reports[2].drift.apparent_scale_ppm == pytest.approx(500.0, abs=40.0)


def test_when_16_bit_recording_is_replayed_then_every_frame_passes(
    tmp_path: Path, configuration: SystemConfiguration
) -> None:
    source = SyntheticSequenceSource(configuration, SequencePlan(2, 0.5), seed=9)
    paths = [write_frame(tmp_path, frame) for frame in source.frames()]
    pipeline = full_pipeline(configuration)
    replay = ImageSequenceSource(paths, configuration.camera.frame_rate_hz)
    assert all(pipeline.process(frame).report.meets_precision_target for frame in replay.frames())
