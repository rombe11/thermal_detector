from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.support import blank_frame, cpu_pipeline
from thermal_board.geometry.planar_homography import project_points
from thermal_board.pipeline.board_measurement_pipeline import (
    ACQUISITION_MODE,
    TRACKING_MODE,
    BoardNotFoundError,
)
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource

if TYPE_CHECKING:
    from thermal_board.acquisition.thermal_frame import ThermalFrame
    from thermal_board.config.system_settings import SystemConfiguration


def drifting_sequence(
    configuration: SystemConfiguration, count: int
) -> tuple[SyntheticSequenceSource, list[ThermalFrame]]:
    source = SyntheticSequenceSource(
        configuration, SequencePlan(count, 0.4, drift_per_frame_px=(0.3, 0.0))
    )
    return source, list(source.frames())


def test_when_sequence_is_processed_then_first_frame_acquires_and_later_frames_track(
    small_configuration: SystemConfiguration,
) -> None:
    pipeline = cpu_pipeline(small_configuration)
    _, frames = drifting_sequence(small_configuration, 3)
    modes = [pipeline.process(frame).localization.mode for frame in frames]
    assert modes == [ACQUISITION_MODE, TRACKING_MODE, TRACKING_MODE]


def test_when_board_moves_then_measured_pose_follows_ground_truth(
    small_configuration: SystemConfiguration,
) -> None:
    pipeline = cpu_pipeline(small_configuration)
    source, frames = drifting_sequence(small_configuration, 3)
    centres = pipeline.components.model.cell_centers_mm
    for frame in frames:
        measurement = pipeline.process(frame)
        estimated = project_points(measurement.localization.metric.homography, centres)
        truth = project_points(source.ground_truth_homography(frame.index), centres)
        np.testing.assert_allclose(estimated, truth, atol=0.05)
        assert measurement.report.meets_precision_target


def test_when_frame_is_blank_then_board_not_found_is_raised(
    small_configuration: SystemConfiguration,
) -> None:
    with pytest.raises(BoardNotFoundError):
        cpu_pipeline(small_configuration).process(blank_frame(small_configuration))


def test_when_board_is_lost_then_the_next_frame_reacquires(
    small_configuration: SystemConfiguration,
) -> None:
    pipeline = cpu_pipeline(small_configuration)
    _, frames = drifting_sequence(small_configuration, 2)
    pipeline.process(frames[0])
    with pytest.raises(BoardNotFoundError):
        pipeline.process(blank_frame(small_configuration))
    assert pipeline.process(frames[1]).localization.mode == ACQUISITION_MODE
