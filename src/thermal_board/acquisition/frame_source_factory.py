from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from thermal_board.acquisition.recorded_sources import (
    SUPPORTED_IMAGE_SUFFIXES,
    ImageSequenceSource,
    VideoFileSource,
    list_image_files,
)
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource

if TYPE_CHECKING:
    from thermal_board.acquisition.thermal_frame import FrameSource
    from thermal_board.config.system_settings import SystemConfiguration

SYNTHETIC_SOURCE_NAME = "synthetic"


def create_frame_source(
    source_specification: str, configuration: SystemConfiguration, synthetic_frame_count: int = 60
) -> FrameSource:
    if source_specification == SYNTHETIC_SOURCE_NAME:
        return create_synthetic_source(configuration, synthetic_frame_count)
    path = Path(source_specification)
    frame_rate_hz = configuration.camera.frame_rate_hz
    if path.is_dir():
        return ImageSequenceSource(list_image_files(path), frame_rate_hz)
    if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        return ImageSequenceSource([path], frame_rate_hz)
    return VideoFileSource(path)


def create_synthetic_source(configuration: SystemConfiguration, frame_count: int) -> FrameSource:
    amplitude = configuration.simulation.vibration_amplitude_px
    return SyntheticSequenceSource(configuration, SequencePlan(frame_count, amplitude))
