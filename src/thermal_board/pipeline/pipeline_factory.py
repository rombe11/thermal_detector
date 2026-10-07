from __future__ import annotations

from typing import TYPE_CHECKING

from thermal_board.compute.array_backend import select_compute_backend
from thermal_board.detection.grid_registration import GridRegistrar
from thermal_board.geometry.board_model import BoardModel
from thermal_board.geometry.pinhole_camera import PinholeCameraModel
from thermal_board.pipeline.board_measurement_pipeline import (
    BoardMeasurementPipeline,
    PipelineComponents,
)
from thermal_board.preprocessing.thermal_normalizer import ThermalNormalizer
from thermal_board.refinement.subpixel_corner_refiner import SubpixelCornerRefiner
from thermal_board.tracking.corner_smoothing import ExponentialCornerSmoother
from thermal_board.tracking.drift_monitor import DriftMonitor
from thermal_board.tracking.phase_correlation_motion import PhaseCorrelationMotionEstimator

if TYPE_CHECKING:
    from thermal_board.compute.array_backend import ComputeBackend
    from thermal_board.config.system_settings import SystemConfiguration, TrackingSettings


def build_tracking_stage(
    model: BoardModel, tracking: TrackingSettings
) -> tuple[ExponentialCornerSmoother, DriftMonitor, PhaseCorrelationMotionEstimator]:
    smoother = ExponentialCornerSmoother(tracking.smoothing_factor)
    drift_monitor = DriftMonitor(model.board_center_mm, tracking.vibration_window_frames)
    return smoother, drift_monitor, PhaseCorrelationMotionEstimator()


def build_pipeline_components(
    configuration: SystemConfiguration, backend: ComputeBackend
) -> PipelineComponents:
    model, detection = BoardModel(configuration.board), configuration.detection
    geometry_stage = (model, PinholeCameraModel(configuration.camera))
    refiner = SubpixelCornerRefiner(configuration.refinement, backend)
    vision_stage = (ThermalNormalizer(detection), GridRegistrar(model, detection), refiner)
    tracking_stage = build_tracking_stage(model, configuration.tracking)
    stages = (*geometry_stage, *vision_stage, *tracking_stage)
    return PipelineComponents(configuration, *stages, backend.name)


def build_measurement_pipeline(
    configuration: SystemConfiguration, backend: ComputeBackend | None = None
) -> BoardMeasurementPipeline:
    selected = backend or select_compute_backend(configuration.compute.prefer_gpu)
    return BoardMeasurementPipeline(build_pipeline_components(configuration, selected))
