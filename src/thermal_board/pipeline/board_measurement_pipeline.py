from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from thermal_board.detection.cell_blob_detector import CellBlobs, detect_cell_blobs
from thermal_board.detection.grid_registration import (
    GridRegistrar,
    GridRegistrationError,
    estimate_pitch_px,
)
from thermal_board.detection.thermal_level_estimation import estimate_thermal_levels
from thermal_board.geometry.planar_homography import (
    HomographyEstimationError,
    pixels_per_millimetre,
    project_points,
    translation_homography,
)
from thermal_board.measurement.accuracy_report import (
    AccuracyReport,
    ReportInputs,
    build_accuracy_report,
)
from thermal_board.measurement.cell_metrology import (
    CellMeasurements,
    MetricRegistration,
    fit_metric_registration,
    measure_cells,
)
from thermal_board.measurement.cell_radiometry import measure_cell_levels

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.acquisition.thermal_frame import ThermalFrame
    from thermal_board.config.system_settings import SystemConfiguration
    from thermal_board.geometry.board_model import BoardModel
    from thermal_board.geometry.pinhole_camera import PinholeCameraModel
    from thermal_board.preprocessing.thermal_normalizer import ThermalNormalizer
    from thermal_board.refinement.subpixel_corner_refiner import (
        RefinedCorners,
        SubpixelCornerRefiner,
    )
    from thermal_board.tracking.corner_smoothing import ExponentialCornerSmoother
    from thermal_board.tracking.drift_monitor import DriftMonitor
    from thermal_board.tracking.phase_correlation_motion import (
        FrameShift,
        PhaseCorrelationMotionEstimator,
    )


class BoardNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PipelineComponents:
    configuration: SystemConfiguration
    model: BoardModel
    camera: PinholeCameraModel
    normalizer: ThermalNormalizer
    registrar: GridRegistrar
    refiner: SubpixelCornerRefiner
    smoother: ExponentialCornerSmoother
    drift_monitor: DriftMonitor
    motion_estimator: PhaseCorrelationMotionEstimator
    compute_backend_name: str


ACQUISITION_MODE = "acquisition"
TRACKING_MODE = "tracking"


@dataclass(frozen=True, slots=True)
class BoardLocalization:
    mode: str
    prior_homography: NDArray[np.float64]
    detected_cell_ratio: float
    refined: RefinedCorners
    metric: MetricRegistration


@dataclass(frozen=True, slots=True)
class FrameMeasurement:
    frame_index: int
    timestamp_s: float
    normalized_image: NDArray[np.float32]
    localization: BoardLocalization
    cells: CellMeasurements
    report: AccuracyReport


class TrackingMemory:
    def __init__(self) -> None:
        self.homography: NDArray[np.float64] | None = None

    def predicted_homography(self, shift: FrameShift, limit_px: float) -> NDArray[np.float64]:
        if self.homography is None:
            raise BoardNotFoundError("no previous board pose to track from")
        if math.hypot(shift.shift_x_px, shift.shift_y_px) > limit_px:
            return self.homography
        return translation_homography(shift.shift_x_px, shift.shift_y_px) @ self.homography


class BoardMeasurementPipeline:
    def __init__(self, components: PipelineComponents) -> None:
        self.components = components
        self.memory = TrackingMemory()

    def detect_blobs(self, image: NDArray[np.float32]) -> CellBlobs:
        settings = self.components.configuration.detection
        blobs = detect_cell_blobs(image, estimate_thermal_levels(image), settings)
        return CellBlobs(
            self.components.camera.undistort(blobs.centroids), blobs.is_hot, blobs.areas
        )

    def search_half_width_px(self, homography: NDArray[np.float64]) -> float:
        model, settings = self.components.model, self.components.configuration.refinement
        scale = pixels_per_millimetre(homography, model.board_center_mm)
        narrowest_feature_mm = float(min(model.geometry.gap_mm, *model.cell_size_mm))
        return settings.profile_half_length_ratio * narrowest_feature_mm * scale / 2.0

    def refine(self, image: NDArray[np.float32], prior: NDArray[np.float64]) -> RefinedCorners:
        model_corners = self.components.model.cell_corners_mm
        predicted = self.components.camera.distort(project_points(prior, model_corners))
        return self.components.refiner.refine(image, predicted, self.search_half_width_px(prior))

    def complete(
        self,
        image: NDArray[np.float32],
        mode: str,
        prior: NDArray[np.float64],
        detected_ratio: float | None,
    ) -> BoardLocalization:
        refined = self.refine(image, prior)
        undistorted = self.components.camera.undistort(refined.corners_px)
        threshold = self.components.configuration.refinement.metric_outlier_threshold_px
        metric = fit_metric_registration(
            self.components.model, undistorted, refined.valid, threshold
        )
        ratio = float(refined.valid.mean()) if detected_ratio is None else detected_ratio
        return BoardLocalization(mode, prior, ratio, refined, metric)

    def acquire(self, image: NDArray[np.float32]) -> BoardLocalization:
        registration = self.components.registrar.register(self.detect_blobs(image))
        return self.complete(
            image, ACQUISITION_MODE, registration.homography, registration.match_ratio
        )

    def maximum_frame_shift_px(self, homography: NDArray[np.float64]) -> float:
        ratio = self.components.configuration.tracking.maximum_frame_shift_pitch_ratio
        return ratio * estimate_pitch_px(self.components.model, homography)

    def board_motion_px(self, previous: NDArray[np.float64], current: NDArray[np.float64]) -> float:
        centre = self.components.model.board_center_mm
        displacement = project_points(current, centre) - project_points(previous, centre)
        return float(np.linalg.norm(displacement))

    def is_consistent_track(
        self, localization: BoardLocalization, previous: NDArray[np.float64]
    ) -> bool:
        minimum_ratio = self.components.configuration.detection.minimum_match_ratio
        enough_cells = float(localization.metric.valid.mean()) >= minimum_ratio
        motion_px = self.board_motion_px(previous, localization.metric.homography)
        return enough_cells and motion_px <= self.maximum_frame_shift_px(previous)

    def track(
        self, image: NDArray[np.float32], previous: NDArray[np.float64], shift: FrameShift
    ) -> BoardLocalization | None:
        prior = self.memory.predicted_homography(shift, self.maximum_frame_shift_px(previous))
        try:
            localization = self.complete(image, TRACKING_MODE, prior, None)
        except HomographyEstimationError:
            return None
        return localization if self.is_consistent_track(localization, previous) else None

    def localize(self, image: NDArray[np.float32]) -> BoardLocalization:
        shift = self.components.motion_estimator.estimate(image)
        previous = self.memory.homography
        tracked = None if previous is None else self.track(image, previous, shift)
        localization = tracked or self.acquire(image)
        self.memory.homography = localization.metric.homography
        return localization

    def localize_or_raise(self, image: NDArray[np.float32]) -> BoardLocalization:
        try:
            return self.localize(image)
        except (GridRegistrationError, HomographyEstimationError) as error:
            self.reset()
            raise BoardNotFoundError(str(error)) from error

    def measure(self, metric: MetricRegistration, raw_pixels: NDArray[Any]) -> CellMeasurements:
        model, camera = self.components.model, self.components.camera
        smoothed = self.components.smoother.update(metric.board_corners_mm, metric.valid)
        levels = measure_cell_levels(raw_pixels, metric.homography, model, camera)
        return dataclasses.replace(
            measure_cells(model, smoothed, metric.valid), measured_levels=levels
        )

    def report(
        self,
        frame: ThermalFrame,
        localization: BoardLocalization,
        cells: CellMeasurements,
        started_s: float,
    ) -> AccuracyReport:
        components = self.components
        drift = components.drift_monitor.update(localization.metric.homography)
        latency_ms = (time.perf_counter() - started_s) * 1000.0
        refined, detected = localization.refined, localization.detected_cell_ratio
        measured = (localization.metric, cells, refined.edge_rms_px, detected)
        runtime = (drift, latency_ms, components.compute_backend_name)
        inputs = ReportInputs(frame.index, components.model, *measured, *runtime)
        return build_accuracy_report(inputs, components.configuration.accuracy)

    def process(self, frame: ThermalFrame) -> FrameMeasurement:
        started_s = time.perf_counter()
        image = self.components.normalizer.normalize(frame.pixels)
        localization = self.localize_or_raise(image)
        cells = self.measure(localization.metric, frame.pixels)
        report = self.report(frame, localization, cells, started_s)
        return FrameMeasurement(frame.index, frame.timestamp_s, image, localization, cells, report)

    def reset(self) -> None:
        self.memory.homography = None
        self.components.motion_estimator.reset()
        self.components.smoother.reset()
