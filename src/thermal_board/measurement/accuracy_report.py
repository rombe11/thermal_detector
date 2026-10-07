from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from thermal_board.geometry.planar_homography import project_points

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import AccuracySettings
    from thermal_board.geometry.board_model import BoardModel
    from thermal_board.measurement.cell_metrology import CellMeasurements, MetricRegistration
    from thermal_board.tracking.drift_monitor import DriftState


@dataclass(frozen=True, slots=True)
class DeviationStatistics:
    mean_mm: float
    rms_mm: float
    max_abs_mm: float

    @classmethod
    def of(cls, deviations_mm: NDArray[np.float64]) -> DeviationStatistics:
        if deviations_mm.size == 0:
            return cls(float("nan"), float("nan"), float("nan"))
        rms = float(np.sqrt(np.mean(np.square(deviations_mm))))
        return cls(float(np.mean(deviations_mm)), rms, float(np.max(np.abs(deviations_mm))))


@dataclass(frozen=True, slots=True)
class CoverageQuality:
    detected_cell_ratio: float
    measured_cell_ratio: float
    cells_out_of_tolerance: int


@dataclass(frozen=True, slots=True)
class GeometricQuality:
    reprojection_rms_px: float
    edge_fit_rms_px: float
    millimetres_per_pixel: float


@dataclass(frozen=True, slots=True)
class MetricDeviations:
    corner_position: DeviationStatistics
    center_position: DeviationStatistics
    width_error: DeviationStatistics
    height_error: DeviationStatistics


@dataclass(frozen=True, slots=True)
class AccuracyReport:
    frame_index: int
    coverage: CoverageQuality
    geometry: GeometricQuality
    deviations: MetricDeviations
    drift: DriftState
    latency_ms: float
    compute_backend: str
    meets_precision_target: bool


@dataclass(frozen=True, slots=True)
class ReportInputs:
    frame_index: int
    model: BoardModel
    registration: MetricRegistration
    cells: CellMeasurements
    edge_rms_px: NDArray[np.float64]
    detected_cell_ratio: float
    drift: DriftState
    latency_ms: float
    compute_backend: str


def reprojection_rms_px(model: BoardModel, registration: MetricRegistration) -> float:
    predicted = project_points(registration.homography, model.cell_corners_mm[registration.valid])
    residuals = predicted - registration.image_corners_px[registration.valid]
    return (
        float(np.sqrt(np.mean(np.sum(np.square(residuals), axis=-1))))
        if residuals.size
        else float("nan")
    )


def millimetres_per_pixel(model: BoardModel, registration: MetricRegistration) -> float:
    corners = registration.image_corners_px
    image_span = np.linalg.norm(corners[-1, 2] - corners[0, 0])
    board_span = np.linalg.norm(model.cell_corners_mm[-1, 2] - model.cell_corners_mm[0, 0])
    return float(board_span / max(float(image_span), 1e-9))


def median_edge_rms(inputs: ReportInputs) -> float:
    valid_rms = inputs.edge_rms_px[inputs.cells.valid]
    return float(np.median(valid_rms)) if valid_rms.size else float("nan")


def geometric_quality(inputs: ReportInputs) -> GeometricQuality:
    reprojection = reprojection_rms_px(inputs.model, inputs.registration)
    scale = millimetres_per_pixel(inputs.model, inputs.registration)
    return GeometricQuality(reprojection, median_edge_rms(inputs), scale)


def metric_deviations(model: BoardModel, cells: CellMeasurements) -> MetricDeviations:
    corner = DeviationStatistics.of(cells.corner_deviation_mm[cells.valid].ravel())
    center = DeviationStatistics.of(np.linalg.norm(cells.center_deviation_mm[cells.valid], axis=-1))
    width = DeviationStatistics.of(cells.widths_mm[cells.valid] - model.geometry.cell_width_mm)
    height = DeviationStatistics.of(cells.heights_mm[cells.valid] - model.geometry.cell_height_mm)
    return MetricDeviations(corner, center, width, height)


def count_out_of_tolerance(cells: CellMeasurements, settings: AccuracySettings) -> int:
    worst_corner = cells.corner_deviation_mm.max(axis=1)
    return int(np.count_nonzero(cells.valid & (worst_corner > settings.tolerance_mm)))


def precision_target_met(
    deviations: MetricDeviations, coverage: CoverageQuality, settings: AccuracySettings
) -> bool:
    corner = deviations.corner_position
    within_target = corner.rms_mm <= settings.target_precision_mm
    within_tolerance = corner.max_abs_mm <= settings.tolerance_mm
    return bool(within_target and within_tolerance and coverage.cells_out_of_tolerance == 0)


def build_accuracy_report(inputs: ReportInputs, settings: AccuracySettings) -> AccuracyReport:
    out_of_tolerance = count_out_of_tolerance(inputs.cells, settings)
    coverage = CoverageQuality(
        inputs.detected_cell_ratio, inputs.cells.valid_ratio, out_of_tolerance
    )
    deviations = metric_deviations(inputs.model, inputs.cells)
    meets_target = precision_target_met(deviations, coverage, settings)
    measured = (coverage, geometric_quality(inputs), deviations)
    runtime = (inputs.drift, inputs.latency_ms, inputs.compute_backend)
    return AccuracyReport(inputs.frame_index, *measured, *runtime, meets_target)
