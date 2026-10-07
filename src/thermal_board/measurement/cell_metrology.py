from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from thermal_board.geometry.planar_homography import HomographyFit, fit_homography, project_points

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.geometry.board_model import BoardModel


@dataclass(frozen=True, slots=True)
class MetricRegistration:
    homography: NDArray[np.float64]
    image_corners_px: NDArray[np.float64]
    valid: NDArray[np.bool_]

    @property
    def board_corners_mm(self) -> NDArray[np.float64]:
        return project_points(np.linalg.inv(self.homography), self.image_corners_px)


@dataclass(frozen=True, slots=True)
class CellMeasurements:
    corners_mm: NDArray[np.float64]
    centers_mm: NDArray[np.float64]
    widths_mm: NDArray[np.float64]
    heights_mm: NDArray[np.float64]
    center_deviation_mm: NDArray[np.float64]
    corner_deviation_mm: NDArray[np.float64]
    valid: NDArray[np.bool_]
    measured_levels: NDArray[np.float64] | None = None

    @property
    def valid_ratio(self) -> float:
        return float(np.mean(self.valid)) if self.valid.size else 0.0


def fit_metric_registration(
    model: BoardModel,
    image_corners_px: NDArray[np.float64],
    valid: NDArray[np.bool_],
    ransac_threshold_px: float,
) -> MetricRegistration:
    model_points = model.cell_corners_mm[valid].reshape(-1, 2)
    image_points = image_corners_px[valid].reshape(-1, 2)
    fit: HomographyFit = fit_homography(model_points, image_points, ransac_threshold_px)
    corner_inliers = fit.inlier_mask.reshape(-1, 4).all(axis=1)
    refined_valid = valid.copy()
    refined_valid[np.flatnonzero(valid)] = corner_inliers
    return MetricRegistration(fit.matrix, image_corners_px, refined_valid)


def edge_lengths_mm(corners_mm: NDArray[np.float64]) -> NDArray[np.float64]:
    following = np.roll(corners_mm, shift=-1, axis=1)
    return np.asarray(np.linalg.norm(following - corners_mm, axis=-1))


def measure_cells(
    model: BoardModel, corners_mm: NDArray[np.float64], valid: NDArray[np.bool_]
) -> CellMeasurements:
    lengths = edge_lengths_mm(corners_mm)
    widths = (lengths[:, 0] + lengths[:, 2]) / 2.0
    heights = (lengths[:, 1] + lengths[:, 3]) / 2.0
    centers = corners_mm.mean(axis=1)
    center_deviation = centers - model.cell_centers_mm
    corner_deviation = np.linalg.norm(corners_mm - model.cell_corners_mm, axis=-1)
    return CellMeasurements(
        corners_mm, centers, widths, heights, center_deviation, corner_deviation, valid
    )
