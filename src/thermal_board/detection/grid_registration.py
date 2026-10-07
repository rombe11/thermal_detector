from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from thermal_board.geometry.planar_homography import (
    HomographyEstimationError,
    fit_homography,
    local_jacobian,
    project_points,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import DetectionSettings
    from thermal_board.detection.cell_blob_detector import CellBlobs
    from thermal_board.geometry.board_model import BoardModel

REGISTRATION_ITERATIONS = 2


class GridRegistrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CellCorrespondences:
    cell_indices: NDArray[np.int64]
    blob_indices: NDArray[np.int64]

    def subset(self, selection: NDArray[np.bool_]) -> CellCorrespondences:
        return CellCorrespondences(self.cell_indices[selection], self.blob_indices[selection])


@dataclass(frozen=True, slots=True)
class GridRegistration:
    homography: NDArray[np.float64]
    correspondences: CellCorrespondences
    match_ratio: float
    pitch_px: float


def extreme_corner_centroids(centroids: NDArray[np.float64]) -> NDArray[np.float64]:
    coordinate_sum = centroids[:, 0] + centroids[:, 1]
    coordinate_difference = centroids[:, 0] - centroids[:, 1]
    indices = [
        np.argmin(coordinate_sum),
        np.argmax(coordinate_difference),
        np.argmax(coordinate_sum),
        np.argmin(coordinate_difference),
    ]
    return centroids[indices]


def coarse_board_homography(model: BoardModel, blobs: CellBlobs) -> NDArray[np.float64]:
    if len(blobs) < len(model.outer_corner_cell_indices):
        raise GridRegistrationError("too few thermal cells detected to locate the board")
    model_corners = model.cell_centers_mm[model.outer_corner_cell_indices]
    return fit_homography(model_corners, extreme_corner_centroids(blobs.centroids)).matrix


def estimate_pitch_px(model: BoardModel, homography: NDArray[np.float64]) -> float:
    jacobian = local_jacobian(homography, model.board_center_mm)
    pitch_x = model.geometry.pitch_x_mm * float(np.linalg.norm(jacobian[:, 0]))
    pitch_y = model.geometry.pitch_y_mm * float(np.linalg.norm(jacobian[:, 1]))
    return min(pitch_x, pitch_y)


def keep_unique_blob_assignments(
    cells: NDArray[np.int64], blob_indices: NDArray[np.int64], distances: NDArray[np.float64]
) -> CellCorrespondences:
    order = np.argsort(distances)
    _, first_occurrence = np.unique(blob_indices[order], return_index=True)
    keep = np.sort(order[first_occurrence])
    return CellCorrespondences(cells[keep], blob_indices[keep])


def match_cells_to_blobs(
    model: BoardModel, homography: NDArray[np.float64], blobs: CellBlobs, tolerance_px: float
) -> CellCorrespondences:
    predicted_centers = project_points(homography, model.cell_centers_mm)
    distances, nearest = cKDTree(blobs.centroids).query(
        predicted_centers, distance_upper_bound=tolerance_px
    )
    cells = np.flatnonzero(np.isfinite(distances))
    blob_indices = nearest[cells].astype(np.int64)
    same_polarity = blobs.is_hot[blob_indices] == model.hot_cell_mask[cells]
    return keep_unique_blob_assignments(
        cells[same_polarity], blob_indices[same_polarity], distances[cells][same_polarity]
    )


class GridRegistrar:
    def __init__(self, model: BoardModel, settings: DetectionSettings) -> None:
        self.model = model
        self.settings = settings

    def fit_matched(
        self, blobs: CellBlobs, matches: CellCorrespondences
    ) -> tuple[NDArray[np.float64], CellCorrespondences]:
        model_points = self.model.cell_centers_mm[matches.cell_indices]
        image_points = blobs.centroids[matches.blob_indices]
        fit = fit_homography(model_points, image_points, self.settings.ransac_threshold_px)
        return fit.matrix, matches.subset(fit.inlier_mask)

    def refine(self, blobs: CellBlobs, homography: NDArray[np.float64]) -> GridRegistration:
        matches = CellCorrespondences(np.zeros(0, np.int64), np.zeros(0, np.int64))
        for _ in range(REGISTRATION_ITERATIONS):
            matches = match_cells_to_blobs(
                self.model, homography, blobs, self.tolerance(homography)
            )
            homography, matches = self.fit_matched(blobs, matches)
        ratio = len(matches.cell_indices) / self.model.cell_count
        pitch_px = estimate_pitch_px(self.model, homography)
        return GridRegistration(homography, matches, ratio, pitch_px)

    def tolerance(self, homography: NDArray[np.float64]) -> float:
        pitch_px = estimate_pitch_px(self.model, homography)
        return pitch_px * self.settings.match_tolerance_pitch_ratio

    def register(self, blobs: CellBlobs) -> GridRegistration:
        try:
            registration = self.refine(blobs, coarse_board_homography(self.model, blobs))
        except HomographyEstimationError as error:
            raise GridRegistrationError(str(error)) from error
        if registration.match_ratio < self.settings.minimum_match_ratio:
            raise GridRegistrationError(f"only {registration.match_ratio:.1%} of cells matched")
        return registration
