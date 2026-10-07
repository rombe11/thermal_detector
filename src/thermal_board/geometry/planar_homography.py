from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MINIMUM_HOMOGRAPHY_POINTS = 4


class HomographyEstimationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HomographyFit:
    matrix: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]


def project_points(
    homography: NDArray[np.float64], points: NDArray[np.float64]
) -> NDArray[np.float64]:
    flat_points = points.reshape(-1, 2)
    homogeneous = flat_points @ homography[:, :2].T + homography[:, 2]
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    return projected.reshape(points.shape)


def translation_homography(shift_x: float, shift_y: float) -> NDArray[np.float64]:
    return np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]])


def require_enough_points(source_points: NDArray[np.float64]) -> None:
    if len(source_points) < MINIMUM_HOMOGRAPHY_POINTS:
        raise HomographyEstimationError("at least four correspondences are required")


def fit_homography(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
    ransac_threshold_px: float | None = None,
) -> HomographyFit:
    require_enough_points(source_points)
    method = 0 if ransac_threshold_px is None else cv2.RANSAC
    threshold = 0.0 if ransac_threshold_px is None else ransac_threshold_px
    source, target = source_points.astype(np.float64), target_points.astype(np.float64)
    matrix, mask = cv2.findHomography(source, target, method, threshold, maxIters=4000)
    if matrix is None:
        raise HomographyEstimationError("homography estimation failed")
    return HomographyFit(matrix / matrix[2, 2], mask.ravel().astype(bool))


def local_jacobian(
    homography: NDArray[np.float64], point: NDArray[np.float64]
) -> NDArray[np.float64]:
    step = 1.0
    offsets = np.array([[0.0, 0.0], [step, 0.0], [0.0, step]])
    projected = project_points(homography, point + offsets)
    return np.stack([projected[1] - projected[0], projected[2] - projected[0]], axis=1) / step


def pixels_per_millimetre(homography: NDArray[np.float64], point_mm: NDArray[np.float64]) -> float:
    return float(np.sqrt(abs(np.linalg.det(local_jacobian(homography, point_mm)))))
