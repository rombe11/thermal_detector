from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import CameraSpecification


class PinholeCameraModel:
    def __init__(self, specification: CameraSpecification) -> None:
        self.specification = specification
        self.intrinsic_matrix = build_intrinsic_matrix(specification)
        self.distortion = np.asarray(specification.distortion_coefficients, dtype=np.float64)

    def undistort(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.specification.has_lens_distortion or points.size == 0:
            return points
        flat = points.reshape(-1, 1, 2).astype(np.float64)
        matrix = self.intrinsic_matrix
        corrected = cv2.undistortPoints(flat, matrix, self.distortion, P=matrix)
        return np.asarray(corrected, dtype=np.float64).reshape(points.shape)

    def distort(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.specification.has_lens_distortion or points.size == 0:
            return points
        flat = points.reshape(-1, 2)
        normalized = (flat - self.intrinsic_matrix[:2, 2]) / np.diag(self.intrinsic_matrix)[:2]
        rays = np.column_stack([normalized, np.ones(len(normalized))])
        zero_vector = np.zeros(3)
        projected, _ = cv2.projectPoints(
            rays, zero_vector, zero_vector, self.intrinsic_matrix, self.distortion
        )
        return np.asarray(projected, dtype=np.float64).reshape(points.shape)


def build_intrinsic_matrix(specification: CameraSpecification) -> NDArray[np.float64]:
    focal = specification.focal_length_px
    center_x = (specification.width_px - 1) / 2.0
    center_y = (specification.height_px - 1) / 2.0
    return np.array([[focal, 0.0, center_x], [0.0, focal, center_y], [0.0, 0.0, 1.0]])
