from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from thermal_board.compute.array_backend import OpenCvBilinearSampler
from thermal_board.geometry.planar_homography import project_points

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.geometry.board_model import BoardModel
    from thermal_board.geometry.pinhole_camera import PinholeCameraModel

INTERIOR_SAMPLE_FRACTIONS = np.array([0.3, 0.5, 0.7])


def cell_interior_points_mm(model: BoardModel) -> NDArray[np.float64]:
    fraction_x, fraction_y = np.meshgrid(INTERIOR_SAMPLE_FRACTIONS, INTERIOR_SAMPLE_FRACTIONS)
    unit_points = np.stack([fraction_x.ravel(), fraction_y.ravel()], axis=1)
    interior = unit_points[np.newaxis, :, :] * model.cell_size_mm
    return model.cell_origins_mm[:, np.newaxis, :] + interior


def measure_cell_levels(
    raw_pixels: NDArray[Any],
    homography: NDArray[np.float64],
    model: BoardModel,
    camera: PinholeCameraModel,
) -> NDArray[np.float64]:
    image_points = camera.distort(project_points(homography, cell_interior_points_mm(model)))
    image = np.asarray(raw_pixels, dtype=np.float32)
    samples = OpenCvBilinearSampler().sample(image, image_points[..., 0], image_points[..., 1])
    return np.asarray(samples, dtype=np.float64).mean(axis=1)
