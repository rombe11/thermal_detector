from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import erf

from thermal_board.geometry.planar_homography import project_points

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import (
        BoardGeometry,
        CameraSpecification,
        SimulationSettings,
    )


def blurred_interval(
    values: NDArray[np.float64], start: Any, end: Any, sigma: float
) -> NDArray[np.float64]:
    if sigma <= 0:
        return np.asarray((values >= start) & (values < end), dtype=np.float64)
    scale = 1.0 / (np.sqrt(2.0) * sigma)
    return np.asarray(0.5 * (erf((values - start) * scale) - erf((values - end) * scale)))


class CheckerboardRadianceField:
    def __init__(
        self, geometry: BoardGeometry, settings: SimulationSettings, blur_sigma_mm: float
    ) -> None:
        self.geometry = geometry
        self.settings = settings
        self.blur_sigma_mm = blur_sigma_mm

    def nearest_cell_indices(
        self, board_x_mm: NDArray[np.float64], board_y_mm: NDArray[np.float64]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        geometry = self.geometry
        grid_x = board_x_mm - geometry.border_x_mm - geometry.cell_width_mm / 2.0
        grid_y = board_y_mm - geometry.border_y_mm - geometry.cell_height_mm / 2.0
        columns = np.clip(np.rint(grid_x / geometry.pitch_x_mm), 0, geometry.columns - 1)
        rows = np.clip(np.rint(grid_y / geometry.pitch_y_mm), 0, geometry.rows - 1)
        return rows.astype(np.int64), columns.astype(np.int64)

    def cell_origins(
        self,
        rows: NDArray[np.int64],
        columns: NDArray[np.int64],
        offsets_mm: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        origin_x = np.asarray(
            self.geometry.border_x_mm + columns * self.geometry.pitch_x_mm, np.float64
        )
        origin_y = np.asarray(
            self.geometry.border_y_mm + rows * self.geometry.pitch_y_mm, np.float64
        )
        if offsets_mm is None:
            return origin_x, origin_y
        return origin_x + offsets_mm[rows, columns, 0], origin_y + offsets_mm[rows, columns, 1]

    def cell_coverage(
        self,
        board_x_mm: NDArray[np.float64],
        board_y_mm: NDArray[np.float64],
        offsets_mm: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        rows, columns = self.nearest_cell_indices(board_x_mm, board_y_mm)
        origin_x, origin_y = self.cell_origins(rows, columns, offsets_mm)
        width, height, sigma = (
            self.geometry.cell_width_mm,
            self.geometry.cell_height_mm,
            self.blur_sigma_mm,
        )
        coverage_x = blurred_interval(board_x_mm, origin_x, origin_x + width, sigma)
        coverage = coverage_x * blurred_interval(board_y_mm, origin_y, origin_y + height, sigma)
        return coverage, np.asarray((rows + columns) % 2 == 0)

    def board_coverage(
        self, board_x_mm: NDArray[np.float64], board_y_mm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        sigma = self.blur_sigma_mm
        coverage_x = blurred_interval(board_x_mm, 0.0, self.geometry.board_width_mm, sigma)
        return coverage_x * blurred_interval(board_y_mm, 0.0, self.geometry.board_height_mm, sigma)

    def radiance(
        self, board_points_mm: NDArray[np.float64], offsets_mm: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        board_x, board_y = board_points_mm[..., 0], board_points_mm[..., 1]
        cell_coverage, is_hot = self.cell_coverage(board_x, board_y, offsets_mm)
        levels = self.settings
        cell_level = np.where(is_hot, levels.hot_level, levels.cold_level)
        board_level = levels.substrate_level + (cell_level - levels.substrate_level) * cell_coverage
        board_coverage = self.board_coverage(board_x, board_y)
        return levels.background_level + (board_level - levels.background_level) * board_coverage


class ThermalBoardRenderer:
    def __init__(
        self,
        geometry: BoardGeometry,
        settings: SimulationSettings,
        camera: CameraSpecification,
        seed: int = 0,
    ) -> None:
        blur_sigma_mm = settings.optical_blur_sigma_px * camera.ground_sample_distance_mm
        self.field = CheckerboardRadianceField(geometry, settings, blur_sigma_mm)
        self.settings = settings
        self.camera = camera
        self.random_generator = np.random.default_rng(seed)

    def pixel_grid(self) -> NDArray[np.float64]:
        rows, columns = np.indices((self.camera.height_px, self.camera.width_px), dtype=np.float64)
        return np.stack([columns, rows], axis=-1)

    def subpixel_offsets(self) -> list[tuple[float, float]]:
        count = self.settings.supersampling
        steps = [(index + 0.5) / count - 0.5 for index in range(count)]
        return [(step_x, step_y) for step_y in steps for step_x in steps]

    def render_noise_free(
        self, homography: NDArray[np.float64], offsets_mm: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        inverse = np.linalg.inv(homography)
        pixels = self.pixel_grid()
        accumulated = np.zeros(pixels.shape[:2])
        for offset in self.subpixel_offsets():
            board_points = project_points(inverse, pixels + np.asarray(offset))
            accumulated += self.field.radiance(board_points, offsets_mm)
        return accumulated / len(self.subpixel_offsets())

    def render(
        self, homography: NDArray[np.float64], offsets_mm: NDArray[np.float64] | None = None
    ) -> NDArray[np.uint16]:
        radiance = self.render_noise_free(homography, offsets_mm)
        return self.digitize(radiance)

    def digitize(self, radiance: NDArray[np.float64]) -> NDArray[np.uint16]:
        noise = self.random_generator.normal(0.0, self.settings.noise_sigma, radiance.shape)
        maximum_count = 2**self.camera.bit_depth - 1
        return np.clip(np.rint(radiance + noise), 0, maximum_count).astype(np.uint16)
