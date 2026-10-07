from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import erf

from thermal_board.geometry.board_model import BoardModel
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


@dataclass(frozen=True, slots=True)
class CellPattern:
    levels: NDArray[np.float64] | None = None
    offsets_mm: NDArray[np.float64] | None = None


def checkerboard_levels(model: BoardModel, settings: SimulationSettings) -> NDArray[np.float64]:
    return np.where(model.hot_cell_mask, settings.hot_level, settings.cold_level)


class CheckerboardRadianceField:
    def __init__(
        self, geometry: BoardGeometry, settings: SimulationSettings, blur_sigma_mm: float
    ) -> None:
        self.geometry = geometry
        self.settings = settings
        self.blur_sigma_mm = blur_sigma_mm
        self.model = BoardModel(geometry)

    def nearest_cell_numbers(
        self, board_x_mm: NDArray[np.float64], board_y_mm: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        geometry = self.geometry
        grid_x = board_x_mm - geometry.border_x_mm - geometry.cell_width_mm / 2.0
        grid_y = board_y_mm - geometry.border_y_mm - geometry.cell_height_mm / 2.0
        columns = np.clip(np.rint(grid_x / geometry.pitch_x_mm), 0, geometry.columns - 1)
        rows = np.clip(np.rint(grid_y / geometry.pitch_y_mm), 0, geometry.rows - 1)
        return self.model.cell_number_grid[rows.astype(np.int64), columns.astype(np.int64)]

    def cell_origins(self, numbers: NDArray[np.int64], pattern: CellPattern) -> NDArray[np.float64]:
        origins = self.model.cell_origins_mm[numbers]
        if pattern.offsets_mm is None:
            return origins
        return origins + pattern.offsets_mm[numbers]

    def cell_levels(self, numbers: NDArray[np.int64], pattern: CellPattern) -> NDArray[np.float64]:
        levels = pattern.levels
        if levels is None:
            levels = checkerboard_levels(self.model, self.settings)
        return np.asarray(levels, dtype=np.float64)[numbers]

    def cell_coverage(
        self, board_x_mm: NDArray[np.float64], board_y_mm: NDArray[np.float64], pattern: CellPattern
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        numbers = self.nearest_cell_numbers(board_x_mm, board_y_mm)
        origins, sigma = self.cell_origins(numbers, pattern), self.blur_sigma_mm
        origin_x, origin_y = origins[..., 0], origins[..., 1]
        width, height = self.geometry.cell_width_mm, self.geometry.cell_height_mm
        coverage_x = blurred_interval(board_x_mm, origin_x, origin_x + width, sigma)
        coverage = coverage_x * blurred_interval(board_y_mm, origin_y, origin_y + height, sigma)
        return coverage, self.cell_levels(numbers, pattern)

    def board_coverage(
        self, board_x_mm: NDArray[np.float64], board_y_mm: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        sigma = self.blur_sigma_mm
        coverage_x = blurred_interval(board_x_mm, 0.0, self.geometry.board_width_mm, sigma)
        return coverage_x * blurred_interval(board_y_mm, 0.0, self.geometry.board_height_mm, sigma)

    def radiance(
        self, board_points_mm: NDArray[np.float64], pattern: CellPattern | None = None
    ) -> NDArray[np.float64]:
        board_x, board_y = board_points_mm[..., 0], board_points_mm[..., 1]
        cell_coverage, cell_level = self.cell_coverage(board_x, board_y, pattern or CellPattern())
        substrate, background = self.settings.substrate_level, self.settings.background_level
        board_level = substrate + (cell_level - substrate) * cell_coverage
        board_coverage = self.board_coverage(board_x, board_y)
        return background + (board_level - background) * board_coverage


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
        self, homography: NDArray[np.float64], pattern: CellPattern | None = None
    ) -> NDArray[np.float64]:
        inverse = np.linalg.inv(homography)
        pixels = self.pixel_grid()
        accumulated = np.zeros(pixels.shape[:2])
        for offset in self.subpixel_offsets():
            board_points = project_points(inverse, pixels + np.asarray(offset))
            accumulated += self.field.radiance(board_points, pattern)
        return accumulated / len(self.subpixel_offsets())

    def render(
        self, homography: NDArray[np.float64], pattern: CellPattern | None = None
    ) -> NDArray[np.uint16]:
        radiance = self.render_noise_free(homography, pattern)
        return self.digitize(radiance)

    def digitize(self, radiance: NDArray[np.float64]) -> NDArray[np.uint16]:
        noise = self.random_generator.normal(0.0, self.settings.noise_sigma, radiance.shape)
        maximum_count = 2**self.camera.bit_depth - 1
        return np.clip(np.rint(radiance + noise), 0, maximum_count).astype(np.uint16)
