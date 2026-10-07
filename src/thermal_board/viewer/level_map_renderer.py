from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.geometry.board_model import BoardModel
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement

MAXIMUM_MAP_WIDTH_PX = 1280
MINIMUM_CELL_PX = 4
MINIMUM_LABELLED_CELL_PX = 56
HEADER_HEIGHT_PX = 56
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_BACKGROUND = (34, 30, 28)
COLOR_GRID = (60, 55, 52)
COLOR_TEXT = (245, 245, 245)
COLOR_MUTED = (165, 160, 155)
COLOR_INVALID = (60, 60, 150)
COLOR_HOVER = (40, 200, 240)
COLOR_DARK_TEXT = (20, 20, 20)
BRIGHT_CELL_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class LevelMapLayout:
    rows: int
    columns: int
    cell_px: int

    @classmethod
    def for_model(cls, model: BoardModel, maximum_width_px: int = MAXIMUM_MAP_WIDTH_PX) -> Self:
        columns = model.geometry.columns
        cell_px = max(MINIMUM_CELL_PX, maximum_width_px // columns)
        return cls(model.geometry.rows, columns, cell_px)

    @property
    def size_px(self) -> tuple[int, int]:
        return HEADER_HEIGHT_PX + self.rows * self.cell_px, self.columns * self.cell_px

    def origin_px(self, row: int, column: int) -> tuple[int, int]:
        return column * self.cell_px, HEADER_HEIGHT_PX + row * self.cell_px

    def grid_position_at(self, x_px: int, y_px: int) -> tuple[int, int] | None:
        row, column = (y_px - HEADER_HEIGHT_PX) // self.cell_px, x_px // self.cell_px
        inside = y_px >= HEADER_HEIGHT_PX and 0 <= row < self.rows and 0 <= column < self.columns
        return (row, column) if inside else None


def cell_number_at(model: BoardModel, layout: LevelMapLayout, x_px: int, y_px: int) -> int | None:
    position = layout.grid_position_at(x_px, y_px)
    return None if position is None else int(model.cell_number_grid[position])


def level_grid(measurement: FrameMeasurement, model: BoardModel) -> NDArray[np.float64]:
    grid = np.full((model.geometry.rows, model.geometry.columns), np.nan)
    levels = measurement.cells.measured_levels
    if levels is not None:
        rows, columns = model.cell_rows_and_columns.T
        grid[rows, columns] = np.where(measurement.cells.valid, levels, np.nan)
    return grid


def level_range(grid: NDArray[np.float64]) -> tuple[float, float]:
    if np.isnan(grid).all():
        return 0.0, 1.0
    low, high = float(np.nanmin(grid)), float(np.nanmax(grid))
    return low, max(high, low + 1e-9)


def shaded_cells(grid: NDArray[np.float64], layout: LevelMapLayout) -> NDArray[np.uint8]:
    low, high = level_range(grid)
    shades = np.nan_to_num((grid - low) / (high - low) * 255.0).astype(np.uint8)
    colored = cv2.cvtColor(shades, cv2.COLOR_GRAY2BGR)
    colored[np.isnan(grid)] = COLOR_INVALID
    enlarged = np.kron(colored, np.ones((layout.cell_px, layout.cell_px, 1), dtype=np.uint8))
    enlarged[:: layout.cell_px, :], enlarged[:, :: layout.cell_px] = COLOR_GRID, COLOR_GRID
    return np.asarray(enlarged, dtype=np.uint8)


def draw_header(canvas: NDArray[np.uint8], grid: NDArray[np.float64]) -> None:
    low, high = level_range(grid)
    cv2.putText(canvas, "CELL GRAY LEVELS", (14, 24), FONT, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
    summary = f"min {low:.0f}   max {high:.0f}   hover a cell for its value"
    cv2.putText(canvas, summary, (14, 46), FONT, 0.45, COLOR_MUTED, 1, cv2.LINE_AA)


def contrasting_text_color(value: float, grid: NDArray[np.float64]) -> tuple[int, int, int]:
    low, high = level_range(grid)
    return COLOR_DARK_TEXT if (value - low) / (high - low) > BRIGHT_CELL_FRACTION else COLOR_TEXT


def draw_cell_labels(
    canvas: NDArray[np.uint8], grid: NDArray[np.float64], model: BoardModel, layout: LevelMapLayout
) -> None:
    for number, (row, column) in enumerate(model.cell_rows_and_columns.tolist()):
        value, (x_px, y_px) = grid[row, column], layout.origin_px(row, column)
        color = COLOR_TEXT if np.isnan(value) else contrasting_text_color(float(value), grid)
        text = "--" if np.isnan(value) else f"{value:.0f}"
        cv2.putText(canvas, f"#{number}", (x_px + 4, y_px + 16), FONT, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (x_px + 4, y_px + 34), FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_hover(
    canvas: NDArray[np.uint8],
    grid: NDArray[np.float64],
    model: BoardModel,
    layout: LevelMapLayout,
    number: int,
) -> None:
    row, column = (int(value) for value in model.cell_rows_and_columns[number])
    x_px, y_px = layout.origin_px(row, column)
    end = (x_px + layout.cell_px, y_px + layout.cell_px)
    cv2.rectangle(canvas, (x_px, y_px), end, COLOR_HOVER, 2)
    value = grid[row, column]
    reading = "not measured" if np.isnan(value) else f"level {value:.1f}"
    tooltip = f"cell #{number}   {reading}   (row {row}, column {column})"
    cv2.putText(
        canvas, tooltip, (canvas.shape[1] - 470, 24), FONT, 0.5, COLOR_HOVER, 1, cv2.LINE_AA
    )


def render_level_map(
    measurement: FrameMeasurement,
    model: BoardModel,
    layout: LevelMapLayout,
    hovered: int | None = None,
) -> NDArray[np.uint8]:
    grid = level_grid(measurement, model)
    canvas = np.full((*layout.size_px, 3), COLOR_BACKGROUND, dtype=np.uint8)
    canvas[HEADER_HEIGHT_PX:] = shaded_cells(grid, layout)
    draw_header(canvas, grid)
    if layout.cell_px >= MINIMUM_LABELLED_CELL_PX:
        draw_cell_labels(canvas, grid, model, layout)
    if hovered is not None:
        draw_hover(canvas, grid, model, layout, hovered)
    return canvas
