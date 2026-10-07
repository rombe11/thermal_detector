from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import BoardGeometry

CORNER_UNIT_OFFSETS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
BOTTOM_TO_TOP = "bottom_to_top"
RIGHT_TO_LEFT = "right_to_left"


def ordered_axis(count: int, reverse: bool) -> NDArray[np.int64]:
    indices = np.arange(count, dtype=np.int64)
    return indices[::-1] if reverse else indices


class BoardModel:
    def __init__(self, geometry: BoardGeometry) -> None:
        self.geometry = geometry

    @cached_property
    def cell_rows_and_columns(self) -> NDArray[np.int64]:
        geometry = self.geometry
        rows = ordered_axis(geometry.rows, geometry.numbering_rows == BOTTOM_TO_TOP)
        columns = ordered_axis(geometry.columns, geometry.numbering_columns == RIGHT_TO_LEFT)
        row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")
        return np.stack([row_grid.ravel(), column_grid.ravel()], axis=1)

    @cached_property
    def cell_number_grid(self) -> NDArray[np.int64]:
        grid = np.empty((self.geometry.rows, self.geometry.columns), dtype=np.int64)
        rows, columns = self.cell_rows_and_columns.T
        grid[rows, columns] = np.arange(self.cell_count, dtype=np.int64)
        return grid

    @cached_property
    def cell_origins_mm(self) -> NDArray[np.float64]:
        rows_and_columns = self.cell_rows_and_columns.astype(np.float64)
        origin_x = self.geometry.border_x_mm + rows_and_columns[:, 1] * self.geometry.pitch_x_mm
        origin_y = self.geometry.border_y_mm + rows_and_columns[:, 0] * self.geometry.pitch_y_mm
        return np.stack([origin_x, origin_y], axis=1)

    @cached_property
    def cell_size_mm(self) -> NDArray[np.float64]:
        return np.array([self.geometry.cell_width_mm, self.geometry.cell_height_mm])

    @cached_property
    def cell_corners_mm(self) -> NDArray[np.float64]:
        corner_offsets = CORNER_UNIT_OFFSETS * self.cell_size_mm
        return self.cell_origins_mm[:, np.newaxis, :] + corner_offsets[np.newaxis, :, :]

    @cached_property
    def cell_centers_mm(self) -> NDArray[np.float64]:
        return self.cell_origins_mm + self.cell_size_mm / 2.0

    @cached_property
    def hot_cell_mask(self) -> NDArray[np.bool_]:
        return np.asarray(self.cell_rows_and_columns.sum(axis=1) % 2 == 0)

    @cached_property
    def outer_corner_cell_indices(self) -> NDArray[np.int64]:
        last_row, last_column = self.geometry.rows - 1, self.geometry.columns - 1
        rows, columns = [0, 0, last_row, last_row], [0, last_column, last_column, 0]
        return self.cell_number_grid[rows, columns]

    @property
    def cell_count(self) -> int:
        return self.geometry.cell_count

    @property
    def board_center_mm(self) -> NDArray[np.float64]:
        return np.array([self.geometry.board_width_mm, self.geometry.board_height_mm]) / 2.0
