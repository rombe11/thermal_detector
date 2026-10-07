from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import BoardGeometry

CORNER_UNIT_OFFSETS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


class BoardModel:
    def __init__(self, geometry: BoardGeometry) -> None:
        self.geometry = geometry

    @cached_property
    def cell_rows_and_columns(self) -> NDArray[np.int64]:
        rows, columns = np.indices((self.geometry.rows, self.geometry.columns))
        return np.stack([rows.ravel(), columns.ravel()], axis=1).astype(np.int64)

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
        flat_index = np.array([0, last_column, last_row * self.geometry.columns + last_column])
        bottom_left = last_row * self.geometry.columns
        return np.append(flat_index, bottom_left).astype(np.int64)

    @property
    def cell_count(self) -> int:
        return self.geometry.cell_count

    @property
    def board_center_mm(self) -> NDArray[np.float64]:
        return np.array([self.geometry.board_width_mm, self.geometry.board_height_mm]) / 2.0
