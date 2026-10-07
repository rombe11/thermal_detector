from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cv2

from thermal_board.viewer.level_map_renderer import LevelMapLayout, cell_number_at, render_level_map

if TYPE_CHECKING:
    from thermal_board.geometry.board_model import BoardModel
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement

LEVEL_WINDOW_NAME = "Cell gray levels"


class LevelDisplay(Protocol):
    def show(self, measurement: FrameMeasurement) -> None: ...

    def close(self) -> None: ...


class CellLevelWindow:
    def __init__(self, model: BoardModel, window_name: str = LEVEL_WINDOW_NAME) -> None:
        self.model = model
        self.layout = LevelMapLayout.for_model(model)
        self.window_name = window_name
        self.hovered_cell: int | None = None
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.on_mouse)

    def on_mouse(self, event: int, x_px: int, y_px: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_MOUSEMOVE:
            self.hovered_cell = cell_number_at(self.model, self.layout, x_px, y_px)

    def show(self, measurement: FrameMeasurement) -> None:
        canvas = render_level_map(measurement, self.model, self.layout, self.hovered_cell)
        cv2.imshow(self.window_name, canvas)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
