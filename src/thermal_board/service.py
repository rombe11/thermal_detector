from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Self

from thermal_board.acquisition.thermal_frame import ThermalFrame
from thermal_board.config.configuration_loader import load_configuration
from thermal_board.pipeline.board_measurement_pipeline import BoardNotFoundError
from thermal_board.pipeline.pipeline_factory import build_measurement_pipeline
from thermal_board.viewer.level_map_renderer import (
    LevelMapLayout,
    cell_number_at,
    render_level_map,
)
from thermal_board.viewer.overlay_renderer import render_lost_frame, render_measurement_overlay

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from thermal_board.compute.array_backend import ComputeBackend
    from thermal_board.config.system_settings import SystemConfiguration
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement


class BoardMetrologyService:
    def __init__(
        self, configuration: SystemConfiguration, backend: ComputeBackend | None = None
    ) -> None:
        self.configuration = configuration
        self.pipeline = build_measurement_pipeline(configuration, backend)
        self.frame_numbers = itertools.count()

    @classmethod
    def from_configuration_file(cls, path: Path) -> Self:
        return cls(load_configuration(path))

    @property
    def cell_count(self) -> int:
        return self.pipeline.components.model.cell_count

    def measure(self, pixels: NDArray[Any], timestamp_s: float = 0.0) -> FrameMeasurement | None:
        frame = ThermalFrame(next(self.frame_numbers), timestamp_s, pixels)
        try:
            return self.pipeline.process(frame)
        except BoardNotFoundError:
            return None

    def annotate(self, measurement: FrameMeasurement) -> NDArray[np.uint8]:
        model, accuracy = self.pipeline.components.model, self.configuration.accuracy
        return render_measurement_overlay(measurement, model, accuracy)

    @property
    def level_map_layout(self) -> LevelMapLayout:
        return LevelMapLayout.for_model(self.pipeline.components.model)

    def render_level_map(
        self, measurement: FrameMeasurement, hovered_cell: int | None = None
    ) -> NDArray[np.uint8]:
        model = self.pipeline.components.model
        return render_level_map(measurement, model, self.level_map_layout, hovered_cell)

    def cell_at_level_map(self, x_px: int, y_px: int) -> int | None:
        return cell_number_at(self.pipeline.components.model, self.level_map_layout, x_px, y_px)

    def annotate_lost(
        self, pixels: NDArray[Any], reason: str = "board not visible"
    ) -> NDArray[np.uint8]:
        normalized = self.pipeline.components.normalizer.normalize(pixels)
        return render_lost_frame(normalized, reason)
