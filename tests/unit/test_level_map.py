from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.support import cpu_pipeline
from thermal_board.geometry.board_model import BoardModel
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource
from thermal_board.simulation.thermal_board_renderer import CellPattern, checkerboard_levels
from thermal_board.viewer.level_map_renderer import (
    COLOR_INVALID,
    HEADER_HEIGHT_PX,
    LevelMapLayout,
    cell_number_at,
    render_level_map,
)
from thermal_board.viewer.measurement_session import MeasurementSession

if TYPE_CHECKING:
    from thermal_board.config.system_settings import SystemConfiguration
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement


class RecordingLevelDisplay:
    def __init__(self) -> None:
        self.shown: list[int] = []
        self.closed = False

    def show(self, measurement: FrameMeasurement) -> None:
        self.shown.append(measurement.frame_index)

    def close(self) -> None:
        self.closed = True


class AcceptingDisplay:
    def show(self, canvas: np.ndarray) -> bool:
        return canvas.size > 0

    def close(self) -> None:
        return None


class DiscardingReports:
    def write(self, report: object) -> None:
        self.last = report


@pytest.fixture(scope="module")
def model(small_configuration: SystemConfiguration) -> BoardModel:
    return BoardModel(small_configuration.board)


@pytest.fixture(scope="module")
def measurement(small_configuration: SystemConfiguration, model: BoardModel) -> FrameMeasurement:
    levels = checkerboard_levels(model, small_configuration.simulation)
    levels[0], levels[2] = 10800.0, 9200.0
    plan = SequencePlan(1, 0.0, cell_pattern=CellPattern(levels=levels))
    frame = next(iter(SyntheticSequenceSource(small_configuration, plan).frames()))
    return cpu_pipeline(small_configuration).process(frame)


def cell_pixel(model: BoardModel, layout: LevelMapLayout, number: int) -> tuple[int, int]:
    row, column = (int(value) for value in model.cell_rows_and_columns[number])
    x_px, y_px = layout.origin_px(row, column)
    return y_px + layout.cell_px - 6, x_px + layout.cell_px - 6


def test_when_a_cell_is_hotter_then_it_is_drawn_brighter_on_the_level_map(
    measurement: FrameMeasurement, model: BoardModel
) -> None:
    layout = LevelMapLayout.for_model(model)
    canvas = render_level_map(measurement, model, layout)
    assert canvas.shape[:2] == layout.size_px
    assert int(canvas[cell_pixel(model, layout, 0)][0]) > int(
        canvas[cell_pixel(model, layout, 2)][0]
    )


def test_when_a_cell_is_not_measured_then_it_is_drawn_as_invalid(
    measurement: FrameMeasurement, model: BoardModel
) -> None:
    valid = measurement.cells.valid.copy()
    valid[5] = False
    cells = dataclasses.replace(measurement.cells, valid=valid)
    unmeasured = dataclasses.replace(measurement, cells=cells)
    layout = LevelMapLayout.for_model(model)
    canvas = render_level_map(unmeasured, model, layout)
    assert tuple(int(value) for value in canvas[cell_pixel(model, layout, 5)]) == COLOR_INVALID


def test_when_the_pointer_is_over_a_cell_then_its_number_is_returned(model: BoardModel) -> None:
    layout = LevelMapLayout.for_model(model)
    y_px, x_px = cell_pixel(model, layout, 13)
    assert cell_number_at(model, layout, x_px, y_px) == 13
    assert cell_number_at(model, layout, 5, HEADER_HEIGHT_PX - 1) is None
    assert cell_number_at(model, layout, layout.size_px[1] + 5, y_px) is None


def test_when_a_cell_is_hovered_then_the_map_shows_its_readout(
    measurement: FrameMeasurement, model: BoardModel
) -> None:
    layout = LevelMapLayout.for_model(model)
    plain = render_level_map(measurement, model, layout)
    hovered = render_level_map(measurement, model, layout, hovered=0)
    assert not np.array_equal(plain[:HEADER_HEIGHT_PX], hovered[:HEADER_HEIGHT_PX])


def test_when_a_level_display_is_attached_then_every_measured_frame_reaches_it(
    small_configuration: SystemConfiguration,
) -> None:
    levels = RecordingLevelDisplay()
    pipeline = cpu_pipeline(small_configuration)
    session = MeasurementSession(pipeline, AcceptingDisplay(), DiscardingReports(), levels)
    session.run(SyntheticSequenceSource(small_configuration, SequencePlan(2, 0.2)))
    assert levels.shown == [0, 1]
    assert levels.closed
