from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import blank_frame
from thermal_board.compute.array_backend import cpu_backend
from thermal_board.service import BoardMetrologyService
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource
from thermal_board.viewer.overlay_renderer import PANEL_WIDTH

if TYPE_CHECKING:
    from thermal_board.config.system_settings import SystemConfiguration


@pytest.fixture
def service(small_configuration: SystemConfiguration) -> BoardMetrologyService:
    return BoardMetrologyService(small_configuration, cpu_backend())


def test_when_frame_is_measured_through_the_service_then_cell_results_are_indexed_by_number(
    service: BoardMetrologyService, small_configuration: SystemConfiguration
) -> None:
    frame = next(iter(SyntheticSequenceSource(small_configuration, SequencePlan(1, 0.0)).frames()))
    measurement = service.measure(frame.pixels)
    assert measurement is not None
    assert measurement.cells.center_deviation_mm.shape == (service.cell_count, 2)
    assert measurement.report.meets_precision_target
    assert (
        service.annotate(measurement).shape[1] == small_configuration.camera.width_px + PANEL_WIDTH
    )


def test_when_board_is_not_visible_then_the_service_returns_nothing(
    service: BoardMetrologyService, small_configuration: SystemConfiguration
) -> None:
    pixels = blank_frame(small_configuration).pixels
    assert service.measure(pixels) is None
    assert (
        service.annotate_lost(pixels).shape[1] == small_configuration.camera.width_px + PANEL_WIDTH
    )


def test_when_frames_are_measured_then_frame_numbers_increase(
    service: BoardMetrologyService, small_configuration: SystemConfiguration
) -> None:
    frames = list(SyntheticSequenceSource(small_configuration, SequencePlan(2, 0.0)).frames())
    indices = [service.measure(frame.pixels) for frame in frames]
    assert [measurement.frame_index for measurement in indices if measurement] == [0, 1]


def test_when_a_level_map_is_requested_then_its_cells_can_be_looked_up_by_pixel(
    service: BoardMetrologyService, small_configuration: SystemConfiguration
) -> None:
    frame = next(iter(SyntheticSequenceSource(small_configuration, SequencePlan(1, 0.0)).frames()))
    measurement = service.measure(frame.pixels)
    assert measurement is not None
    assert service.render_level_map(measurement).shape[:2] == service.level_map_layout.size_px
    assert service.cell_at_level_map(2, service.level_map_layout.size_px[0] - 2) is not None
