from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from thermal_board.geometry.board_model import BoardModel
from thermal_board.simulation.cell_level_table import CellLevelTableError, read_cell_levels
from thermal_board.simulation.thermal_board_renderer import checkerboard_levels

if TYPE_CHECKING:
    from pathlib import Path

    from thermal_board.config.system_settings import SystemConfiguration


def write_table(directory: Path, text: str) -> Path:
    path = directory / "levels.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_when_some_cells_are_listed_then_only_those_cells_change(
    tmp_path: Path, small_configuration: SystemConfiguration
) -> None:
    model, simulation = BoardModel(small_configuration.board), small_configuration.simulation
    levels = read_cell_levels(
        write_table(tmp_path, "cell,level\n3,9100\n7,6400\n"), model, simulation
    )
    defaults = checkerboard_levels(model, simulation)
    changed = [number for number in range(model.cell_count) if levels[number] != defaults[number]]
    assert changed == [3, 7]
    assert (levels[3], levels[7]) == (9100.0, 6400.0)


def test_when_a_cell_number_is_out_of_range_then_the_table_is_rejected(
    tmp_path: Path, small_configuration: SystemConfiguration
) -> None:
    model = BoardModel(small_configuration.board)
    with pytest.raises(CellLevelTableError, match="outside"):
        read_cell_levels(write_table(tmp_path, "500,9000\n"), model, small_configuration.simulation)
