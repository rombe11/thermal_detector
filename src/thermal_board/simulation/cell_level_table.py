from __future__ import annotations

import csv
from typing import TYPE_CHECKING

from thermal_board.simulation.thermal_board_renderer import checkerboard_levels

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import SimulationSettings
    from thermal_board.geometry.board_model import BoardModel

CELL_TABLE_HEADER = ("cell", "row", "column", "center_x_mm", "center_y_mm", "level")


class CellLevelTableError(ValueError):
    pass


def parse_level_rows(text: str) -> list[tuple[int, float]]:
    rows = [line.split(",") for line in text.splitlines() if line.strip()]
    numeric_rows = [row for row in rows if row[0].strip().isdigit()]
    return [(int(row[0]), float(row[-1])) for row in numeric_rows]


def read_cell_levels(
    path: Path, model: BoardModel, settings: SimulationSettings
) -> NDArray[np.float64]:
    levels = checkerboard_levels(model, settings).astype(float)
    for number, level in parse_level_rows(path.read_text(encoding="utf-8")):
        if not 0 <= number < model.cell_count:
            raise CellLevelTableError(f"cell {number} is outside 0..{model.cell_count - 1}")
        levels[number] = level
    return levels


def cell_table_rows(model: BoardModel, settings: SimulationSettings) -> list[list[object]]:
    levels = checkerboard_levels(model, settings)
    positions = zip(
        model.cell_rows_and_columns.tolist(), model.cell_centers_mm.tolist(), strict=True
    )
    return [
        [number, row, column, round(x_mm, 3), round(y_mm, 3), float(levels[number])]
        for number, ((row, column), (x_mm, y_mm)) in enumerate(positions)
    ]


def write_cell_table(path: Path, model: BoardModel, settings: SimulationSettings) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CELL_TABLE_HEADER)
        writer.writerows(cell_table_rows(model, settings))
