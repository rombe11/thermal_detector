from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.support import CONFIGURATION_PATH
from thermal_board.config.configuration_loader import build_system_configuration
from thermal_board.config.configuration_validation import (
    ConfigurationError,
    validate_board_geometry,
    validate_camera_specification,
)

if TYPE_CHECKING:
    from thermal_board.config.system_settings import SystemConfiguration


def repository_document() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(CONFIGURATION_PATH.read_text(encoding="utf-8"))
    return document


def test_when_repository_config_is_loaded_then_board_has_6144_cells_of_50_mm(
    configuration: SystemConfiguration,
) -> None:
    board = configuration.board
    assert board.cell_count == 6144
    assert (board.cell_width_mm, board.cell_height_mm) == (50.0, 50.0)
    assert board.border_x_mm >= 0
    assert board.border_y_mm >= 0


def test_when_section_is_missing_then_configuration_is_rejected() -> None:
    document = repository_document()
    del document["camera"]
    with pytest.raises(ConfigurationError, match="camera"):
        build_system_configuration(document)


def test_when_parameter_is_unknown_then_configuration_is_rejected() -> None:
    document = repository_document()
    document["board"]["unexpected_parameter"] = 1
    with pytest.raises(ConfigurationError, match="BoardGeometry"):
        build_system_configuration(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width_px", 640),
        ("height_px", 512),
        ("frame_rate_hz", 25.0),
        ("sensor_type", "MWIR"),
        ("spectral_band_um", (3.0, 5.0)),
    ],
)
def test_when_camera_is_below_specification_then_it_is_rejected(
    configuration: SystemConfiguration, field: str, value: object
) -> None:
    camera = dataclasses.replace(configuration.camera, **{field: value})
    with pytest.raises(ConfigurationError):
        validate_camera_specification(camera)


@pytest.mark.parametrize(("field", "value"), [("columns", 200), ("rows", 200), ("gap_mm", 0.0)])
def test_when_cell_grid_does_not_fit_board_then_it_is_rejected(
    configuration: SystemConfiguration, field: str, value: object
) -> None:
    board = dataclasses.replace(configuration.board, **{field: value})
    with pytest.raises(ConfigurationError):
        validate_board_geometry(board)
