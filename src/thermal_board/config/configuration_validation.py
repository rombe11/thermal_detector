from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thermal_board.config.system_settings import (
        BoardGeometry,
        CameraSpecification,
        SystemConfiguration,
    )

MINIMUM_SENSOR_WIDTH_PX = 1280
MINIMUM_SENSOR_HEIGHT_PX = 1024
FRAME_RATE_RANGE_HZ = (30.0, 60.0)
SUPPORTED_SENSOR_TYPE = "LWIR"
LWIR_BAND_LIMITS_UM = (7.0, 15.0)
MINIMUM_CELLS_PER_AXIS = 2
MINIMUM_PROFILES_PER_EDGE = 2


class ConfigurationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def validate_board_geometry(board: BoardGeometry) -> None:
    dimensions = (board.cell_width_mm, board.cell_height_mm, board.board_width_mm)
    require(all(value > 0 for value in dimensions), "board dimensions must be positive")
    require(board.gap_mm > 0, "gap between cells must be positive")
    require(min(board.columns, board.rows) >= MINIMUM_CELLS_PER_AXIS, "grid too small")
    require(board.border_x_mm >= 0, "cell grid is wider than the board")
    require(board.border_y_mm >= 0, "cell grid is taller than the board")


def validate_camera_specification(camera: CameraSpecification) -> None:
    require(camera.sensor_type == SUPPORTED_SENSOR_TYPE, "camera must be an LWIR sensor")
    require(camera.width_px >= MINIMUM_SENSOR_WIDTH_PX, "sensor width below SXGA")
    require(camera.height_px >= MINIMUM_SENSOR_HEIGHT_PX, "sensor height below SXGA")
    lowest_rate, highest_rate = FRAME_RATE_RANGE_HZ
    require(lowest_rate <= camera.frame_rate_hz <= highest_rate, "frame rate outside 30-60 Hz")
    band_start, band_end = camera.spectral_band_um
    lowest_band, highest_band = LWIR_BAND_LIMITS_UM
    require(lowest_band <= band_start < band_end <= highest_band, "band is not long-wave IR")
    require(camera.focal_length_mm > 0 and camera.pixel_pitch_um > 0, "optics must be positive")


def validate_system_configuration(configuration: SystemConfiguration) -> None:
    validate_board_geometry(configuration.board)
    validate_camera_specification(configuration.camera)
    smoothing = configuration.tracking.smoothing_factor
    require(0.0 < smoothing <= 1.0, "smoothing factor must be in (0, 1]")
    require(
        configuration.refinement.profiles_per_edge >= MINIMUM_PROFILES_PER_EDGE, "too few profiles"
    )
    require(configuration.simulation.supersampling >= 1, "supersampling must be at least 1")
