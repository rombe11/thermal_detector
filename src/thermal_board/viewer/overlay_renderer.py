from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from thermal_board.reporting.report_sinks import summary_fields

if TYPE_CHECKING:
    from typing import TypeAlias

    from numpy.typing import NDArray

    from thermal_board.config.system_settings import AccuracySettings
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement

    CellGroup: TypeAlias = tuple[NDArray[np.bool_], tuple[int, int, int]]

SUBPIXEL_SHIFT_BITS = 4
SUBPIXEL_SCALE = 2**SUBPIXEL_SHIFT_BITS
COLOR_WITHIN_TARGET = (80, 220, 80)
COLOR_WITHIN_TOLERANCE = (0, 220, 255)
COLOR_OUT_OF_TOLERANCE = (0, 0, 255)
COLOR_INVALID = (160, 160, 160)
COLOR_TEXT = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)
GROUP_COLORS = (COLOR_INVALID, COLOR_WITHIN_TARGET, COLOR_WITHIN_TOLERANCE, COLOR_OUT_OF_TOLERANCE)
HUD_ORIGIN = (12, 24)
HUD_LINE_HEIGHT = 22
HUD_FONT_SCALE = 0.55


def colorize_thermal(normalized_image: NDArray[np.float32]) -> NDArray[np.uint8]:
    grayscale = np.clip(normalized_image * 255.0, 0, 255).astype(np.uint8)
    return np.asarray(cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO), dtype=np.uint8)


def classify_cells(measurement: FrameMeasurement, settings: AccuracySettings) -> list[CellGroup]:
    worst = measurement.cells.corner_deviation_mm.max(axis=1)
    valid = measurement.cells.valid
    within_target = valid & (worst <= settings.target_precision_mm)
    within_tolerance = valid & ~within_target & (worst <= settings.tolerance_mm)
    out_of_tolerance = valid & (worst > settings.tolerance_mm)
    selections = (~valid, within_target, within_tolerance, out_of_tolerance)
    return list(zip(selections, GROUP_COLORS, strict=True))


def draw_cell_outlines(
    canvas: NDArray[np.uint8], corners_px: NDArray[np.float64], groups: list[CellGroup]
) -> None:
    fixed_point = np.rint(corners_px * SUBPIXEL_SCALE).astype(np.int32)
    for selection, color in groups:
        polygons = list(fixed_point[selection])
        if polygons:
            cv2.polylines(canvas, polygons, isClosed=True, color=color, shift=SUBPIXEL_SHIFT_BITS)


def draw_text_lines(canvas: NDArray[np.uint8], lines: list[str]) -> None:
    origin_x, origin_y = HUD_ORIGIN
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line_number, text in enumerate(lines):
        position = (origin_x, origin_y + line_number * HUD_LINE_HEIGHT)
        cv2.putText(canvas, text, position, font, HUD_FONT_SCALE, COLOR_SHADOW, 3, cv2.LINE_AA)
        cv2.putText(canvas, text, position, font, HUD_FONT_SCALE, COLOR_TEXT, 1, cv2.LINE_AA)


def render_measurement_overlay(
    measurement: FrameMeasurement, settings: AccuracySettings
) -> NDArray[np.uint8]:
    canvas = colorize_thermal(measurement.normalized_image)
    corners_px = measurement.localization.refined.corners_px
    draw_cell_outlines(canvas, corners_px, classify_cells(measurement, settings))
    draw_text_lines(canvas, summary_fields(measurement.report))
    return canvas


def render_lost_frame(normalized_image: NDArray[np.float32], message: str) -> NDArray[np.uint8]:
    canvas = colorize_thermal(normalized_image)
    draw_text_lines(canvas, ["BOARD NOT FOUND", message])
    return canvas
