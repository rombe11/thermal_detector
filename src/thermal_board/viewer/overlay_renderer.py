from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from typing import TypeAlias

    from numpy.typing import NDArray

    from thermal_board.config.system_settings import AccuracySettings
    from thermal_board.geometry.board_model import BoardModel
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement

    Color: TypeAlias = tuple[int, int, int]
    CellGroup: TypeAlias = tuple[NDArray[np.bool_], Color]

PANEL_WIDTH = 360
PANEL_MINIMUM_HEIGHT = 1000
PANEL_MARGIN = 22
LINE_HEIGHT = 24
SECTION_GAP = 14
MAP_MAXIMUM_HEIGHT = 210
SUBPIXEL_SHIFT_BITS = 4
SUBPIXEL_SCALE = 2**SUBPIXEL_SHIFT_BITS
FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX

COLOR_PANEL: Color = (34, 30, 28)
COLOR_DIVIDER: Color = (70, 64, 60)
COLOR_SECTION: Color = (200, 170, 90)
COLOR_LABEL: Color = (165, 160, 155)
COLOR_VALUE: Color = (245, 245, 245)
COLOR_PASS: Color = (96, 190, 72)
COLOR_FAIL: Color = (70, 70, 225)
COLOR_WITHIN_TARGET: Color = (96, 190, 72)
COLOR_WITHIN_TOLERANCE: Color = (40, 200, 240)
COLOR_OUT_OF_TOLERANCE: Color = (70, 70, 225)
COLOR_INVALID: Color = (140, 140, 140)
COLOR_HIGHLIGHT: Color = (255, 255, 255)
GROUP_COLORS = (COLOR_INVALID, COLOR_WITHIN_TARGET, COLOR_WITHIN_TOLERANCE, COLOR_OUT_OF_TOLERANCE)
OUTLINED_COLORS = frozenset({COLOR_INVALID, COLOR_WITHIN_TOLERANCE, COLOR_OUT_OF_TOLERANCE})
LEGEND = (
    (COLOR_WITHIN_TARGET, "within target"),
    (COLOR_WITHIN_TOLERANCE, "within tolerance"),
    (COLOR_OUT_OF_TOLERANCE, "out of tolerance"),
    (COLOR_INVALID, "not measured"),
)


@dataclass
class InformationPanel:
    canvas: NDArray[np.uint8]
    cursor_y: int = PANEL_MARGIN

    @classmethod
    def blank(cls, height: int) -> InformationPanel:
        canvas = np.full((height, PANEL_WIDTH, 3), COLOR_PANEL, dtype=np.uint8)
        return cls(canvas)

    def text(self, text: str, x: int, color: Color, scale: float = 0.5, font: int = FONT) -> None:
        cv2.putText(self.canvas, text, (x, self.cursor_y), font, scale, color, 1, cv2.LINE_AA)

    def title(self, text: str, subtitle: str) -> None:
        self.cursor_y += 8
        self.text(text, PANEL_MARGIN, COLOR_VALUE, 0.75, TITLE_FONT)
        self.cursor_y += LINE_HEIGHT
        self.text(subtitle, PANEL_MARGIN, COLOR_LABEL, 0.45)
        self.cursor_y += LINE_HEIGHT

    def badge(self, text: str, color: Color, caption: str) -> None:
        top, bottom = self.cursor_y, self.cursor_y + 40
        cv2.rectangle(self.canvas, (PANEL_MARGIN, top), (PANEL_MARGIN + 120, bottom), color, -1)
        self.cursor_y = bottom - 12
        self.text(text, PANEL_MARGIN + 22, COLOR_VALUE, 0.8, TITLE_FONT)
        self.text(caption, PANEL_MARGIN + 140, COLOR_LABEL, 0.5)
        self.cursor_y = bottom + SECTION_GAP + 8

    def section(self, heading: str) -> None:
        right = PANEL_WIDTH - PANEL_MARGIN
        line_y = self.cursor_y - 16
        cv2.line(
            self.canvas, (PANEL_MARGIN, line_y), (right, line_y), COLOR_DIVIDER, 1, cv2.LINE_AA
        )
        self.cursor_y += 6
        self.text(heading.upper(), PANEL_MARGIN, COLOR_SECTION, 0.45)
        self.cursor_y += LINE_HEIGHT

    def row(self, label: str, value: str) -> None:
        self.text(label, PANEL_MARGIN, COLOR_LABEL)
        (value_width, _), _ = cv2.getTextSize(value, FONT, 0.5, 1)
        self.text(value, PANEL_WIDTH - PANEL_MARGIN - value_width, COLOR_VALUE)
        self.cursor_y += LINE_HEIGHT

    def image(self, picture: NDArray[np.uint8]) -> None:
        top = self.cursor_y - 12
        visible = picture[: max(self.canvas.shape[0] - top, 0)]
        height, width = visible.shape[:2]
        self.canvas[top : top + height, PANEL_MARGIN : PANEL_MARGIN + width] = visible
        self.cursor_y = top + picture.shape[0] + LINE_HEIGHT

    def legend(self) -> None:
        for color, label in LEGEND:
            top = self.cursor_y - 11
            cv2.rectangle(
                self.canvas, (PANEL_MARGIN, top), (PANEL_MARGIN + 14, top + 12), color, -1
            )
            self.text(label, PANEL_MARGIN + 26, COLOR_LABEL)
            self.cursor_y += LINE_HEIGHT - 4


def colorize_thermal(normalized_image: NDArray[np.float32]) -> NDArray[np.uint8]:
    grayscale = np.clip(normalized_image * 220.0 + 20.0, 0, 255).astype(np.uint8)
    return np.asarray(cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR), dtype=np.uint8)


def worst_corner_deviation(measurement: FrameMeasurement) -> NDArray[np.float64]:
    worst = measurement.cells.corner_deviation_mm.max(axis=1)
    return np.where(measurement.cells.valid, worst, np.nan)


def classify_cells(measurement: FrameMeasurement, settings: AccuracySettings) -> list[CellGroup]:
    worst, valid = worst_corner_deviation(measurement), measurement.cells.valid
    within_target = valid & (worst <= settings.target_precision_mm)
    within_tolerance = valid & ~within_target & (worst <= settings.tolerance_mm)
    out_of_tolerance = valid & (worst > settings.tolerance_mm)
    selections = (~valid, within_target, within_tolerance, out_of_tolerance)
    return list(zip(selections, GROUP_COLORS, strict=True))


def fixed_point(corners_px: NDArray[np.float64]) -> NDArray[np.int32]:
    return np.asarray(np.rint(corners_px * SUBPIXEL_SCALE), dtype=np.int32)


def draw_polygons(canvas: NDArray[np.uint8], polygons: NDArray[np.int32], color: Color) -> None:
    thickness = 2 if color == COLOR_OUT_OF_TOLERANCE else 1
    options = {"thickness": thickness, "lineType": cv2.LINE_AA, "shift": SUBPIXEL_SHIFT_BITS}
    cv2.polylines(canvas, list(polygons), isClosed=True, color=color, **options)


def draw_cell_outlines(
    canvas: NDArray[np.uint8], corners_px: NDArray[np.float64], groups: list[CellGroup]
) -> None:
    polygons = fixed_point(corners_px)
    for selection, color in groups:
        if color in OUTLINED_COLORS and selection.any():
            draw_polygons(canvas, polygons[selection], color)


def worst_cell_number(measurement: FrameMeasurement) -> int | None:
    worst = worst_corner_deviation(measurement)
    return None if np.isnan(worst).all() else int(np.nanargmax(worst))


def highlight_cell(canvas: NDArray[np.uint8], corners_px: NDArray[np.float64], number: int) -> None:
    centre = corners_px[number].mean(axis=0)
    radius = int(np.ptp(corners_px[number], axis=0).max() * 1.3) + 4
    centre_x, centre_y = (int(value) for value in np.rint(centre))
    centre_px = (centre_x, centre_y)
    cv2.circle(canvas, centre_px, radius, COLOR_HIGHLIGHT, 2, cv2.LINE_AA)
    label_position = (centre_px[0] + radius + 4, centre_px[1] + 5)
    cv2.putText(canvas, f"#{number}", label_position, FONT, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"#{number}", label_position, FONT, 0.55, COLOR_HIGHLIGHT, 1, cv2.LINE_AA)


def deviation_map(
    measurement: FrameMeasurement, model: BoardModel, settings: AccuracySettings
) -> NDArray[np.uint8]:
    rows, columns = model.cell_rows_and_columns.T
    grid = np.full((model.geometry.rows, model.geometry.columns), np.nan)
    grid[rows, columns] = worst_corner_deviation(measurement)
    scaled = np.nan_to_num(np.clip(grid / settings.tolerance_mm, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = np.asarray(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), dtype=np.uint8)
    colored[np.isnan(grid)] = COLOR_INVALID
    return fit_to_panel(colored)


def fit_to_panel(picture: NDArray[np.uint8]) -> NDArray[np.uint8]:
    available_width = PANEL_WIDTH - 2 * PANEL_MARGIN
    scale = min(available_width / picture.shape[1], MAP_MAXIMUM_HEIGHT / picture.shape[0])
    size = (max(1, round(picture.shape[1] * scale)), max(1, round(picture.shape[0] * scale)))
    return np.asarray(cv2.resize(picture, size, interpolation=cv2.INTER_NEAREST), np.uint8)


def colour_bar() -> NDArray[np.uint8]:
    ramp = np.linspace(0, 255, PANEL_WIDTH - 2 * PANEL_MARGIN).astype(np.uint8)[np.newaxis, :]
    bar = cv2.applyColorMap(np.repeat(ramp, 8, axis=0), cv2.COLORMAP_TURBO)
    return np.asarray(bar, dtype=np.uint8)


def write_accuracy(panel: InformationPanel, measurement: FrameMeasurement) -> None:
    deviations = measurement.report.deviations
    panel.section("accuracy")
    panel.row("corner error rms", f"{deviations.corner_position.rms_mm:.3f} mm")
    panel.row("corner error max", f"{deviations.corner_position.max_abs_mm:.3f} mm")
    panel.row("width error rms", f"{deviations.width_error.rms_mm:.3f} mm")
    panel.row("height error rms", f"{deviations.height_error.rms_mm:.3f} mm")


def write_coverage(panel: InformationPanel, measurement: FrameMeasurement) -> None:
    coverage, geometry = measurement.report.coverage, measurement.report.geometry
    panel.section("coverage")
    panel.row(
        "cells measured", f"{coverage.measured_cell_ratio:.1%} of {measurement.cells.valid.size}"
    )
    panel.row("out of tolerance", str(coverage.cells_out_of_tolerance))
    panel.row("model residual", f"{geometry.reprojection_rms_px:.3f} px")


def write_motion(panel: InformationPanel, measurement: FrameMeasurement) -> None:
    drift = measurement.report.drift
    panel.section("motion")
    panel.row("drift x / y", f"{drift.translation_px[0]:+.2f} / {drift.translation_px[1]:+.2f} px")
    panel.row("vibration rms", f"{drift.vibration_rms_px:.2f} px")
    panel.row("apparent scale", f"{round(drift.apparent_scale_ppm):+d} ppm")
    panel.row("mode", measurement.localization.mode)


def write_system(panel: InformationPanel, measurement: FrameMeasurement) -> None:
    report = measurement.report
    panel.section("system")
    panel.row(
        "latency", f"{report.latency_ms:.1f} ms  ({1000.0 / max(report.latency_ms, 1e-6):.1f} fps)"
    )
    panel.row("backend", report.compute_backend)


def write_deviation_map(
    panel: InformationPanel,
    measurement: FrameMeasurement,
    model: BoardModel,
    settings: AccuracySettings,
) -> None:
    panel.section("deviation map")
    panel.image(deviation_map(measurement, model, settings))
    panel.image(colour_bar())
    panel.cursor_y -= 6
    panel.text("0 mm", PANEL_MARGIN, COLOR_LABEL, 0.4)
    panel.text(f"{settings.tolerance_mm:g} mm", PANEL_WIDTH - PANEL_MARGIN - 40, COLOR_LABEL, 0.4)
    panel.cursor_y += LINE_HEIGHT


def write_worst_cell(
    panel: InformationPanel, measurement: FrameMeasurement, number: int | None
) -> None:
    panel.section("worst cell")
    if number is None:
        panel.row("cell", "none measured")
        return
    panel.row(f"cell #{number}", f"{worst_corner_deviation(measurement)[number]:.3f} mm")


def measurement_panel(
    measurement: FrameMeasurement, model: BoardModel, settings: AccuracySettings, height: int
) -> NDArray[np.uint8]:
    panel, report = InformationPanel.blank(height), measurement.report
    panel.title("THERMAL BOARD", "real-time metrology")
    verdict = ("PASS", COLOR_PASS) if report.meets_precision_target else ("FAIL", COLOR_FAIL)
    panel.badge(*verdict, f"frame {report.frame_index}")
    for writer in (write_accuracy, write_coverage, write_motion, write_system):
        writer(panel, measurement)
    write_deviation_map(panel, measurement, model, settings)
    write_worst_cell(panel, measurement, worst_cell_number(measurement))
    panel.legend()
    return panel.canvas


def compose(view: NDArray[np.uint8], panel: NDArray[np.uint8]) -> NDArray[np.uint8]:
    height = panel.shape[0]
    padded = np.zeros((height, view.shape[1], 3), dtype=np.uint8)
    padded[: view.shape[0]] = view
    return np.hstack([padded, panel])


def render_measurement_overlay(
    measurement: FrameMeasurement, model: BoardModel, settings: AccuracySettings
) -> NDArray[np.uint8]:
    view = colorize_thermal(measurement.normalized_image)
    corners_px, worst = measurement.localization.refined.corners_px, worst_cell_number(measurement)
    draw_cell_outlines(view, corners_px, classify_cells(measurement, settings))
    if worst is not None:
        highlight_cell(view, corners_px, worst)
    height = max(view.shape[0], PANEL_MINIMUM_HEIGHT)
    return compose(view, measurement_panel(measurement, model, settings, height))


def lost_panel(reason: str, height: int) -> NDArray[np.uint8]:
    panel = InformationPanel.blank(height)
    panel.title("THERMAL BOARD", "real-time metrology")
    panel.badge("LOST", COLOR_FAIL, "board not found")
    panel.section("reason")
    panel.text(reason[:44], PANEL_MARGIN, COLOR_VALUE)
    return panel.canvas


def render_lost_frame(normalized_image: NDArray[np.float32], message: str) -> NDArray[np.uint8]:
    view = (colorize_thermal(normalized_image) * 0.45).astype(np.uint8)
    height = max(view.shape[0], PANEL_MINIMUM_HEIGHT)
    return compose(view, lost_panel(message, height))
