from __future__ import annotations

import dataclasses
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from thermal_board.measurement.accuracy_report import AccuracyReport

LOGGER = logging.getLogger("thermal_board")


class ReportSink(Protocol):
    def write(self, report: AccuracyReport) -> None: ...


def report_as_dictionary(report: AccuracyReport) -> dict[str, Any]:
    return dataclasses.asdict(report)


def accuracy_fields(report: AccuracyReport) -> list[str]:
    deviations, status = report.deviations, "PASS" if report.meets_precision_target else "FAIL"
    corner, width = deviations.corner_position, deviations.width_error
    height = deviations.height_error
    return [
        f"{status} frame {report.frame_index} [{report.compute_backend}]",
        f"corner rms {corner.rms_mm:.3f} mm max {corner.max_abs_mm:.3f} mm",
        f"size rms w {width.rms_mm:.3f} mm h {height.rms_mm:.3f} mm",
    ]


def runtime_fields(report: AccuracyReport) -> list[str]:
    drift, coverage, geometry = report.drift, report.coverage, report.geometry
    shift_x, shift_y = drift.translation_px
    return [
        f"cells {coverage.measured_cell_ratio:.1%} reproj {geometry.reprojection_rms_px:.3f} px",
        f"drift {shift_x:+.2f},{shift_y:+.2f} px vibration {drift.vibration_rms_px:.2f} px",
        f"scale {drift.apparent_scale_ppm:+.0f} ppm latency {report.latency_ms:.1f} ms",
    ]


def summary_fields(report: AccuracyReport) -> list[str]:
    return accuracy_fields(report) + runtime_fields(report)


def format_report_summary(report: AccuracyReport) -> str:
    return " | ".join(summary_fields(report))


class LoggingReportSink:
    def write(self, report: AccuracyReport) -> None:
        LOGGER.info(format_report_summary(report))


class JsonLinesReportSink:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("w", encoding="utf-8")

    def write(self, report: AccuracyReport) -> None:
        self.stream.write(json.dumps(report_as_dictionary(report)) + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


class CompositeReportSink:
    def __init__(self, sinks: list[ReportSink]) -> None:
        self.sinks = sinks

    def write(self, report: AccuracyReport) -> None:
        for sink in self.sinks:
            sink.write(report)
