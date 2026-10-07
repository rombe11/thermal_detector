from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from thermal_board.pipeline.board_measurement_pipeline import BoardNotFoundError
from thermal_board.viewer.overlay_renderer import render_lost_frame, render_measurement_overlay

if TYPE_CHECKING:
    from thermal_board.acquisition.thermal_frame import FrameSource, ThermalFrame
    from thermal_board.measurement.accuracy_report import AccuracyReport
    from thermal_board.pipeline.board_measurement_pipeline import BoardMeasurementPipeline
    from thermal_board.reporting.report_sinks import ReportSink
    from thermal_board.viewer.display_sinks import FrameSink


@dataclass(slots=True)
class SessionSummary:
    frames_measured: int = 0
    frames_lost: int = 0
    frames_meeting_target: int = 0
    worst_corner_rms_mm: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, report: AccuracyReport) -> None:
        self.frames_measured += 1
        self.frames_meeting_target += int(report.meets_precision_target)
        corner_rms = report.deviations.corner_position.rms_mm
        self.worst_corner_rms_mm = max(self.worst_corner_rms_mm, corner_rms)
        self.latencies_ms.append(report.latency_ms)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else math.nan

    @property
    def all_frames_met_target(self) -> bool:
        return self.frames_measured > 0 and self.frames_meeting_target == self.frames_measured


class MeasurementSession:
    def __init__(
        self, pipeline: BoardMeasurementPipeline, frame_sink: FrameSink, report_sink: ReportSink
    ) -> None:
        self.pipeline = pipeline
        self.frame_sink = frame_sink
        self.report_sink = report_sink
        self.summary = SessionSummary()

    def handle_frame(self, frame: ThermalFrame) -> bool:
        accuracy = self.pipeline.components.configuration.accuracy
        try:
            measurement = self.pipeline.process(frame)
        except BoardNotFoundError as error:
            return self.handle_lost_frame(frame, str(error))
        self.summary.record(measurement.report)
        self.report_sink.write(measurement.report)
        return self.frame_sink.show(render_measurement_overlay(measurement, accuracy))

    def handle_lost_frame(self, frame: ThermalFrame, message: str) -> bool:
        self.summary.frames_lost += 1
        normalized = self.pipeline.components.normalizer.normalize(frame.pixels)
        return self.frame_sink.show(render_lost_frame(normalized, message))

    def run(self, source: FrameSource, max_frames: int | None = None) -> SessionSummary:
        try:
            for frame in itertools.islice(source.frames(), max_frames):
                if not self.handle_frame(frame):
                    break
        finally:
            self.frame_sink.close()
        return self.summary
