from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.support import CONFIGURATION_PATH, blank_frame, cpu_pipeline
from thermal_board import cli
from thermal_board.reporting.report_sinks import JsonLinesReportSink, format_report_summary
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource
from thermal_board.viewer.display_sinks import VideoRecordingSink
from thermal_board.viewer.measurement_session import MeasurementSession
from thermal_board.viewer.overlay_renderer import render_lost_frame, render_measurement_overlay

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from thermal_board.acquisition.thermal_frame import ThermalFrame
    from thermal_board.config.system_settings import SystemConfiguration
    from thermal_board.measurement.accuracy_report import AccuracyReport
    from thermal_board.pipeline.board_measurement_pipeline import FrameMeasurement


class QuittingDisplay:
    def __init__(self, frames_before_quit: int) -> None:
        self.frames_before_quit, self.shown, self.closed = frames_before_quit, 0, False

    def show(self, canvas: np.ndarray) -> bool:
        self.shown += int(canvas.ndim == 3)
        return self.shown < self.frames_before_quit

    def close(self) -> None:
        self.closed = True


class DiscardingReportSink:
    def write(self, report: AccuracyReport) -> None:
        self.last_report = report


@dataclasses.dataclass
class ScriptedSource:
    scripted_frames: list[ThermalFrame]

    def frames(self) -> Iterator[ThermalFrame]:
        yield from self.scripted_frames


@pytest.fixture(scope="module")
def measurement(small_configuration: SystemConfiguration) -> FrameMeasurement:
    frame = next(iter(SyntheticSequenceSource(small_configuration, SequencePlan(1, 0.0)).frames()))
    return cpu_pipeline(small_configuration).process(frame)


def run_session(
    configuration: SystemConfiguration, display: QuittingDisplay, frames: list[ThermalFrame]
) -> MeasurementSession:
    session = MeasurementSession(cpu_pipeline(configuration), display, DiscardingReportSink())
    session.run(ScriptedSource(frames))
    return session


def run_cli(command_line: str) -> int:
    return cli.main(f"--config {CONFIGURATION_PATH} {command_line}".split())


def test_when_report_is_summarised_then_verdict_accuracy_and_drift_are_shown(
    measurement: FrameMeasurement,
) -> None:
    summary = format_report_summary(measurement.report)
    assert summary.startswith("PASS frame 0")
    assert "corner rms" in summary
    assert "drift +0.00,+0.00 px" in summary


def test_when_reports_are_written_as_json_lines_then_each_line_parses(
    tmp_path: Path, measurement: FrameMeasurement
) -> None:
    path = tmp_path / "reports.jsonl"
    with JsonLinesReportSink(path) as sink:
        sink.write(measurement.report)
        sink.write(measurement.report)
    documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [document["meets_precision_target"] for document in documents] == [True, True]


def test_when_measurement_is_rendered_then_overlay_has_sensor_size(
    measurement: FrameMeasurement, small_configuration: SystemConfiguration
) -> None:
    canvas = render_measurement_overlay(measurement, small_configuration.accuracy)
    assert canvas.shape == (
        small_configuration.camera.height_px,
        small_configuration.camera.width_px,
        3,
    )
    assert render_lost_frame(measurement.normalized_image, "lost").shape == canvas.shape


def test_when_overlays_are_recorded_then_a_video_file_is_written(tmp_path: Path) -> None:
    sink = VideoRecordingSink(tmp_path / "overlay.mp4", 30.0)
    sink.show(np.zeros((64, 80, 3), np.uint8))
    sink.close()
    assert (tmp_path / "overlay.mp4").stat().st_size > 0


def test_when_board_is_lost_mid_sequence_then_session_counts_the_lost_frame(
    small_configuration: SystemConfiguration,
) -> None:
    frames = list(SyntheticSequenceSource(small_configuration, SequencePlan(2, 0.3)).frames())
    display = QuittingDisplay(frames_before_quit=10)
    summary = run_session(
        small_configuration, display, [frames[0], blank_frame(small_configuration), frames[1]]
    ).summary
    assert (summary.frames_measured, summary.frames_lost, display.shown) == (2, 1, 3)
    assert summary.all_frames_met_target
    assert display.closed


def test_when_viewer_quits_then_session_stops(small_configuration: SystemConfiguration) -> None:
    frames = list(SyntheticSequenceSource(small_configuration, SequencePlan(3, 0.3)).frames())
    assert (
        run_session(
            small_configuration, QuittingDisplay(frames_before_quit=1), frames
        ).summary.frames_measured
        == 1
    )


def test_when_recording_is_simulated_and_replayed_then_every_frame_passes(tmp_path: Path) -> None:
    recording, report = tmp_path / "recording", tmp_path / "reports.jsonl"
    assert run_cli(f"simulate --output {recording} --frames 2") == 0
    assert run_cli(f"run --source {recording} --headless --report {report}") == 0
    documents = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert [document["meets_precision_target"] for document in documents] == [True, True]


def test_when_board_is_never_found_then_run_exits_with_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "blank.npy"
    np.save(image_path, np.full((1024, 1280), 8000, np.uint16))
    assert run_cli(f"run --source {image_path} --headless") == 1
