from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from thermal_board.acquisition.frame_source_factory import (
    SYNTHETIC_SOURCE_NAME,
    create_frame_source,
)
from thermal_board.config.configuration_loader import load_configuration
from thermal_board.pipeline.pipeline_factory import build_measurement_pipeline
from thermal_board.reporting.report_sinks import (
    CompositeReportSink,
    JsonLinesReportSink,
    LoggingReportSink,
    ReportSink,
)
from thermal_board.simulation.synthetic_sequence import SequencePlan, SyntheticSequenceSource
from thermal_board.viewer.display_sinks import (
    CompositeFrameSink,
    FrameSink,
    VideoRecordingSink,
    WindowFrameSink,
)
from thermal_board.viewer.measurement_session import MeasurementSession, SessionSummary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from thermal_board.config.system_settings import SystemConfiguration

DEFAULT_CONFIGURATION_PATH = Path("config.yaml")
DEFAULT_SYNTHETIC_FRAME_COUNT = 60
PROGRAM_DESCRIPTION = "Sub-millimetre metrology of an LWIR thermal calibration board"
SOURCE_HELP = "'synthetic', a video file, a thermal image or a folder of images"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thermal-board", description=PROGRAM_DESCRIPTION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIGURATION_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    add_run_command(commands)
    add_simulate_command(commands)
    return parser


def add_run_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = commands.add_parser("run", help="measure a recorded or synthetic feed")
    run.add_argument("--source", default=SYNTHETIC_SOURCE_NAME, help=SOURCE_HELP)
    run.add_argument("--frames", type=int, default=None)
    run.add_argument("--headless", action="store_true")
    run.add_argument("--record", type=Path, default=None)
    run.add_argument("--report", type=Path, default=None)


def add_simulate_command(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    simulate = commands.add_parser("simulate", help="render a 16-bit synthetic recording to disk")
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--frames", type=int, default=30)
    simulate.add_argument("--seed", type=int, default=0)


def build_frame_sink(
    arguments: argparse.Namespace, configuration: SystemConfiguration
) -> FrameSink:
    sinks: list[FrameSink] = [] if arguments.headless else [WindowFrameSink()]
    if arguments.record is not None:
        sinks.append(VideoRecordingSink(arguments.record, configuration.camera.frame_rate_hz))
    return CompositeFrameSink(sinks)


def build_report_sink(arguments: argparse.Namespace, resources: contextlib.ExitStack) -> ReportSink:
    sinks: list[ReportSink] = [LoggingReportSink()]
    if arguments.report is not None:
        sinks.append(resources.enter_context(JsonLinesReportSink(arguments.report)))
    return CompositeReportSink(sinks)


def summarize(summary: SessionSummary) -> str:
    counts = f"measured={summary.frames_measured} lost={summary.frames_lost}"
    target = f"meeting_target={summary.frames_meeting_target}"
    accuracy = f"worst_corner_rms={summary.worst_corner_rms_mm:.3f}mm"
    return f"{counts} {target} {accuracy} mean_latency={summary.mean_latency_ms:.1f}ms\n"


def build_session(
    arguments: argparse.Namespace,
    configuration: SystemConfiguration,
    resources: contextlib.ExitStack,
) -> MeasurementSession:
    report_sink = build_report_sink(arguments, resources)
    frame_sink = build_frame_sink(arguments, configuration)
    return MeasurementSession(build_measurement_pipeline(configuration), frame_sink, report_sink)


def run_measurement(arguments: argparse.Namespace, configuration: SystemConfiguration) -> int:
    frame_count = arguments.frames or DEFAULT_SYNTHETIC_FRAME_COUNT
    source = create_frame_source(arguments.source, configuration, frame_count)
    with contextlib.ExitStack() as resources:
        summary = build_session(arguments, configuration, resources).run(source, arguments.frames)
    sys.stdout.write(summarize(summary))
    return 0 if summary.all_frames_met_target else 1


def run_simulation(arguments: argparse.Namespace, configuration: SystemConfiguration) -> int:
    arguments.output.mkdir(parents=True, exist_ok=True)
    plan = SequencePlan(arguments.frames, configuration.simulation.vibration_amplitude_px)
    source = SyntheticSequenceSource(configuration, plan, arguments.seed)
    for frame in source.frames():
        cv2.imwrite(str(arguments.output / f"frame_{frame.index:05d}.png"), frame.pixels)
    sys.stdout.write(f"wrote {arguments.frames} frames to {arguments.output}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    arguments = build_argument_parser().parse_args(argv)
    configuration = load_configuration(arguments.config)
    if arguments.command == "simulate":
        return run_simulation(arguments, configuration)
    return run_measurement(arguments, configuration)
