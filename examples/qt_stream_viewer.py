from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from thermal_board.acquisition.frame_source_factory import create_frame_source
from thermal_board.service import BoardMetrologyService

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from thermal_board.acquisition.thermal_frame import FrameSource


class MeasurementWorker(QObject):
    frame_ready = Signal(object)
    finished = Signal()

    def __init__(self, service: BoardMetrologyService, source: FrameSource) -> None:
        super().__init__()
        self.service = service
        self.source = source
        self.running = True

    @Slot()
    def run(self) -> None:
        for frame in self.source.frames():
            if not self.running:
                break
            self.frame_ready.emit(self.annotated(frame.pixels, frame.timestamp_s))
        self.finished.emit()

    def annotated(self, pixels: NDArray[Any], timestamp_s: float) -> NDArray[np.uint8]:
        measurement = self.service.measure(pixels, timestamp_s)
        if measurement is None:
            return self.service.annotate_lost(pixels)
        return self.service.annotate(measurement)

    def stop(self) -> None:
        self.running = False


def to_pixmap(canvas: NDArray[np.uint8]) -> QPixmap:
    height, width = canvas.shape[:2]
    image = QImage(canvas.data, width, height, canvas.strides[0], QImage.Format.Format_BGR888)
    return QPixmap.fromImage(image.copy())


class StreamWindow(QMainWindow):
    def __init__(self, worker: MeasurementWorker, quit_when_done: bool = False) -> None:
        super().__init__()
        self.quit_when_done = quit_when_done
        self.setWindowTitle("Thermal board metrology")
        self.view = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(820, 512)
        self.setCentralWidget(self.view)
        self.worker, self.thread = worker, QThread(self)
        self.connect_worker()

    def connect_worker(self) -> None:
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.finished.connect(self.on_stream_finished)

    @Slot()
    def on_stream_finished(self) -> None:
        self.thread.quit()
        if self.quit_when_done:
            QApplication.quit()

    @Slot(object)
    def show_frame(self, canvas: NDArray[np.uint8]) -> None:
        pixmap = to_pixmap(canvas)
        scaled = pixmap.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view.setPixmap(scaled)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed thermal-board metrology in a Qt window")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--source", default="synthetic")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--quit-when-done", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    arguments, application = parse_arguments(argv[1:]), QApplication(argv)
    service = BoardMetrologyService.from_configuration_file(arguments.config)
    source = create_frame_source(arguments.source, service.configuration, arguments.frames)
    window = StreamWindow(MeasurementWorker(service, source), arguments.quit_when_done)
    application.aboutToQuit.connect(window.stop)
    window.show()
    window.start()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
