from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import cv2

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

QUIT_KEYS = frozenset({ord("q"), 27})
VIDEO_CODEC = "mp4v"


class FrameSink(Protocol):
    def show(self, canvas: NDArray[np.uint8]) -> bool: ...

    def close(self) -> None: ...


class NullFrameSink:
    def show(self, canvas: NDArray[np.uint8]) -> bool:
        return canvas is not None

    def close(self) -> None:
        return None


class WindowFrameSink:
    def __init__(self, window_name: str = "Thermal Board Metrology") -> None:
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show(self, canvas: NDArray[np.uint8]) -> bool:
        cv2.imshow(self.window_name, canvas)
        return (cv2.waitKey(1) & 0xFF) not in QUIT_KEYS

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)


class VideoRecordingSink:
    def __init__(self, path: Path, frame_rate_hz: float) -> None:
        self.path = path
        self.frame_rate_hz = frame_rate_hz
        self.writer: cv2.VideoWriter | None = None

    def open_writer(self, canvas: NDArray[np.uint8]) -> cv2.VideoWriter:
        height, width = canvas.shape[:2]
        codec = cv2.VideoWriter.fourcc(*VIDEO_CODEC)
        return cv2.VideoWriter(str(self.path), codec, self.frame_rate_hz, (width, height))

    def show(self, canvas: NDArray[np.uint8]) -> bool:
        if self.writer is None:
            self.writer = self.open_writer(canvas)
        self.writer.write(canvas)
        return True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


class CompositeFrameSink:
    def __init__(self, sinks: list[FrameSink]) -> None:
        self.sinks = sinks

    def show(self, canvas: NDArray[np.uint8]) -> bool:
        results = [sink.show(canvas) for sink in self.sinks]
        return all(results)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()
