from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from thermal_board.acquisition.thermal_frame import FrameSourceError, ThermalFrame

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".pgm", ".bmp", ".jpg", ".npy"})
COLOR_CHANNEL_AXIS_LENGTH = 3


def to_single_channel(pixels: NDArray[Any]) -> NDArray[Any]:
    if pixels.ndim == COLOR_CHANNEL_AXIS_LENGTH:
        return np.asarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY))
    return pixels


def read_thermal_image(path: Path) -> NDArray[Any]:
    if path.suffix.lower() == ".npy":
        return to_single_channel(np.load(path))
    pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if pixels is None:
        raise FrameSourceError(f"cannot read thermal image {path}")
    return to_single_channel(np.asarray(pixels))


class ImageSequenceSource:
    def __init__(self, image_paths: Sequence[Path], frame_rate_hz: float) -> None:
        if not image_paths:
            raise FrameSourceError("image sequence is empty")
        self.image_paths = list(image_paths)
        self.frame_period_s = 1.0 / frame_rate_hz

    def frames(self) -> Iterator[ThermalFrame]:
        for index, path in enumerate(self.image_paths):
            yield ThermalFrame(index, index * self.frame_period_s, read_thermal_image(path))


class VideoFileSource:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path

    def open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise FrameSourceError(f"cannot open video {self.video_path}")
        return capture

    def frames(self) -> Iterator[ThermalFrame]:
        capture = self.open_capture()
        try:
            yield from self.read_all(capture)
        finally:
            capture.release()

    def read_all(self, capture: cv2.VideoCapture) -> Iterator[ThermalFrame]:
        index = 0
        has_frame, pixels = capture.read()
        while has_frame:
            timestamp_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield ThermalFrame(index, timestamp_s, to_single_channel(np.asarray(pixels)))
            index += 1
            has_frame, pixels = capture.read()


def list_image_files(directory: Path) -> list[Path]:
    candidates = sorted(directory.iterdir())
    return [path for path in candidates if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
