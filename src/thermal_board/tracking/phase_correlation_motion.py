from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

DOWNSAMPLING_FACTOR = 2


@dataclass(frozen=True, slots=True)
class FrameShift:
    shift_x_px: float
    shift_y_px: float
    confidence: float


class PhaseCorrelationMotionEstimator:
    def __init__(self) -> None:
        self.previous_image: NDArray[np.float32] | None = None
        self.window: NDArray[np.float32] | None = None

    def downsample(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        height, width = image.shape[:2]
        size = (width // DOWNSAMPLING_FACTOR, height // DOWNSAMPLING_FACTOR)
        return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.float32)

    def hanning_window(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.window is None or self.window.shape != image.shape:
            height, width = image.shape
            self.window = np.asarray(cv2.createHanningWindow((width, height), cv2.CV_32F))
        return self.window

    def estimate(self, image: NDArray[np.float32]) -> FrameShift:
        current = self.downsample(image)
        previous, self.previous_image = self.previous_image, current
        if previous is None or previous.shape != current.shape:
            return FrameShift(0.0, 0.0, 0.0)
        (shift_x, shift_y), response = cv2.phaseCorrelate(
            previous, current, self.hanning_window(current)
        )
        scale = float(DOWNSAMPLING_FACTOR)
        return FrameShift(shift_x * scale, shift_y * scale, float(response))

    def reset(self) -> None:
        self.previous_image = None
