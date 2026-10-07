from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import DetectionSettings

MINIMUM_DYNAMIC_RANGE = 1e-6
PERCENTILE_SUBSAMPLING_STRIDE = 4


class ThermalNormalizer:
    def __init__(self, settings: DetectionSettings) -> None:
        self.settings = settings

    def intensity_window(self, pixels: NDArray[np.float32]) -> tuple[float, float]:
        stride = PERCENTILE_SUBSAMPLING_STRIDE
        sampled = pixels[::stride, ::stride]
        percentiles = (
            self.settings.normalization_low_percentile,
            self.settings.normalization_high_percentile,
        )
        low, high = np.percentile(sampled, percentiles)
        return float(low), float(max(high, low + MINIMUM_DYNAMIC_RANGE))

    def normalize(self, raw_pixels: NDArray[Any]) -> NDArray[np.float32]:
        pixels = np.asarray(raw_pixels, dtype=np.float32)
        low, high = self.intensity_window(pixels)
        normalized = (pixels - low) / (high - low)
        return self.denoise(np.clip(normalized, 0.0, 1.0).astype(np.float32))

    def denoise(self, pixels: NDArray[np.float32]) -> NDArray[np.float32]:
        sigma = self.settings.denoise_sigma_px
        if sigma <= 0:
            return pixels
        return np.asarray(cv2.GaussianBlur(pixels, (0, 0), sigma), dtype=np.float32)
