from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import DetectionSettings
    from thermal_board.detection.thermal_level_estimation import ThermalLevels

MINIMUM_BLOB_AREA_PX = 4
BACKGROUND_LABEL_COUNT = 1


@dataclass(frozen=True, slots=True)
class CellBlobs:
    centroids: NDArray[np.float64]
    is_hot: NDArray[np.bool_]
    areas: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.centroids)

    def subset(self, selection: NDArray[np.bool_] | NDArray[np.int64]) -> CellBlobs:
        return CellBlobs(self.centroids[selection], self.is_hot[selection], self.areas[selection])


def concatenate_blobs(first: CellBlobs, second: CellBlobs) -> CellBlobs:
    centroids = np.concatenate([first.centroids, second.centroids])
    is_hot = np.concatenate([first.is_hot, second.is_hot])
    return CellBlobs(centroids, is_hot, np.concatenate([first.areas, second.areas]))


def connected_blobs(mask: NDArray[np.bool_], is_hot: bool) -> CellBlobs:
    label_image = mask.astype(np.uint8)
    _, _, statistics, centroids = cv2.connectedComponentsWithStats(label_image, connectivity=4)
    areas = statistics[BACKGROUND_LABEL_COUNT:, cv2.CC_STAT_AREA].astype(np.float64)
    polarity = np.full(len(areas), fill_value=is_hot)
    return CellBlobs(np.asarray(centroids[BACKGROUND_LABEL_COUNT:], np.float64), polarity, areas)


def keep_cell_sized_blobs(blobs: CellBlobs, settings: DetectionSettings) -> CellBlobs:
    plausible = blobs.areas >= MINIMUM_BLOB_AREA_PX
    if not np.any(plausible):
        return blobs.subset(plausible)
    typical_area = float(np.median(blobs.areas[plausible]))
    lower = typical_area * settings.blob_area_min_ratio
    upper = typical_area * settings.blob_area_max_ratio
    return blobs.subset((blobs.areas >= lower) & (blobs.areas <= upper) & plausible)


def detect_cell_blobs(
    image: NDArray[np.float32], levels: ThermalLevels, settings: DetectionSettings
) -> CellBlobs:
    hot_blobs = connected_blobs(image > levels.hot_threshold, is_hot=True)
    cold_blobs = connected_blobs(image < levels.cold_threshold, is_hot=False)
    hot_cells = keep_cell_sized_blobs(hot_blobs, settings)
    return concatenate_blobs(hot_cells, keep_cell_sized_blobs(cold_blobs, settings))
