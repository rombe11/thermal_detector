from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from thermal_board.geometry.planar_homography import local_jacobian, project_points

if TYPE_CHECKING:
    from numpy.typing import NDArray

PARTS_PER_MILLION = 1e6


@dataclass(frozen=True, slots=True)
class DriftState:
    translation_px: tuple[float, float]
    rotation_deg: float
    apparent_scale_ppm: float
    frame_motion_px: float
    vibration_rms_px: float


@dataclass(frozen=True, slots=True)
class BoardImagePlacement:
    center_px: NDArray[np.float64]
    jacobian: NDArray[np.float64]


def relative_rotation_and_scale(
    reference: NDArray[np.float64], current: NDArray[np.float64]
) -> tuple[float, float]:
    relative = current @ np.linalg.inv(reference)
    rotation = math.degrees(
        math.atan2(relative[1, 0] - relative[0, 1], relative[0, 0] + relative[1, 1])
    )
    scale = math.sqrt(abs(float(np.linalg.det(relative))))
    return rotation, (scale - 1.0) * PARTS_PER_MILLION


class DriftMonitor:
    def __init__(self, board_center_mm: NDArray[np.float64], vibration_window_frames: int) -> None:
        self.board_center_mm = board_center_mm
        self.reference: BoardImagePlacement | None = None
        self.previous: BoardImagePlacement | None = None
        self.recent_motion_px: deque[float] = deque(maxlen=vibration_window_frames)

    def placement(self, homography: NDArray[np.float64]) -> BoardImagePlacement:
        center = project_points(homography, self.board_center_mm)
        return BoardImagePlacement(center, local_jacobian(homography, self.board_center_mm))

    def frame_motion(self, placement: BoardImagePlacement) -> float:
        previous, self.previous = self.previous, placement
        motion = (
            0.0
            if previous is None
            else float(np.linalg.norm(placement.center_px - previous.center_px))
        )
        self.recent_motion_px.append(motion)
        return motion

    def reference_for(self, placement: BoardImagePlacement) -> BoardImagePlacement:
        if self.reference is None:
            self.reference = placement
        return self.reference

    def update(self, homography: NDArray[np.float64]) -> DriftState:
        placement = self.placement(homography)
        reference = self.reference_for(placement)
        shift_x, shift_y = (float(value) for value in placement.center_px - reference.center_px)
        rotation, scale_ppm = relative_rotation_and_scale(reference.jacobian, placement.jacobian)
        motion = self.frame_motion(placement)
        vibration = math.sqrt(float(np.mean(np.square(self.recent_motion_px))))
        return DriftState((shift_x, shift_y), rotation, scale_ppm, motion, vibration)

    def reset_reference(self) -> None:
        self.reference = None
        self.previous = None
        self.recent_motion_px.clear()
