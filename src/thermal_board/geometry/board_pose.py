from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BoardPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    distance_mm: float
    lateral_offset_mm: tuple[float, float] = (0.0, 0.0)

    def rotation_matrix(self) -> NDArray[np.float64]:
        angles = np.radians([self.pitch_deg, self.yaw_deg, self.roll_deg])
        rotation, _ = cv2.Rodrigues(np.array([angles[0], 0.0, 0.0]))
        yaw_rotation, _ = cv2.Rodrigues(np.array([0.0, angles[1], 0.0]))
        roll_rotation, _ = cv2.Rodrigues(np.array([0.0, 0.0, angles[2]]))
        return np.asarray(roll_rotation @ yaw_rotation @ rotation, dtype=np.float64)


def board_centering_transform(
    board_center_mm: NDArray[np.float64], board_scale: float
) -> NDArray[np.float64]:
    offset_x, offset_y = -board_scale * board_center_mm
    return np.array([[board_scale, 0.0, offset_x], [0.0, board_scale, offset_y], [0.0, 0.0, 1.0]])


def board_to_image_homography(
    pose: BoardPose,
    intrinsic_matrix: NDArray[np.float64],
    board_center_mm: NDArray[np.float64],
    board_scale: float = 1.0,
) -> NDArray[np.float64]:
    rotation = pose.rotation_matrix()
    translation = np.array([*pose.lateral_offset_mm, pose.distance_mm])
    extrinsic = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    centering = board_centering_transform(board_center_mm, board_scale)
    homography = intrinsic_matrix @ extrinsic @ centering
    return np.asarray(homography / homography[2, 2], dtype=np.float64)
