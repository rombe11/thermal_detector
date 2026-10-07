from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from thermal_board.acquisition.thermal_frame import ThermalFrame
from thermal_board.geometry.board_model import BoardModel
from thermal_board.geometry.board_pose import BoardPose, board_to_image_homography
from thermal_board.geometry.pinhole_camera import build_intrinsic_matrix
from thermal_board.geometry.planar_homography import translation_homography
from thermal_board.simulation.thermal_board_renderer import ThermalBoardRenderer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from thermal_board.config.system_settings import SystemConfiguration

PARTS_PER_MILLION = 1e-6


@dataclass(frozen=True, slots=True)
class SequencePlan:
    frame_count: int
    vibration_amplitude_px: float
    drift_per_frame_px: tuple[float, float] = (0.0, 0.0)
    expansion_ppm_per_frame: float = 0.0


class SyntheticScene:
    def __init__(self, configuration: SystemConfiguration) -> None:
        self.configuration = configuration
        self.intrinsic_matrix = build_intrinsic_matrix(configuration.camera)
        self.board_center_mm = BoardModel(configuration.board).board_center_mm
        self.pose = nominal_board_pose(configuration)

    def homography(self, board_scale: float = 1.0) -> NDArray[np.float64]:
        return board_to_image_homography(
            self.pose, self.intrinsic_matrix, self.board_center_mm, board_scale
        )


def nominal_board_pose(configuration: SystemConfiguration) -> BoardPose:
    simulation = configuration.simulation
    distance_mm = configuration.camera.working_distance_m * 1000.0
    return BoardPose(simulation.yaw_deg, simulation.pitch_deg, simulation.roll_deg, distance_mm)


class SyntheticSequenceSource:
    def __init__(
        self, configuration: SystemConfiguration, plan: SequencePlan, seed: int = 0
    ) -> None:
        self.scene = SyntheticScene(configuration)
        self.renderer = ThermalBoardRenderer(
            configuration.board, configuration.simulation, configuration.camera, seed
        )
        self.plan = plan
        self.frame_period_s = 1.0 / configuration.camera.frame_rate_hz
        self.vibrations = self.sample_vibrations(seed)

    def sample_vibrations(self, seed: int) -> NDArray[np.float64]:
        generator = np.random.default_rng(seed + 1)
        shape = (self.plan.frame_count, 2)
        return generator.uniform(-1.0, 1.0, shape) * self.plan.vibration_amplitude_px

    def ground_truth_homography(self, index: int) -> NDArray[np.float64]:
        shift = np.asarray(self.plan.drift_per_frame_px) * index + self.vibrations[index]
        scale = 1.0 + self.plan.expansion_ppm_per_frame * index * PARTS_PER_MILLION
        return translation_homography(*shift) @ self.scene.homography(scale)

    def frames(self) -> Iterator[ThermalFrame]:
        for index in range(self.plan.frame_count):
            pixels = self.renderer.render(self.ground_truth_homography(index))
            yield ThermalFrame(index, index * self.frame_period_s, pixels)
