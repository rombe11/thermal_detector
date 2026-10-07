from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from thermal_board.acquisition.thermal_frame import ThermalFrame
from thermal_board.compute.array_backend import cpu_backend
from thermal_board.geometry.board_model import BoardModel
from thermal_board.geometry.planar_homography import project_points
from thermal_board.pipeline.pipeline_factory import build_measurement_pipeline
from thermal_board.preprocessing.thermal_normalizer import ThermalNormalizer
from thermal_board.simulation.synthetic_sequence import SyntheticScene
from thermal_board.simulation.thermal_board_renderer import ThermalBoardRenderer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from thermal_board.config.system_settings import SystemConfiguration
    from thermal_board.pipeline.board_measurement_pipeline import BoardMeasurementPipeline

CONFIGURATION_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
SUBSTRATE_COUNTS = 8000


@dataclasses.dataclass(frozen=True)
class RenderedScene:
    configuration: SystemConfiguration
    model: BoardModel
    homography: NDArray[np.float64]
    raw_image: NDArray[np.uint16]
    normalized_image: NDArray[np.float32]

    @property
    def true_corners_px(self) -> NDArray[np.float64]:
        return project_points(self.homography, self.model.cell_corners_mm)


def shrink_configuration(configuration: SystemConfiguration) -> SystemConfiguration:
    board = dataclasses.replace(
        configuration.board, columns=12, rows=10, board_width_mm=975.0, board_height_mm=825.0
    )
    camera = dataclasses.replace(configuration.camera, width_px=320, height_px=256)
    compute = dataclasses.replace(configuration.compute, prefer_gpu=False)
    return dataclasses.replace(configuration, board=board, camera=camera, compute=compute)


def render_scene(configuration: SystemConfiguration) -> RenderedScene:
    homography = SyntheticScene(configuration).homography()
    simulation, camera = configuration.simulation, configuration.camera
    raw_image = ThermalBoardRenderer(configuration.board, simulation, camera).render(homography)
    normalized = ThermalNormalizer(configuration.detection).normalize(raw_image)
    model = BoardModel(configuration.board)
    return RenderedScene(configuration, model, homography, raw_image, normalized)


def cpu_pipeline(configuration: SystemConfiguration) -> BoardMeasurementPipeline:
    return build_measurement_pipeline(configuration, cpu_backend())


def blank_frame(configuration: SystemConfiguration) -> ThermalFrame:
    shape = (configuration.camera.height_px, configuration.camera.width_px)
    return ThermalFrame(0, 0.0, np.full(shape, SUBSTRATE_COUNTS, dtype=np.uint16))
