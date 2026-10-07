from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import yaml

from thermal_board.config.configuration_validation import (
    ConfigurationError,
    validate_system_configuration,
)
from thermal_board.config.system_settings import (
    AccuracySettings,
    BoardGeometry,
    CameraSpecification,
    ComputeSettings,
    DetectionSettings,
    RefinementSettings,
    SimulationSettings,
    SystemConfiguration,
    TrackingSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

SectionType = TypeVar("SectionType")


def build_section(section_type: type[SectionType], values: dict[str, Any]) -> SectionType:
    try:
        return section_type(**values)
    except TypeError as error:
        raise ConfigurationError(f"{section_type.__name__}: {error}") from error


def build_camera_specification(values: dict[str, Any]) -> CameraSpecification:
    normalized = dict(values)
    normalized["spectral_band_um"] = tuple(values.get("spectral_band_um", ()))
    normalized["distortion_coefficients"] = tuple(values.get("distortion_coefficients", ()))
    return build_section(CameraSpecification, normalized)


def read_section(document: dict[str, Any], name: str) -> dict[str, Any]:
    section = document.get(name)
    if not isinstance(section, dict):
        raise ConfigurationError(f"missing configuration section '{name}'")
    return section


def build_system_configuration(document: dict[str, Any]) -> SystemConfiguration:
    return SystemConfiguration(
        board=build_section(BoardGeometry, read_section(document, "board")),
        camera=build_camera_specification(read_section(document, "camera")),
        detection=build_section(DetectionSettings, read_section(document, "detection")),
        refinement=build_section(RefinementSettings, read_section(document, "refinement")),
        tracking=build_section(TrackingSettings, read_section(document, "tracking")),
        accuracy=build_section(AccuracySettings, read_section(document, "accuracy")),
        compute=build_section(ComputeSettings, read_section(document, "compute")),
        simulation=build_section(SimulationSettings, read_section(document, "simulation")),
    )


def load_configuration(path: Path) -> SystemConfiguration:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ConfigurationError(f"configuration file {path} is not a mapping")
    configuration = build_system_configuration(document)
    validate_system_configuration(configuration)
    return configuration
