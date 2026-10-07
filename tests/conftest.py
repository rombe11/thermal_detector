from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support import CONFIGURATION_PATH, RenderedScene, render_scene, shrink_configuration
from thermal_board.config.configuration_loader import load_configuration

if TYPE_CHECKING:
    from thermal_board.config.system_settings import SystemConfiguration


@pytest.fixture(scope="session")
def configuration() -> SystemConfiguration:
    return load_configuration(CONFIGURATION_PATH)


@pytest.fixture(scope="session")
def small_configuration(configuration: SystemConfiguration) -> SystemConfiguration:
    return shrink_configuration(configuration)


@pytest.fixture(scope="session")
def small_scene(small_configuration: SystemConfiguration) -> RenderedScene:
    return render_scene(small_configuration)


@pytest.fixture(scope="session")
def full_scene(configuration: SystemConfiguration) -> RenderedScene:
    return render_scene(configuration)
