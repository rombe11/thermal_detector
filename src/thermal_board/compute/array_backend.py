from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray


class SubpixelSampler(Protocol):
    def sample(self, image: Any, x_coordinates: Any, y_coordinates: Any) -> Any: ...


REMAP_ROW_LENGTH = 1024


def tile_for_remap(coordinates: NDArray[Any]) -> NDArray[np.float32]:
    flat = np.asarray(coordinates, dtype=np.float32).ravel()
    padding = (-len(flat)) % REMAP_ROW_LENGTH
    return np.pad(flat, (0, padding), mode="edge").reshape(-1, REMAP_ROW_LENGTH)


class OpenCvBilinearSampler:
    def sample(self, image: Any, x_coordinates: Any, y_coordinates: Any) -> Any:
        if x_coordinates.size == 0:
            return np.zeros(x_coordinates.shape, dtype=np.float32)
        map_x, map_y = tile_for_remap(x_coordinates), tile_for_remap(y_coordinates)
        sampled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        flat = np.asarray(sampled, dtype=np.float32).ravel()[: x_coordinates.size]
        return flat.reshape(x_coordinates.shape)


class CupyBilinearSampler:
    def __init__(self, cupy_module: ModuleType) -> None:
        self.cupy = cupy_module
        self.ndimage = importlib.import_module("cupyx.scipy.ndimage")

    def sample(self, image: Any, x_coordinates: Any, y_coordinates: Any) -> Any:
        row_column = self.cupy.stack([y_coordinates.ravel(), x_coordinates.ravel()])
        sampled = self.ndimage.map_coordinates(image, row_column, order=1, mode="nearest")
        return sampled.reshape(x_coordinates.shape)


@dataclass(frozen=True, slots=True)
class ComputeBackend:
    name: str
    array_module: ModuleType
    sampler: SubpixelSampler

    def to_device(self, array: NDArray[Any]) -> Any:
        return self.array_module.asarray(array)

    def to_host(self, array: Any) -> NDArray[Any]:
        if self.array_module is np:
            return np.asarray(array)
        return np.asarray(self.array_module.asnumpy(array))


def cpu_backend() -> ComputeBackend:
    return ComputeBackend("cpu-numpy-opencv", np, OpenCvBilinearSampler())


def try_gpu_backend() -> ComputeBackend | None:
    try:
        cupy_module = importlib.import_module("cupy")
        if cupy_module.cuda.runtime.getDeviceCount() == 0:
            return None
        return ComputeBackend("gpu-cupy-cuda", cupy_module, CupyBilinearSampler(cupy_module))
    except (ImportError, RuntimeError, AttributeError):
        return None


def select_compute_backend(prefer_gpu: bool) -> ComputeBackend:
    gpu_backend = try_gpu_backend() if prefer_gpu else None
    return gpu_backend if gpu_backend is not None else cpu_backend()
