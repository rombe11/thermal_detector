from __future__ import annotations

import numpy as np
import pytest

from thermal_board.compute.array_backend import (
    OpenCvBilinearSampler,
    select_compute_backend,
    try_gpu_backend,
)

RAMP_IMAGE = np.add.outer(np.arange(30.0) * 3.0, np.arange(40.0) * 2.0).astype(np.float32)
GPU_BACKEND = try_gpu_backend()


def ramp_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    return generator.uniform(1, 38, count), generator.uniform(1, 28, count)


def test_when_linear_ramp_is_sampled_then_bilinear_values_are_exact() -> None:
    x_coordinates, y_coordinates = ramp_points(5000)
    sampled = OpenCvBilinearSampler().sample(RAMP_IMAGE, x_coordinates, y_coordinates)
    np.testing.assert_allclose(sampled, 2.0 * x_coordinates + 3.0 * y_coordinates, atol=0.1)


def test_when_nothing_is_requested_then_sampling_returns_empty_result() -> None:
    empty = np.zeros((0, 3))
    assert OpenCvBilinearSampler().sample(RAMP_IMAGE, empty, empty).shape == (0, 3)


def test_when_gpu_is_not_preferred_then_cpu_backend_is_selected() -> None:
    assert select_compute_backend(prefer_gpu=False).name == "cpu-numpy-opencv"


@pytest.mark.skipif(GPU_BACKEND is None, reason="no CUDA device with CuPy")
def test_when_gpu_is_available_then_it_samples_like_the_cpu() -> None:
    assert GPU_BACKEND is not None
    x_coordinates, y_coordinates = ramp_points(100)
    device = [GPU_BACKEND.to_device(array) for array in (RAMP_IMAGE, x_coordinates, y_coordinates)]
    sampled = GPU_BACKEND.to_host(GPU_BACKEND.sampler.sample(*device))
    np.testing.assert_allclose(sampled, 2.0 * x_coordinates + 3.0 * y_coordinates, atol=0.1)
