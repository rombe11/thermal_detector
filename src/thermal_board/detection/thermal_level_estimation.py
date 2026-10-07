from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

HISTOGRAM_BIN_COUNT = 128
HISTOGRAM_STRIDE = 2


@dataclass(frozen=True, slots=True)
class ThermalLevels:
    cold: float
    substrate: float
    hot: float

    @property
    def hot_threshold(self) -> float:
        return (self.hot + self.substrate) / 2.0

    @property
    def cold_threshold(self) -> float:
        return (self.cold + self.substrate) / 2.0

    @property
    def contrast(self) -> float:
        return self.hot - self.cold


def cumulative_moments(
    histogram: NDArray[np.float64], centers: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weights = np.concatenate([[0.0], np.cumsum(histogram)])
    moments = np.concatenate([[0.0], np.cumsum(histogram * centers)])
    return weights, moments


def class_variance_term(
    weights: NDArray[np.float64],
    moments: NDArray[np.float64],
    start: NDArray[np.intp],
    end: NDArray[np.intp],
) -> NDArray[np.float64]:
    class_weight = weights[end] - weights[start]
    class_moment = moments[end] - moments[start]
    empty = np.zeros_like(class_moment)
    return np.divide(class_moment**2, class_weight, out=empty, where=class_weight > 0)


def best_thresholds(weights: NDArray[np.float64], moments: NDArray[np.float64]) -> tuple[int, int]:
    lower, upper = np.triu_indices(len(weights) - 1, k=1)
    first, last = np.zeros_like(lower), np.full_like(upper, len(weights) - 1)
    variance = class_variance_term(weights, moments, first, lower)
    variance += class_variance_term(weights, moments, lower, upper)
    variance += class_variance_term(weights, moments, upper, last)
    best = int(np.argmax(variance))
    return int(lower[best]), int(upper[best])


def class_means(
    histogram: NDArray[np.float64], centers: NDArray[np.float64], thresholds: tuple[int, int]
) -> list[float]:
    edges = [0, thresholds[0], thresholds[1], len(histogram)]
    means = []
    for start, end in itertools.pairwise(edges):
        weight = max(float(histogram[start:end].sum()), 1e-12)
        means.append(float((histogram[start:end] * centers[start:end]).sum()) / weight)
    return means


def normalized_histogram(
    image: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    samples = image[::HISTOGRAM_STRIDE, ::HISTOGRAM_STRIDE].ravel()
    counts, edges = np.histogram(samples, bins=HISTOGRAM_BIN_COUNT)
    histogram = counts.astype(np.float64) / max(int(counts.sum()), 1)
    return histogram, np.asarray((edges[:-1] + edges[1:]) / 2.0, dtype=np.float64)


def estimate_thermal_levels(image: NDArray[np.float32]) -> ThermalLevels:
    histogram, centers = normalized_histogram(image)
    thresholds = best_thresholds(*cumulative_moments(histogram, centers))
    cold, substrate, hot = class_means(histogram, centers, thresholds)
    return ThermalLevels(cold, substrate, hot)
