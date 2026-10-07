from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MINIMUM_WEIGHT_SUM = 1e-12


@dataclass(frozen=True, slots=True)
class FittedLines:
    coefficients: Any
    residual_rms_px: Any


def weighted_average(values: Any, weights: Any, weight_sum: Any) -> Any:
    return (values * weights).sum(axis=-1) / weight_sum


def minor_axis(xx: Any, xy: Any, yy: Any, xp: Any) -> tuple[Any, Any]:
    major_angle = 0.5 * xp.arctan2(2.0 * xy, xx - yy)
    normal = xp.stack([-xp.sin(major_angle), xp.cos(major_angle)], axis=-1)
    half_trace, half_difference = (xx + yy) / 2.0, (xx - yy) / 2.0
    smallest_eigenvalue = half_trace - xp.sqrt(half_difference**2 + xy**2)
    return normal, xp.maximum(smallest_eigenvalue, 0.0)


def weighted_moments(x_values: Any, y_values: Any, weights: Any, xp: Any) -> tuple[Any, ...]:
    weight_sum = xp.maximum(weights.sum(axis=-1), MINIMUM_WEIGHT_SUM)
    mean_x = weighted_average(x_values, weights, weight_sum)
    mean_y = weighted_average(y_values, weights, weight_sum)
    centered_x, centered_y = x_values - mean_x[..., None], y_values - mean_y[..., None]
    xx = weighted_average(centered_x * centered_x, weights, weight_sum)
    xy = weighted_average(centered_x * centered_y, weights, weight_sum)
    yy = weighted_average(centered_y * centered_y, weights, weight_sum)
    return mean_x, mean_y, xx, xy, yy


def fit_lines(x_values: Any, y_values: Any, weights: Any, xp: Any) -> FittedLines:
    mean_x, mean_y, xx, xy, yy = weighted_moments(x_values, y_values, weights, xp)
    normal, residual_variance = minor_axis(xx, xy, yy, xp)
    offset = -(normal[..., 0] * mean_x + normal[..., 1] * mean_y)
    coefficients = xp.concatenate([normal, offset[..., None]], axis=-1)
    return FittedLines(coefficients, xp.sqrt(residual_variance))


def intersect_consecutive_lines(lines: Any, xp: Any) -> Any:
    previous_lines = xp.roll(lines, shift=1, axis=-2)
    homogeneous = xp.cross(previous_lines, lines)
    scale = homogeneous[..., 2:3]
    safe_scale = xp.where(xp.abs(scale) < MINIMUM_WEIGHT_SUM, MINIMUM_WEIGHT_SUM, scale)
    return homogeneous[..., :2] / safe_scale
