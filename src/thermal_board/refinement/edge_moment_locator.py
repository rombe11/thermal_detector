from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MINIMUM_STEP_CONTRAST = 1e-9


@dataclass(frozen=True, slots=True)
class EdgeCrossings:
    offsets_px: Any
    step_contrast: Any
    inside_window: Any


@dataclass(frozen=True, slots=True)
class SampledProfiles:
    values: Any
    cumulative_area: Any
    columns: Any
    first_offset_px: float
    step_px: float

    def interpolate(self, positions_px: Any, xp: Any) -> tuple[Any, Any]:
        index_float = (positions_px - self.first_offset_px) / self.step_px
        index = xp.clip(xp.floor(index_float), 0, self.values.shape[0] - 2).astype(xp.int64)
        fraction = index_float - index
        left, right = self.values[index, self.columns], self.values[index + 1, self.columns]
        slope = right - left
        area = self.cumulative_area[index, self.columns] + self.step_px * fraction * (
            left + slope * fraction / 2.0
        )
        return left + slope * fraction, area


def cumulative_trapezoid(sample_major_profiles: Any, step_px: float, xp: Any) -> Any:
    cumulative = xp.zeros_like(sample_major_profiles)
    pair_sums = sample_major_profiles[1:] + sample_major_profiles[:-1]
    xp.cumsum(pair_sums, axis=0, out=cumulative[1:])
    return cumulative * (step_px / 2.0)


def windowed_moment(
    profiles: SampledProfiles, centre_px: Any, half_width_px: float, xp: Any
) -> tuple[Any, Any]:
    low_value, low_area = profiles.interpolate(centre_px - half_width_px, xp)
    high_value, high_area = profiles.interpolate(centre_px + half_width_px, xp)
    contrast = high_value - low_value
    safe_contrast = xp.where(
        xp.abs(contrast) < MINIMUM_STEP_CONTRAST, MINIMUM_STEP_CONTRAST, contrast
    )
    normalized_area = (high_area - low_area - low_value * 2.0 * half_width_px) / safe_contrast
    return centre_px + half_width_px - normalized_area, contrast


def sampled_profiles(profiles: Any, offsets_px: Any, xp: Any) -> SampledProfiles:
    sample_major = profiles.reshape(profiles.shape[0], -1)
    step_px = float(offsets_px[1] - offsets_px[0])
    cumulative = cumulative_trapezoid(sample_major, step_px, xp)
    columns = xp.arange(sample_major.shape[1])
    return SampledProfiles(sample_major, cumulative, columns, float(offsets_px[0]), step_px)


def recentre_windows(
    sampled: SampledProfiles, half_width_px: float, iterations: int, xp: Any
) -> tuple[Any, Any]:
    capture_px = max(sampled.first_offset_px * -1.0 - half_width_px, 0.0)
    centre = xp.zeros(sampled.values.shape[1], dtype=sampled.values.dtype)
    for _ in range(max(iterations, 1)):
        estimate, contrast = windowed_moment(sampled, centre, half_width_px, xp)
        centre = xp.clip(estimate, -capture_px, capture_px)
    return estimate, contrast


def locate_edge_crossings(
    profiles: Any, offsets_px: Any, half_width_px: float, iterations: int, xp: Any
) -> EdgeCrossings:
    sampled = sampled_profiles(profiles, offsets_px, xp)
    estimate, contrast = recentre_windows(sampled, half_width_px, iterations, xp)
    inside = xp.abs(estimate) < float(offsets_px[-1])
    shape = profiles.shape[1:]
    return EdgeCrossings(
        estimate.reshape(shape), xp.abs(contrast).reshape(shape), inside.reshape(shape)
    )
