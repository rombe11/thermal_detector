from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from thermal_board.refinement.edge_moment_locator import EdgeCrossings, locate_edge_crossings
from thermal_board.refinement.edge_profile_sampling import EdgeProfileLayout, build_profile_layout
from thermal_board.refinement.line_fitting import (
    FittedLines,
    fit_lines,
    intersect_consecutive_lines,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from thermal_board.compute.array_backend import ComputeBackend
    from thermal_board.config.system_settings import RefinementSettings

MINIMUM_PROFILES_PER_FITTED_EDGE = 2


@dataclass(frozen=True, slots=True)
class RefinedCorners:
    corners_px: NDArray[np.float64]
    edge_rms_px: NDArray[np.float64]
    valid: NDArray[np.bool_]


class SubpixelCornerRefiner:
    def __init__(self, settings: RefinementSettings, backend: ComputeBackend) -> None:
        self.settings = settings
        self.backend = backend
        self.xp = backend.array_module

    def profile_fractions(self) -> Any:
        margin = self.settings.edge_margin_ratio
        return self.xp.linspace(margin, 1.0 - margin, self.settings.profiles_per_edge)

    def layout(self, corners: Any, moment_half_width_px: float) -> EdgeProfileLayout:
        sampled_half_length = moment_half_width_px + self.settings.profile_capture_margin_px
        fractions, step = self.profile_fractions(), self.settings.profile_step_px
        return build_profile_layout(corners, fractions, sampled_half_length, step, self.xp)

    def crossings(
        self, image: Any, layout: EdgeProfileLayout, moment_half_width_px: float
    ) -> EdgeCrossings:
        profiles = self.backend.sampler.sample(image, *layout.sample_coordinates())
        iterations = self.settings.edge_recentering_iterations
        return locate_edge_crossings(
            profiles, layout.offsets_px, moment_half_width_px, iterations, self.xp
        )

    def profile_weights(self, crossings: EdgeCrossings) -> Any:
        strong = crossings.step_contrast >= self.settings.minimum_edge_contrast
        return (strong & crossings.inside_window).astype(self.xp.float64)

    def fit_edges(
        self, layout: EdgeProfileLayout, crossings: EdgeCrossings
    ) -> tuple[FittedLines, Any]:
        anchors, normals, offsets = layout.anchor_points, layout.unit_normals, crossings.offsets_px
        edge_x = anchors[..., 0] + offsets * normals[:, :, None, 0]
        edge_y = anchors[..., 1] + offsets * normals[:, :, None, 1]
        weights = self.profile_weights(crossings)
        return fit_lines(edge_x, edge_y, weights, self.xp), weights.sum(axis=-1)

    def validity(self, lines: FittedLines, support: Any, corner_shift: Any) -> Any:
        enough_support = (support >= MINIMUM_PROFILES_PER_FITTED_EDGE).all(axis=-1)
        straight = (lines.residual_rms_px <= self.settings.maximum_edge_rms_px).all(axis=-1)
        stable = (corner_shift <= self.settings.maximum_corner_shift_px).all(axis=-1)
        return enough_support & straight & stable

    def refine_once(
        self, image: Any, corners: Any, search_half_width_px: float
    ) -> tuple[Any, Any, FittedLines]:
        layout = self.layout(corners, search_half_width_px)
        crossings = self.crossings(image, layout, search_half_width_px)
        lines, support = self.fit_edges(layout, crossings)
        return intersect_consecutive_lines(lines.coefficients, self.xp), support, lines

    def refine(
        self,
        image: NDArray[np.float32],
        predicted_corners: NDArray[np.float64],
        search_half_width_px: float,
    ) -> RefinedCorners:
        device_image = self.backend.to_device(image)
        corners = predicted = self.backend.to_device(predicted_corners)
        for _ in range(self.settings.refinement_passes):
            corners, support, lines = self.refine_once(device_image, corners, search_half_width_px)
        corner_shift = self.xp.linalg.norm(corners - predicted, axis=-1)
        valid = self.validity(lines, support, corner_shift)
        edge_rms = self.xp.sqrt((lines.residual_rms_px**2).mean(axis=-1))
        return RefinedCorners(
            *(self.backend.to_host(array) for array in (corners, edge_rms, valid))
        )
