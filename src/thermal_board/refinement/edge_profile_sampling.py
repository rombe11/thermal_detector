from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EDGE_COUNT = 4


@dataclass(frozen=True, slots=True)
class EdgeProfileLayout:
    anchor_points: Any
    unit_normals: Any
    offsets_px: Any

    def sample_coordinates(self) -> tuple[Any, Any]:
        anchors = self.anchor_points.astype(self.offsets_px.dtype)
        offsets = self.offsets_px[:, None, None, None]
        normals = self.unit_normals[None, :, :, None, :]
        return anchors[None, ..., 0] + offsets * normals[..., 0], anchors[
            None, ..., 1
        ] + offsets * normals[..., 1]


def edge_endpoints(corners: Any, xp: Any) -> tuple[Any, Any]:
    following_corners = xp.roll(corners, shift=-1, axis=1)
    return corners, following_corners


def edge_unit_normals(starts: Any, ends: Any, xp: Any) -> Any:
    direction = ends - starts
    length = xp.linalg.norm(direction, axis=-1, keepdims=True)
    unit_direction = direction / xp.maximum(length, 1e-12)
    return xp.stack([-unit_direction[..., 1], unit_direction[..., 0]], axis=-1)


def edge_anchor_points(starts: Any, ends: Any, fractions: Any) -> Any:
    weights = fractions[None, None, :, None]
    return starts[:, :, None, :] * (1.0 - weights) + ends[:, :, None, :] * weights


def build_profile_layout(
    corners: Any, fractions: Any, half_length_px: float, step_px: float, xp: Any
) -> EdgeProfileLayout:
    starts, ends = edge_endpoints(corners, xp)
    sample_count = 2 * round(half_length_px / step_px) + 1
    offsets = (xp.arange(sample_count, dtype=xp.float32) - (sample_count - 1) / 2.0) * step_px
    anchors = edge_anchor_points(starts, ends, fractions)
    normals = edge_unit_normals(starts, ends, xp).astype(xp.float32)
    return EdgeProfileLayout(anchors, normals, offsets.astype(xp.float32))
