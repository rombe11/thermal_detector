from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ExponentialCornerSmoother:
    def __init__(self, smoothing_factor: float) -> None:
        self.smoothing_factor = smoothing_factor
        self.state: NDArray[np.float64] | None = None

    def update(
        self, corners_mm: NDArray[np.float64], valid: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        if self.state is None or self.state.shape != corners_mm.shape:
            self.state = corners_mm.copy()
            return self.state.copy()
        blended = self.smoothing_factor * corners_mm + (1.0 - self.smoothing_factor) * self.state
        self.state = np.where(valid[:, None, None], blended, self.state)
        return self.state.copy()

    def reset(self) -> None:
        self.state = None
