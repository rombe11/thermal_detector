from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ThermalFrame:
    index: int
    timestamp_s: float
    pixels: NDArray[Any]


class FrameSource(Protocol):
    def frames(self) -> Iterator[ThermalFrame]: ...


class FrameSourceError(RuntimeError):
    pass
