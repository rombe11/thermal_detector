from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from thermal_board.acquisition.frame_source_factory import create_frame_source
from thermal_board.acquisition.recorded_sources import (
    ImageSequenceSource,
    VideoFileSource,
    list_image_files,
    read_thermal_image,
)
from thermal_board.acquisition.thermal_frame import FrameSourceError
from thermal_board.simulation.synthetic_sequence import SyntheticSequenceSource

if TYPE_CHECKING:
    from pathlib import Path

    from thermal_board.config.system_settings import SystemConfiguration

RADIOMETRIC_IMAGE = (np.arange(64 * 48).reshape(48, 64) * 5).astype(np.uint16)


def write_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (64, 48))
    for index in range(frame_count):
        writer.write(np.full((48, 64, 3), index * 40, dtype=np.uint8))
    writer.release()


@pytest.mark.parametrize("suffix", [".png", ".tiff", ".npy"])
def test_when_16_bit_frame_is_read_then_radiometric_counts_are_preserved(
    tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"frame{suffix}"
    if suffix == ".npy":
        np.save(path, RADIOMETRIC_IMAGE)
    else:
        cv2.imwrite(str(path), RADIOMETRIC_IMAGE)
    np.testing.assert_array_equal(read_thermal_image(path), RADIOMETRIC_IMAGE)


def test_when_colour_frame_is_read_then_it_becomes_single_channel(tmp_path: Path) -> None:
    path = tmp_path / "colour.png"
    cv2.imwrite(str(path), np.full((8, 8, 3), 90, np.uint8))
    assert read_thermal_image(path).shape == (8, 8)


def test_when_image_is_missing_then_frame_source_error_is_raised(tmp_path: Path) -> None:
    with pytest.raises(FrameSourceError, match="cannot read"):
        read_thermal_image(tmp_path / "missing.png")


def test_when_video_is_read_then_grayscale_frames_arrive_in_order(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    write_video(path, 3)
    frames = list(VideoFileSource(path).frames())
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert frames[0].pixels.ndim == 2
    assert frames[2].pixels.mean() > frames[0].pixels.mean()


def test_when_folder_is_listed_then_only_images_are_kept_in_order(tmp_path: Path) -> None:
    for name in ("b.png", "a.tiff", "notes.txt", "c.npy"):
        (tmp_path / name).write_bytes(b"")
    assert [path.name for path in list_image_files(tmp_path)] == ["a.tiff", "b.png", "c.npy"]


@pytest.mark.parametrize(
    ("specification", "expected_type"),
    [
        ("synthetic", SyntheticSequenceSource),
        ("{folder}", ImageSequenceSource),
        ("{folder}/frame.png", ImageSequenceSource),
        ("{folder}/clip.mp4", VideoFileSource),
    ],
)
def test_when_source_is_specified_then_matching_source_is_created(
    tmp_path: Path,
    small_configuration: SystemConfiguration,
    specification: str,
    expected_type: type,
) -> None:
    cv2.imwrite(str(tmp_path / "frame.png"), RADIOMETRIC_IMAGE)
    source = create_frame_source(specification.format(folder=tmp_path), small_configuration)
    assert isinstance(source, expected_type)
