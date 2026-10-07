from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    board_width_mm: float
    board_height_mm: float
    cell_width_mm: float
    cell_height_mm: float
    gap_mm: float
    columns: int
    rows: int

    @property
    def pitch_x_mm(self) -> float:
        return self.cell_width_mm + self.gap_mm

    @property
    def pitch_y_mm(self) -> float:
        return self.cell_height_mm + self.gap_mm

    @property
    def grid_width_mm(self) -> float:
        return self.columns * self.pitch_x_mm - self.gap_mm

    @property
    def grid_height_mm(self) -> float:
        return self.rows * self.pitch_y_mm - self.gap_mm

    @property
    def border_x_mm(self) -> float:
        return (self.board_width_mm - self.grid_width_mm) / 2.0

    @property
    def border_y_mm(self) -> float:
        return (self.board_height_mm - self.grid_height_mm) / 2.0

    @property
    def cell_count(self) -> int:
        return self.columns * self.rows


@dataclass(frozen=True, slots=True)
class CameraSpecification:
    sensor_type: str
    spectral_band_um: tuple[float, float]
    width_px: int
    height_px: int
    frame_rate_hz: float
    bit_depth: int
    pixel_pitch_um: float
    focal_length_mm: float
    working_distance_m: float
    distortion_coefficients: tuple[float, ...]

    @property
    def focal_length_px(self) -> float:
        return self.focal_length_mm * 1000.0 / self.pixel_pitch_um

    @property
    def ground_sample_distance_mm(self) -> float:
        return self.working_distance_m * 1000.0 / self.focal_length_px

    @property
    def has_lens_distortion(self) -> bool:
        return any(coefficient != 0.0 for coefficient in self.distortion_coefficients)


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    normalization_low_percentile: float
    normalization_high_percentile: float
    denoise_sigma_px: float
    blob_area_min_ratio: float
    blob_area_max_ratio: float
    match_tolerance_pitch_ratio: float
    ransac_threshold_px: float
    minimum_match_ratio: float


@dataclass(frozen=True, slots=True)
class RefinementSettings:
    profiles_per_edge: int
    edge_margin_ratio: float
    profile_half_length_ratio: float
    profile_step_px: float
    minimum_edge_contrast: float
    refinement_passes: int
    profile_capture_margin_px: float
    edge_recentering_iterations: int
    maximum_edge_rms_px: float
    maximum_corner_shift_px: float
    metric_outlier_threshold_px: float


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    smoothing_factor: float
    vibration_window_frames: int
    maximum_frame_shift_pitch_ratio: float


@dataclass(frozen=True, slots=True)
class AccuracySettings:
    target_precision_mm: float
    tolerance_mm: float


@dataclass(frozen=True, slots=True)
class ComputeSettings:
    prefer_gpu: bool


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    substrate_level: float
    hot_level: float
    cold_level: float
    background_level: float
    noise_sigma: float
    optical_blur_sigma_px: float
    supersampling: int
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    vibration_amplitude_px: float


@dataclass(frozen=True, slots=True)
class SystemConfiguration:
    board: BoardGeometry
    camera: CameraSpecification
    detection: DetectionSettings
    refinement: RefinementSettings
    tracking: TrackingSettings
    accuracy: AccuracySettings
    compute: ComputeSettings
    simulation: SimulationSettings
