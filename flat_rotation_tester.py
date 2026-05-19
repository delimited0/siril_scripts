#!/usr/bin/env python3
"""
Flat Rotation Tester for Siril

GUI tool for diagnosing flat-frame rotation mismatches in astrophotography data.
It builds a master flat from a directory of FITS flats, derives a dust-feature
mask from that flat, rotates both the flat and dust mask through a candidate
angle range, flat-corrects a selected FITS light, scores residual artifacts only
inside the dust-feature regions, and saves one diagnostic PNG per angle without
modifying the original FITS files.

Requirements:
- PyQt5
- numpy
- astropy
- scipy
- pillow
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from PIL import Image
from PyQt5.QtCore import QThread, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage

try:
    import sirilpy

    SIRILPY_AVAILABLE = True
except ImportError:
    SIRILPY_AVAILABLE = False


SUPPORTED_FITS_SUFFIXES = {".fit", ".fits", ".fts"}


@dataclass
class DiagnosticResult:
    angle: float
    score: float
    png_path: Path


@dataclass
class FITSFrame:
    data: np.ndarray
    bayer_pattern: Optional[str]


def collect_fits_files(directory: Path) -> List[Path]:
    """Return sorted FITS files from a directory."""
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FITS_SUFFIXES
    )


def canonicalize_fits_array(data: np.ndarray) -> np.ndarray:
    """Normalize FITS array layout to either 2D or channel-first 3D."""
    array = np.asarray(data, dtype=np.float32)
    array = np.squeeze(array)

    if array.ndim == 2:
        return array

    if array.ndim != 3:
        raise ValueError(f"Unsupported FITS dimensions: {array.shape}")

    if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
        return array

    if array.shape[2] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
        return np.moveaxis(array, -1, 0)

    raise ValueError(f"Could not determine image axes for FITS data shape {array.shape}")


def bayer_channel_offsets(pattern: str) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, int]]:
    """Return channel offsets for a Bayer pattern."""
    normalized = pattern.strip().upper()
    if normalized == "RGGB":
        return (0, 0), ((0, 1), (1, 0)), (1, 1)
    if normalized == "BGGR":
        return (1, 1), ((0, 1), (1, 0)), (0, 0)
    if normalized == "GRBG":
        return (0, 1), ((0, 0), (1, 1)), (1, 0)
    if normalized == "GBRG":
        return (1, 0), ((0, 0), (1, 1)), (0, 1)
    raise ValueError(f"Unsupported Bayer pattern: {pattern}")


def bilinear_reconstruct(mosaic: np.ndarray, offsets: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    """Reconstruct a Bayer channel using normalized bilinear interpolation."""
    plane = np.zeros_like(mosaic, dtype=np.float32)
    mask = np.zeros_like(mosaic, dtype=np.float32)

    for row_offset, col_offset in offsets:
        plane[row_offset::2, col_offset::2] = mosaic[row_offset::2, col_offset::2]
        mask[row_offset::2, col_offset::2] = 1.0

    kernel = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )

    weighted_values = ndimage.convolve(plane, kernel, mode="mirror")
    weighted_mask = ndimage.convolve(mask, kernel, mode="mirror")
    reconstructed = weighted_values / np.clip(weighted_mask, 1e-6, None)
    return reconstructed.astype(np.float32, copy=False)


def debayer_image(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """Debayer a 2D FITS mosaic to channel-first RGB."""
    red_offset, green_offsets, blue_offset = bayer_channel_offsets(pattern)
    red = bilinear_reconstruct(mosaic, (red_offset,))
    green = bilinear_reconstruct(mosaic, green_offsets)
    blue = bilinear_reconstruct(mosaic, (blue_offset,))
    return np.stack([red, green, blue], axis=0)


def read_fits_frame(path: Path, debayer: bool = False) -> FITSFrame:
    """Read the first image HDU from a FITS file, optionally debayering CFA data."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                data = canonicalize_fits_array(hdu.data)
                bayer_pattern = hdu.header.get("BAYERPAT")
                if debayer and data.ndim == 2 and bayer_pattern:
                    data = debayer_image(data, str(bayer_pattern))
                return FITSFrame(data=data, bayer_pattern=str(bayer_pattern).upper() if bayer_pattern else None)
    raise ValueError(f"No image data found in {path}")


def save_fits_frame(path: Path, data: np.ndarray, bayer_pattern: Optional[str] = None) -> None:
    """Save a FITS frame used by the analysis."""
    header = fits.Header()
    if bayer_pattern:
        header["BAYERPAT"] = bayer_pattern
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)


def robust_positive_median(data: np.ndarray) -> float:
    """Return a robust positive median for normalization."""
    finite = np.isfinite(data)
    positive = finite & (data > 0)

    if np.any(positive):
        return float(np.median(data[positive]))

    if np.any(finite):
        return float(np.median(data[finite]))

    raise ValueError("Image contains no finite values")


def normalize_flat(flat_data: np.ndarray) -> np.ndarray:
    """Normalize a flat field to a median of 1.0."""
    if flat_data.ndim == 3:
        scales = np.array([robust_positive_median(channel) for channel in flat_data], dtype=np.float32)
        if np.any(~np.isfinite(scales)) or np.any(scales == 0):
            raise ValueError("Master flat normalization failed")
        return flat_data / scales[:, np.newaxis, np.newaxis]

    scale = robust_positive_median(flat_data)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Master flat normalization failed")
    return flat_data / scale


def rotate_image(data: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate an image around its center while keeping its original size."""
    if data.ndim == 2:
        rotated = ndimage.rotate(
            data,
            angle_degrees,
            reshape=False,
            order=3,
            mode="nearest",
            prefilter=True,
        )
        return rotated.astype(np.float32, copy=False)

    channels = [
        ndimage.rotate(
            channel,
            angle_degrees,
            reshape=False,
            order=3,
            mode="nearest",
            prefilter=True,
        ).astype(np.float32, copy=False)
        for channel in data
    ]
    return np.stack(channels, axis=0)


def normalize_light_for_stack(light_data: np.ndarray) -> np.ndarray:
    """Normalize a corrected light before combining diagnostics."""
    if light_data.ndim == 3:
        scales = np.array([robust_positive_median(channel) for channel in light_data], dtype=np.float32)
        if np.any(~np.isfinite(scales)) or np.any(scales == 0):
            raise ValueError("Light normalization failed")
        return light_data / scales[:, np.newaxis, np.newaxis]

    scale = robust_positive_median(light_data)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Light normalization failed")
    return light_data / scale


def corrected_light(light_data: np.ndarray, rotated_flat: np.ndarray) -> np.ndarray:
    """Apply flat correction using a safely clipped flat."""
    if light_data.ndim == 3:
        safe_channels = []
        for channel in rotated_flat:
            finite = np.isfinite(channel)
            positive = finite & (channel > 0)

            if not np.any(positive):
                raise ValueError("Rotated flat contains no positive finite values")

            floor = max(float(np.percentile(channel[positive], 1)), 1e-6)
            safe_channels.append(np.where(positive, np.clip(channel, floor, None), floor))

        safe_flat = np.stack(safe_channels, axis=0)
        corrected = light_data / safe_flat
        return corrected.astype(np.float32, copy=False)

    finite = np.isfinite(rotated_flat)
    positive = finite & (rotated_flat > 0)

    if not np.any(positive):
        raise ValueError("Rotated flat contains no positive finite values")

    floor = max(float(np.percentile(rotated_flat[positive], 1)), 1e-6)
    safe_flat = np.where(positive, np.clip(rotated_flat, floor, None), floor)
    corrected = light_data / safe_flat
    return corrected.astype(np.float32, copy=False)


def reference_plane(data: np.ndarray) -> np.ndarray:
    """Collapse channel-first data to a single analysis plane."""
    if data.ndim == 2:
        return np.asarray(data, dtype=np.float32)
    return np.nanmean(np.asarray(data, dtype=np.float32), axis=0).astype(np.float32, copy=False)


def robust_mad_sigma(values: np.ndarray) -> float:
    """Estimate a robust standard deviation using the median absolute deviation."""
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Residual metric received no finite values")

    center = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - center)))
    return max(1.4826 * mad, 1e-6)


def circular_structure(radius: int) -> np.ndarray:
    """Return a circular boolean structuring element."""
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def derive_dust_feature_mask(master_flat: np.ndarray) -> np.ndarray:
    """Detect dust motes and strong localized flat features in the normalized master flat."""
    normalized_flat = normalize_flat(master_flat)
    feature_plane = reference_plane(normalized_flat)
    finite_mask = np.isfinite(feature_plane)
    finite_values = feature_plane[finite_mask]
    if finite_values.size < 32:
        raise ValueError("Not enough finite flat pixels to build a dust mask")

    median_level = float(np.median(finite_values))
    filled_plane = np.where(finite_mask, feature_plane, median_level)
    smooth_sigma = max(24.0, min(feature_plane.shape) / 18.0)
    smooth_flat = ndimage.gaussian_filter(filled_plane, sigma=smooth_sigma, mode="nearest")
    detail = filled_plane - smooth_flat
    detail_values = detail[finite_mask]
    detail_center = float(np.median(detail_values))
    detail_sigma = robust_mad_sigma(detail_values)

    dark_seed = finite_mask & (detail <= detail_center - 2.5 * detail_sigma)
    strong_seed = finite_mask & (np.abs(detail - detail_center) >= 4.0 * detail_sigma)
    feature_seed = dark_seed | strong_seed
    if np.count_nonzero(feature_seed) < 16:
        feature_seed = finite_mask & (detail <= detail_center - 1.8 * detail_sigma)
    if np.count_nonzero(feature_seed) < 16:
        raise ValueError("Could not detect enough dust features in the master flat")

    dilation_radius = max(3, int(round(min(feature_plane.shape) / 256.0)))
    feature_mask = ndimage.binary_dilation(feature_seed, structure=circular_structure(dilation_radius))
    feature_mask = ndimage.binary_fill_holes(feature_mask)
    return feature_mask.astype(bool, copy=False)


def rotate_mask(mask: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate a boolean mask while keeping its original size."""
    rotated = ndimage.rotate(
        mask.astype(np.float32),
        angle_degrees,
        reshape=False,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return rotated >= 0.5


def compute_masked_residual_score(corrected_data: np.ndarray, dust_mask: np.ndarray) -> Tuple[float, int]:
    """Score residual artifacts only inside dust-feature regions."""
    normalized = normalize_light_for_stack(corrected_data)
    analysis_plane = reference_plane(normalized)
    finite_mask = np.isfinite(analysis_plane)
    candidate_mask = finite_mask & dust_mask
    if np.count_nonzero(candidate_mask) < 32:
        raise ValueError("Rotated dust mask does not cover enough valid pixels")

    candidate_values = analysis_plane[candidate_mask]
    median_level = float(np.median(candidate_values))
    global_sigma = robust_mad_sigma(candidate_values)
    saturation_floor = median_level + 12.0 * global_sigma
    saturation_percentile = float(np.percentile(candidate_values, 99.9))
    saturation_mask = finite_mask & (analysis_plane >= max(saturation_floor, saturation_percentile))

    local_fill = np.where(finite_mask, analysis_plane, median_level)
    star_sigma = max(2.0, min(analysis_plane.shape) / 256.0)
    local_background = ndimage.gaussian_filter(local_fill, sigma=star_sigma, mode="nearest")
    local_residual = analysis_plane - local_background
    local_values = local_residual[candidate_mask]
    local_center = float(np.median(local_values))
    local_sigma = robust_mad_sigma(local_values)
    star_seed = finite_mask & (local_residual >= local_center + 6.0 * local_sigma)
    star_mask = ndimage.binary_dilation(
        star_seed,
        structure=circular_structure(max(1, int(round(min(analysis_plane.shape) / 512.0)))),
    )

    analysis_mask = candidate_mask & ~saturation_mask & ~star_mask
    if np.count_nonzero(analysis_mask) < 32:
        analysis_mask = candidate_mask & ~saturation_mask
    if np.count_nonzero(analysis_mask) < 32:
        analysis_mask = candidate_mask

    fill_level = float(np.median(analysis_plane[analysis_mask]))
    filled_plane = np.where(analysis_mask, analysis_plane, fill_level)
    background_sigma = max(12.0, min(analysis_plane.shape) / 24.0)
    smooth_background = ndimage.gaussian_filter(filled_plane, sigma=background_sigma, mode="nearest")
    residual = analysis_plane - smooth_background
    return robust_mad_sigma(residual[analysis_mask]), int(np.count_nonzero(analysis_mask))


def angle_series(min_angle: float, max_angle: float, step: float) -> List[float]:
    """Build an inclusive candidate angle list."""
    if step <= 0:
        raise ValueError("Angle step must be greater than zero")
    if max_angle < min_angle:
        raise ValueError("Angle range maximum must be greater than or equal to minimum")

    angles: List[float] = []
    current = min_angle
    epsilon = step / 1000.0

    while current <= max_angle + epsilon:
        angles.append(round(current, 6))
        current += step

    return angles


def stretch_for_png(data: np.ndarray) -> np.ndarray:
    """Convert normalized data to an 8-bit array for PNG export."""
    if data.ndim == 2:
        finite = np.isfinite(data)
        if not np.any(finite):
            raise ValueError("Diagnostic image contains no finite values")
        values = data[finite]
        low = float(np.percentile(values, 1))
        high = float(np.percentile(values, 99.5))
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        stretched = np.clip((data - low) / (high - low), 0, 1)
        return (stretched * 255).astype(np.uint8)

    channels = []
    for channel in data[:3]:
        finite = np.isfinite(channel)
        if not np.any(finite):
            raise ValueError("Diagnostic image contains no finite channel values")
        values = channel[finite]
        low = float(np.percentile(values, 1))
        high = float(np.percentile(values, 99.5))
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        stretched = np.clip((channel - low) / (high - low), 0, 1)
        channels.append((stretched * 255).astype(np.uint8))

    rgb = np.stack(channels, axis=-1)
    if rgb.shape[-1] == 1:
        return rgb[..., 0]
    return rgb


def save_png(data: np.ndarray, path: Path) -> None:
    """Save a 2D or RGB array as PNG."""
    png_data = stretch_for_png(data)
    image = Image.fromarray(png_data)
    max_dimension = 1600
    if max(image.size) > max_dimension:
        scale = max_dimension / float(max(image.size))
        resized = (
            max(1, int(round(image.size[0] * scale))),
            max(1, int(round(image.size[1] * scale))),
        )
        image = image.resize(resized, Image.Resampling.BILINEAR)
    if image.mode == "RGB":
        image = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    image.save(path, optimize=True, compress_level=9)


class ProcessingWorker(QThread):
    """Background worker for rotation testing."""

    log_message = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, str, object, str)

    def __init__(
        self,
        flat_dir: Path,
        light_file: Path,
        min_angle: float,
        max_angle: float,
        step: float,
        output_dir: Path,
    ):
        super().__init__()
        self.flat_dir = flat_dir
        self.light_file = light_file
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.step = step
        self.output_dir = output_dir

    def emit_log(self, message: str, color: str = "black") -> None:
        self.log_message.emit(message, color)

    def run(self) -> None:
        try:
            results = self.process()
            best_result = min(results, key=lambda result: result.score)
            self.finished.emit(
                True,
                f"Best angle: {best_result.angle:+.3f} degrees (score {best_result.score:.6f})",
                results,
                str(self.output_dir),
            )
        except Exception as exc:
            self.finished.emit(False, str(exc), [], str(self.output_dir))

    def process(self) -> List[DiagnosticResult]:
        flat_files = collect_fits_files(self.flat_dir)

        if not flat_files:
            raise ValueError("No FITS flat files found in the selected flat directory")
        if not self.light_file.exists() or self.light_file.suffix.lower() not in SUPPORTED_FITS_SUFFIXES:
            raise ValueError("Selected light file is not a valid FITS file")

        angles = angle_series(self.min_angle, self.max_angle, self.step)
        total_steps = len(angles) + len(flat_files) + 1
        completed_steps = 0

        self.emit_log(f"Found {len(flat_files)} flat files", "blue")
        self.emit_log(f"Using light file {self.light_file.name}", "blue")
        self.emit_log(f"Testing {len(angles)} rotation angles", "blue")

        flat_stack = []
        expected_shape: Optional[Tuple[int, ...]] = None
        flat_bayer_patterns = set()

        for flat_file in flat_files:
            flat_frame = read_fits_frame(flat_file, debayer=False)
            flat_data = flat_frame.data
            if expected_shape is None:
                expected_shape = flat_data.shape
            elif flat_data.shape != expected_shape:
                raise ValueError(
                    f"Flat shape mismatch: {flat_file.name} has shape {flat_data.shape},"
                    f" expected {expected_shape}"
                )

            flat_stack.append(flat_data)
            if flat_frame.bayer_pattern:
                flat_bayer_patterns.add(flat_frame.bayer_pattern)
            completed_steps += 1
            self.progress_update.emit(int(completed_steps / total_steps * 100))

        if len(flat_bayer_patterns) > 1:
            raise ValueError(f"Flat Bayer pattern mismatch: {sorted(flat_bayer_patterns)}")

        results: List[DiagnosticResult] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        master_flat = np.nanmedian(np.stack(flat_stack, axis=0), axis=0).astype(np.float32)
        master_flat = normalize_flat(master_flat)
        master_flat_path = self.output_dir / "analysis_master_flat.fit"
        master_bayer_pattern = next(iter(flat_bayer_patterns)) if flat_bayer_patterns else None
        save_fits_frame(master_flat_path, master_flat, master_bayer_pattern)
        if flat_bayer_patterns:
            self.emit_log(
                f"Built raw CFA master flat from {len(flat_stack)} files using {master_bayer_pattern}",
                "green",
            )
        else:
            self.emit_log(f"Built master flat from {len(flat_stack)} files", "green")
        self.emit_log(f"Saved analysis master flat to {master_flat_path}", "green")

        dust_feature_mask = derive_dust_feature_mask(master_flat)
        dust_pixels = int(np.count_nonzero(dust_feature_mask))
        self.emit_log(f"Derived dust-feature mask with {dust_pixels} pixels", "green")

        light_frame = read_fits_frame(self.light_file, debayer=False)
        light_data = light_frame.data
        if light_data.shape != master_flat.shape:
            raise ValueError(
                f"Light shape mismatch: {self.light_file.name} has shape {light_data.shape},"
                f" expected {master_flat.shape}"
            )

        light_bayer_patterns = {light_frame.bayer_pattern} if light_frame.bayer_pattern else set()
        if flat_bayer_patterns and light_bayer_patterns and flat_bayer_patterns != light_bayer_patterns:
            raise ValueError(
                f"Flat and light Bayer patterns differ: flats={sorted(flat_bayer_patterns)},"
                f" lights={sorted(light_bayer_patterns)}"
            )

        completed_steps += 1
        self.progress_update.emit(int(completed_steps / total_steps * 100))

        manifest = {
            "flat_directory": str(self.flat_dir),
            "light_file": str(self.light_file),
            "analysis_master_flat": master_flat_path.name,
            "angles": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "processed_as_raw_cfa": bool(flat_bayer_patterns),
            "bayer_pattern": master_bayer_pattern,
            "dust_mask_pixels": dust_pixels,
            "best_angle_degrees": None,
            "best_score": None,
        }

        for angle in angles:
            self.emit_log(f"Processing angle {angle:+.3f} degrees", "blue")
            rotated_flat = normalize_flat(rotate_image(master_flat, angle))
            rotated_mask = rotate_mask(dust_feature_mask, angle)
            corrected = corrected_light(light_data, rotated_flat)
            normalized_corrected = normalize_light_for_stack(corrected)
            score, analysis_pixels = compute_masked_residual_score(corrected, rotated_mask)
            png_name = f"flat_rotation_{angle:+08.3f}deg.png".replace("+", "p").replace("-", "m")
            png_path = self.output_dir / png_name
            save_png(normalized_corrected, png_path)

            results.append(DiagnosticResult(angle=angle, score=score, png_path=png_path))
            manifest["angles"].append(
                {
                    "angle_degrees": angle,
                    "score": score,
                    "mask_pixels": int(np.count_nonzero(rotated_mask)),
                    "analysis_pixels": analysis_pixels,
                    "png": png_name,
                }
            )
            self.emit_log(
                f"Angle {angle:+.3f} score: {score:.6f} using {analysis_pixels} dust-mask pixels",
                "green",
            )

            completed_steps += 1
            self.progress_update.emit(int(completed_steps / total_steps * 100))

        best_result = min(results, key=lambda result: result.score)
        manifest["best_angle_degrees"] = best_result.angle
        manifest["best_score"] = best_result.score
        self.emit_log(
            f"Best angle: {best_result.angle:+.3f} degrees (score {best_result.score:.6f})",
            "green",
        )

        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        self.progress_update.emit(100)
        return results


class FlatRotationTesterGUI(QMainWindow):
    """Main application window."""

    def __init__(self, siril_instance=None):
        super().__init__()
        self.siril = siril_instance
        self.siril_home_dir: Optional[Path] = None
        self.worker: Optional[ProcessingWorker] = None
        self.results: List[DiagnosticResult] = []
        self.output_dir: Optional[Path] = None

        self.init_ui()
        if self.siril:
            try:
                wd = self.siril.get_siril_wd()
                if wd:
                    wd_path = Path(wd)
                    if wd_path.exists():
                        self.siril_home_dir = wd_path
                        self.log(f"Siril home directory detected: {wd_path}", "green")
            except AttributeError as exc:
                self.log(f"Siril not connected properly: {exc}", "red")
            except Exception as exc:
                self.log(f"Could not detect Siril home directory: {exc}", "orange")

        self.load_settings()
        self.update_output_dir_label()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Flat Rotation Tester for Siril")
        self.setGeometry(100, 100, 1200, 820)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        input_group = QGroupBox("Input Data")
        input_layout = QGridLayout()

        self.flat_dir_label = QLabel("No flat directory selected")
        self.flat_dir_label.setWordWrap(True)
        flat_button = QPushButton("Select Flat Directory")
        flat_button.clicked.connect(self.select_flat_directory)
        input_layout.addWidget(QLabel("Flat FITS Directory:"), 0, 0)
        input_layout.addWidget(self.flat_dir_label, 0, 1)
        input_layout.addWidget(flat_button, 0, 2)

        self.light_file_label = QLabel("No light FITS file selected")
        self.light_file_label.setWordWrap(True)
        light_button = QPushButton("Select Light File")
        light_button.clicked.connect(self.select_light_file)
        input_layout.addWidget(QLabel("Light FITS File:"), 1, 0)
        input_layout.addWidget(self.light_file_label, 1, 1)
        input_layout.addWidget(light_button, 1, 2)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        settings_group = QGroupBox("Rotation Search")
        settings_layout = QGridLayout()

        self.min_angle_spin = QDoubleSpinBox()
        self.min_angle_spin.setRange(-360.0, 360.0)
        self.min_angle_spin.setDecimals(3)
        self.min_angle_spin.setSingleStep(0.5)
        self.min_angle_spin.setValue(-5.0)
        settings_layout.addWidget(QLabel("Angle Range Start (deg):"), 0, 0)
        settings_layout.addWidget(self.min_angle_spin, 0, 1)

        self.max_angle_spin = QDoubleSpinBox()
        self.max_angle_spin.setRange(-360.0, 360.0)
        self.max_angle_spin.setDecimals(3)
        self.max_angle_spin.setSingleStep(0.5)
        self.max_angle_spin.setValue(5.0)
        settings_layout.addWidget(QLabel("Angle Range End (deg):"), 0, 2)
        settings_layout.addWidget(self.max_angle_spin, 0, 3)

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.001, 180.0)
        self.step_spin.setDecimals(3)
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setValue(0.5)
        settings_layout.addWidget(QLabel("Angle Step (deg):"), 1, 0)
        settings_layout.addWidget(self.step_spin, 1, 1)

        self.output_dir_label = QLabel("Output directory will be created in Siril's current home directory")
        self.output_dir_label.setWordWrap(True)
        settings_layout.addWidget(QLabel("Diagnostic Output:"), 1, 2)
        settings_layout.addWidget(self.output_dir_label, 1, 3)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Search Best Rotation")
        self.start_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.start_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_button)

        self.prev_button = QPushButton("Previous Result")
        self.prev_button.clicked.connect(self.show_previous_result)
        self.prev_button.setEnabled(False)
        button_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Result")
        self.next_button.clicked.connect(self.show_next_result)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)

        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self.open_output_directory)
        self.open_output_button.setEnabled(False)
        button_layout.addWidget(self.open_output_button)

        main_layout.addLayout(button_layout)

        viewer_group = QGroupBox("Diagnostic Review")
        viewer_layout = QHBoxLayout()

        self.result_list = QListWidget()
        self.result_list.setMinimumWidth(220)
        self.result_list.currentRowChanged.connect(self.on_result_selected)
        viewer_layout.addWidget(self.result_list)

        preview_layout = QVBoxLayout()
        self.preview_info_label = QLabel("Search rotations to review the flat-corrected light and residual scores.")
        self.preview_info_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview_info_label)

        self.preview_label = QLabel("No diagnostic image loaded")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(700, 420)
        self.preview_label.setStyleSheet("border: 1px solid #888; background-color: #111; color: #ddd;")
        preview_layout.addWidget(self.preview_label, 1)

        viewer_layout.addLayout(preview_layout, 1)
        viewer_group.setLayout(viewer_layout)
        main_layout.addWidget(viewer_group, 1)

        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setMinimumHeight(180)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        self.log("Flat Rotation Tester initialized", "green")

    def log(self, message: str, color: str = "black") -> None:
        """Append a colorized message to the log view."""
        color_map = {
            "black": "#000000",
            "red": "#CC0000",
            "green": "#008800",
            "blue": "#0055CC",
            "orange": "#CC7700",
        }
        hex_color = color_map.get(color, color_map["black"])
        self.log_text.append(f'<span style="color: {hex_color};">{message}</span>')

    def select_flat_directory(self) -> None:
        """Choose the flat FITS directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Flat FITS Directory")
        if directory:
            self.flat_dir_label.setText(directory)
            self.update_output_dir_label()

    def select_light_file(self) -> None:
        """Choose the light FITS file to score against the rotated flats."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Light FITS File",
            "",
            "FITS Files (*.fit *.fits *.fts);;All Files (*)",
        )
        if file_path:
            self.light_file_label.setText(file_path)
            self.update_output_dir_label()

    def update_output_dir_label(self) -> None:
        """Refresh the derived output directory label."""
        light_file = self.selected_light_file()
        base_dir = self.selected_output_base_dir()

        if base_dir is None:
            self.output_dir_label.setText("Output directory will be created next to the selected light file")
            return

        if light_file is None:
            self.output_dir_label.setText(str(base_dir / "flat_rotation_diagnostics_<timestamp>"))
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"{light_file.stem}_flat_rotation_diagnostics_{timestamp}"
        self.output_dir_label.setText(str(output_dir))

    def selected_flat_dir(self) -> Optional[Path]:
        """Return the selected flat directory, if any."""
        text = self.flat_dir_label.text().strip()
        if not text or text == "No flat directory selected":
            return None
        return Path(text)

    def selected_light_file(self) -> Optional[Path]:
        """Return the selected light FITS file, if any."""
        text = self.light_file_label.text().strip()
        if not text or text == "No light FITS file selected":
            return None
        return Path(text)

    def selected_output_base_dir(self) -> Optional[Path]:
        """Return the Siril home directory when available, otherwise fall back to the light file folder."""
        if self.siril_home_dir is not None:
            return self.siril_home_dir

        light_file = self.selected_light_file()
        if light_file is not None:
            return light_file.parent

        return None

    def build_output_dir(self) -> Path:
        """Create the derived output directory path for the current run."""
        light_file = self.selected_light_file()
        if light_file is None:
            raise ValueError("No light FITS file selected")
        base_dir = self.selected_output_base_dir()
        if base_dir is None:
            raise ValueError("Could not determine an output directory")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_dir / f"{light_file.stem}_flat_rotation_diagnostics_{timestamp}"

    def start_processing(self) -> None:
        """Validate inputs and start the worker thread."""
        flat_dir = self.selected_flat_dir()
        light_file = self.selected_light_file()

        if flat_dir is None or not flat_dir.exists():
            QMessageBox.warning(self, "Missing Flats", "Please select a valid flat FITS directory.")
            return

        if light_file is None or not light_file.exists() or light_file.suffix.lower() not in SUPPORTED_FITS_SUFFIXES:
            QMessageBox.warning(self, "Missing Light", "Please select a valid light FITS file.")
            return

        min_angle = self.min_angle_spin.value()
        max_angle = self.max_angle_spin.value()
        step = self.step_spin.value()

        try:
            angles = angle_series(min_angle, max_angle, step)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Angle Settings", str(exc))
            return

        if len(angles) > 2000:
            QMessageBox.warning(
                self,
                "Too Many Angles",
                "The requested range produces more than 2000 diagnostic angles. Use a larger step or smaller range.",
            )
            return

        output_dir = self.build_output_dir()
        self.output_dir = output_dir
        self.output_dir_label.setText(str(output_dir))

        self.start_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results = []
        self.result_list.clear()
        self.preview_label.setText("Generating diagnostic images...")
        self.preview_label.setPixmap(QPixmap())
        self.preview_info_label.setText("Working...")

        self.log(
            f"Starting flat rotation scan from {min_angle:+.3f} to {max_angle:+.3f} degrees in {step:.3f} degree steps",
            "green",
        )

        self.worker = ProcessingWorker(flat_dir, light_file, min_angle, max_angle, step, output_dir)
        self.worker.log_message.connect(self.log)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self, success: bool, message: str, results: object, output_dir: str) -> None:
        """Handle worker completion."""
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100 if success else self.progress_bar.value())

        if not success:
            self.log(f"Error: {message}", "red")
            QMessageBox.critical(self, "Processing Failed", message)
            self.preview_label.setText("No diagnostic image loaded")
            self.preview_info_label.setText("Diagnostic generation failed.")
            return

        self.results = list(results)
        self.output_dir = Path(output_dir)
        self.open_output_button.setEnabled(True)
        self.populate_results()
        self.log(f"Saved {len(self.results)} diagnostic PNG files to {self.output_dir}", "green")
        QMessageBox.information(self, "Diagnostics Ready", message)

    def populate_results(self) -> None:
        """Load result list entries into the review panel."""
        self.result_list.clear()
        for result in self.results:
            self.result_list.addItem(f"{result.angle:+.3f} deg   score {result.score:.6f}")

        enabled = len(self.results) > 1
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)

        if self.results:
            best_index = min(range(len(self.results)), key=lambda index: self.results[index].score)
            self.result_list.setCurrentRow(best_index)
        else:
            self.preview_label.setText("No diagnostic image loaded")
            self.preview_info_label.setText("No results were generated.")

    def on_result_selected(self, row: int) -> None:
        """Display the selected diagnostic PNG."""
        if row < 0 or row >= len(self.results):
            return

        result = self.results[row]
        pixmap = QPixmap(str(result.png_path))
        if pixmap.isNull():
            self.preview_label.setText(f"Could not load {result.png_path.name}")
            return

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)
        self.preview_info_label.setText(
            f"Angle: {result.angle:+.3f} deg    Score: {result.score:.6f}    File: {result.png_path.name}"
        )

    def resizeEvent(self, event) -> None:
        """Keep the preview scaled when the window size changes."""
        super().resizeEvent(event)
        current_row = self.result_list.currentRow()
        if current_row >= 0:
            self.on_result_selected(current_row)

    def show_previous_result(self) -> None:
        """Select the previous diagnostic image."""
        current = self.result_list.currentRow()
        if current > 0:
            self.result_list.setCurrentRow(current - 1)

    def show_next_result(self) -> None:
        """Select the next diagnostic image."""
        current = self.result_list.currentRow()
        if 0 <= current < self.result_list.count() - 1:
            self.result_list.setCurrentRow(current + 1)

    def open_output_directory(self) -> None:
        """Open the diagnostic output folder in the desktop file browser."""
        if self.output_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir)))

    def save_settings(self) -> None:
        """Persist GUI settings between runs."""
        config_path = Path.home() / ".flat_rotation_tester_config.json"
        settings = {
            "flat_directory": str(self.selected_flat_dir()) if self.selected_flat_dir() else "",
            "light_file": str(self.selected_light_file()) if self.selected_light_file() else "",
            "min_angle": self.min_angle_spin.value(),
            "max_angle": self.max_angle_spin.value(),
            "step": self.step_spin.value(),
        }

        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)

    def load_settings(self) -> None:
        """Restore GUI settings from the previous run."""
        config_path = Path.home() / ".flat_rotation_tester_config.json"
        if not config_path.exists():
            return

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                settings = json.load(handle)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            self.log(f"Could not load saved settings: {exc}", "orange")
            return

        flat_directory = settings.get("flat_directory", "").strip()
        light_file = settings.get("light_file", settings.get("light_directory", "")).strip()

        if flat_directory:
            self.flat_dir_label.setText(flat_directory)
        if light_file and Path(light_file).suffix.lower() in SUPPORTED_FITS_SUFFIXES:
            self.light_file_label.setText(light_file)

        self.min_angle_spin.setValue(float(settings.get("min_angle", -5.0)))
        self.max_angle_spin.setValue(float(settings.get("max_angle", 5.0)))
        self.step_spin.setValue(float(settings.get("step", 0.5)))
        self.update_output_dir_label()

    def closeEvent(self, event) -> None:
        """Persist settings when the window closes."""
        try:
            self.save_settings()
        except OSError as exc:
            self.log(f"Could not save settings: {exc}", "orange")
        event.accept()


def ensure_dependencies() -> None:
    """Ask Siril to install missing packages when possible."""
    if not SIRILPY_AVAILABLE:
        return

    sirilpy.ensure_installed(
        "PyQt5",
        "numpy",
        "astropy",
        "scipy",
        "pillow",
        version_constraints=[None, ">=1.20.0", ">=4.0", ">=1.6.0", None],
    )


def main() -> int:
    """Main entry point."""
    try:
        ensure_dependencies()
    except Exception as exc:
        print(f"Error ensuring dependencies: {exc}")
        return 1

    siril = None
    if SIRILPY_AVAILABLE:
        try:
            siril = sirilpy.SirilInterface()
            siril.connect()
        except Exception as exc:
            print(f"Warning: could not connect to Siril, using local output fallback: {exc}")
            siril = None

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = FlatRotationTesterGUI(siril_instance=siril)
    window.show()

    result = app.exec_()
    return 0 if result == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
