#!/usr/bin/env python3
"""
Flat Alignment Fix for Siril

Interactive GUI tool for manually aligning a stacked master flat to a light
frame. It builds a master flat from a directory of FITS flats, lets you rotate
and shift that flat without modifying the source files, previews the effect on
the selected light, and writes transformed outputs into a separate working
directory when applied.

Requirements:
- PyQt5
- numpy
- astropy
- scipy
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy.io import fits
from PyQt5.QtCore import QPointF, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
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
PREVIEW_MAX_DIMENSION = 800
PREVIEW_OVERLAY_ALPHA = 88


@dataclass
class FITSFrame:
    data: np.ndarray
    bayer_pattern: Optional[str]


@dataclass
class LoadedData:
    flat_dir: Path
    flat_files: List[Path]
    light_input_path: Path
    light_files: List[Path]
    light_output_stem: str
    master_flat: np.ndarray
    light_data: np.ndarray
    bayer_pattern: Optional[str]
    preview_flat: np.ndarray
    preview_light: np.ndarray
    preview_downsample: int
    output_dir: Path


@dataclass
class AppliedOutputs:
    transformed_flat_path: Path
    corrected_light_path: Path
    manifest_path: Path


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


def read_fits_frame(path: Path) -> FITSFrame:
    """Read the first image HDU from a FITS file."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None:
                data = canonicalize_fits_array(hdu.data)
                bayer_pattern = hdu.header.get("BAYERPAT")
                return FITSFrame(
                    data=data,
                    bayer_pattern=str(bayer_pattern).upper() if bayer_pattern else None,
                )
    raise ValueError(f"No image data found in {path}")


def save_fits_frame(path: Path, data: np.ndarray, bayer_pattern: Optional[str] = None) -> None:
    """Save a FITS frame used by the tool."""
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


def robust_mad_sigma(values: np.ndarray) -> float:
    """Estimate a robust standard deviation using the median absolute deviation."""
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Residual metric received no finite values")

    center = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - center)))
    return max(1.4826 * mad, 1e-6)


def reference_plane(data: np.ndarray) -> np.ndarray:
    """Collapse channel-first data to a single analysis plane."""
    if data.ndim == 2:
        return np.asarray(data, dtype=np.float32)
    return np.nanmean(np.asarray(data, dtype=np.float32), axis=0).astype(np.float32, copy=False)


def downsample_array(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample an image or channel-first image by integer slicing."""
    factor = max(1, int(factor))
    if factor == 1:
        return np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        return np.asarray(data[::factor, ::factor], dtype=np.float32)
    return np.asarray(data[:, ::factor, ::factor], dtype=np.float32)


def choose_preview_downsample(shape: Sequence[int]) -> int:
    """Select an integer downsample factor for interactive previews."""
    height = int(shape[-2])
    width = int(shape[-1])
    largest = max(height, width)
    return max(1, int(np.ceil(largest / PREVIEW_MAX_DIMENSION)))


def transform_image(
    data: np.ndarray,
    angle_degrees: float,
    shift_x: float,
    shift_y: float,
    order: int,
    cval: float = 1.0,
) -> np.ndarray:
    """Rotate and shift an image while preserving its original size."""
    shift_vector = (shift_y, shift_x)

    if data.ndim == 2:
        rotated = ndimage.rotate(
            data,
            angle_degrees,
            reshape=False,
            order=order,
            mode="nearest" if order > 0 else "constant",
            cval=cval,
            prefilter=order > 1,
        )
        shifted = ndimage.shift(
            rotated,
            shift=shift_vector,
            order=order,
            mode="nearest" if order > 0 else "constant",
            cval=cval,
            prefilter=order > 1,
        )
        return shifted.astype(np.float32, copy=False)

    channels = [
        transform_image(channel, angle_degrees, shift_x, shift_y, order=order, cval=cval)
        for channel in data
    ]
    return np.stack(channels, axis=0)


def corrected_light(light_data: np.ndarray, transformed_flat: np.ndarray) -> np.ndarray:
    """Apply flat correction using a safely clipped flat."""
    if light_data.ndim == 3:
        safe_channels = []
        for channel in transformed_flat:
            finite = np.isfinite(channel)
            positive = finite & (channel > 0)
            if not np.any(positive):
                raise ValueError("Transformed flat contains no positive finite values")
            floor = max(float(np.percentile(channel[positive], 1)), 1e-6)
            safe_channels.append(np.where(positive, np.clip(channel, floor, None), floor))
        safe_flat = np.stack(safe_channels, axis=0)
        corrected = light_data / safe_flat
        return corrected.astype(np.float32, copy=False)

    finite = np.isfinite(transformed_flat)
    positive = finite & (transformed_flat > 0)
    if not np.any(positive):
        raise ValueError("Transformed flat contains no positive finite values")

    floor = max(float(np.percentile(transformed_flat[positive], 1)), 1e-6)
    safe_flat = np.where(positive, np.clip(transformed_flat, floor, None), floor)
    corrected = light_data / safe_flat
    return corrected.astype(np.float32, copy=False)


DisplayLimits = Union[Tuple[float, float, float], List[Tuple[float, float, float]]]


def compute_display_limits(data: np.ndarray) -> DisplayLimits:
    """Compute a stable display stretch for preview rendering."""
    if data.ndim == 2:
        finite = np.isfinite(data)
        if not np.any(finite):
            raise ValueError("Preview image contains no finite values")
        values = data[finite]
        median = float(np.median(values))
        sigma = robust_mad_sigma(values)
        low = float(np.percentile(values, 0.1))
        high = min(float(np.percentile(values, 99.9)), median + 12.0 * sigma)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        return (low, high, median)

    limits: List[Tuple[float, float, float]] = []
    for channel in data[:3]:
        finite = np.isfinite(channel)
        if not np.any(finite):
            raise ValueError("Preview image contains no finite channel values")
        values = channel[finite]
        median = float(np.median(values))
        sigma = robust_mad_sigma(values)
        low = float(np.percentile(values, 0.1))
        high = min(float(np.percentile(values, 99.9)), median + 12.0 * sigma)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        limits.append((low, high, median))
    return limits


def stretch_with_limits(
    data: np.ndarray,
    limits: DisplayLimits,
    shadow_clip: float,
    highlight_clip: float,
    gamma: float,
) -> np.ndarray:
    """Convert a 2D or channel-first preview array into an 8-bit display image."""
    if data.ndim == 2:
        low, high, median = limits  # type: ignore[misc]
        stretched = np.clip((np.nan_to_num(data, nan=median) - low) / (high - low), 0, 1)
        stretched = np.clip((stretched - shadow_clip) / max(highlight_clip - shadow_clip, 1e-6), 0, 1)
        stretched = np.power(stretched, gamma)
        return (stretched * 255).astype(np.uint8)

    channels = []
    channel_limits = limits  # type: ignore[assignment]
    for index, channel in enumerate(data[:3]):
        low, high, median = channel_limits[index]
        stretched = np.clip((np.nan_to_num(channel, nan=median) - low) / (high - low), 0, 1)
        stretched = np.clip((stretched - shadow_clip) / max(highlight_clip - shadow_clip, 1e-6), 0, 1)
        stretched = np.power(stretched, gamma)
        channels.append((stretched * 255).astype(np.uint8))

    while len(channels) < 3:
        channels.append(channels[-1])
    return np.stack(channels[:3], axis=-1)


def overlay_rgba(flat_data: np.ndarray) -> np.ndarray:
    """Build a semi-transparent red overlay from the transformed flat pattern itself."""
    plane = reference_plane(normalize_flat(flat_data))
    finite = np.isfinite(plane)
    if not np.any(finite):
        return np.zeros((plane.shape[0], plane.shape[1], 4), dtype=np.uint8)

    values = plane[finite]
    median = float(np.median(values))
    sigma = robust_mad_sigma(values)
    deviation = np.abs(plane - median) / max(3.0 * sigma, 1e-6)
    alpha = np.clip(deviation, 0.0, 1.0)
    alpha = np.power(alpha, 0.7)

    rgba = np.zeros((plane.shape[0], plane.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = np.where(finite, alpha * PREVIEW_OVERLAY_ALPHA, 0).astype(np.uint8)
    return rgba


def array_to_qimage(data: np.ndarray) -> QImage:
    """Convert a grayscale, RGB, or RGBA uint8 array into a QImage."""
    if data.ndim == 2:
        contiguous = np.ascontiguousarray(data)
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.shape[1],
            QImage.Format_Grayscale8,
        )
        return image.copy()

    contiguous = np.ascontiguousarray(data)
    channels = contiguous.shape[2]
    if channels == 3:
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.shape[1] * 3,
            QImage.Format_RGB888,
        )
        return image.copy()
    if channels == 4:
        image = QImage(
            contiguous.data,
            contiguous.shape[1],
            contiguous.shape[0],
            contiguous.shape[1] * 4,
            QImage.Format_RGBA8888,
        )
        return image.copy()
    raise ValueError(f"Unsupported preview channel count: {channels}")


def build_output_dir(base_dir: Path, light_file: Path) -> Path:
    """Return the stable output directory for iterative apply operations."""
    output_stem = light_file.stem if light_file.is_file() else light_file.name
    return base_dir / "flat_alignment_fix_outputs" / output_stem


class ImagePreviewWidget(QWidget):
    """Image preview widget with optional drag interaction and overlay rendering."""

    drag_delta = pyqtSignal(float, float)

    def __init__(self, interactive: bool = False, parent=None):
        super().__init__(parent)
        self.interactive = interactive
        self.base_pixmap: Optional[QPixmap] = None
        self.overlay_pixmap: Optional[QPixmap] = None
        self._image_rect = None
        self._dragging = False
        self._last_pos = QPointF()
        self.setMinimumSize(460, 460)
        self.setStyleSheet("border: 1px solid #777; background-color: #111;")
        if self.interactive:
            self.setCursor(Qt.OpenHandCursor)

    def set_images(self, base_image: np.ndarray, overlay_image: Optional[np.ndarray] = None) -> None:
        """Load the background image and optional overlay."""
        self.base_pixmap = QPixmap.fromImage(array_to_qimage(base_image))
        self.overlay_pixmap = None
        if overlay_image is not None:
            self.overlay_pixmap = QPixmap.fromImage(array_to_qimage(overlay_image))
        self.update()

    def _target_rect(self):
        """Return the target rect used when painting the pixmap."""
        if self.base_pixmap is None or self.base_pixmap.isNull():
            self._image_rect = None
            return None

        pixmap_width = self.base_pixmap.width()
        pixmap_height = self.base_pixmap.height()
        widget_width = max(1, self.width())
        widget_height = max(1, self.height())
        scale = min(widget_width / pixmap_width, widget_height / pixmap_height)
        target_width = int(round(pixmap_width * scale))
        target_height = int(round(pixmap_height * scale))
        left = (widget_width - target_width) // 2
        top = (widget_height - target_height) // 2
        self._image_rect = (left, top, target_width, target_height)
        return self._image_rect

    def paintEvent(self, event) -> None:
        """Draw the preview image and overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        target_rect = self._target_rect()
        if target_rect is None or self.base_pixmap is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No preview loaded")
            return

        left, top, width, height = target_rect
        painter.drawPixmap(left, top, width, height, self.base_pixmap)
        if self.overlay_pixmap is not None:
            painter.drawPixmap(left, top, width, height, self.overlay_pixmap)

    def mousePressEvent(self, event) -> None:
        """Start drag-based shifting when the cursor is over the image."""
        if not self.interactive or self.base_pixmap is None or event.button() != Qt.LeftButton:
            return
        target_rect = self._target_rect()
        if target_rect is None:
            return
        left, top, width, height = target_rect
        if left <= event.x() <= left + width and top <= event.y() <= top + height:
            self._dragging = True
            self._last_pos = QPointF(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        """Emit image-space drag deltas while dragging."""
        if not self._dragging or self.base_pixmap is None or self._image_rect is None:
            super().mouseMoveEvent(event)
            return

        left, top, width, height = self._image_rect
        if width <= 0 or height <= 0:
            return

        delta = QPointF(event.pos()) - self._last_pos
        self._last_pos = QPointF(event.pos())

        image_scale_x = self.base_pixmap.width() / float(width)
        image_scale_y = self.base_pixmap.height() / float(height)
        self.drag_delta.emit(delta.x() * image_scale_x, delta.y() * image_scale_y)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Finish drag-based shifting."""
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.OpenHandCursor if self.interactive else Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class LoadDataWorker(QThread):
    """Background worker that builds the master flat and preview assets."""

    log_message = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, str, object)

    def __init__(self, flat_dir: Path, light_input_path: Path, output_dir: Path):
        super().__init__()
        self.flat_dir = flat_dir
        self.light_input_path = light_input_path
        self.output_dir = output_dir

    def emit_log(self, message: str, color: str = "black") -> None:
        self.log_message.emit(message, color)

    def run(self) -> None:
        try:
            result = self.process()
            self.finished.emit(True, "Loaded the light and built a master flat for manual alignment.", result)
        except Exception as exc:
            self.finished.emit(False, str(exc), None)

    def process(self) -> LoadedData:
        flat_files = collect_fits_files(self.flat_dir)
        if not flat_files:
            raise ValueError("No FITS flat files found in the selected flat directory")

        if not self.light_input_path.exists():
            raise ValueError("Selected light input does not exist")

        if self.light_input_path.is_dir():
            light_files = collect_fits_files(self.light_input_path)
            if not light_files:
                raise ValueError("No FITS light files found in the selected light directory")
            light_output_stem = f"{self.light_input_path.name}_stack"
            self.emit_log(f"Found {len(light_files)} light files in {self.light_input_path}", "blue")
        else:
            if self.light_input_path.suffix.lower() not in SUPPORTED_FITS_SUFFIXES:
                raise ValueError("Selected light file is not a valid FITS file")
            light_files = [self.light_input_path]
            light_output_stem = self.light_input_path.stem
            self.emit_log(f"Using light file {self.light_input_path}", "blue")

        self.emit_log(f"Found {len(flat_files)} flat files in {self.flat_dir}", "blue")
        self.progress_update.emit(5)

        flat_stack = []
        flat_bayer_patterns = set()
        expected_shape: Optional[Tuple[int, ...]] = None

        total_files = len(flat_files)
        for index, flat_file in enumerate(flat_files, start=1):
            flat_frame = read_fits_frame(flat_file)
            if expected_shape is None:
                expected_shape = flat_frame.data.shape
            elif flat_frame.data.shape != expected_shape:
                raise ValueError(
                    f"Flat shape mismatch: {flat_file.name} has shape {flat_frame.data.shape}, "
                    f"expected {expected_shape}"
                )
            flat_stack.append(flat_frame.data)
            if flat_frame.bayer_pattern:
                flat_bayer_patterns.add(flat_frame.bayer_pattern)
            self.progress_update.emit(5 + int(index / max(total_files, 1) * 40))

        if len(flat_bayer_patterns) > 1:
            raise ValueError(f"Flat Bayer pattern mismatch: {sorted(flat_bayer_patterns)}")

        master_flat = np.nanmedian(np.stack(flat_stack, axis=0), axis=0).astype(np.float32)
        master_flat = normalize_flat(master_flat)
        self.emit_log("Built normalized master flat", "green")
        self.progress_update.emit(50)

        light_stack = []
        light_bayer_patterns = set()
        total_light_files = len(light_files)
        for index, light_file in enumerate(light_files, start=1):
            light_frame = read_fits_frame(light_file)
            if light_frame.data.shape != master_flat.shape:
                raise ValueError(
                    f"Light shape mismatch: {light_file.name} has shape {light_frame.data.shape}, "
                    f"expected {master_flat.shape}"
                )
            light_stack.append(light_frame.data)
            if light_frame.bayer_pattern:
                light_bayer_patterns.add(light_frame.bayer_pattern)
            self.progress_update.emit(50 + int(index / max(total_light_files, 1) * 15))

        if flat_bayer_patterns and light_bayer_patterns and flat_bayer_patterns != light_bayer_patterns:
            raise ValueError(
                f"Flat and light Bayer patterns differ: flats={sorted(flat_bayer_patterns)}, "
                f"lights={sorted(light_bayer_patterns)}"
            )

        stacked_light = np.nanmedian(np.stack(light_stack, axis=0), axis=0).astype(np.float32)
        if len(light_files) > 1:
            self.emit_log(f"Built stacked light from {len(light_files)} files", "green")
        else:
            self.emit_log("Loaded single light for comparison", "green")
        self.progress_update.emit(70)

        preview_downsample = choose_preview_downsample(master_flat.shape)
        preview_flat = downsample_array(master_flat, preview_downsample)
        preview_light = downsample_array(stacked_light, preview_downsample)
        self.emit_log(
            f"Prepared fast preview data at {preview_light.shape[-1]}x{preview_light.shape[-2]} "
            f"(downsample {preview_downsample}x)",
            "green",
        )
        self.progress_update.emit(100)

        bayer_pattern = next(iter(flat_bayer_patterns)) if flat_bayer_patterns else light_frame.bayer_pattern
        return LoadedData(
            flat_dir=self.flat_dir,
            flat_files=flat_files,
            light_input_path=self.light_input_path,
            light_files=light_files,
            light_output_stem=light_output_stem,
            master_flat=master_flat,
            light_data=stacked_light.astype(np.float32, copy=False),
            bayer_pattern=bayer_pattern,
            preview_flat=preview_flat,
            preview_light=preview_light,
            preview_downsample=preview_downsample,
            output_dir=self.output_dir,
        )


class ApplyTransformWorker(QThread):
    """Background worker that writes the transformed flat and corrected result."""

    log_message = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(bool, str, object)

    def __init__(
        self,
        loaded_data: LoadedData,
        rotation_degrees: float,
        shift_x: float,
        shift_y: float,
    ):
        super().__init__()
        self.loaded_data = loaded_data
        self.rotation_degrees = rotation_degrees
        self.shift_x = shift_x
        self.shift_y = shift_y

    def emit_log(self, message: str, color: str = "black") -> None:
        self.log_message.emit(message, color)

    def run(self) -> None:
        try:
            result = self.process()
            self.finished.emit(True, "Applied the flat transform and wrote updated outputs.", result)
        except Exception as exc:
            self.finished.emit(False, str(exc), None)

    def process(self) -> AppliedOutputs:
        output_dir = self.loaded_data.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_update.emit(5)

        self.emit_log(
            f"Applying transform: rotation {self.rotation_degrees:+.3f} deg, "
            f"shift X {self.shift_x:+.2f} px, shift Y {self.shift_y:+.2f} px",
            "blue",
        )

        transformed_flat = transform_image(
            self.loaded_data.master_flat,
            self.rotation_degrees,
            self.shift_x,
            self.shift_y,
            order=3,
            cval=1.0,
        )
        transformed_flat = normalize_flat(transformed_flat)
        self.progress_update.emit(45)

        corrected = corrected_light(self.loaded_data.light_data, transformed_flat)
        self.progress_update.emit(70)

        transformed_flat_path = output_dir / "transformed_master_flat.fit"
        corrected_light_path = output_dir / f"{self.loaded_data.light_output_stem}_flat_corrected.fit"
        manifest_path = output_dir / "manifest.json"

        save_fits_frame(transformed_flat_path, transformed_flat, self.loaded_data.bayer_pattern)
        save_fits_frame(corrected_light_path, corrected, self.loaded_data.bayer_pattern)
        self.progress_update.emit(90)

        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "flat_directory": str(self.loaded_data.flat_dir),
            "flat_files": [path.name for path in self.loaded_data.flat_files],
            "light_input": str(self.loaded_data.light_input_path),
            "light_files": [path.name for path in self.loaded_data.light_files],
            "bayer_pattern": self.loaded_data.bayer_pattern,
            "rotation_degrees": self.rotation_degrees,
            "shift_x_pixels": self.shift_x,
            "shift_y_pixels": self.shift_y,
            "transformed_master_flat": transformed_flat_path.name,
            "corrected_light": corrected_light_path.name,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self.progress_update.emit(100)

        self.emit_log(f"Wrote transformed flat to {transformed_flat_path}", "green")
        self.emit_log(f"Wrote corrected result to {corrected_light_path}", "green")
        return AppliedOutputs(
            transformed_flat_path=transformed_flat_path,
            corrected_light_path=corrected_light_path,
            manifest_path=manifest_path,
        )


class FlatAlignmentFixGUI(QMainWindow):
    """Main application window."""

    def __init__(self, siril_instance=None):
        super().__init__()
        self.siril = siril_instance
        self.siril_wd: Optional[Path] = None
        self.load_worker: Optional[LoadDataWorker] = None
        self.apply_worker: Optional[ApplyTransformWorker] = None
        self.loaded_data: Optional[LoadedData] = None
        self.preview_limits: Optional[DisplayLimits] = None

        self.init_ui()
        self.detect_siril_working_directory()
        self.load_settings()
        self.update_output_dir_label()
        self.update_preview()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Flat Alignment Fix for Siril")
        self.setGeometry(100, 100, 1450, 980)

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

        self.light_input_label = QLabel("No light FITS file or directory selected")
        self.light_input_label.setWordWrap(True)
        light_button = QPushButton("Select Light File")
        light_button.clicked.connect(self.select_light_file)
        light_dir_button = QPushButton("Select Light Directory")
        light_dir_button.clicked.connect(self.select_light_directory)
        input_layout.addWidget(QLabel("Light FITS File or Directory:"), 1, 0)
        input_layout.addWidget(self.light_input_label, 1, 1)
        light_button_row = QHBoxLayout()
        light_button_row.addWidget(light_button)
        light_button_row.addWidget(light_dir_button)
        input_layout.addLayout(light_button_row, 1, 2)

        self.output_dir_label = QLabel("Output directory will be created next to the selected light input")
        self.output_dir_label.setWordWrap(True)
        input_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        input_layout.addWidget(self.output_dir_label, 2, 1, 1, 2)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        control_group = QGroupBox("Alignment Controls")
        control_layout = QGridLayout()

        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-360.0, 360.0)
        self.rotation_spin.setDecimals(3)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.valueChanged.connect(self.on_transform_changed)
        control_layout.addWidget(QLabel("Rotation (deg):"), 0, 0)
        control_layout.addWidget(self.rotation_spin, 0, 1)

        self.rotation_step_spin = QDoubleSpinBox()
        self.rotation_step_spin.setRange(0.001, 30.0)
        self.rotation_step_spin.setDecimals(3)
        self.rotation_step_spin.setValue(0.1)
        self.rotation_step_spin.setSingleStep(0.05)
        control_layout.addWidget(QLabel("Rotation Step:"), 0, 2)
        control_layout.addWidget(self.rotation_step_spin, 0, 3)

        rotate_minus_button = QPushButton("Rotate -")
        rotate_minus_button.clicked.connect(lambda: self.adjust_rotation(-self.rotation_step_spin.value()))
        control_layout.addWidget(rotate_minus_button, 0, 4)

        rotate_plus_button = QPushButton("Rotate +")
        rotate_plus_button.clicked.connect(lambda: self.adjust_rotation(self.rotation_step_spin.value()))
        control_layout.addWidget(rotate_plus_button, 0, 5)

        self.shift_x_spin = QDoubleSpinBox()
        self.shift_x_spin.setRange(-100000.0, 100000.0)
        self.shift_x_spin.setDecimals(2)
        self.shift_x_spin.setSingleStep(1.0)
        self.shift_x_spin.valueChanged.connect(self.on_transform_changed)
        control_layout.addWidget(QLabel("Shift X (px):"), 1, 0)
        control_layout.addWidget(self.shift_x_spin, 1, 1)

        self.shift_y_spin = QDoubleSpinBox()
        self.shift_y_spin.setRange(-100000.0, 100000.0)
        self.shift_y_spin.setDecimals(2)
        self.shift_y_spin.setSingleStep(1.0)
        self.shift_y_spin.valueChanged.connect(self.on_transform_changed)
        control_layout.addWidget(QLabel("Shift Y (px):"), 1, 2)
        control_layout.addWidget(self.shift_y_spin, 1, 3)

        self.load_button = QPushButton("Load Data")
        self.load_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.load_button.clicked.connect(self.start_loading)
        control_layout.addWidget(self.load_button, 1, 4)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_transform)
        control_layout.addWidget(self.reset_button, 1, 5)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.apply_button.setText("Apply to FITS / Siril")
        self.apply_button.setToolTip(
            "Write the transformed master flat and corrected light or light-stack FITS files to the output folder, "
            "then load the corrected result into Siril when connected."
        )
        self.apply_button.clicked.connect(self.apply_transform)
        control_layout.addWidget(self.apply_button, 2, 4, 1, 2)

        self.transform_label = QLabel("Rotation +0.000 deg | Shift X +0.00 px | Shift Y +0.00 px")
        self.transform_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(self.transform_label, 2, 0, 1, 4)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        stretch_group = QGroupBox("Shared Preview Stretch")
        stretch_layout = QVBoxLayout()

        self.shadow_clip_slider = QSlider(Qt.Horizontal)
        self.shadow_clip_slider.setRange(0, 950)
        self.shadow_clip_slider.setValue(0)
        self.shadow_clip_slider.setToolTip("Clip this fraction from the dark end for both preview panes.")
        self.shadow_clip_slider.valueChanged.connect(self.on_transform_changed)
        self.shadow_clip_value_label = QLabel("0.000")
        shadow_row = QHBoxLayout()
        shadow_row.addWidget(QLabel("Shadow Clip:"))
        shadow_row.addWidget(self.shadow_clip_slider, 1)
        shadow_row.addWidget(self.shadow_clip_value_label)
        stretch_layout.addLayout(shadow_row)

        self.highlight_clip_slider = QSlider(Qt.Horizontal)
        self.highlight_clip_slider.setRange(50, 1000)
        self.highlight_clip_slider.setValue(1000)
        self.highlight_clip_slider.setToolTip("Keep this fraction of the bright end for both preview panes.")
        self.highlight_clip_slider.valueChanged.connect(self.on_transform_changed)
        self.highlight_clip_value_label = QLabel("1.000")
        highlight_row = QHBoxLayout()
        highlight_row.addWidget(QLabel("Highlight Clip:"))
        highlight_row.addWidget(self.highlight_clip_slider, 1)
        highlight_row.addWidget(self.highlight_clip_value_label)
        stretch_layout.addLayout(highlight_row)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 400)
        self.gamma_slider.setValue(50)
        self.gamma_slider.setToolTip("Preview gamma for both panes. Lower values brighten midtones.")
        self.gamma_slider.valueChanged.connect(self.on_transform_changed)
        self.gamma_value_label = QLabel("0.500")
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        gamma_row.addWidget(self.gamma_slider, 1)
        gamma_row.addWidget(self.gamma_value_label)
        stretch_layout.addLayout(gamma_row)

        stretch_button_row = QHBoxLayout()
        self.reset_stretch_button = QPushButton("Reset Stretch")
        self.reset_stretch_button.clicked.connect(self.reset_stretch)
        stretch_button_row.addWidget(self.reset_stretch_button)
        stretch_button_row.addStretch(1)
        stretch_layout.addLayout(stretch_button_row)

        self.stretch_label = QLabel("Shadow 0.000 | Highlight 1.000 | Gamma 0.500")
        self.stretch_label.setStyleSheet("font-weight: bold;")
        stretch_layout.addWidget(self.stretch_label)

        stretch_group.setLayout(stretch_layout)
        main_layout.addWidget(stretch_group)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        preview_group = QGroupBox("Interactive Preview")
        preview_layout = QVBoxLayout()

        help_label = QLabel(
            "Load Data builds a master flat from the selected flat folder and prepares the preview. "
            "Drag the red flat overlay in the left preview to shift the flat, use the rotation controls for angle changes, "
            "adjust the shared stretch sliders to change both panes together, then Apply to write outputs and load the corrected result in Siril."
        )
        help_label.setWordWrap(True)
        preview_layout.addWidget(help_label)

        viewer_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_title = QLabel("Alignment View")
        left_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_title)
        self.alignment_view = ImagePreviewWidget(interactive=True)
        self.alignment_view.drag_delta.connect(self.on_alignment_dragged)
        left_layout.addWidget(self.alignment_view, 1)
        viewer_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        right_title = QLabel("Corrected Preview")
        right_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_title)
        self.corrected_view = ImagePreviewWidget(interactive=False)
        right_layout.addWidget(self.corrected_view, 1)
        viewer_layout.addLayout(right_layout, 1)

        preview_layout.addLayout(viewer_layout, 1)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, 1)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setMinimumHeight(190)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        self.set_controls_enabled(False)
        self.log("Flat Alignment Fix initialized", "green")

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

    def detect_siril_working_directory(self) -> None:
        """Capture Siril's working directory when available."""
        if not self.siril:
            return
        try:
            wd = self.siril.get_siril_wd()
            if wd:
                wd_path = Path(wd)
                if wd_path.exists():
                    self.siril_wd = wd_path
                    self.log(f"Siril working directory detected: {wd_path}", "green")
        except AttributeError as exc:
            self.log(f"Siril not connected properly: {exc}", "red")
        except Exception as exc:
            self.log(f"Could not detect Siril working directory: {exc}", "orange")

    def set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable transform controls."""
        self.rotation_spin.setEnabled(enabled)
        self.rotation_step_spin.setEnabled(enabled)
        self.shift_x_spin.setEnabled(enabled)
        self.shift_y_spin.setEnabled(enabled)
        self.shadow_clip_slider.setEnabled(enabled)
        self.highlight_clip_slider.setEnabled(enabled)
        self.gamma_slider.setEnabled(enabled)
        self.reset_stretch_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        self.alignment_view.setEnabled(enabled)

    def selected_flat_dir(self) -> Optional[Path]:
        """Return the selected flat directory."""
        text = self.flat_dir_label.text().strip()
        if not text or text == "No flat directory selected":
            return None
        return Path(text)

    def selected_light_input(self) -> Optional[Path]:
        """Return the selected light FITS file or directory."""
        text = self.light_input_label.text().strip()
        if not text or text == "No light FITS file or directory selected":
            return None
        return Path(text)

    def selected_output_base_dir(self) -> Optional[Path]:
        """Return the preferred output base directory."""
        if self.siril_wd is not None:
            return self.siril_wd
        light_input = self.selected_light_input()
        if light_input is not None:
            return light_input.parent if light_input.is_file() else light_input.parent
        return None

    def current_output_dir(self) -> Optional[Path]:
        """Return the current derived output directory."""
        base_dir = self.selected_output_base_dir()
        light_input = self.selected_light_input()
        if base_dir is None or light_input is None:
            return None
        return build_output_dir(base_dir, light_input)

    def update_output_dir_label(self) -> None:
        """Refresh the derived output directory label."""
        output_dir = self.current_output_dir()
        if output_dir is None:
            self.output_dir_label.setText("Output directory will be created next to the selected light input")
            return
        self.output_dir_label.setText(str(output_dir))

    def select_flat_directory(self) -> None:
        """Choose the flat FITS directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Flat FITS Directory")
        if directory:
            self.flat_dir_label.setText(directory)
            self.update_output_dir_label()

    def select_light_file(self) -> None:
        """Choose the light FITS file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Light FITS File",
            "",
            "FITS Files (*.fit *.fits *.fts);;All Files (*)",
        )
        if file_path:
            self.light_input_label.setText(file_path)
            self.update_output_dir_label()

    def select_light_directory(self) -> None:
        """Choose a directory of light FITS files to stack for preview and apply."""
        directory = QFileDialog.getExistingDirectory(self, "Select Light FITS Directory")
        if directory:
            self.light_input_label.setText(directory)
            self.update_output_dir_label()

    def adjust_rotation(self, delta: float) -> None:
        """Nudge the rotation value."""
        self.rotation_spin.setValue(self.rotation_spin.value() + delta)

    def reset_transform(self) -> None:
        """Reset rotation and shifts back to zero."""
        self.rotation_spin.setValue(0.0)
        self.shift_x_spin.setValue(0.0)
        self.shift_y_spin.setValue(0.0)

    def reset_stretch(self) -> None:
        """Reset the shared preview stretch back to defaults."""
        self.shadow_clip_slider.blockSignals(True)
        self.highlight_clip_slider.blockSignals(True)
        self.gamma_slider.blockSignals(True)
        self.shadow_clip_slider.setValue(0)
        self.highlight_clip_slider.setValue(1000)
        self.gamma_slider.setValue(50)
        self.shadow_clip_slider.blockSignals(False)
        self.highlight_clip_slider.blockSignals(False)
        self.gamma_slider.blockSignals(False)
        self.on_transform_changed()

    def shadow_clip_value(self) -> float:
        """Return the current shadow clip fraction."""
        return self.shadow_clip_slider.value() / 1000.0

    def highlight_clip_value(self) -> float:
        """Return the current highlight clip fraction."""
        return self.highlight_clip_slider.value() / 1000.0

    def gamma_value(self) -> float:
        """Return the current preview gamma."""
        return self.gamma_slider.value() / 100.0

    def set_shadow_clip_value(self, value: float) -> None:
        """Set the shadow clip slider from a float value."""
        self.shadow_clip_slider.setValue(int(round(np.clip(value, 0.0, 0.95) * 1000.0)))

    def set_highlight_clip_value(self, value: float) -> None:
        """Set the highlight clip slider from a float value."""
        self.highlight_clip_slider.setValue(int(round(np.clip(value, 0.05, 1.0) * 1000.0)))

    def set_gamma_value(self, value: float) -> None:
        """Set the gamma slider from a float value."""
        self.gamma_slider.setValue(int(round(np.clip(value, 0.1, 4.0) * 100.0)))

    def on_alignment_dragged(self, preview_dx: float, preview_dy: float) -> None:
        """Convert drag deltas from preview pixels into full-resolution pixels."""
        if self.loaded_data is None:
            return
        factor = float(self.loaded_data.preview_downsample)
        new_shift_x = self.shift_x_spin.value() + preview_dx * factor
        new_shift_y = self.shift_y_spin.value() + preview_dy * factor
        self.shift_x_spin.blockSignals(True)
        self.shift_y_spin.blockSignals(True)
        self.shift_x_spin.setValue(new_shift_x)
        self.shift_y_spin.setValue(new_shift_y)
        self.shift_x_spin.blockSignals(False)
        self.shift_y_spin.blockSignals(False)
        self.on_transform_changed()

    def on_transform_changed(self) -> None:
        """Update transform status and refresh the previews."""
        self.transform_label.setText(
            f"Rotation {self.rotation_spin.value():+0.3f} deg | "
            f"Shift X {self.shift_x_spin.value():+0.2f} px | "
            f"Shift Y {self.shift_y_spin.value():+0.2f} px"
        )
        if self.highlight_clip_slider.value() <= self.shadow_clip_slider.value():
            self.highlight_clip_slider.blockSignals(True)
            self.highlight_clip_slider.setValue(min(1000, self.shadow_clip_slider.value() + 10))
            self.highlight_clip_slider.blockSignals(False)
        self.shadow_clip_value_label.setText(f"{self.shadow_clip_value():0.3f}")
        self.highlight_clip_value_label.setText(f"{self.highlight_clip_value():0.3f}")
        self.gamma_value_label.setText(f"{self.gamma_value():0.3f}")
        self.stretch_label.setText(
            f"Shadow {self.shadow_clip_value():0.3f} | "
            f"Highlight {self.highlight_clip_value():0.3f} | "
            f"Gamma {self.gamma_value():0.3f}"
        )
        self.update_preview()

    def start_loading(self) -> None:
        """Validate inputs and build the master flat plus preview assets."""
        flat_dir = self.selected_flat_dir()
        light_input = self.selected_light_input()
        output_dir = self.current_output_dir()

        if flat_dir is None or not flat_dir.exists():
            QMessageBox.warning(self, "Missing Flats", "Please select a valid flat FITS directory.")
            return
        if light_input is None or not light_input.exists():
            QMessageBox.warning(self, "Missing Light", "Please select a valid light FITS file or directory.")
            return
        if light_input.is_file() and light_input.suffix.lower() not in SUPPORTED_FITS_SUFFIXES:
            QMessageBox.warning(self, "Missing Light", "Please select a valid light FITS file or directory.")
            return
        if output_dir is None:
            QMessageBox.warning(self, "Missing Output", "Could not determine an output directory.")
            return

        self.load_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.loaded_data = None
        self.preview_limits = None
        self.update_preview()

        self.log(f"Building master flat from {flat_dir}", "green")
        if light_input.is_dir():
            self.log(f"Building stacked light from {light_input}", "green")
        else:
            self.log(f"Using light {light_input}", "green")

        self.load_worker = LoadDataWorker(flat_dir, light_input, output_dir)
        self.load_worker.log_message.connect(self.log)
        self.load_worker.progress_update.connect(self.progress_bar.setValue)
        self.load_worker.finished.connect(self.on_loading_finished)
        self.load_worker.start()

    def on_loading_finished(self, success: bool, message: str, result: object) -> None:
        """Handle completion of the preview-data worker."""
        self.load_button.setEnabled(True)
        if not success:
            self.progress_bar.setValue(0)
            self.log(f"Error: {message}", "red")
            QMessageBox.critical(self, "Load Failed", message)
            self.update_preview()
            return

        self.loaded_data = result
        self.preview_limits = compute_display_limits(self.loaded_data.preview_light)
        self.progress_bar.setValue(100)
        self.set_controls_enabled(True)
        self.log(message, "green")
        self.reset_transform()
        self.update_preview()

    def update_preview(self) -> None:
        """Render the current alignment and corrected previews."""
        if self.loaded_data is None or self.preview_limits is None:
            placeholder = np.zeros((32, 32), dtype=np.uint8)
            self.alignment_view.set_images(placeholder, None)
            self.corrected_view.set_images(placeholder, None)
            return

        preview_shift_x = self.shift_x_spin.value() / float(self.loaded_data.preview_downsample)
        preview_shift_y = self.shift_y_spin.value() / float(self.loaded_data.preview_downsample)
        angle = self.rotation_spin.value()

        transformed_flat = transform_image(
            self.loaded_data.preview_flat,
            angle,
            preview_shift_x,
            preview_shift_y,
            order=1,
            cval=1.0,
        )
        transformed_flat = normalize_flat(transformed_flat)
        corrected_preview = corrected_light(self.loaded_data.preview_light, transformed_flat)

        alignment_image = stretch_with_limits(
            self.loaded_data.preview_light,
            self.preview_limits,
            self.shadow_clip_value(),
            self.highlight_clip_value(),
            self.gamma_value(),
        )
        corrected_image = stretch_with_limits(
            corrected_preview,
            self.preview_limits,
            self.shadow_clip_value(),
            self.highlight_clip_value(),
            self.gamma_value(),
        )
        overlay_image = overlay_rgba(transformed_flat)

        self.alignment_view.set_images(alignment_image, overlay_image)
        self.corrected_view.set_images(corrected_image, None)

    def apply_transform(self) -> None:
        """Write the transformed flat and corrected result to the output directory."""
        if self.loaded_data is None:
            QMessageBox.warning(self, "No Data", "Load the flat directory and light file or directory first.")
            return

        self.apply_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.apply_worker = ApplyTransformWorker(
            self.loaded_data,
            self.rotation_spin.value(),
            self.shift_x_spin.value(),
            self.shift_y_spin.value(),
        )
        self.apply_worker.log_message.connect(self.log)
        self.apply_worker.progress_update.connect(self.progress_bar.setValue)
        self.apply_worker.finished.connect(self.on_apply_finished)
        self.apply_worker.start()

    def on_apply_finished(self, success: bool, message: str, result: object) -> None:
        """Handle completion of the apply worker."""
        self.apply_button.setEnabled(True)
        self.load_button.setEnabled(True)

        if not success:
            self.log(f"Error: {message}", "red")
            QMessageBox.critical(self, "Apply Failed", message)
            return

        outputs: AppliedOutputs = result
        self.progress_bar.setValue(100)
        self.log(message, "green")
        self.log(f"Manifest written to {outputs.manifest_path}", "green")
        self.load_corrected_result_in_siril(outputs.corrected_light_path)

    def load_corrected_result_in_siril(self, corrected_light_path: Path) -> None:
        """Load the corrected result into Siril's main preview when connected."""
        if not self.siril:
            self.log("Siril is not connected, so the corrected result is available only in the output directory.", "orange")
            return

        output_dir = corrected_light_path.parent
        previous_wd = None
        try:
            if hasattr(self.siril, "get_siril_wd"):
                previous_wd = self.siril.get_siril_wd()
            self.siril.cmd("cd", str(output_dir))
            self.siril.cmd("load", corrected_light_path.stem)
            self.log(f"Loaded corrected result into Siril: {corrected_light_path.name}", "green")
        except Exception as exc:
            self.log(f"Could not load corrected result into Siril: {exc}", "orange")
        finally:
            if previous_wd:
                try:
                    self.siril.cmd("cd", str(previous_wd))
                except Exception as exc:
                    self.log(f"Could not restore Siril working directory: {exc}", "orange")

    def save_settings(self) -> None:
        """Persist GUI settings between runs."""
        config_path = Path.home() / ".flat_alignment_fix_config.json"
        settings = {
            "flat_directory": str(self.selected_flat_dir()) if self.selected_flat_dir() else "",
            "light_input": str(self.selected_light_input()) if self.selected_light_input() else "",
            "rotation": self.rotation_spin.value(),
            "rotation_step": self.rotation_step_spin.value(),
            "shift_x": self.shift_x_spin.value(),
            "shift_y": self.shift_y_spin.value(),
            "shadow_clip": self.shadow_clip_value(),
            "highlight_clip": self.highlight_clip_value(),
            "gamma": self.gamma_value(),
        }
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)

    def load_settings(self) -> None:
        """Restore GUI settings from the previous run."""
        config_path = Path.home() / ".flat_alignment_fix_config.json"
        if not config_path.exists():
            return

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                settings = json.load(handle)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            self.log(f"Could not load saved settings: {exc}", "orange")
            return

        flat_directory = settings.get("flat_directory", "").strip()
        light_input = settings.get("light_input", settings.get("light_file", "")).strip()
        if flat_directory:
            self.flat_dir_label.setText(flat_directory)
        if light_input:
            light_path = Path(light_input)
            if light_path.is_dir() or light_path.suffix.lower() in SUPPORTED_FITS_SUFFIXES:
                self.light_input_label.setText(light_input)

        self.rotation_spin.setValue(float(settings.get("rotation", 0.0)))
        self.rotation_step_spin.setValue(float(settings.get("rotation_step", 0.1)))
        self.shift_x_spin.setValue(float(settings.get("shift_x", 0.0)))
        self.shift_y_spin.setValue(float(settings.get("shift_y", 0.0)))
        self.set_shadow_clip_value(float(settings.get("shadow_clip", 0.0)))
        self.set_highlight_clip_value(float(settings.get("highlight_clip", 1.0)))
        self.set_gamma_value(float(settings.get("gamma", 0.5)))
        self.update_output_dir_label()

    def closeEvent(self, event) -> None:
        """Persist settings on close."""
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
        version_constraints=[None, ">=1.20.0", ">=4.0", ">=1.6.0"],
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
            print(f"Warning: could not connect to Siril, using local-only mode: {exc}")
            siril = None

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = FlatAlignmentFixGUI(siril_instance=siril)
    window.show()

    result = app.exec_()
    return 0 if result == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
