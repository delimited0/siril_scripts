#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
#
# RNC Color Stretch GUI for Siril.
# Substantially modified for Siril in 2026 by Patrick Ding.
# Adapted from Roger N. Clark's rnc-color-stretch version 0.975 by
# Johannes H. Gjeraker: https://github.com/jhgjeraker/rnc-color-stretch
#
# Copyright (c) 2016, Roger N. Clark, clarkvision.com
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# Redistributions must retain the above copyright notice, this list of
# conditions, and the following disclaimer. Neither Roger N. Clark,
# clarkvision.com, nor the names of contributors may be used to endorse or
# promote products derived from this software without prior written permission.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for details.
#
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.

"""
RNC Color Stretch GUI for Siril.

Interactive wrapper around Roger N. Clark's color-preserving stretch workflow,
adapted from the Python reference implementation by Johannes H. Gjeraker:
https://github.com/jhgjeraker/rnc-color-stretch

The tool reads Siril's currently loaded image, previews parameter changes on a
downsampled copy, and applies the result back to Siril's active image data.

Requirements:
- sirilpy when launched from Siril
- PyQt5
- numpy
- astropy
- pillow
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

cv2 = None

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

try:
    from skimage.restoration import richardson_lucy
except ImportError:
    richardson_lucy = None

try:
    import sirilpy

    SIRILPY_AVAILABLE = True
except ImportError:
    SIRILPY_AVAILABLE = False

if SIRILPY_AVAILABLE:
    try:
        sirilpy.ensure_installed(
            "PyQt5",
            "numpy",
            "astropy",
            "pillow",
            "opencv-python",
            "scipy",
            "scikit-image",
            version_constraints=[None, ">=1.20.0", ">=4.0", None, None, None, None],
        )
    except Exception as exc:
        raise RuntimeError(f"Error ensuring dependencies: {exc}") from exc

from astropy.io import fits
from PIL import Image
from PyQt5.QtCore import QPoint, QRect, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


SUPPORTED_FITS_SUFFIXES = {".fit", ".fits", ".fts"}
SUPPORTED_TIFF_SUFFIXES = {".tif", ".tiff"}
PREVIEW_MAX_DIMENSION = 1400
DISPLAY_EPSILON = 1e-8
DEFAULT_ZERO_SKY = 4096
CONFIG_PATH = Path.home() / ".siril_rnc_color_stretch.json"
DEFAULT_TONE_CURVE = False
DEFAULT_SKY_LEVEL_FACTOR = 0.060
DEFAULT_ROOTPOWER = 6
DEFAULT_ROOTPOWER2 = 1
DEFAULT_ROOTITER = 1
DEFAULT_STRETCH_TYPE = "root"
DEFAULT_ASINH_K1 = 100
DEFAULT_ASINH_K2 = 5
DEFAULT_LOG_K1 = 100
DEFAULT_LOG_K2 = 5
DEFAULT_S_CURVE_INDEX = 0
DEFAULT_COLOR_CORRECTION_MODE = "ratio"
DEFAULT_COLOR_ENHANCE = 1.0
DEFAULT_COLOR_GAMMA = 5.0
DEFAULT_SETMIN = False
DEFAULT_SETMIN_VALUE = 0
DEFAULT_SKY_REGION_MODE = "full"
DEFAULT_SKY_X = 1000
DEFAULT_SKY_Y = 1000
DEFAULT_SKY_WIDTH = 500
DEFAULT_SKY_HEIGHT = 700
DEFAULT_SKY_STEP_FRACTION = 10
DEFAULT_HSV_ADJUST = False
DEFAULT_HUE_ADJUST = 0.0
DEFAULT_SAT_ADJUST = 1.0
DEFAULT_VAL_ADJUST = 1.0
DEFAULT_VIB_ADJUST = 0.0
DEFAULT_WHITE_BALANCE = "none"
DEFAULT_TEMP = 1.0
DEFAULT_TINT = 1.0
DEFAULT_CA_CORRECT = False
DEFAULT_VIGNETTE_CORRECT = False
DEFAULT_VIGNETTE_STRENGTH = 30
DEFAULT_GRADIENT_CORRECT = False
DEFAULT_GRADIENT_STRENGTH = 100
DEFAULT_STAR_REDUCTION = False
DEFAULT_STAR_REDUCTION_STRENGTH = 1.0
DEFAULT_RL_DECONVOLVE = False
DEFAULT_RL_ITERATIONS = 15
DEFAULT_RL_SIGMA = 0.8
PARAMETER_TOOLTIPS = {
    "tone_curve": (
        "Applies Clark's optional pre-stretch tone curve before sky detection. "
        "Use it when the image is so dark that the histogram sky level cannot be found. "
        "It can over-brighten data that is already well exposed."
    ),
    "sky": (
        "Fraction of the smoothed histogram peak used to find the dark-sky level on the left side of the histogram. "
        "Lower values search farther into the shadows and usually subtract less background. "
        "Higher values choose a brighter sky point and can make the background darker or clip faint signal."
    ),
    "zero_r": (
        "Target red-channel sky level after sky subtraction, in 16-bit ADU. "
        "Higher values leave a brighter/redder background; lower values push red shadows darker."
    ),
    "zero_g": (
        "Target green-channel sky level after sky subtraction, in 16-bit ADU. "
        "This is the main sky reference used by the algorithm. Higher values leave a brighter background; lower values darken it."
    ),
    "zero_b": (
        "Target blue-channel sky level after sky subtraction, in 16-bit ADU. "
        "Higher values leave a brighter/bluer background; lower values push blue shadows darker."
    ),
    "zero_rgb": (
        "Master control for the red, green, and blue zero-sky levels. "
        "Moving this slider sets all three channel sliders together; the individual channel sliders can still be adjusted afterward."
    ),
    "rootpower": (
        "Main root stretch exponent denominator. The algorithm applies output = input^(1/root power) after sky subtraction. "
        "Higher values brighten shadows and faint nebulosity more strongly; lower values are milder."
    ),
    "rootpower2": (
        "Root power used on additional root-stretch passes after the first. "
        "Only matters when root iterations is greater than 1. Higher values make later passes more aggressive."
    ),
    "rootiter": (
        "Number of root-stretch and sky-rebalance passes. "
        "More passes can reveal faint signal but can also amplify noise and make the image look harsh."
    ),
    "s_curve": (
        "Optional contrast curve after the root stretch. "
        "S-curve 1 increases contrast around midtones and darkens low levels; S-curve 2 brightens more broadly with less low-end effect. "
        "Stacked options apply both."
    ),
    "color_correction": (
        "Color recovery method after stretching: none, ratio recovery, or HSV recovery from the original hue/saturation."
    ),
    "color_enhance": (
        "Multiplier for the signal-dependent color recovery. "
        "Higher values increase saturation/color recovery, especially in brighter structures; lower values reduce color correction and noise risk."
    ),
    "setmin": (
        "Raises pixels below the selected minimum levels. "
        "This can hide very dark color artifacts or chromatic star halos, but too much will lift the black floor."
    ),
    "min_r": (
        "Minimum red output level used when Set minimum is enabled. "
        "Higher values lift very dark red pixels and can suppress red shadow artifacts."
    ),
    "min_g": (
        "Minimum green output level used when Set minimum is enabled. "
        "Higher values lift very dark green pixels and can suppress green shadow artifacts."
    ),
    "min_b": (
        "Minimum blue output level used when Set minimum is enabled. "
        "Higher values lift very dark blue pixels and can suppress blue shadow artifacts."
    ),
}

STRETCH_TYPES = ("none", "root", "asinh", "log")
COLOR_CORRECTION_MODES = ("none", "ratio", "hsv")
SKY_REGION_MODES = ("full", "auto", "manual")
WHITE_BALANCE_MODES = ("none", "gray", "temp_tint")


@dataclass(frozen=True)
class StretchParameters:
    tone_curve: bool
    sky_region_mode: str
    sky_x: int
    sky_y: int
    sky_width: int
    sky_height: int
    sky_step_fraction: int
    skylevelfactor: float
    zerosky_r: int
    zerosky_g: int
    zerosky_b: int
    stretch_type: str
    rootpower: int
    rootpower2: int
    rootiter: int
    asinh_k1: int
    asinh_k2: int
    asinh_iter: int
    log_k1: int
    log_k2: int
    log_iter: int
    s_curve: int
    setmin: bool
    setmin_r: int
    setmin_g: int
    setmin_b: int
    color_correction_mode: str
    colorenhance: float
    color_gamma: float
    hsv_adjust: bool
    hue_adjust: float
    sat_adjust: float
    val_adjust: float
    vib_adjust: float
    white_balance: str
    temp: float
    tint: float
    ca_correct: bool
    vignette_correct: bool
    vignette_strength: int
    gradient_correct: bool
    gradient_strength: int
    star_reduction: bool
    star_reduction_strength: float
    rl_deconvolve: bool
    rl_iterations: int
    rl_sigma: float


@dataclass
class LoadedImage:
    path: Path
    rgb: np.ndarray
    header: fits.Header
    source_is_current: bool
    is_color: bool
    original_shape: Tuple[int, ...]
    original_dtype: str
    preview_input: np.ndarray
    preview_scale: int
    display_limits: Tuple[float, float]
    siril_data_shape: Optional[Tuple[int, ...]] = None
    siril_data_dtype: Optional[str] = None


@dataclass
class StretchResult:
    rgb: np.ndarray
    log: str


class ParameterSlider(QWidget):
    valueChanged = pyqtSignal()

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 0,
        single_step: float = 1.0,
        page_step: float = 10.0,
    ):
        super().__init__()
        self.decimals = decimals
        self.scale = 10**decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(self._to_slider(minimum), self._to_slider(maximum))
        self.slider.setSingleStep(max(1, self._to_slider(single_step)))
        self.slider.setPageStep(max(1, self._to_slider(page_step)))
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(54)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.setValue(value)

    def _to_slider(self, value: float) -> int:
        return int(round(float(value) * self.scale))

    def _from_slider(self, value: int) -> float:
        scaled = float(value) / float(self.scale)
        if self.decimals == 0:
            return int(round(scaled))
        return scaled

    def _format_value(self) -> str:
        value = self.value()
        if self.decimals == 0:
            return f"{int(value)}"
        return f"{float(value):.{self.decimals}f}"

    def _on_slider_changed(self) -> None:
        self.value_label.setText(self._format_value())
        self.valueChanged.emit()

    def value(self):
        return self._from_slider(self.slider.value())

    def setValue(self, value: float) -> None:
        self.slider.setValue(self._to_slider(value))
        self.value_label.setText(self._format_value())


def canonicalize_fits_array(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unsupported FITS dimensions: {array.shape}")
    if array.shape[0] >= 3 and array.shape[0] <= 4:
        return np.moveaxis(array[:3], 0, -1)
    if array.shape[-1] >= 3 and array.shape[-1] <= 4:
        return array[:, :, :3]
    raise ValueError(f"Could not determine RGB axes for FITS data shape {array.shape}")


def debayer_image(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    data = np.asarray(mosaic, dtype=np.float32)
    pattern = pattern.upper()
    if len(pattern) != 4 or any(ch not in "RGB" for ch in pattern):
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    height, width = data.shape
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width, 3), dtype=np.float32)
    channel_for = {"R": 0, "G": 1, "B": 2}

    for yoff in range(2):
        for xoff in range(2):
            channel = channel_for[pattern[yoff * 2 + xoff]]
            rgb[yoff::2, xoff::2, channel] = data[yoff::2, xoff::2]
            counts[yoff::2, xoff::2, channel] = 1.0

    for channel in range(3):
        plane = rgb[:, :, channel]
        mask = counts[:, :, channel]
        filled = interpolate_missing_samples(plane, mask)
        rgb[:, :, channel] = filled
    return rgb


def interpolate_missing_samples(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask > 0):
        return np.zeros_like(plane, dtype=np.float32)

    filled = plane.astype(np.float32, copy=True)
    weights = mask.astype(np.float32, copy=True)
    for _ in range(8):
        padded_values = np.pad(filled * weights, 1, mode="edge")
        padded_weights = np.pad(weights, 1, mode="edge")
        value_sum = (
            padded_values[:-2, 1:-1]
            + padded_values[2:, 1:-1]
            + padded_values[1:-1, :-2]
            + padded_values[1:-1, 2:]
            + padded_values[:-2, :-2]
            + padded_values[:-2, 2:]
            + padded_values[2:, :-2]
            + padded_values[2:, 2:]
        )
        weight_sum = (
            padded_weights[:-2, 1:-1]
            + padded_weights[2:, 1:-1]
            + padded_weights[1:-1, :-2]
            + padded_weights[1:-1, 2:]
            + padded_weights[:-2, :-2]
            + padded_weights[:-2, 2:]
            + padded_weights[2:, :-2]
            + padded_weights[2:, 2:]
        )
        fillable = (weights == 0.0) & (weight_sum > 0.0)
        if not np.any(fillable):
            break
        filled[fillable] = value_sum[fillable] / weight_sum[fillable]
        weights[fillable] = 1.0

    if np.any(weights == 0.0):
        filled[weights == 0.0] = float(np.mean(filled[weights > 0.0]))
    return filled


def read_fits_rgb(path: Path) -> Tuple[np.ndarray, fits.Header, bool, Tuple[int, ...], str]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is None:
                continue
            raw = np.asarray(hdu.data)
            header = hdu.header.copy()
            bayer_pattern = header.get("BAYERPAT")
            if raw.ndim == 2 and bayer_pattern:
                return debayer_image(raw, str(bayer_pattern)), header, True, raw.shape, str(raw.dtype)

            array = canonicalize_fits_array(raw)
            if array.ndim == 2:
                rgb = np.repeat(array[:, :, np.newaxis], 3, axis=2)
                return rgb.astype(np.float32), header, False, raw.shape, str(raw.dtype)
            return array.astype(np.float32), header, True, raw.shape, str(raw.dtype)
    raise ValueError(f"No image data found in {path}")


def read_tiff_rgb(path: Path) -> Tuple[np.ndarray, fits.Header, bool, Tuple[int, ...], str]:
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)
    header = fits.Header()
    header["HISTORY"] = f"Loaded from TIFF: {path.name}"
    if array.ndim == 2:
        rgb = np.repeat(array[:, :, np.newaxis], 3, axis=2)
        return rgb.astype(np.float32), header, False, array.shape, str(array.dtype)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported TIFF dimensions: {array.shape}")
    return array[:, :, :3].astype(np.float32), header, True, array.shape, str(array.dtype)


def read_image_file(path: Path, source_is_current: bool = False) -> LoadedImage:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_FITS_SUFFIXES:
        rgb, header, is_color, original_shape, original_dtype = read_fits_rgb(path)
    elif suffix in SUPPORTED_TIFF_SUFFIXES:
        rgb, header, is_color, original_shape, original_dtype = read_tiff_rgb(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    preview_scale = choose_preview_downsample(rgb.shape[:2])
    preview_input = rgb[::preview_scale, ::preview_scale, :].astype(np.float32, copy=True)
    display_limits = linear_display_limits(rgb, original_dtype)
    return LoadedImage(
        path=path,
        rgb=rgb.astype(np.float32, copy=False),
        header=header,
        source_is_current=source_is_current,
        is_color=is_color,
        original_shape=original_shape,
        original_dtype=original_dtype,
        preview_input=preview_input,
        preview_scale=preview_scale,
        display_limits=display_limits,
    )


def header_from_siril(header_text: Optional[str]) -> fits.Header:
    if not header_text:
        return fits.Header()
    try:
        return fits.Header.fromstring(str(header_text), sep="\n")
    except Exception:
        header = fits.Header()
        header["HISTORY"] = "Siril FITS header could not be parsed by Astropy."
        return header


def siril_data_to_rgb(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    array = np.asarray(data)
    if array.ndim == 2:
        rgb = np.repeat(array[:, :, np.newaxis], 3, axis=2)
        return rgb.astype(np.float32), False
    if array.ndim == 3 and array.shape[0] in (1, 3):
        if array.shape[0] == 1:
            plane = array[0, :, :]
            rgb = np.repeat(plane[:, :, np.newaxis], 3, axis=2)
            return rgb.astype(np.float32), False
        return np.moveaxis(array, 0, -1).astype(np.float32), True
    if array.ndim == 3 and array.shape[-1] in (1, 3):
        if array.shape[-1] == 1:
            rgb = np.repeat(array[:, :, :1], 3, axis=2)
            return rgb.astype(np.float32), False
        return array[:, :, :3].astype(np.float32), True
    raise ValueError(f"Unsupported Siril image data shape: {array.shape}")


def loaded_image_from_siril_fit(fit, filename: Optional[str] = None) -> LoadedImage:
    data = getattr(fit, "data", None)
    if data is None:
        raise ValueError("Siril returned image metadata without pixel data.")

    array = np.asarray(data)
    rgb, is_color = siril_data_to_rgb(array)
    preview_scale = choose_preview_downsample(rgb.shape[:2])
    preview_input = rgb[::preview_scale, ::preview_scale, :].astype(np.float32, copy=True)
    original_dtype = str(array.dtype)
    display_limits = linear_display_limits(rgb, original_dtype)
    path = Path(filename) if filename else Path("current_siril_image.fit")
    return LoadedImage(
        path=path,
        rgb=rgb.astype(np.float32, copy=False),
        header=header_from_siril(getattr(fit, "header", "")),
        source_is_current=True,
        is_color=is_color,
        original_shape=tuple(array.shape),
        original_dtype=original_dtype,
        preview_input=preview_input,
        preview_scale=preview_scale,
        display_limits=display_limits,
        siril_data_shape=tuple(array.shape),
        siril_data_dtype=original_dtype,
    )


def result_rgb_to_siril_data(rgb: np.ndarray, source: LoadedImage) -> np.ndarray:
    data = np.asarray(rgb)
    dtype_name = (source.siril_data_dtype or source.original_dtype).lower()
    if "uint16" in dtype_name:
        converted = np.clip(np.rint(data), 0, 65535).astype(np.uint16)
    else:
        converted = np.clip(data, 0.0, 1.0).astype(np.float32)

    shape = source.siril_data_shape or source.original_shape
    if len(shape) == 2:
        return converted[:, :, 1] if converted.ndim == 3 else converted
    if len(shape) == 3 and shape[0] == 1:
        plane = converted[:, :, 1] if converted.ndim == 3 else converted
        return np.ascontiguousarray(plane[np.newaxis, :, :])
    if len(shape) == 3 and shape[0] == 3:
        return np.ascontiguousarray(np.moveaxis(converted[:, :, :3], -1, 0))
    if len(shape) == 3 and shape[-1] == 1:
        return np.ascontiguousarray(converted[:, :, 1:2] if converted.ndim == 3 else converted[:, :, np.newaxis])
    if len(shape) == 3 and shape[-1] == 3:
        return np.ascontiguousarray(converted[:, :, :3])
    return np.ascontiguousarray(np.moveaxis(converted[:, :, :3], -1, 0))


def choose_preview_downsample(shape: Sequence[int]) -> int:
    height, width = int(shape[0]), int(shape[1])
    return max(1, int(np.ceil(max(height, width) / PREVIEW_MAX_DIMENSION)))


def scale_to_16bit_range(rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(rgb, dtype=np.float32).copy()
    finite_max = float(np.nanmax(img))
    finite_min = float(np.nanmin(img))
    if not np.isfinite(finite_max) or not np.isfinite(finite_min):
        raise ValueError("Image contains no finite data")
    img[~np.isfinite(img)] = 0.0
    if finite_max <= 1.00001:
        img *= 65535.0
    elif finite_max < 8000.0:
        img *= 65000.0 / max(finite_max, 1.0)
    img[img < 0.0] = 0.0
    return img


def restore_source_scale(result_16: np.ndarray, source_rgb: np.ndarray) -> np.ndarray:
    source_max = float(np.nanmax(source_rgb))
    if source_max <= 1.00001 and np.issubdtype(source_rgb.dtype, np.floating):
        return np.clip(result_16 / 65535.0, 0.0, 1.0).astype(np.float32)
    return np.clip(result_16, 0.0, 65535.0).astype(np.float32)


def histogram_channel(img: np.ndarray, channel: int) -> np.ndarray:
    values = np.clip(np.rint(img[:, :, channel]), 0, 65535).astype(np.uint16, copy=False)
    return np.bincount(values.ravel(), minlength=65536).astype(np.float64)


def smooth_histogram(hist: np.ndarray, width: int = 601) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(hist, kernel, mode="same")


def require_cv2(feature: str) -> None:
    global cv2
    if cv2 is None:
        qt_plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
        qt_plugins_path = os.environ.get("QT_PLUGIN_PATH")
        try:
            import cv2 as imported_cv2
        except ImportError as exc:
            raise RuntimeError(f"{feature} requires OpenCV. Install opencv-python in Siril's Python environment.") from exc
        finally:
            if qt_plugin_path is None:
                os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
            else:
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_plugin_path
            if qt_plugins_path is None:
                os.environ.pop("QT_PLUGIN_PATH", None)
            else:
                os.environ["QT_PLUGIN_PATH"] = qt_plugins_path
        cv2 = imported_cv2


def require_rl_deps() -> None:
    if gaussian_filter is None or richardson_lucy is None:
        raise RuntimeError("Richardson-Lucy deconvolution requires scipy and scikit-image.")


def rgb_channels(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return img[:, :, 0], img[:, :, 1], img[:, :, 2]


def merge_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([r, g, b], axis=2)


def selected_sky_region(img: np.ndarray, params: StretchParameters, lines: list[str]) -> Tuple[int, int, int, int]:
    height, width = img.shape[:2]
    if params.sky_region_mode == "full":
        return 0, 0, width, height

    win_width = min(max(1, params.sky_width), width)
    win_height = min(max(1, params.sky_height), height)

    if params.sky_region_mode == "manual":
        x1 = min(max(0, params.sky_x), max(0, width - win_width))
        y1 = min(max(0, params.sky_y), max(0, height - win_height))
        lines.append(f"Manual sky region: x={x1}, y={y1}, w={win_width}, h={win_height}")
        return x1, y1, x1 + win_width, y1 + win_height

    if params.sky_region_mode == "auto":
        luminance = np.mean(img, axis=2)
        step_fraction = max(1, params.sky_step_fraction)
        step_x = max(1, win_width // step_fraction)
        step_y = max(1, win_height // step_fraction)
        best_mean = float("inf")
        best_x, best_y = 0, 0
        for y in range(0, height - win_height + 1, step_y):
            for x in range(0, width - win_width + 1, step_x):
                region_mean = float(np.mean(luminance[y : y + win_height, x : x + win_width]))
                if region_mean < best_mean:
                    best_mean = region_mean
                    best_x, best_y = x, y
        lines.append(
            f"Auto sky region: x={best_x}, y={best_y}, w={win_width}, h={win_height}, mean {best_mean:.1f}"
        )
        return best_x, best_y, best_x + win_width, best_y + win_height

    raise ValueError(f"Unknown sky region mode: {params.sky_region_mode}")


def append_stats(lines: list[str], label: str, img: np.ndarray) -> None:
    mins = np.nanmin(img, axis=(0, 1))
    maxs = np.nanmax(img, axis=(0, 1))
    means = np.nanmean(img, axis=(0, 1))
    lines.append(
        f"{label}: R {mins[0]:.1f}-{maxs[0]:.1f} mean {means[0]:.1f}, "
        f"G {mins[1]:.1f}-{maxs[1]:.1f} mean {means[1]:.1f}, "
        f"B {mins[2]:.1f}-{maxs[2]:.1f} mean {means[2]:.1f}"
    )


def tone_curve(img: np.ndarray) -> np.ndarray:
    return img * 12.0 * ((1.0 / 12.0) ** ((img / 65535.0) ** 0.4))


def smooth_and_subtract(
    in_img: np.ndarray,
    params: StretchParameters,
    lines: list[str],
    label: str,
    passes: int = 2,
) -> np.ndarray:
    img = np.array(in_img, dtype=np.float32, copy=True)
    x1, y1, x2, y2 = selected_sky_region(img, params, lines)
    for pass_index in range(passes):
        peaks = []
        sky_indexes = []
        selected = img[y1:y2, x1:x2, :]
        for channel in range(3):
            hist = histogram_channel(selected, channel)
            smoothed = smooth_histogram(hist)
            peak = int(np.argmax(smoothed[400:65501]) + 400)
            peaks.append(peak)

        green_hist = histogram_channel(selected, 1)
        green_smoothed = smooth_histogram(green_hist)
        green_peak = int(np.argmax(green_smoothed[400:65501]) + 400)
        green_target = float(green_smoothed[green_peak] * params.skylevelfactor)

        for channel in range(3):
            hist = histogram_channel(selected, channel)
            smoothed = smooth_histogram(hist)
            peak = peaks[channel]
            sky_index = 0
            for idx in range(peak, 0, -1):
                if smoothed[idx] >= green_target and smoothed[idx - 1] <= green_target:
                    sky_index = idx
                    break
            if sky_index == 0:
                raise ValueError(
                    f"Could not find {label} sky level for channel {channel + 1}. "
                    "Try enabling tone curve or lowering the sky-level factor."
                )
            sky_indexes.append(sky_index)

        zeros = np.array([params.zerosky_r, params.zerosky_g, params.zerosky_b], dtype=np.float32)
        subtract = np.array(sky_indexes, dtype=np.float32) - zeros
        denom = np.maximum(65535.0 - subtract, 1.0)
        img = (img - subtract.reshape(1, 1, 3)) * (65535.0 / denom.reshape(1, 1, 3))
        img[img < 0.0] = 0.0
        lines.append(
            f"{label} sky pass {pass_index + 1}: peaks RGB {peaks}, "
            f"sky RGB {sky_indexes}, subtract RGB {[int(v) for v in subtract]}"
        )
    return img


def root_stretch(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    img = np.array(in_img, dtype=np.float32, copy=True)
    powers = [params.rootpower]
    if params.rootiter >= 2:
        powers.extend([params.rootpower2] * (params.rootiter - 1))
    for index, power in enumerate(powers[: params.rootiter], start=1):
        exponent = 1.0 / float(max(power, 1))
        stretched = 65535.0 * (((img + 1.0) / 65536.0) ** exponent)
        minimum = max(int(np.nanmin(stretched)) - 4095, 0)
        stretched = (stretched - float(minimum)) / max(65535.0 - float(minimum), 1.0)
        img = np.clip(65535.0 * stretched, 0.0, 65535.0)
        lines.append(f"Root stretch pass {index}: power {power}, subtracted minimum {minimum}")
        sky_passes = 3 if power > 60 else 2
        img = smooth_and_subtract(img, params, lines, f"root pass {index}", passes=sky_passes)
    return img


def asinh_stretch(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    img = np.array(in_img, dtype=np.float32, copy=True)
    factors = [params.asinh_k1]
    if params.asinh_iter >= 2:
        factors.extend([params.asinh_k2] * (params.asinh_iter - 1))
    for index, factor in enumerate(factors[: params.asinh_iter], start=1):
        kf = float(max(factor, 1))
        stretched = 65535.0 * (np.arcsinh(kf * (img + 1.0) / 65536.0) / np.arcsinh(kf))
        minimum = max(int(np.nanmin(stretched)) - 4095, 0)
        stretched = (stretched - float(minimum)) / max(65535.0 - float(minimum), 1.0)
        img = np.clip(65535.0 * stretched, 0.0, 65535.0)
        lines.append(f"ASINH stretch pass {index}: K {factor}, subtracted minimum {minimum}")
        sky_passes = 3 if factor > 100 else 2
        img = smooth_and_subtract(img, params, lines, f"asinh pass {index}", passes=sky_passes)
    return img


def log_stretch(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    img = np.array(in_img, dtype=np.float32, copy=True)
    factors = [params.log_k1]
    if params.log_iter >= 2:
        factors.extend([params.log_k2] * (params.log_iter - 1))
    for index, factor in enumerate(factors[: params.log_iter], start=1):
        kf = float(max(factor, 1))
        normalized = (img + 1.0) / 65536.0
        stretched = 65535.0 * (np.log1p(kf * normalized) / np.log1p(kf))
        minimum = max(int(np.nanmin(stretched)) - 4095, 0)
        stretched = (stretched - float(minimum)) / max(65535.0 - float(minimum), 1.0)
        img = np.clip(65535.0 * stretched, 0.0, 65535.0)
        lines.append(f"Log stretch pass {index}: K {factor}, subtracted minimum {minimum}")
        sky_passes = 3 if factor > 100 else 2
        img = smooth_and_subtract(img, params, lines, f"log pass {index}", passes=sky_passes)
    return img


def s_curve(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    img = np.array(in_img, dtype=np.float32, copy=True)
    for index in range(params.s_curve):
        if index + 1 in (2, 4):
            xfactor = 3.0
            xoffset = 0.22
        else:
            xfactor = 5.0
            xoffset = 0.42

        scurvemin = xfactor / (1.0 + np.exp(-((0.0 - xoffset) * xfactor))) - (1.0 - xoffset)
        scurvemax = xfactor / (1.0 + np.exp(-((1.0 - xoffset) * xfactor))) - (1.0 - xoffset)
        scurveminsc = scurvemin / scurvemax
        sc = img / 65535.0
        sc = xfactor / (1.0 + np.exp(-((sc - xoffset) * xfactor))) - (1.0 - xoffset)
        sc = sc / scurvemax
        sc = sc - scurveminsc
        img = 65535.0 * sc / (1.0 - scurveminsc)
        lines.append(f"S-curve pass {index + 1}: factor {xfactor:.1f}, offset {xoffset:.2f}")

    return smooth_and_subtract(img, params, lines, "s-curve")


def set_minimum(in_img: np.ndarray, params: StretchParameters) -> np.ndarray:
    img = np.array(in_img, dtype=np.float32, copy=True)
    minimums = np.array([params.setmin_r, params.setmin_g, params.setmin_b], dtype=np.float32)
    for channel in range(3):
        mask = img[:, :, channel] < minimums[channel]
        img[:, :, channel] = np.where(mask, minimums[channel] + 0.2 * img[:, :, channel], img[:, :, channel])
    return img


def color_correct(
    in_img: np.ndarray,
    original_subtracted: np.ndarray,
    params: StretchParameters,
    lines: list[str],
) -> np.ndarray:
    img = np.asarray(in_img, dtype=np.float32).copy()
    original = np.asarray(original_subtracted, dtype=np.float32).copy()
    zeros = np.array([params.zerosky_r, params.zerosky_g, params.zerosky_b], dtype=np.float32)
    original -= zeros.reshape(1, 1, 3)
    original[original < 10.0] = 10.0
    img[img < 10.0] = 10.0

    gr = (original[:, :, 1] / original[:, :, 0]) / (img[:, :, 1] / img[:, :, 0])
    br = (original[:, :, 2] / original[:, :, 0]) / (img[:, :, 2] / img[:, :, 0])
    rg = (original[:, :, 0] / original[:, :, 1]) / (img[:, :, 0] / img[:, :, 1])
    bg = (original[:, :, 2] / original[:, :, 1]) / (img[:, :, 2] / img[:, :, 1])
    gb = (original[:, :, 1] / original[:, :, 2]) / (img[:, :, 1] / img[:, :, 2])
    rb = (original[:, :, 0] / original[:, :, 2]) / (img[:, :, 0] / img[:, :, 2])

    ratios = [np.clip(ratio, 0.2, 1.0) for ratio in (gr, br, rg, bg, gb, rb)]
    gr, br, rg, bg, gb, rb = ratios

    cavgn = np.mean(img, axis=2) / 65535.0
    max_cavgn = float(np.nanmax(cavgn))
    if max_cavgn > 0.0 and max_cavgn < 1.0:
        cavgn = cavgn / max_cavgn
    cavgn = np.clip(cavgn, 0.0, None) ** 0.2
    cavgn = (cavgn + 0.3) / 1.3
    cfe = 1.2 * params.colorenhance * cavgn

    gr = 1.0 + cfe * (gr - 1.0)
    br = 1.0 + cfe * (br - 1.0)
    rg = 1.0 + cfe * (rg - 1.0)
    bg = 1.0 + cfe * (bg - 1.0)
    gb = 1.0 + cfe * (gb - 1.0)
    rb = 1.0 + cfe * (rb - 1.0)

    c2gr = img[:, :, 1] * gr
    c3br = img[:, :, 2] * br
    c1rg = img[:, :, 0] * rg
    c3bg = img[:, :, 2] * bg
    c1rb = img[:, :, 0] * rb
    c2gb = img[:, :, 1] * gb

    max_channel = np.argmax(img, axis=2)
    red_max = max_channel == 0
    green_max = max_channel == 1
    blue_max = max_channel == 2
    img[:, :, 1] = np.where(red_max, c2gr, img[:, :, 1])
    img[:, :, 2] = np.where(red_max, c3br, img[:, :, 2])
    img[:, :, 0] = np.where(green_max, c1rg, img[:, :, 0])
    img[:, :, 2] = np.where(green_max, c3bg, img[:, :, 2])
    img[:, :, 0] = np.where(blue_max, c1rb, img[:, :, 0])
    img[:, :, 1] = np.where(blue_max, c2gb, img[:, :, 1])
    lines.append(f"Color correction: enhancement {params.colorenhance:.2f}, cfe mean {float(np.nanmean(cfe)):.2f}")
    return img


def hsv_color_correct(
    in_img: np.ndarray,
    original_subtracted: np.ndarray,
    params: StretchParameters,
    lines: list[str],
) -> np.ndarray:
    require_cv2("HSV color correction")
    original = np.float32(np.clip(original_subtracted / 65535.0, 0.0, 1.0))
    stretched = np.float32(np.clip(in_img / 65535.0, 0.0, 1.0))
    hsv_orig = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)
    hsv_stretched = cv2.cvtColor(stretched, cv2.COLOR_RGB2HSV)
    hue = hsv_orig[:, :, 0]
    saturation = hsv_orig[:, :, 1]
    value = hsv_stretched[:, :, 2]
    gamma = max(params.color_gamma, 0.1)
    saturation_boost = 1.0 + params.colorenhance * (value ** (1.0 / gamma))
    hsv_new = np.stack([hue, np.clip(saturation * saturation_boost, 0.0, 1.0), value], axis=2).astype(np.float32)
    lines.append(f"HSV color correction: enhancement {params.colorenhance:.2f}, gamma {gamma:.2f}")
    return np.clip(cv2.cvtColor(hsv_new, cv2.COLOR_HSV2RGB) * 65535.0, 0.0, 65535.0)


def hsv_adjust(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    if not params.hsv_adjust:
        return in_img
    require_cv2("HSV adjustment")
    rgb = np.float32(np.clip(in_img / 65535.0, 0.0, 1.0))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + params.hue_adjust) % 360.0
    hsv[:, :, 1] = hsv[:, :, 1] * params.sat_adjust
    if params.vib_adjust != 0:
        saturation = hsv[:, :, 1]
        hsv[:, :, 1] = saturation + params.vib_adjust * saturation * (1.0 - saturation)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0.0, 1.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * params.val_adjust, 0.0, 1.0)
    lines.append(
        f"HSV adjust: hue {params.hue_adjust:.1f}, sat {params.sat_adjust:.2f}, "
        f"value {params.val_adjust:.2f}, vibrance {params.vib_adjust:.2f}"
    )
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 65535.0, 0.0, 65535.0)


def apply_white_balance(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    if params.white_balance == "none":
        return in_img
    img = np.array(in_img, dtype=np.float32, copy=True)
    r, g, b = rgb_channels(img)
    if params.white_balance == "gray":
        means = np.array([np.mean(r), np.mean(g), np.mean(b)], dtype=np.float32)
        gray = float(np.mean(means))
        scales = gray / np.maximum(means, 1e-6)
        img *= scales.reshape(1, 1, 3)
        lines.append(f"White balance: gray world scales R/G/B {scales[0]:.3f}/{scales[1]:.3f}/{scales[2]:.3f}")
    elif params.white_balance == "temp_tint":
        temp = max(params.temp, 0.01)
        tint = max(params.tint, 0.01)
        magenta_boost = math.sqrt(tint)
        img[:, :, 0] *= temp * magenta_boost
        img[:, :, 1] *= 1.0 / tint
        img[:, :, 2] *= (1.0 / temp) * magenta_boost
        lines.append(f"White balance: temp {temp:.2f}, tint {tint:.2f}")
    else:
        raise ValueError(f"Unknown white balance mode: {params.white_balance}")
    return np.clip(img, 0.0, 65535.0)


def align_channels(in_img: np.ndarray, lines: list[str]) -> np.ndarray:
    require_cv2("Chromatic aberration correction")
    img = np.asarray(in_img, dtype=np.float32)
    r, g, b = rgb_channels(img)

    def align(ref: np.ndarray, target: np.ndarray) -> np.ndarray:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        _, warp_matrix = cv2.findTransformECC(
            ref.astype(np.float32), target.astype(np.float32), warp_matrix, cv2.MOTION_TRANSLATION
        )
        return cv2.warpAffine(
            target.astype(np.float32),
            warp_matrix,
            (target.shape[1], target.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )

    try:
        result = merge_rgb(align(g, r), g, align(g, b))
        lines.append("Chromatic aberration correction: aligned red and blue to green")
        return np.clip(result, 0.0, 65535.0)
    except Exception as exc:
        lines.append(f"Chromatic aberration correction skipped: {exc}")
        return in_img


def fix_channel_bounds(channel: np.ndarray) -> np.ndarray:
    return np.clip(channel, 0.0, 65535.0)


def radial_vignette_correction(in_img: np.ndarray, strength: int, lines: list[str]) -> np.ndarray:
    img = np.asarray(in_img, dtype=np.float32)
    height, width = img.shape[:2]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    radius = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2)
    design = np.c_[radius.ravel(), (radius**2).ravel(), np.ones(radius.size)]
    factor = np.clip(float(strength) / 100.0, 0.0, 1.0)
    corrected = np.empty_like(img)
    for channel in range(3):
        coeffs, _, _, _ = np.linalg.lstsq(design, img[:, :, channel].ravel(), rcond=None)
        background = coeffs[0] * radius + coeffs[1] * radius**2 + coeffs[2]
        plane = img[:, :, channel] - factor * background
        plane -= np.nanmin(plane)
        corrected[:, :, channel] = fix_channel_bounds(plane)
    lines.append(f"Radial vignette correction: strength {strength}%")
    return corrected


def linear_gradient_correction(in_img: np.ndarray, strength: int, lines: list[str]) -> np.ndarray:
    img = np.asarray(in_img, dtype=np.float32)
    height, width = img.shape[:2]
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    design = np.c_[x.ravel(), y.ravel(), np.ones(x.size)]
    factor = np.clip(float(strength) / 100.0, 0.0, 1.0)
    corrected = np.empty_like(img)
    for channel in range(3):
        coeffs, _, _, _ = np.linalg.lstsq(design, img[:, :, channel].ravel(), rcond=None)
        background = coeffs[0] * x + coeffs[1] * y + coeffs[2]
        plane = img[:, :, channel] - factor * background
        plane -= np.nanmin(plane)
        corrected[:, :, channel] = fix_channel_bounds(plane)
    lines.append(f"Linear gradient correction: strength {strength}%")
    return corrected


def reduce_star_sizes(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    if not params.star_reduction or params.star_reduction_strength <= 0:
        return in_img
    require_cv2("Star size reduction")
    img = np.asarray(in_img, dtype=np.float32)
    gray = np.mean(img, axis=2) / 65535.0
    blur_small = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_large = cv2.GaussianBlur(gray, (15, 15), 0)
    dog = blur_small - blur_large
    mean, std = cv2.meanStdDev(dog)
    _, binary = cv2.threshold(dog, float(mean[0][0] + 1.5 * std[0][0]), 1.0, cv2.THRESH_BINARY)
    mask = (binary * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= 5)[0] + 1
    star_mask = np.isin(labels, keep).astype(np.uint8)
    if not np.any(star_mask):
        lines.append("Star reduction: no stars detected")
        return in_img

    result = img.copy()
    strength = float(np.clip(params.star_reduction_strength, 0.0, 1.0))
    passes = (
        (int(1 + strength * 2), 60 + strength * 10, int(3 + strength * 4)),
        (int(1 + strength * 2), 75 + strength * 10, int(5 + strength * 4)),
        (int(1 + strength), 85 + strength * 7, int(5 + strength * 4)),
    )
    gray_full = np.mean(result, axis=2)
    star_pixels = gray_full[star_mask > 0]
    for erosion_size, percentile, feather_size in passes:
        if feather_size % 2 == 0:
            feather_size += 1
        core_threshold = np.percentile(star_pixels, percentile)
        halo_mask = ((gray_full < core_threshold) & (star_mask > 0)).astype(np.float32)
        feathered = np.clip(cv2.GaussianBlur(halo_mask, (feather_size, feather_size), 0), 0.0, 1.0)
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, erosion_size), max(1, erosion_size)))
        for channel in range(3):
            eroded = cv2.erode(result[:, :, channel], erosion_kernel, iterations=1)
            result[:, :, channel] = result[:, :, channel] * (1.0 - feathered) + eroded * feathered
    lines.append(f"Star reduction: strength {strength:.2f}")
    return np.clip(result, 0.0, 65535.0)


def richardson_lucy_deconvolution(in_img: np.ndarray, params: StretchParameters, lines: list[str]) -> np.ndarray:
    if not params.rl_deconvolve:
        return in_img
    require_rl_deps()
    sigma = max(float(params.rl_sigma), 0.1)
    iterations = max(1, int(params.rl_iterations))
    img = np.asarray(in_img, dtype=np.float32) / 65535.0
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    luminance = gaussian_filter(luminance, sigma=0.5)
    luminance = np.clip(luminance, 1e-6, 1.0)
    size = max(3, int(np.ceil(sigma * 8)))
    if size % 2 == 0:
        size += 1
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    beta = 4.0
    fwhm = 2.355 * sigma
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    psf = (1.0 + ((x**2 + y**2) / alpha**2)) ** (-beta)
    psf /= np.sum(psf)
    pad = size // 2
    lum_pad = np.pad(luminance, pad, mode="reflect")
    try:
        processed = richardson_lucy(lum_pad, psf, num_iter=iterations, clip=True)
    except TypeError:
        processed = richardson_lucy(lum_pad, psf, iterations=iterations, clip=True)
    processed = processed[pad:-pad, pad:-pad]
    ratio = np.clip(processed / luminance, 0.0, 5.0)
    result = np.clip(img * ratio[:, :, np.newaxis], 0.0, 1.0) * 65535.0
    lines.append(f"Richardson-Lucy deconvolution: iterations {iterations}, sigma {sigma:.2f}, kernel {size}")
    return result.astype(np.float32)


def apply_rnc_stretch(source_rgb: np.ndarray, params: StretchParameters) -> StretchResult:
    lines: list[str] = []
    img = scale_to_16bit_range(source_rgb)
    append_stats(lines, "Input scaled to 16-bit range", img)

    if params.ca_correct:
        img = align_channels(img, lines)
        append_stats(lines, "After chromatic aberration correction", img)

    if params.vignette_correct and params.vignette_strength > 0:
        img = radial_vignette_correction(img, params.vignette_strength, lines)
        append_stats(lines, "After vignette correction", img)

    if params.gradient_correct and params.gradient_strength > 0:
        img = linear_gradient_correction(img, params.gradient_strength, lines)
        append_stats(lines, "After gradient correction", img)

    if params.tone_curve:
        img = tone_curve(img)
        append_stats(lines, "After tone curve", img)

    img = smooth_and_subtract(img, params, lines, "initial")
    append_stats(lines, "After initial sky subtraction", img)
    original_subtracted = img.copy()

    if params.stretch_type == "none":
        lines.append("Stretch: none")
    elif params.stretch_type == "root":
        img = root_stretch(img, params, lines)
        append_stats(lines, "After root stretch", img)
    elif params.stretch_type == "asinh":
        img = asinh_stretch(img, params, lines)
        append_stats(lines, "After ASINH stretch", img)
    elif params.stretch_type == "log":
        img = log_stretch(img, params, lines)
        append_stats(lines, "After log stretch", img)
    else:
        raise ValueError(f"Unknown stretch type: {params.stretch_type}")

    if params.s_curve > 0:
        img = s_curve(img, params, lines)
        append_stats(lines, "After S-curve", img)

    if params.setmin:
        img = set_minimum(img, params)
        append_stats(lines, "After set minimum", img)

    if params.color_correction_mode == "ratio":
        img = color_correct(img, original_subtracted, params, lines)
        append_stats(lines, "After color correction", img)
    elif params.color_correction_mode == "hsv":
        img = hsv_color_correct(img, original_subtracted, params, lines)
        append_stats(lines, "After HSV color correction", img)
    elif params.color_correction_mode != "none":
        raise ValueError(f"Unknown color correction mode: {params.color_correction_mode}")

    img = hsv_adjust(img, params, lines)
    if params.hsv_adjust:
        append_stats(lines, "After HSV adjustment", img)

    img = apply_white_balance(img, params, lines)
    if params.white_balance != "none":
        append_stats(lines, "After white balance", img)

    img = richardson_lucy_deconvolution(img, params, lines)
    if params.rl_deconvolve:
        append_stats(lines, "After Richardson-Lucy deconvolution", img)

    img = reduce_star_sizes(img, params, lines)
    if params.star_reduction:
        append_stats(lines, "After star reduction", img)

    if params.setmin:
        img = set_minimum(img, params)
        append_stats(lines, "After final set minimum", img)

    img = np.clip(img, 0.0, 65535.0)
    append_stats(lines, "Output", img)
    return StretchResult(rgb=restore_source_scale(img, source_rgb), log="\n".join(lines))


def linear_display_limits(rgb: np.ndarray, original_dtype: str = "") -> Tuple[float, float]:
    data = np.asarray(rgb, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    max_value = float(np.max(finite))
    dtype_name = original_dtype.lower()
    if "float" in dtype_name and max_value <= 1.00001:
        return 0.0, 1.0
    if "uint8" in dtype_name:
        return 0.0, 255.0
    return 0.0, 65535.0


def rgb_to_display(rgb: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    data = np.asarray(rgb, dtype=np.float32)
    lo, hi = limits
    scaled = np.clip((data - lo) / max(hi - lo, DISPLAY_EPSILON), 0.0, 1.0)
    return np.ascontiguousarray(np.rint(scaled * 255.0).astype(np.uint8))


def array_to_qimage(rgb8: np.ndarray) -> QImage:
    data = np.ascontiguousarray(rgb8)
    height, width, channels = data.shape
    if channels != 3:
        raise ValueError("Expected RGB data")
    image = QImage(data.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()


def save_result_fits(path: Path, rgb: np.ndarray, source: LoadedImage, params: StretchParameters) -> None:
    header = source.header.copy()
    for key in ("BAYERPAT", "XBAYROFF", "YBAYROFF", "BSCALE", "BZERO", "MIPS-FLO", "MIPS-FHI"):
        if key in header:
            del header[key]
    header["HISTORY"] = "RNC color stretch applied by rnc_color_stretch.py"
    header["HISTORY"] = (
        f"sky={params.skylevelfactor} region={params.sky_region_mode} stretch={params.stretch_type}"
    )
    header["HISTORY"] = f"s_curve={params.s_curve} color_correct={params.color_correction_mode}"
    rgb_data = np.asarray(rgb, dtype=np.float32)
    rgb_max = float(np.nanmax(rgb_data))
    if rgb_max > 1.00001:
        rgb_data = rgb_data / 65535.0
    rgb_data = np.clip(rgb_data, 0.0, 1.0).astype(np.float32, copy=False)
    header["BUNIT"] = "normalized"
    header["MIPS-FLO"] = 0.0
    header["MIPS-FHI"] = 1.0
    header["HISTORY"] = "Saved as normalized 32-bit float data in the 0..1 range for Siril."
    data = np.moveaxis(rgb_data, -1, 0)
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


class ImagePane(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.pixmap: Optional[QPixmap] = None
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.pan_start = QPoint()
        self.setMinimumSize(360, 360)
        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        self.canvas = _ImageCanvas(self)
        layout.addWidget(self.canvas, 1)

    def set_image(self, rgb8: np.ndarray, preserve_view: bool = False) -> None:
        self.canvas.set_pixmap(QPixmap.fromImage(array_to_qimage(rgb8)), preserve_view=preserve_view)

    def clear(self) -> None:
        self.canvas.set_pixmap(None)

    def reset_view(self) -> None:
        self.canvas.fit_to_window()
        self.canvas.update()


class _ImageCanvas(QWidget):
    def __init__(self, parent: ImagePane):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.pan_start = QPoint()
        self.setMinimumSize(320, 320)
        self.setMouseTracking(True)

    def set_pixmap(self, pixmap: Optional[QPixmap], preserve_view: bool = False) -> None:
        old_size = self.pixmap.size() if self.pixmap and not self.pixmap.isNull() else None
        old_zoom = self.zoom_level
        old_pan = QPoint(self.pan_offset)
        self.pixmap = pixmap
        can_preserve = (
            preserve_view
            and pixmap is not None
            and not pixmap.isNull()
            and old_size is not None
            and old_size == pixmap.size()
        )
        if can_preserve:
            self.zoom_level = old_zoom
            self.pan_offset = old_pan
        else:
            self.fit_to_window()
        self.update()

    def fit_to_window(self) -> None:
        if not self.pixmap or self.pixmap.isNull():
            self.zoom_level = 1.0
            self.pan_offset = QPoint(0, 0)
            return
        self.zoom_level = min(self.width() / self.pixmap.width(), self.height() / self.pixmap.height(), 1.0)
        self.pan_offset = QPoint(0, 0)

    def _origin(self) -> QPoint:
        if not self.pixmap:
            return QPoint(0, 0)
        width = int(round(self.pixmap.width() * self.zoom_level))
        height = int(round(self.pixmap.height() * self.zoom_level))
        return QPoint((self.width() - width) // 2 + self.pan_offset.x(), (self.height() - height) // 2 + self.pan_offset.y())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 18))
        if not self.pixmap or self.pixmap.isNull():
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image")
            return
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        origin = self._origin()
        width = int(round(self.pixmap.width() * self.zoom_level))
        height = int(round(self.pixmap.height() * self.zoom_level))
        painter.drawPixmap(QRect(origin.x(), origin.y(), width, height), self.pixmap)
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawRect(QRect(origin.x(), origin.y(), width, height))

    def wheelEvent(self, event) -> None:
        if not self.pixmap:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.zoom_level = max(0.05, min(20.0, self.zoom_level * factor))
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.MiddleButton, Qt.LeftButton):
            self.is_panning = True
            self.pan_start = event.pos()

    def mouseMoveEvent(self, event) -> None:
        if self.is_panning:
            delta = event.pos() - self.pan_start
            self.pan_offset += delta
            self.pan_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.RightButton, Qt.MiddleButton, Qt.LeftButton):
            self.is_panning = False

    def resizeEvent(self, event) -> None:
        if self.pixmap and self.zoom_level <= 1.0:
            self.fit_to_window()
        super().resizeEvent(event)


class PreviewWorker(QThread):
    finished = pyqtSignal(bool, object, str)

    def __init__(self, rgb: np.ndarray, params: StretchParameters):
        super().__init__()
        self.rgb = rgb
        self.params = params

    def run(self) -> None:
        try:
            result = apply_rnc_stretch(self.rgb, self.params)
            self.finished.emit(True, result, "")
        except Exception as exc:
            self.finished.emit(False, None, str(exc))


class ApplyWorker(QThread):
    finished = pyqtSignal(bool, object, str)

    def __init__(self, image: LoadedImage, params: StretchParameters):
        super().__init__()
        self.image = image
        self.params = params

    def run(self) -> None:
        try:
            result = apply_rnc_stretch(self.image.rgb, self.params)
            self.finished.emit(True, result, "")
        except Exception as exc:
            self.finished.emit(False, None, str(exc))


class RNCColorStretchGUI(QMainWindow):
    def __init__(self, siril_instance=None):
        super().__init__()
        self.siril = siril_instance
        self.siril_wd: Optional[Path] = None
        self.loaded_image: Optional[LoadedImage] = None
        self.preview_worker: Optional[PreviewWorker] = None
        self.apply_worker: Optional[ApplyWorker] = None
        self.last_run_log = ""
        self.status_messages: list[str] = []
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(450)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)

        self.detect_siril_working_directory()
        self.setWindowTitle("RNC Color Stretch for Siril")
        self.resize(1280, 820)
        self.build_ui()
        self.load_settings()
        if self.siril is not None:
            QTimer.singleShot(100, self.load_current_siril_image)

    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_panel.setMinimumWidth(370)
        left_layout = QVBoxLayout(left_panel)

        source_box = QGroupBox("Source")
        source_layout = QGridLayout(source_box)
        self.source_label = QLabel("No image loaded")
        self.source_label.setWordWrap(True)
        capture_button = QPushButton("Use Current Siril Image")
        capture_button.clicked.connect(self.load_current_siril_image)
        browse_button = QPushButton("Choose Image")
        browse_button.clicked.connect(self.select_image)
        source_layout.addWidget(self.source_label, 0, 0, 1, 2)
        source_layout.addWidget(capture_button, 1, 0)
        source_layout.addWidget(browse_button, 1, 1)
        left_layout.addWidget(source_box)

        params_box = QGroupBox("Stretch Parameters")
        params_layout = QGridLayout(params_box)
        params_layout.setColumnStretch(1, 1)
        row = 0
        self.tone_curve_check = QCheckBox("Tone curve")
        self.tone_curve_check.setToolTip(PARAMETER_TOOLTIPS["tone_curve"])
        params_layout.addWidget(self.tone_curve_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.tone_curve_check.setChecked(DEFAULT_TONE_CURVE), PARAMETER_TOOLTIPS["tone_curve"])
        row += 1

        self.sky_spin = ParameterSlider(0.005, 0.200, DEFAULT_SKY_LEVEL_FACTOR, decimals=3, single_step=0.001, page_step=0.010)
        self.add_parameter_label(params_layout, row, "Sky level factor", PARAMETER_TOOLTIPS["sky"])
        self.sky_spin.setToolTip(PARAMETER_TOOLTIPS["sky"])
        params_layout.addWidget(self.sky_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.sky_spin.setValue(DEFAULT_SKY_LEVEL_FACTOR), PARAMETER_TOOLTIPS["sky"])
        row += 1

        self.sky_region_combo = QComboBox()
        self.sky_region_combo.addItem("Full image", "full")
        self.sky_region_combo.addItem("Auto darkest window", "auto")
        self.sky_region_combo.addItem("Manual window", "manual")
        self.add_parameter_label(params_layout, row, "Sky region", "Region used to compute RGB sky-zero histograms.")
        params_layout.addWidget(self.sky_region_combo, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.sky_region_combo.setCurrentIndex(0), "")
        row += 1
        self.sky_x_spin = self.make_int_spin(0, 200000, DEFAULT_SKY_X)
        self.sky_y_spin = self.make_int_spin(0, 200000, DEFAULT_SKY_Y)
        self.sky_width_spin = self.make_int_spin(1, 200000, DEFAULT_SKY_WIDTH)
        self.sky_height_spin = self.make_int_spin(1, 200000, DEFAULT_SKY_HEIGHT)
        self.sky_step_spin = self.make_int_spin(1, 20, DEFAULT_SKY_STEP_FRACTION)
        for label_text, widget in (
            ("Sky X", self.sky_x_spin),
            ("Sky Y", self.sky_y_spin),
            ("Sky width", self.sky_width_spin),
            ("Sky height", self.sky_height_spin),
            ("Auto step fraction", self.sky_step_spin),
        ):
            self.add_parameter_label(params_layout, row, label_text, "Sky-region coordinate or scan parameter.")
            params_layout.addWidget(widget, row, 1)
            row += 1

        self.zero_r_spin = ParameterSlider(0, 20000, DEFAULT_ZERO_SKY, single_step=64, page_step=512)
        self.zero_g_spin = ParameterSlider(0, 20000, DEFAULT_ZERO_SKY, single_step=64, page_step=512)
        self.zero_b_spin = ParameterSlider(0, 20000, DEFAULT_ZERO_SKY, single_step=64, page_step=512)
        self.zero_rgb_spin = ParameterSlider(0, 20000, DEFAULT_ZERO_SKY, single_step=64, page_step=512)
        self.add_parameter_label(params_layout, row, "Zero sky RGB", PARAMETER_TOOLTIPS["zero_rgb"])
        self.zero_rgb_spin.setToolTip(PARAMETER_TOOLTIPS["zero_rgb"])
        params_layout.addWidget(self.zero_rgb_spin, row, 1)
        self.add_reset_button(params_layout, row, self.reset_zero_rgb, PARAMETER_TOOLTIPS["zero_rgb"])
        row += 1
        self.add_parameter_label(params_layout, row, "Zero sky R", PARAMETER_TOOLTIPS["zero_r"])
        self.zero_r_spin.setToolTip(PARAMETER_TOOLTIPS["zero_r"])
        params_layout.addWidget(self.zero_r_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.zero_r_spin.setValue(DEFAULT_ZERO_SKY), PARAMETER_TOOLTIPS["zero_r"])
        row += 1
        self.add_parameter_label(params_layout, row, "Zero sky G", PARAMETER_TOOLTIPS["zero_g"])
        self.zero_g_spin.setToolTip(PARAMETER_TOOLTIPS["zero_g"])
        params_layout.addWidget(self.zero_g_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.zero_g_spin.setValue(DEFAULT_ZERO_SKY), PARAMETER_TOOLTIPS["zero_g"])
        row += 1
        self.add_parameter_label(params_layout, row, "Zero sky B", PARAMETER_TOOLTIPS["zero_b"])
        self.zero_b_spin.setToolTip(PARAMETER_TOOLTIPS["zero_b"])
        params_layout.addWidget(self.zero_b_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.zero_b_spin.setValue(DEFAULT_ZERO_SKY), PARAMETER_TOOLTIPS["zero_b"])
        row += 1

        self.stretch_type_combo = QComboBox()
        self.stretch_type_combo.addItem("No stretch", "none")
        self.stretch_type_combo.addItem("Root power", "root")
        self.stretch_type_combo.addItem("ASINH", "asinh")
        self.stretch_type_combo.addItem("Log", "log")
        self.stretch_type_combo.setCurrentIndex(1)
        self.add_parameter_label(params_layout, row, "Stretch type", "Stretch algorithm to apply after initial sky-zero.")
        params_layout.addWidget(self.stretch_type_combo, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.stretch_type_combo.setCurrentIndex(1), "")
        row += 1

        self.rootpower_spin = ParameterSlider(1, 600, DEFAULT_ROOTPOWER, single_step=1, page_step=10)
        self.rootpower2_spin = ParameterSlider(1, 600, DEFAULT_ROOTPOWER2, single_step=1, page_step=10)
        self.rootiter_spin = self.make_int_spin(1, 4, DEFAULT_ROOTITER)
        self.add_parameter_label(params_layout, row, "Root power", PARAMETER_TOOLTIPS["rootpower"])
        self.rootpower_spin.setToolTip(PARAMETER_TOOLTIPS["rootpower"])
        params_layout.addWidget(self.rootpower_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.rootpower_spin.setValue(DEFAULT_ROOTPOWER), PARAMETER_TOOLTIPS["rootpower"])
        row += 1
        self.add_parameter_label(params_layout, row, "Root power 2", PARAMETER_TOOLTIPS["rootpower2"])
        self.rootpower2_spin.setToolTip(PARAMETER_TOOLTIPS["rootpower2"])
        params_layout.addWidget(self.rootpower2_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.rootpower2_spin.setValue(DEFAULT_ROOTPOWER2), PARAMETER_TOOLTIPS["rootpower2"])
        row += 1
        self.add_parameter_label(params_layout, row, "Root iterations", PARAMETER_TOOLTIPS["rootiter"])
        self.rootiter_spin.setToolTip(PARAMETER_TOOLTIPS["rootiter"])
        params_layout.addWidget(self.rootiter_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.rootiter_spin.setValue(DEFAULT_ROOTITER), PARAMETER_TOOLTIPS["rootiter"])
        row += 1

        self.asinh_k1_spin = ParameterSlider(1, 1000, DEFAULT_ASINH_K1, single_step=1, page_step=25)
        self.asinh_k2_spin = ParameterSlider(1, 1000, DEFAULT_ASINH_K2, single_step=1, page_step=25)
        self.asinh_iter_spin = self.make_int_spin(1, 4, DEFAULT_ROOTITER)
        self.log_k1_spin = ParameterSlider(1, 500, DEFAULT_LOG_K1, single_step=1, page_step=25)
        self.log_k2_spin = ParameterSlider(1, 500, DEFAULT_LOG_K2, single_step=1, page_step=25)
        self.log_iter_spin = self.make_int_spin(1, 4, DEFAULT_ROOTITER)
        for label_text, widget, default in (
            ("ASINH K1", self.asinh_k1_spin, DEFAULT_ASINH_K1),
            ("ASINH K2", self.asinh_k2_spin, DEFAULT_ASINH_K2),
            ("ASINH iterations", self.asinh_iter_spin, DEFAULT_ROOTITER),
            ("Log K1", self.log_k1_spin, DEFAULT_LOG_K1),
            ("Log K2", self.log_k2_spin, DEFAULT_LOG_K2),
            ("Log iterations", self.log_iter_spin, DEFAULT_ROOTITER),
        ):
            self.add_parameter_label(params_layout, row, label_text, "Parameters for ASINH or logarithmic stretch modes.")
            params_layout.addWidget(widget, row, 1)
            if hasattr(widget, "setValue"):
                self.add_reset_button(params_layout, row, lambda w=widget, v=default: w.setValue(v), "")
            row += 1

        self.scurve_combo = QComboBox()
        self.scurve_combo.addItem("None", 0)
        self.scurve_combo.addItem("S-curve 1", 1)
        self.scurve_combo.addItem("S-curve 2", 2)
        self.scurve_combo.addItem("S-curve 1 then 2", 3)
        self.scurve_combo.addItem("S-curve 2 then 1", 4)
        self.add_parameter_label(params_layout, row, "S-curve", PARAMETER_TOOLTIPS["s_curve"])
        self.scurve_combo.setToolTip(PARAMETER_TOOLTIPS["s_curve"])
        params_layout.addWidget(self.scurve_combo, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.scurve_combo.setCurrentIndex(DEFAULT_S_CURVE_INDEX), PARAMETER_TOOLTIPS["s_curve"])
        row += 1

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItem("None", "none")
        self.color_mode_combo.addItem("Ratio", "ratio")
        self.color_mode_combo.addItem("HSV", "hsv")
        self.color_mode_combo.setCurrentIndex(1)
        self.add_parameter_label(params_layout, row, "Color correction", PARAMETER_TOOLTIPS["color_correction"])
        params_layout.addWidget(self.color_mode_combo, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.color_mode_combo.setCurrentIndex(1), PARAMETER_TOOLTIPS["color_correction"])
        row += 1

        self.color_enhance_spin = ParameterSlider(0.0, 3.0, DEFAULT_COLOR_ENHANCE, decimals=2, single_step=0.05, page_step=0.25)
        self.add_parameter_label(params_layout, row, "Color enhance", PARAMETER_TOOLTIPS["color_enhance"])
        self.color_enhance_spin.setToolTip(PARAMETER_TOOLTIPS["color_enhance"])
        params_layout.addWidget(self.color_enhance_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.color_enhance_spin.setValue(DEFAULT_COLOR_ENHANCE), PARAMETER_TOOLTIPS["color_enhance"])
        row += 1

        self.color_gamma_spin = ParameterSlider(0.1, 10.0, DEFAULT_COLOR_GAMMA, decimals=1, single_step=0.1, page_step=1.0)
        self.add_parameter_label(params_layout, row, "HSV gamma", "Gamma used by HSV color correction.")
        params_layout.addWidget(self.color_gamma_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.color_gamma_spin.setValue(DEFAULT_COLOR_GAMMA), "")
        row += 1

        self.hsv_adjust_check = QCheckBox("HSV post adjust")
        params_layout.addWidget(self.hsv_adjust_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.hsv_adjust_check.setChecked(DEFAULT_HSV_ADJUST), "")
        row += 1
        self.hue_adjust_spin = ParameterSlider(-180, 180, DEFAULT_HUE_ADJUST, decimals=0, single_step=1, page_step=15)
        self.sat_adjust_spin = ParameterSlider(0.0, 2.0, DEFAULT_SAT_ADJUST, decimals=2, single_step=0.05, page_step=0.25)
        self.val_adjust_spin = ParameterSlider(0.1, 2.0, DEFAULT_VAL_ADJUST, decimals=2, single_step=0.05, page_step=0.25)
        self.vib_adjust_spin = ParameterSlider(-1.0, 1.0, DEFAULT_VIB_ADJUST, decimals=2, single_step=0.05, page_step=0.25)
        for label_text, widget, default in (
            ("Hue", self.hue_adjust_spin, DEFAULT_HUE_ADJUST),
            ("Saturation", self.sat_adjust_spin, DEFAULT_SAT_ADJUST),
            ("Value", self.val_adjust_spin, DEFAULT_VAL_ADJUST),
            ("Vibrance", self.vib_adjust_spin, DEFAULT_VIB_ADJUST),
        ):
            self.add_parameter_label(params_layout, row, label_text, "HSV post-adjustment control.")
            params_layout.addWidget(widget, row, 1)
            self.add_reset_button(params_layout, row, lambda w=widget, v=default: w.setValue(v), "")
            row += 1

        self.white_balance_combo = QComboBox()
        self.white_balance_combo.addItem("None", "none")
        self.white_balance_combo.addItem("Gray world", "gray")
        self.white_balance_combo.addItem("Temp/tint", "temp_tint")
        self.add_parameter_label(params_layout, row, "White balance", "Optional white-balance adjustment after color correction.")
        params_layout.addWidget(self.white_balance_combo, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.white_balance_combo.setCurrentIndex(0), "")
        row += 1
        self.temp_spin = ParameterSlider(0.8, 1.2, DEFAULT_TEMP, decimals=2, single_step=0.01, page_step=0.05)
        self.tint_spin = ParameterSlider(0.7, 1.3, DEFAULT_TINT, decimals=2, single_step=0.01, page_step=0.05)
        for label_text, widget, default in (("Temperature", self.temp_spin, DEFAULT_TEMP), ("Tint", self.tint_spin, DEFAULT_TINT)):
            self.add_parameter_label(params_layout, row, label_text, "Temp/tint white-balance factor.")
            params_layout.addWidget(widget, row, 1)
            self.add_reset_button(params_layout, row, lambda w=widget, v=default: w.setValue(v), "")
            row += 1

        self.ca_check = QCheckBox("Chromatic aberration correction")
        params_layout.addWidget(self.ca_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.ca_check.setChecked(DEFAULT_CA_CORRECT), "")
        row += 1
        self.vignette_check = QCheckBox("Radial vignette correction")
        params_layout.addWidget(self.vignette_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.vignette_check.setChecked(DEFAULT_VIGNETTE_CORRECT), "")
        row += 1
        self.vignette_strength_spin = ParameterSlider(0, 100, DEFAULT_VIGNETTE_STRENGTH, single_step=1, page_step=10)
        self.add_parameter_label(params_layout, row, "Vignette strength", "Radial vignette correction strength.")
        params_layout.addWidget(self.vignette_strength_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.vignette_strength_spin.setValue(DEFAULT_VIGNETTE_STRENGTH), "")
        row += 1
        self.gradient_check = QCheckBox("Linear gradient correction")
        params_layout.addWidget(self.gradient_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.gradient_check.setChecked(DEFAULT_GRADIENT_CORRECT), "")
        row += 1
        self.gradient_strength_spin = ParameterSlider(0, 100, DEFAULT_GRADIENT_STRENGTH, single_step=1, page_step=10)
        self.add_parameter_label(params_layout, row, "Gradient strength", "Linear gradient correction strength.")
        params_layout.addWidget(self.gradient_strength_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.gradient_strength_spin.setValue(DEFAULT_GRADIENT_STRENGTH), "")
        row += 1

        self.rl_check = QCheckBox("Richardson-Lucy deconvolution")
        params_layout.addWidget(self.rl_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.rl_check.setChecked(DEFAULT_RL_DECONVOLVE), "")
        row += 1
        self.rl_iterations_spin = self.make_int_spin(1, 40, DEFAULT_RL_ITERATIONS)
        self.rl_sigma_spin = ParameterSlider(0.1, 3.0, DEFAULT_RL_SIGMA, decimals=1, single_step=0.1, page_step=0.5)
        self.add_parameter_label(params_layout, row, "RL iterations", "Richardson-Lucy iteration count.")
        params_layout.addWidget(self.rl_iterations_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.rl_iterations_spin.setValue(DEFAULT_RL_ITERATIONS), "")
        row += 1
        self.add_parameter_label(params_layout, row, "RL sigma", "Gaussian-equivalent PSF sigma in pixels.")
        params_layout.addWidget(self.rl_sigma_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.rl_sigma_spin.setValue(DEFAULT_RL_SIGMA), "")
        row += 1

        self.star_reduction_check = QCheckBox("Star size reduction")
        params_layout.addWidget(self.star_reduction_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.star_reduction_check.setChecked(DEFAULT_STAR_REDUCTION), "")
        row += 1
        self.star_strength_spin = ParameterSlider(0.0, 1.0, DEFAULT_STAR_REDUCTION_STRENGTH, decimals=2, single_step=0.05, page_step=0.1)
        self.add_parameter_label(params_layout, row, "Star reduction", "Star halo reduction strength.")
        params_layout.addWidget(self.star_strength_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.star_strength_spin.setValue(DEFAULT_STAR_REDUCTION_STRENGTH), "")
        row += 1

        self.setmin_check = QCheckBox("Set minimum")
        self.setmin_check.setChecked(DEFAULT_SETMIN)
        self.setmin_check.setToolTip(PARAMETER_TOOLTIPS["setmin"])
        params_layout.addWidget(self.setmin_check, row, 0, 1, 2)
        self.add_reset_button(params_layout, row, lambda: self.setmin_check.setChecked(DEFAULT_SETMIN), PARAMETER_TOOLTIPS["setmin"])
        row += 1
        self.min_r_spin = ParameterSlider(0, 20000, DEFAULT_SETMIN_VALUE, single_step=64, page_step=512)
        self.min_g_spin = ParameterSlider(0, 20000, DEFAULT_SETMIN_VALUE, single_step=64, page_step=512)
        self.min_b_spin = ParameterSlider(0, 20000, DEFAULT_SETMIN_VALUE, single_step=64, page_step=512)
        self.add_parameter_label(params_layout, row, "Minimum R", PARAMETER_TOOLTIPS["min_r"])
        self.min_r_spin.setToolTip(PARAMETER_TOOLTIPS["min_r"])
        params_layout.addWidget(self.min_r_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.min_r_spin.setValue(DEFAULT_SETMIN_VALUE), PARAMETER_TOOLTIPS["min_r"])
        row += 1
        self.add_parameter_label(params_layout, row, "Minimum G", PARAMETER_TOOLTIPS["min_g"])
        self.min_g_spin.setToolTip(PARAMETER_TOOLTIPS["min_g"])
        params_layout.addWidget(self.min_g_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.min_g_spin.setValue(DEFAULT_SETMIN_VALUE), PARAMETER_TOOLTIPS["min_g"])
        row += 1
        self.add_parameter_label(params_layout, row, "Minimum B", PARAMETER_TOOLTIPS["min_b"])
        self.min_b_spin.setToolTip(PARAMETER_TOOLTIPS["min_b"])
        params_layout.addWidget(self.min_b_spin, row, 1)
        self.add_reset_button(params_layout, row, lambda: self.min_b_spin.setValue(DEFAULT_SETMIN_VALUE), PARAMETER_TOOLTIPS["min_b"])
        row += 1

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_box)
        left_layout.addWidget(params_scroll, 3)

        output_box = QGroupBox("Apply")
        output_layout = QVBoxLayout(output_box)
        self.output_label = QLabel("Apply updates the active Siril image in memory. Use Siril Save or Save As afterward.")
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label)
        left_layout.addWidget(output_box)

        button_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.update_preview)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_to_full_image)
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_preview_views)
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_view_button)
        left_layout.addLayout(button_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(110)
        left_layout.addWidget(self.status_text, 1)

        main_layout.addWidget(left_panel)

        tabs = QTabWidget()
        preview_area = QWidget()
        preview_layout = QVBoxLayout(preview_area)
        splitter = QSplitter(Qt.Horizontal)
        self.before_pane = ImagePane("Before")
        self.after_pane = ImagePane("After")
        splitter.addWidget(self.before_pane)
        splitter.addWidget(self.after_pane)
        splitter.setSizes([1, 1])
        preview_layout.addWidget(splitter, 1)
        tabs.addTab(preview_area, "Preview")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        tabs.addTab(self.log_text, "Log")
        self.tabs = tabs
        main_layout.addWidget(tabs, 1)

        widgets = [
            self.tone_curve_check,
            self.sky_region_combo,
            self.sky_x_spin,
            self.sky_y_spin,
            self.sky_width_spin,
            self.sky_height_spin,
            self.sky_step_spin,
            self.sky_spin,
            self.zero_r_spin,
            self.zero_g_spin,
            self.zero_b_spin,
            self.stretch_type_combo,
            self.rootpower_spin,
            self.rootpower2_spin,
            self.rootiter_spin,
            self.asinh_k1_spin,
            self.asinh_k2_spin,
            self.asinh_iter_spin,
            self.log_k1_spin,
            self.log_k2_spin,
            self.log_iter_spin,
            self.scurve_combo,
            self.color_mode_combo,
            self.color_enhance_spin,
            self.color_gamma_spin,
            self.hsv_adjust_check,
            self.hue_adjust_spin,
            self.sat_adjust_spin,
            self.val_adjust_spin,
            self.vib_adjust_spin,
            self.white_balance_combo,
            self.temp_spin,
            self.tint_spin,
            self.ca_check,
            self.vignette_check,
            self.vignette_strength_spin,
            self.gradient_check,
            self.gradient_strength_spin,
            self.rl_check,
            self.rl_iterations_spin,
            self.rl_sigma_spin,
            self.star_reduction_check,
            self.star_strength_spin,
            self.setmin_check,
            self.min_r_spin,
            self.min_g_spin,
            self.min_b_spin,
        ]
        for widget in widgets:
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.schedule_preview)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.schedule_preview)
            else:
                widget.valueChanged.connect(self.schedule_preview)
        self.zero_rgb_spin.valueChanged.connect(self.apply_zero_rgb_master)

    def make_int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(128 if maximum > 1000 else 1)
        spin.setValue(value)
        return spin

    def add_parameter_label(self, layout: QGridLayout, row: int, text: str, tooltip: str) -> None:
        label = QLabel(text)
        label.setToolTip(tooltip)
        layout.addWidget(label, row, 0)

    def add_reset_button(self, layout: QGridLayout, row: int, reset_func, tooltip: str = "") -> None:
        button = QPushButton("Reset")
        button.setFixedWidth(58)
        reset_tip = "Reset only this parameter to its default value."
        button.setToolTip(f"{reset_tip}\n\n{tooltip}" if tooltip else reset_tip)
        button.clicked.connect(lambda _checked=False: reset_func())
        layout.addWidget(button, row, 2)

    def reset_preview_views(self) -> None:
        self.before_pane.reset_view()
        self.after_pane.reset_view()

    def apply_zero_rgb_master(self) -> None:
        value = int(self.zero_rgb_spin.value())
        for slider in (self.zero_r_spin, self.zero_g_spin, self.zero_b_spin):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        self.schedule_preview()

    def reset_zero_rgb(self) -> None:
        self.zero_rgb_spin.setValue(DEFAULT_ZERO_SKY)

    def current_params(self) -> StretchParameters:
        return StretchParameters(
            tone_curve=self.tone_curve_check.isChecked(),
            sky_region_mode=str(self.sky_region_combo.currentData()),
            sky_x=int(self.sky_x_spin.value()),
            sky_y=int(self.sky_y_spin.value()),
            sky_width=int(self.sky_width_spin.value()),
            sky_height=int(self.sky_height_spin.value()),
            sky_step_fraction=int(self.sky_step_spin.value()),
            skylevelfactor=float(self.sky_spin.value()),
            zerosky_r=int(self.zero_r_spin.value()),
            zerosky_g=int(self.zero_g_spin.value()),
            zerosky_b=int(self.zero_b_spin.value()),
            stretch_type=str(self.stretch_type_combo.currentData()),
            rootpower=int(self.rootpower_spin.value()),
            rootpower2=int(self.rootpower2_spin.value()),
            rootiter=int(self.rootiter_spin.value()),
            asinh_k1=int(self.asinh_k1_spin.value()),
            asinh_k2=int(self.asinh_k2_spin.value()),
            asinh_iter=int(self.asinh_iter_spin.value()),
            log_k1=int(self.log_k1_spin.value()),
            log_k2=int(self.log_k2_spin.value()),
            log_iter=int(self.log_iter_spin.value()),
            s_curve=int(self.scurve_combo.currentData()),
            setmin=self.setmin_check.isChecked(),
            setmin_r=int(self.min_r_spin.value()),
            setmin_g=int(self.min_g_spin.value()),
            setmin_b=int(self.min_b_spin.value()),
            color_correction_mode=str(self.color_mode_combo.currentData()),
            colorenhance=float(self.color_enhance_spin.value()),
            color_gamma=float(self.color_gamma_spin.value()),
            hsv_adjust=self.hsv_adjust_check.isChecked(),
            hue_adjust=float(self.hue_adjust_spin.value()),
            sat_adjust=float(self.sat_adjust_spin.value()),
            val_adjust=float(self.val_adjust_spin.value()),
            vib_adjust=float(self.vib_adjust_spin.value()),
            white_balance=str(self.white_balance_combo.currentData()),
            temp=float(self.temp_spin.value()),
            tint=float(self.tint_spin.value()),
            ca_correct=self.ca_check.isChecked(),
            vignette_correct=self.vignette_check.isChecked(),
            vignette_strength=int(self.vignette_strength_spin.value()),
            gradient_correct=self.gradient_check.isChecked(),
            gradient_strength=int(self.gradient_strength_spin.value()),
            star_reduction=self.star_reduction_check.isChecked(),
            star_reduction_strength=float(self.star_strength_spin.value()),
            rl_deconvolve=self.rl_check.isChecked(),
            rl_iterations=int(self.rl_iterations_spin.value()),
            rl_sigma=float(self.rl_sigma_spin.value()),
        )

    def detect_siril_working_directory(self) -> None:
        if not self.siril:
            return
        try:
            wd = self.siril.get_siril_wd()
            if wd:
                path = Path(str(wd))
                if path.exists():
                    self.siril_wd = path
        except Exception:
            self.siril_wd = None

    def load_current_siril_image(self) -> None:
        if not self.siril:
            QMessageBox.warning(self, "Siril Not Connected", "Run this script from Siril or choose an image file.")
            return
        try:
            fit = self.siril.get_image(with_pixels=True, preview=False)
            filename = self.siril.get_image_filename() if hasattr(self.siril, "get_image_filename") else None
            image = loaded_image_from_siril_fit(fit, filename)
            self.set_loaded_image(image)
            self.log("Loaded active Siril image without saving a copy.")
        except Exception as exc:
            QMessageBox.warning(self, "Load Failed", f"Could not read the active Siril image:\n{exc}")
            self.log(f"Loading active Siril image failed: {exc}")

    def select_image(self) -> None:
        start_dir = self.siril_wd or (self.loaded_image.path.parent if self.loaded_image else Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose image",
            str(start_dir),
            "Images (*.fit *.fits *.fts *.tif *.tiff);;All files (*)",
        )
        if file_path:
            try:
                self.load_image(Path(file_path), source_is_current=False)
            except Exception as exc:
                QMessageBox.critical(self, "Load Failed", str(exc))

    def load_image(self, path: Path, source_is_current: bool) -> None:
        image = read_image_file(path, source_is_current=source_is_current)
        self.set_loaded_image(image)
        self.log(f"Loaded {path}")

    def set_loaded_image(self, image: LoadedImage) -> None:
        self.loaded_image = image
        height, width = image.rgb.shape[:2]
        origin = "current Siril image" if image.source_is_current else "file"
        color_note = "color" if image.is_color else "grayscale expanded to RGB"
        self.source_label.setText(
            f"{image.path.name}\n{width} x {height}, {color_note}, source: {origin}, preview scale {image.preview_scale}x"
        )
        self.before_pane.set_image(rgb_to_display(image.preview_input, image.display_limits))
        self.after_pane.clear()
        self.update_output_label()
        self.log(f"Preview display is linear, range {image.display_limits[0]:.0f} to {image.display_limits[1]:.0f}")
        self.save_settings()
        self.schedule_preview()

    def schedule_preview(self) -> None:
        if self.loaded_image is not None:
            self.preview_timer.start()

    def update_preview(self) -> None:
        if self.loaded_image is None:
            QMessageBox.warning(self, "Missing Image", "Use the current Siril image or choose an image first.")
            return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            return
        self.set_busy(True, "Previewing...")
        self.preview_worker = PreviewWorker(self.loaded_image.preview_input, self.current_params())
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.start()

    def on_preview_finished(self, success: bool, result: object, message: str) -> None:
        self.set_busy(False, "")
        if not success:
            self.log(f"Preview failed: {message}")
            return
        stretch = result
        assert isinstance(stretch, StretchResult)
        limits = self.loaded_image.display_limits if self.loaded_image is not None else (0.0, 65535.0)
        self.after_pane.set_image(rgb_to_display(stretch.rgb, limits), preserve_view=True)
        self.set_run_log(stretch.log)
        self.save_settings()

    def default_output_path(self) -> Path:
        assert self.loaded_image is not None
        return self.loaded_image.path.with_name(f"{self.loaded_image.path.stem}_rnc_stretched.fit")

    def update_output_label(self) -> None:
        if self.loaded_image is None:
            return
        if self.loaded_image.source_is_current and self.siril is not None:
            self.output_label.setText("Apply updates the active Siril image in memory. Use Siril Save or Save As afterward.")
        else:
            self.output_label.setText(f"Local mode output file: {self.default_output_path()}")

    def apply_to_full_image(self) -> None:
        if self.siril is not None and (self.loaded_image is None or self.loaded_image.source_is_current):
            try:
                fit = self.siril.get_image(with_pixels=True, preview=False)
                filename = self.siril.get_image_filename() if hasattr(self.siril, "get_image_filename") else None
                self.loaded_image = loaded_image_from_siril_fit(fit, filename)
            except Exception as exc:
                QMessageBox.warning(self, "Load Failed", f"Could not read the active Siril image:\n{exc}")
                self.log(f"Loading active Siril image failed: {exc}")
                return
        if self.loaded_image is None:
            QMessageBox.warning(self, "Missing Image", "Use the current Siril image or choose an image first.")
            return
        if self.apply_worker is not None and self.apply_worker.isRunning():
            return
        self.update_output_label()
        self.set_busy(True, "Applying...")
        self.apply_worker = ApplyWorker(self.loaded_image, self.current_params())
        self.apply_worker.finished.connect(self.on_apply_finished)
        self.apply_worker.start()

    def on_apply_finished(self, success: bool, result: object, message: str) -> None:
        self.set_busy(False, "")
        if not success:
            QMessageBox.critical(self, "Apply Failed", message)
            self.log(f"Apply failed: {message}")
            return
        stretch = result
        assert isinstance(stretch, StretchResult)
        if self.loaded_image is not None and self.loaded_image.source_is_current and self.siril is not None:
            try:
                siril_data = result_rgb_to_siril_data(stretch.rgb, self.loaded_image)
                with self.siril.image_lock():
                    if hasattr(self.siril, "undo_save_state"):
                        self.siril.undo_save_state("RNC color stretch")
                    self.siril.set_image_pixeldata(siril_data)
                self.loaded_image.rgb = stretch.rgb.astype(np.float32, copy=False)
                self.loaded_image.preview_input = self.loaded_image.rgb[
                    :: self.loaded_image.preview_scale, :: self.loaded_image.preview_scale, :
                ].astype(np.float32, copy=True)
                self.before_pane.set_image(rgb_to_display(self.loaded_image.preview_input, self.loaded_image.display_limits))
                self.after_pane.clear()
                self.set_run_log(f"Applied to active Siril image.\n\n{stretch.log}")
                self.log("Applied stretch to active Siril image. Save or Save As from Siril to keep it.")
                self.tabs.setCurrentIndex(1)
                QMessageBox.information(
                    self,
                    "RNC Stretch Applied",
                    "Applied to the active Siril image.\nUse Siril Save or Save As to keep the result.",
                )
                return
            except Exception as exc:
                QMessageBox.critical(self, "Apply Failed", f"Could not update the active Siril image:\n{exc}")
                self.log(f"Updating active Siril image failed: {exc}")
                return

        if self.loaded_image is None:
            return
        output_path = self.default_output_path()
        try:
            save_result_fits(output_path, stretch.rgb, self.loaded_image, self.current_params())
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))
            self.log(f"Save failed: {exc}")
            return
        self.set_run_log(f"Output file: {output_path}\n\n{stretch.log}")
        self.log(f"Wrote {output_path}")
        self.tabs.setCurrentIndex(1)
        QMessageBox.information(self, "RNC Stretch Complete", f"Wrote:\n{output_path}")

    def set_busy(self, busy: bool, text: str) -> None:
        self.progress.setRange(0, 0 if busy else 1)
        self.progress.setValue(0 if busy else 1)
        self.preview_button.setEnabled(not busy)
        self.apply_button.setEnabled(not busy)
        self.reset_view_button.setEnabled(not busy)
        if text:
            self.log(text)

    def log(self, message: str) -> None:
        self.status_messages.append(message)
        self.status_messages = self.status_messages[-80:]
        self.status_text.setPlainText("\n".join(self.status_messages))
        self.status_text.moveCursor(QTextCursor.End)

    def set_run_log(self, text: str) -> None:
        self.last_run_log = text
        self.log_text.setPlainText(text)
        self.log_text.moveCursor(QTextCursor.Start)

    def set_combo_to_data(self, combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def save_settings(self) -> None:
        try:
            settings = {
                "tone_curve": self.tone_curve_check.isChecked(),
                "sky_region": self.sky_region_combo.currentData(),
                "sky_x": self.sky_x_spin.value(),
                "sky_y": self.sky_y_spin.value(),
                "sky_width": self.sky_width_spin.value(),
                "sky_height": self.sky_height_spin.value(),
                "sky_step": self.sky_step_spin.value(),
                "sky": self.sky_spin.value(),
                "zero_r": self.zero_r_spin.value(),
                "zero_g": self.zero_g_spin.value(),
                "zero_b": self.zero_b_spin.value(),
                "stretch_type": self.stretch_type_combo.currentData(),
                "rootpower": self.rootpower_spin.value(),
                "rootpower2": self.rootpower2_spin.value(),
                "rootiter": self.rootiter_spin.value(),
                "asinh_k1": self.asinh_k1_spin.value(),
                "asinh_k2": self.asinh_k2_spin.value(),
                "asinh_iter": self.asinh_iter_spin.value(),
                "log_k1": self.log_k1_spin.value(),
                "log_k2": self.log_k2_spin.value(),
                "log_iter": self.log_iter_spin.value(),
                "s_curve": self.scurve_combo.currentIndex(),
                "color_mode": self.color_mode_combo.currentData(),
                "colorenhance": self.color_enhance_spin.value(),
                "color_gamma": self.color_gamma_spin.value(),
                "hsv_adjust": self.hsv_adjust_check.isChecked(),
                "hue_adjust": self.hue_adjust_spin.value(),
                "sat_adjust": self.sat_adjust_spin.value(),
                "val_adjust": self.val_adjust_spin.value(),
                "vib_adjust": self.vib_adjust_spin.value(),
                "white_balance": self.white_balance_combo.currentData(),
                "temp": self.temp_spin.value(),
                "tint": self.tint_spin.value(),
                "ca": self.ca_check.isChecked(),
                "vignette": self.vignette_check.isChecked(),
                "vignette_strength": self.vignette_strength_spin.value(),
                "gradient": self.gradient_check.isChecked(),
                "gradient_strength": self.gradient_strength_spin.value(),
                "rl": self.rl_check.isChecked(),
                "rl_iterations": self.rl_iterations_spin.value(),
                "rl_sigma": self.rl_sigma_spin.value(),
                "star_reduction": self.star_reduction_check.isChecked(),
                "star_strength": self.star_strength_spin.value(),
                "setmin": self.setmin_check.isChecked(),
                "min_r": self.min_r_spin.value(),
                "min_g": self.min_g_spin.value(),
                "min_b": self.min_b_spin.value(),
            }
            CONFIG_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except OSError:
            pass

    def load_settings(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            settings = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.tone_curve_check.setChecked(bool(settings.get("tone_curve", DEFAULT_TONE_CURVE)))
            self.set_combo_to_data(self.sky_region_combo, settings.get("sky_region", DEFAULT_SKY_REGION_MODE))
            self.sky_x_spin.setValue(int(settings.get("sky_x", DEFAULT_SKY_X)))
            self.sky_y_spin.setValue(int(settings.get("sky_y", DEFAULT_SKY_Y)))
            self.sky_width_spin.setValue(int(settings.get("sky_width", DEFAULT_SKY_WIDTH)))
            self.sky_height_spin.setValue(int(settings.get("sky_height", DEFAULT_SKY_HEIGHT)))
            self.sky_step_spin.setValue(int(settings.get("sky_step", DEFAULT_SKY_STEP_FRACTION)))
            self.sky_spin.setValue(float(settings.get("sky", DEFAULT_SKY_LEVEL_FACTOR)))
            self.zero_r_spin.setValue(int(settings.get("zero_r", DEFAULT_ZERO_SKY)))
            self.zero_g_spin.setValue(int(settings.get("zero_g", DEFAULT_ZERO_SKY)))
            self.zero_b_spin.setValue(int(settings.get("zero_b", DEFAULT_ZERO_SKY)))
            self.set_combo_to_data(self.stretch_type_combo, settings.get("stretch_type", DEFAULT_STRETCH_TYPE))
            self.rootpower_spin.setValue(int(settings.get("rootpower", DEFAULT_ROOTPOWER)))
            self.rootpower2_spin.setValue(int(settings.get("rootpower2", DEFAULT_ROOTPOWER2)))
            self.rootiter_spin.setValue(int(settings.get("rootiter", DEFAULT_ROOTITER)))
            self.asinh_k1_spin.setValue(int(settings.get("asinh_k1", DEFAULT_ASINH_K1)))
            self.asinh_k2_spin.setValue(int(settings.get("asinh_k2", DEFAULT_ASINH_K2)))
            self.asinh_iter_spin.setValue(int(settings.get("asinh_iter", DEFAULT_ROOTITER)))
            self.log_k1_spin.setValue(int(settings.get("log_k1", DEFAULT_LOG_K1)))
            self.log_k2_spin.setValue(int(settings.get("log_k2", DEFAULT_LOG_K2)))
            self.log_iter_spin.setValue(int(settings.get("log_iter", DEFAULT_ROOTITER)))
            self.scurve_combo.setCurrentIndex(int(settings.get("s_curve", DEFAULT_S_CURVE_INDEX)))
            if "color_mode" in settings:
                self.set_combo_to_data(self.color_mode_combo, settings.get("color_mode", DEFAULT_COLOR_CORRECTION_MODE))
            else:
                self.set_combo_to_data(self.color_mode_combo, "ratio" if settings.get("color", True) else "none")
            self.color_enhance_spin.setValue(float(settings.get("colorenhance", DEFAULT_COLOR_ENHANCE)))
            self.color_gamma_spin.setValue(float(settings.get("color_gamma", DEFAULT_COLOR_GAMMA)))
            self.hsv_adjust_check.setChecked(bool(settings.get("hsv_adjust", DEFAULT_HSV_ADJUST)))
            self.hue_adjust_spin.setValue(float(settings.get("hue_adjust", DEFAULT_HUE_ADJUST)))
            self.sat_adjust_spin.setValue(float(settings.get("sat_adjust", DEFAULT_SAT_ADJUST)))
            self.val_adjust_spin.setValue(float(settings.get("val_adjust", DEFAULT_VAL_ADJUST)))
            self.vib_adjust_spin.setValue(float(settings.get("vib_adjust", DEFAULT_VIB_ADJUST)))
            self.set_combo_to_data(self.white_balance_combo, settings.get("white_balance", DEFAULT_WHITE_BALANCE))
            self.temp_spin.setValue(float(settings.get("temp", DEFAULT_TEMP)))
            self.tint_spin.setValue(float(settings.get("tint", DEFAULT_TINT)))
            self.ca_check.setChecked(bool(settings.get("ca", DEFAULT_CA_CORRECT)))
            self.vignette_check.setChecked(bool(settings.get("vignette", DEFAULT_VIGNETTE_CORRECT)))
            self.vignette_strength_spin.setValue(int(settings.get("vignette_strength", DEFAULT_VIGNETTE_STRENGTH)))
            self.gradient_check.setChecked(bool(settings.get("gradient", DEFAULT_GRADIENT_CORRECT)))
            self.gradient_strength_spin.setValue(int(settings.get("gradient_strength", DEFAULT_GRADIENT_STRENGTH)))
            self.rl_check.setChecked(bool(settings.get("rl", DEFAULT_RL_DECONVOLVE)))
            self.rl_iterations_spin.setValue(int(settings.get("rl_iterations", DEFAULT_RL_ITERATIONS)))
            self.rl_sigma_spin.setValue(float(settings.get("rl_sigma", DEFAULT_RL_SIGMA)))
            self.star_reduction_check.setChecked(bool(settings.get("star_reduction", DEFAULT_STAR_REDUCTION)))
            self.star_strength_spin.setValue(float(settings.get("star_strength", DEFAULT_STAR_REDUCTION_STRENGTH)))
            self.setmin_check.setChecked(bool(settings.get("setmin", DEFAULT_SETMIN)))
            self.min_r_spin.setValue(int(settings.get("min_r", DEFAULT_SETMIN_VALUE)))
            self.min_g_spin.setValue(int(settings.get("min_g", DEFAULT_SETMIN_VALUE)))
            self.min_b_spin.setValue(int(settings.get("min_b", DEFAULT_SETMIN_VALUE)))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    def closeEvent(self, event) -> None:
        self.save_settings()
        event.accept()


def main() -> int:
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
    window = RNCColorStretchGUI(siril_instance=siril)
    window.show()
    return 0 if app.exec_() == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
