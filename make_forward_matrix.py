#!/usr/bin/env python3
"""
Forward Matrix Builder for Siril.

Load a FITS or TIFF image containing a 24-patch ColorChecker, crop the chart,
sample the 24 patches, derive a white-balance vector from a neutral patch, and
solve for a forward matrix that maps white-balanced camera RGB values to XYZ D50.

The app reports:
- White balance vector
- Forward matrix (white-balanced RGB -> XYZ D50)
- Combined matrix (raw RGB -> XYZ D50) where Combined = Forward * diag(WB)

Requirements:
- PyQt5
- numpy
- astropy
- pillow
- scipy
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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
            "scipy",
            version_constraints=[None, ">=1.20.0", ">=4.0", None, None],
        )
    except Exception as exc:
        raise RuntimeError(f"Error ensuring dependencies: {exc}") from exc

from astropy.io import fits
from PIL import Image
from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage, optimize


SUPPORTED_FITS_SUFFIXES = {".fit", ".fits", ".fts"}
SUPPORTED_TIFF_SUFFIXES = {".tif", ".tiff"}
PREVIEW_MAX_DIMENSION = 1800
PATCH_INSET_FRACTION = 0.18
DEFAULT_PATCH_SAMPLE_FRACTION = 1.0 - (2.0 * PATCH_INSET_FRACTION)
EPSILON = 1e-8
D50_WHITEPOINT = np.array([0.96422, 1.0, 0.82521], dtype=np.float64)
D65_WHITEPOINT = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

BRADFORD_MATRIX = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ],
    dtype=np.float64,
)
BRADFORD_MATRIX_INV = np.linalg.inv(BRADFORD_MATRIX)
XYZ_D65_TO_LINEAR_SRGB_MATRIX = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)

PATCH_NAMES = [
    "Dark Skin",
    "Light Skin",
    "Blue Sky",
    "Foliage",
    "Blue Flower",
    "Bluish Green",
    "Orange",
    "Purplish Blue",
    "Moderate Red",
    "Purple",
    "Yellow Green",
    "Orange Yellow",
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "Magenta",
    "Cyan",
    "White 9.5",
    "Neutral 8",
    "Neutral 6.5",
    "Neutral 5",
    "Neutral 3.5",
    "Black 2",
]

REFERENCE_LAB: Dict[str, np.ndarray] = {
    "After Nov 2014": np.array(
        [
            [37.54, 14.37, 14.92],
            [64.66, 19.27, 17.50],
            [49.32, -3.82, -22.54],
            [43.46, -12.74, 22.72],
            [54.94, 9.61, -24.79],
            [70.48, -32.26, -0.37],
            [62.73, 35.83, 56.50],
            [39.43, 10.75, -45.17],
            [50.57, 48.64, 16.67],
            [30.10, 22.54, -20.87],
            [71.77, -24.13, 58.19],
            [71.51, 18.24, 67.37],
            [28.37, 15.42, -49.80],
            [54.38, -39.72, 32.27],
            [42.43, 51.05, 28.62],
            [81.80, 2.67, 80.41],
            [50.63, 51.28, -14.12],
            [49.57, -29.71, -28.32],
            [95.19, -1.03, 2.93],
            [81.29, -0.57, 0.44],
            [66.89, -0.75, -0.06],
            [50.76, -0.13, 0.14],
            [35.63, -0.46, -0.48],
            [20.64, 0.07, -0.46],
        ],
        dtype=np.float64,
    ),
    "Before Nov 2014": np.array(
        [
            [37.986, 13.555, 14.059],
            [65.711, 18.130, 17.810],
            [49.927, -4.880, -21.905],
            [43.139, -13.095, 21.905],
            [55.112, 8.844, -25.399],
            [70.719, -33.397, -0.199],
            [62.661, 36.067, 57.096],
            [40.020, 10.410, -45.964],
            [51.124, 48.239, 16.248],
            [30.325, 22.976, -21.587],
            [72.532, -23.709, 57.255],
            [71.941, 19.363, 67.857],
            [28.778, 14.179, -50.297],
            [55.261, -38.342, 31.370],
            [42.101, 53.378, 28.190],
            [81.733, 4.039, 79.819],
            [51.935, 49.986, -14.574],
            [51.038, -28.631, -28.638],
            [96.539, -0.425, 1.186],
            [81.257, -0.638, -0.335],
            [66.766, -0.734, -0.504],
            [50.867, -0.153, -0.270],
            [35.656, -0.421, -1.231],
            [20.461, -0.079, -0.973],
        ],
        dtype=np.float64,
    ),
}

WB_PATCH_LABELS = ["Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "White 9.5"]
WB_PATCH_INDEX = {
    "White 9.5": 18,
    "Neutral 8": 19,
    "Neutral 6.5": 20,
    "Neutral 5": 21,
    "Neutral 3.5": 22,
}
WB_REFERENCE_CHANNELS = {"R": 0, "G": 1}

ORIENTATION_OPTIONS = [
    ("Upright", 0),
    ("Rotated 90° CW", 1),
    ("Rotated 180°", 2),
    ("Rotated 270° CW", 3),
]

EDGE_LEFT = 1
EDGE_RIGHT = 2
EDGE_TOP = 4
EDGE_BOTTOM = 8


@dataclass
class LoadedImage:
    path: Path
    rgb: np.ndarray
    preview_rgb: np.ndarray
    preview_scale: int
    is_color: bool


@dataclass
class CalibrationResult:
    wb_vector: np.ndarray
    forward_matrix: np.ndarray
    combined_matrix: np.ndarray
    siril_matrix_after_wb: np.ndarray
    siril_matrix_one_step: np.ndarray
    siril_output_scale: float
    mean_delta_e: float
    max_delta_e: float
    optimization_note: Optional[str]


def canonicalize_fits_array(data: np.ndarray) -> np.ndarray:
    """Normalize FITS data into either 2D or channel-first 3D form."""
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
    """Return RGB channel offsets for a Bayer CFA pattern."""
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
    """Reconstruct one Bayer plane with normalized bilinear interpolation."""
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
    return (weighted_values / np.clip(weighted_mask, EPSILON, None)).astype(np.float32, copy=False)


def debayer_image(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    """Debayer a 2D mosaic into channel-first RGB."""
    red_offset, green_offsets, blue_offset = bayer_channel_offsets(pattern)
    red = bilinear_reconstruct(mosaic, (red_offset,))
    green = bilinear_reconstruct(mosaic, green_offsets)
    blue = bilinear_reconstruct(mosaic, (blue_offset,))
    return np.stack([red, green, blue], axis=0)


def channel_first_to_hwc(data: np.ndarray) -> np.ndarray:
    """Convert channel-first data to HWC RGB."""
    if data.ndim == 2:
        return np.repeat(data[:, :, np.newaxis], 3, axis=2).astype(np.float32, copy=False)
    if data.ndim != 3:
        raise ValueError(f"Unsupported array shape: {data.shape}")
    if data.shape[0] < 3:
        raise ValueError("Need at least three channels for color sampling")
    return np.moveaxis(data[:3], 0, -1).astype(np.float32, copy=False)


def read_fits_rgb(path: Path) -> Tuple[np.ndarray, bool]:
    """Read FITS data and return HWC RGB plus whether the source is genuinely color."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if getattr(hdu, "data", None) is None:
                continue
            data = canonicalize_fits_array(hdu.data)
            bayer_pattern = hdu.header.get("BAYERPAT")
            if data.ndim == 2 and bayer_pattern:
                rgb = channel_first_to_hwc(debayer_image(data, str(bayer_pattern)))
                return rgb, True
            if data.ndim == 2:
                return channel_first_to_hwc(data), False
            return channel_first_to_hwc(data), True
    raise ValueError(f"No image data found in {path}")


def read_tiff_rgb(path: Path) -> Tuple[np.ndarray, bool]:
    """Read TIFF data and return HWC RGB plus whether the source is genuinely color."""
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image)

    if array.ndim == 2:
        rgb = np.repeat(array[:, :, np.newaxis], 3, axis=2)
        return np.asarray(rgb, dtype=np.float32), False

    if array.ndim != 3:
        raise ValueError(f"Unsupported TIFF dimensions: {array.shape}")

    if array.shape[2] == 1:
        rgb = np.repeat(array, 3, axis=2)
        return np.asarray(rgb, dtype=np.float32), False

    if array.shape[2] < 3:
        raise ValueError(f"Unsupported TIFF channel count: {array.shape[2]}")

    return np.asarray(array[:, :, :3], dtype=np.float32), True


def read_image_file(path: Path) -> LoadedImage:
    """Load a FITS or TIFF image and build a downsampled display preview."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_FITS_SUFFIXES:
        rgb, is_color = read_fits_rgb(path)
    elif suffix in SUPPORTED_TIFF_SUFFIXES:
        rgb, is_color = read_tiff_rgb(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    preview_scale = choose_preview_downsample(rgb.shape[:2])
    preview_rgb = build_preview_rgb(rgb[::preview_scale, ::preview_scale, :])
    return LoadedImage(
        path=path,
        rgb=rgb.astype(np.float32, copy=False),
        preview_rgb=preview_rgb,
        preview_scale=preview_scale,
        is_color=is_color,
    )


def choose_preview_downsample(shape: Sequence[int]) -> int:
    """Choose an integer downsample factor for interactive preview display."""
    height, width = int(shape[0]), int(shape[1])
    largest = max(height, width)
    return max(1, int(np.ceil(largest / PREVIEW_MAX_DIMENSION)))


def robust_mad_sigma(values: np.ndarray) -> float:
    """Estimate a robust standard deviation from the median absolute deviation."""
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Preview image contains no finite values")
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    return max(1.4826 * mad, 1e-6)


def build_preview_rgb(rgb: np.ndarray) -> np.ndarray:
    """Stretch linear RGB data into a display-friendly uint8 preview."""
    data = np.asarray(rgb, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] < 3:
        raise ValueError("Preview data must be RGB")

    channels: List[np.ndarray] = []
    for index in range(3):
        channel = data[:, :, index]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            channels.append(np.zeros(channel.shape, dtype=np.uint8))
            continue
        median = float(np.median(finite))
        sigma = robust_mad_sigma(finite)
        low = float(np.percentile(finite, 0.2))
        high = min(float(np.percentile(finite, 99.8)), median + 12.0 * sigma)
        if not np.isfinite(high) or high <= low:
            high = low + 1.0
        stretched = np.clip((np.nan_to_num(channel, nan=median) - low) / (high - low), 0.0, 1.0)
        gamma = 0.55
        channels.append((np.power(stretched, gamma) * 255.0).astype(np.uint8))

    return np.stack(channels, axis=2)


def array_to_qimage(data: np.ndarray) -> QImage:
    """Convert a uint8 RGB image into a QImage."""
    contiguous = np.ascontiguousarray(data)
    image = QImage(
        contiguous.data,
        contiguous.shape[1],
        contiguous.shape[0],
        contiguous.shape[1] * 3,
        QImage.Format_RGB888,
    )
    return image.copy()


def crop_array(data: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop an HWC image with a full-resolution (x, y, w, h) rectangle."""
    x, y, width, height = rect
    if width <= 1 or height <= 1:
        raise ValueError("Crop region is too small")

    max_height, max_width = data.shape[:2]
    x0 = int(np.clip(x, 0, max_width - 1))
    y0 = int(np.clip(y, 0, max_height - 1))
    x1 = int(np.clip(x + width, x0 + 1, max_width))
    y1 = int(np.clip(y + height, y0 + 1, max_height))
    return np.asarray(data[y0:y1, x0:x1, :], dtype=np.float32)


def rotate_crop_to_upright(crop_rgb: np.ndarray, orientation_value: int) -> np.ndarray:
    """Rotate the selected chart crop into the canonical 6x4 upright layout."""
    if orientation_value == 0:
        return crop_rgb
    if orientation_value == 1:
        return np.rot90(crop_rgb, k=1).copy()
    if orientation_value == 2:
        return np.rot90(crop_rgb, k=2).copy()
    if orientation_value == 3:
        return np.rot90(crop_rgb, k=3).copy()
    raise ValueError(f"Unsupported orientation value: {orientation_value}")


def clamp_patch_sample_fraction(sample_fraction: float) -> float:
    """Clamp the patch sampling size fraction to a practical range."""
    return float(np.clip(sample_fraction, 0.10, 0.95))


def patch_inset_fraction(sample_fraction: float) -> float:
    """Convert the kept patch fraction into an inset fraction per side."""
    return 0.5 * (1.0 - clamp_patch_sample_fraction(sample_fraction))


def rotate_grid_position(col: int, row: int, cols: int, rows: int, orientation_value: int) -> Tuple[int, int]:
    """Rotate a grid cell position into the displayed orientation."""
    if orientation_value == 0:
        return col, row
    if orientation_value == 1:
        return rows - 1 - row, col
    if orientation_value == 2:
        return cols - 1 - col, rows - 1 - row
    if orientation_value == 3:
        return row, cols - 1 - col
    raise ValueError(f"Unsupported orientation value: {orientation_value}")


def sample_patch_median(patch_rgb: np.ndarray) -> np.ndarray:
    """Return a robust RGB triplet from a patch crop."""
    if patch_rgb.size == 0:
        raise ValueError("Patch sample is empty")
    flat = patch_rgb.reshape(-1, patch_rgb.shape[2])
    finite = np.all(np.isfinite(flat), axis=1)
    flat = flat[finite]
    if flat.shape[0] == 0:
        raise ValueError("Patch sample contains no finite pixels")
    return np.median(flat[:, :3], axis=0).astype(np.float64)


def sample_colorchecker_patches(
    crop_rgb: np.ndarray,
    orientation_value: int,
    patch_sample_fraction: float,
) -> np.ndarray:
    """Split the chart crop into a 6x4 patch grid and sample each patch."""
    upright = rotate_crop_to_upright(crop_rgb, orientation_value)
    height, width = upright.shape[:2]
    if height < 40 or width < 60:
        raise ValueError("Crop is too small to sample 24 patches reliably")

    row_edges = np.linspace(0, height, 5)
    col_edges = np.linspace(0, width, 7)
    patch_samples: List[np.ndarray] = []
    inset_fraction = patch_inset_fraction(patch_sample_fraction)

    for row in range(4):
        for col in range(6):
            y0 = int(round(row_edges[row]))
            y1 = int(round(row_edges[row + 1]))
            x0 = int(round(col_edges[col]))
            x1 = int(round(col_edges[col + 1]))

            cell_width = max(1, x1 - x0)
            cell_height = max(1, y1 - y0)
            inset_x = max(1, int(round(cell_width * inset_fraction)))
            inset_y = max(1, int(round(cell_height * inset_fraction)))

            inner_x0 = min(x0 + inset_x, x1 - 1)
            inner_x1 = max(inner_x0 + 1, x1 - inset_x)
            inner_y0 = min(y0 + inset_y, y1 - 1)
            inner_y1 = max(inner_y0 + 1, y1 - inset_y)

            patch = upright[inner_y0:inner_y1, inner_x0:inner_x1, :]
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                raise ValueError("Crop selection is too tight for reliable patch sampling")
            patch_samples.append(sample_patch_median(patch))

    return np.stack(patch_samples, axis=0)


def white_balance_from_patch(rgb: np.ndarray, reference_channel: str = "G") -> np.ndarray:
    """Build a white-balance vector normalized to R or G."""
    try:
        reference_index = WB_REFERENCE_CHANNELS[reference_channel]
    except KeyError as exc:
        raise ValueError(f"Unsupported white-balance reference channel: {reference_channel}") from exc

    values = np.asarray(rgb, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("White-balance patch must contain finite positive RGB values")
    reference = float(values[reference_index])
    return reference / values


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab D50 values to XYZ D50."""
    values = np.asarray(lab, dtype=np.float64)
    single = values.ndim == 1
    lab_values = values.reshape(-1, 3)

    fy = (lab_values[:, 0] + 16.0) / 116.0
    fx = fy + (lab_values[:, 1] / 500.0)
    fz = fy - (lab_values[:, 2] / 200.0)

    delta = 6.0 / 29.0

    def f_inv(component: np.ndarray) -> np.ndarray:
        cubic = component ** 3
        linear = 3.0 * (delta ** 2) * (component - 4.0 / 29.0)
        return np.where(component > delta, cubic, linear)

    xyz = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=1)
    xyz *= D50_WHITEPOINT[np.newaxis, :]
    return xyz[0] if single else xyz


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ D50 values to CIE Lab D50."""
    values = np.asarray(xyz, dtype=np.float64)
    single = values.ndim == 1
    xyz_values = values.reshape(-1, 3) / D50_WHITEPOINT[np.newaxis, :]

    delta = 6.0 / 29.0

    def f(component: np.ndarray) -> np.ndarray:
        cubic = np.cbrt(np.clip(component, 0.0, None))
        linear = component / (3.0 * (delta ** 2)) + 4.0 / 29.0
        return np.where(component > delta ** 3, cubic, linear)

    fx = f(xyz_values[:, 0])
    fy = f(xyz_values[:, 1])
    fz = f(xyz_values[:, 2])

    lab = np.stack(
        [
            116.0 * fy - 16.0,
            500.0 * (fx - fy),
            200.0 * (fy - fz),
        ],
        axis=1,
    )
    return lab[0] if single else lab


def adapt_xyz_whitepoint(xyz: np.ndarray, source_white: np.ndarray, target_white: np.ndarray) -> np.ndarray:
    """Adapt XYZ values between whitepoints using the Bradford transform."""
    values = np.asarray(xyz, dtype=np.float64)
    single = values.ndim == 1
    xyz_values = values.reshape(-1, 3)

    source_cone = BRADFORD_MATRIX @ np.asarray(source_white, dtype=np.float64)
    target_cone = BRADFORD_MATRIX @ np.asarray(target_white, dtype=np.float64)
    adaptation = BRADFORD_MATRIX_INV @ np.diag(target_cone / np.clip(source_cone, EPSILON, None)) @ BRADFORD_MATRIX
    adapted = xyz_values @ adaptation.T
    return adapted[0] if single else adapted


def xyz_d50_to_linear_srgb(xyz_d50: np.ndarray) -> np.ndarray:
    """Convert XYZ D50 values to linear sRGB."""
    xyz_d65 = adapt_xyz_whitepoint(xyz_d50, D50_WHITEPOINT, D65_WHITEPOINT)
    values = np.asarray(xyz_d65, dtype=np.float64)
    single = values.ndim == 1
    rgb = values.reshape(-1, 3) @ XYZ_D65_TO_LINEAR_SRGB_MATRIX.T
    return rgb[0] if single else rgb


def xyz_d50_to_linear_srgb_matrix() -> np.ndarray:
    """Return the matrix mapping XYZ D50 directly to linear sRGB."""
    source_cone = BRADFORD_MATRIX @ D50_WHITEPOINT
    target_cone = BRADFORD_MATRIX @ D65_WHITEPOINT
    adaptation = BRADFORD_MATRIX_INV @ np.diag(target_cone / np.clip(source_cone, EPSILON, None)) @ BRADFORD_MATRIX
    return XYZ_D65_TO_LINEAR_SRGB_MATRIX @ adaptation


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 color differences for paired Lab samples."""
    l1, a1, b1 = np.moveaxis(np.asarray(lab1, dtype=np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(lab2, dtype=np.float64), -1, 0)

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0 ** 7 + EPSILON)))

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.sqrt(a1_prime * a1_prime + b1 * b1)
    c2_prime = np.sqrt(a2_prime * a2_prime + b2 * b2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime

    delta_h_prime = h2_prime - h1_prime
    delta_h_prime = np.where(delta_h_prime > 180.0, delta_h_prime - 360.0, delta_h_prime)
    delta_h_prime = np.where(delta_h_prime < -180.0, delta_h_prime + 360.0, delta_h_prime)
    delta_h_prime = np.where((c1_prime * c2_prime) == 0.0, 0.0, delta_h_prime)

    delta_big_h_prime = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

    l_bar_prime = 0.5 * (l1 + l2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    h_bar_prime = 0.5 * (h1_prime + h2_prime)
    h_bar_prime = np.where(np.abs(h1_prime - h2_prime) > 180.0, h_bar_prime + 180.0, h_bar_prime)
    h_bar_prime = np.where((c1_prime * c2_prime) == 0.0, h1_prime + h2_prime, h_bar_prime)
    h_bar_prime %= 360.0

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )

    delta_theta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    c_bar_prime7 = c_bar_prime ** 7
    r_c = 2.0 * np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25.0 ** 7 + EPSILON))
    s_l = 1.0 + (0.015 * ((l_bar_prime - 50.0) ** 2)) / np.sqrt(20.0 + ((l_bar_prime - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    delta_e = np.sqrt(
        (delta_l_prime / s_l) ** 2
        + (delta_c_prime / s_c) ** 2
        + (delta_big_h_prime / s_h) ** 2
        + r_t * (delta_c_prime / s_c) * (delta_big_h_prime / s_h)
    )
    return delta_e


def seed_forward_matrix(rgb_wb: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    """Solve the normal-equation seed matrix from white-balanced RGB to XYZ."""
    solution, _, _, _ = np.linalg.lstsq(rgb_wb, reference_xyz, rcond=None)
    return solution.T.astype(np.float64, copy=False)


def solve_forward_matrix(
    sampled_rgb: np.ndarray,
    reference_lab: np.ndarray,
    wb_vector: np.ndarray,
    siril_output_scale: float,
) -> CalibrationResult:
    """Fit a forward matrix using least squares seeded from the normal equation."""
    rgb_wb = np.asarray(sampled_rgb, dtype=np.float64) * np.asarray(wb_vector, dtype=np.float64)[np.newaxis, :]
    if np.any(~np.isfinite(rgb_wb)) or np.any(rgb_wb <= 0):
        raise ValueError("Sampled RGB values must be finite and positive after white balance")
    if not np.isfinite(siril_output_scale) or siril_output_scale <= 0:
        raise ValueError("Siril output scale must be a finite positive value")

    reference_xyz = lab_to_xyz(reference_lab)
    seed = seed_forward_matrix(rgb_wb, reference_xyz)

    def objective(parameters: np.ndarray) -> float:
        matrix = parameters.reshape(3, 3)
        xyz = rgb_wb @ matrix.T
        xyz = np.clip(xyz, EPSILON, None)
        lab = xyz_to_lab(xyz)
        delta_e = ciede2000(lab, reference_lab)
        if not np.all(np.isfinite(delta_e)):
            return 1e12
        return float(np.mean(delta_e))

    optimization_note = None
    result = optimize.minimize(
        objective,
        seed.reshape(-1),
        method="Powell",
        options={"maxiter": 5000, "maxfev": 80000, "xtol": 1e-7, "ftol": 1e-7},
    )

    if result.success and np.all(np.isfinite(result.x)):
        forward_matrix = result.x.reshape(3, 3)
    else:
        forward_matrix = seed
        message = getattr(result, "message", "unknown optimization failure")
        optimization_note = f"Used the normal-equation seed because the dE00 refinement did not converge: {message}"

    xyz_fit = np.clip(rgb_wb @ forward_matrix.T, EPSILON, None)
    lab_fit = xyz_to_lab(xyz_fit)
    delta_e = ciede2000(lab_fit, reference_lab)
    combined_matrix = forward_matrix @ np.diag(wb_vector)
    xyz_to_srgb_matrix = xyz_d50_to_linear_srgb_matrix()
    siril_matrix_after_wb = siril_output_scale * (xyz_to_srgb_matrix @ forward_matrix)
    siril_matrix_one_step = siril_matrix_after_wb @ np.diag(wb_vector)

    return CalibrationResult(
        wb_vector=wb_vector,
        forward_matrix=forward_matrix,
        combined_matrix=combined_matrix,
        siril_matrix_after_wb=siril_matrix_after_wb,
        siril_matrix_one_step=siril_matrix_one_step,
        siril_output_scale=float(siril_output_scale),
        mean_delta_e=float(np.mean(delta_e)),
        max_delta_e=float(np.max(delta_e)),
        optimization_note=optimization_note,
    )


def format_vector_markdown(title: str, values: np.ndarray) -> str:
    """Format a 3-vector as a markdown table."""
    rows = [
        f"### {title}",
        "",
        "| Channel | Value |",
        "| --- | ---: |",
        f"| R | {values[0]:.8f} |",
        f"| G | {values[1]:.8f} |",
        f"| B | {values[2]:.8f} |",
    ]
    return "\n".join(rows)


def format_matrix_markdown(
    title: str,
    matrix: np.ndarray,
    row_labels: Tuple[str, str, str] = ("X", "Y", "Z"),
) -> str:
    """Format a 3x3 matrix as a markdown table."""
    rows = [
        f"### {title}",
        "",
        "|   | R | G | B |",
        "| --- | ---: | ---: | ---: |",
        f"| {row_labels[0]} | {matrix[0, 0]:.8f} | {matrix[0, 1]:.8f} | {matrix[0, 2]:.8f} |",
        f"| {row_labels[1]} | {matrix[1, 0]:.8f} | {matrix[1, 1]:.8f} | {matrix[1, 2]:.8f} |",
        f"| {row_labels[2]} | {matrix[2, 0]:.8f} | {matrix[2, 1]:.8f} | {matrix[2, 2]:.8f} |",
    ]
    return "\n".join(rows)


def format_output_markdown(
    reference_name: str,
    wb_patch_label: str,
    wb_reference_channel: str,
    result: CalibrationResult,
) -> str:
    """Build the final copy-pastable markdown report."""
    sections = [
        f"Reference: **{reference_name}**",
        f"White-balance patch: **{wb_patch_label}**",
        f"White-balance reference channel: **{wb_reference_channel} = 1.0**",
        f"Fit quality: **mean dE00 {result.mean_delta_e:.3f}**, **max dE00 {result.max_delta_e:.3f}**",
        "",
        format_vector_markdown("White Balance Vector", result.wb_vector),
        "",
        format_matrix_markdown(
            "Siril-Ready Matrix After White Balance (white-balanced RGB -> linear sRGB-like RGB)",
            result.siril_matrix_after_wb,
            row_labels=("R", "G", "B"),
        ),
        "",
        format_matrix_markdown(
            "Siril-Ready One-Step Matrix (raw RGB -> linear sRGB-like RGB)",
            result.siril_matrix_one_step,
            row_labels=("R", "G", "B"),
        ),
        "",
        format_matrix_markdown("Forward Matrix (white-balanced RGB -> XYZ D50)", result.forward_matrix),
        "",
        format_matrix_markdown("Combined Matrix (raw RGB -> XYZ D50)", result.combined_matrix),
        "",
        (
            "Use **either** the Siril-ready matrix after white balance **or** the Siril-ready one-step matrix. "
            "Do not white-balance first and then also use the one-step matrix."
        ),
        (
            f"The Siril-ready matrices are brightness-scaled by the selected WB patch {wb_reference_channel} value "
            f"(`{result.siril_output_scale:.3f}`) so the coefficients stay practical for Siril's GUI."
        ),
        "The intermediate XYZ matrices are still included for reference. Combined XYZ is `Forward × diag(WB)`.",
    ]
    if result.optimization_note:
        sections.extend(["", f"> {result.optimization_note}"])
    return "\n".join(sections)


class CropImageViewer(QWidget):
    """Preview widget with zoom, pan, and crop selection."""

    selection_changed = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: Optional[QPixmap] = None
        self.selection_rect = QRect()
        self.selection_start: Optional[QPoint] = None
        self.is_selecting = False
        self.scale_factor = 1

        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.pan_start = QPoint()
        self.pan_start_selection = QRect()
        self.is_resizing = False
        self.resize_edges = 0
        self.resize_start_image = QPoint()
        self.resize_start_selection = QRect()
        self.mouse_drag_active = False

        self.grid_cols = 6
        self.grid_rows = 4
        self.orientation_value = 0
        self.patch_sample_fraction = DEFAULT_PATCH_SAMPLE_FRACTION
        self.setMinimumSize(560, 560)
        self.setMouseTracking(True)
        self.setStyleSheet("border: 1px solid #777; background-color: #111;")

    def _grab_drag_mouse(self) -> None:
        """Keep receiving move/release events until the active drag finishes."""
        if not self.mouse_drag_active:
            self.grabMouse()
            self.mouse_drag_active = True

    def _release_drag_mouse(self) -> None:
        """Release a drag mouse grab, if this widget owns one."""
        if self.mouse_drag_active:
            self.releaseMouse()
            self.mouse_drag_active = False

    def set_image(self, image_rgb: np.ndarray, scale_factor: int) -> None:
        """Load an RGB preview image."""
        self.pixmap = QPixmap.fromImage(array_to_qimage(image_rgb))
        self.scale_factor = max(1, int(scale_factor))
        self._fit_to_widget()
        self.selection_rect = QRect(0, 0, self.pixmap.width(), self.pixmap.height())
        self.selection_start = None
        self.is_selecting = False
        self.is_resizing = False
        self.is_panning = False
        self._release_drag_mouse()
        self.update()
        self.selection_changed.emit(self.selection_rect)

    def clear(self) -> None:
        """Clear the current preview."""
        self.pixmap = None
        self.selection_rect = QRect()
        self.selection_start = None
        self.is_selecting = False
        self.is_resizing = False
        self.is_panning = False
        self._release_drag_mouse()
        self.update()

    def set_grid_shape(self, cols: int, rows: int) -> None:
        """Update the displayed grid shape."""
        self.grid_cols = max(1, int(cols))
        self.grid_rows = max(1, int(rows))
        self.update()

    def set_orientation(self, orientation_value: int) -> None:
        """Update the displayed chart orientation."""
        self.orientation_value = int(orientation_value)
        self.update()

    def set_patch_sample_fraction(self, sample_fraction: float) -> None:
        """Update the overlay patch sampling size."""
        self.patch_sample_fraction = clamp_patch_sample_fraction(sample_fraction)
        self.update()

    def _fit_to_widget(self) -> None:
        """Scale the image to fit in the current widget."""
        if self.pixmap is None or self.pixmap.isNull():
            return
        scale_x = self.width() / max(1, self.pixmap.width())
        scale_y = self.height() / max(1, self.pixmap.height())
        self.zoom_level = min(scale_x, scale_y, 1.0)
        self.pan_offset = QPoint(0, 0)

    def fit_to_window(self) -> None:
        """Public fit-to-window helper."""
        self._fit_to_widget()
        self.update()

    def reset_selection(self) -> None:
        """Reset selection to the full preview image."""
        if self.pixmap is None:
            return
        self.selection_rect = QRect(0, 0, self.pixmap.width(), self.pixmap.height())
        self.selection_start = None
        self.is_selecting = False
        self.is_resizing = False
        self.is_panning = False
        self._release_drag_mouse()
        self.update()
        self.selection_changed.emit(self.selection_rect)

    def _image_origin(self) -> QPoint:
        """Return the top-left corner of the zoomed image in widget coordinates."""
        if self.pixmap is None:
            return QPoint(0, 0)
        scaled_w = int(round(self.pixmap.width() * self.zoom_level))
        scaled_h = int(round(self.pixmap.height() * self.zoom_level))
        x = (self.width() - scaled_w) // 2 + self.pan_offset.x()
        y = (self.height() - scaled_h) // 2 + self.pan_offset.y()
        return QPoint(x, y)

    def _widget_to_image(self, widget_pos: QPoint) -> QPoint:
        """Convert widget coordinates into preview-image coordinates."""
        origin = self._image_origin()
        x = int((widget_pos.x() - origin.x()) / max(self.zoom_level, EPSILON))
        y = int((widget_pos.y() - origin.y()) / max(self.zoom_level, EPSILON))
        return QPoint(x, y)

    def _image_to_widget(self, image_pos: QPoint) -> QPoint:
        """Convert preview-image coordinates into widget coordinates."""
        origin = self._image_origin()
        x = int(round(image_pos.x() * self.zoom_level)) + origin.x()
        y = int(round(image_pos.y() * self.zoom_level)) + origin.y()
        return QPoint(x, y)

    def current_full_resolution_selection(self) -> Optional[Tuple[int, int, int, int]]:
        """Return the current crop rectangle in full-resolution image coordinates."""
        if self.pixmap is None or self.selection_rect.isNull():
            return None
        x = int(round(self.selection_rect.x() * self.scale_factor))
        y = int(round(self.selection_rect.y() * self.scale_factor))
        width = int(round(self.selection_rect.width() * self.scale_factor))
        height = int(round(self.selection_rect.height() * self.scale_factor))
        return x, y, width, height

    def _clamp_selection_rect(self, rect: QRect) -> QRect:
        """Keep the crop rectangle fully inside the preview image."""
        if self.pixmap is None:
            return rect

        width = min(rect.width(), self.pixmap.width())
        height = min(rect.height(), self.pixmap.height())
        x = min(max(0, rect.x()), self.pixmap.width() - width)
        y = min(max(0, rect.y()), self.pixmap.height() - height)
        return QRect(x, y, width, height)

    def _selection_widget_rect(self) -> QRect:
        """Return the crop rectangle in widget coordinates."""
        if self.pixmap is None or self.selection_rect.isNull():
            return QRect()
        top_left = self._image_to_widget(self.selection_rect.topLeft())
        bottom_right = self._image_to_widget(self.selection_rect.bottomRight())
        return QRect(top_left, bottom_right).normalized()

    def _resize_edges_at_widget_pos(self, widget_pos: QPoint) -> int:
        """Return which crop edges the cursor is hovering over."""
        rect = self._selection_widget_rect()
        if rect.isNull():
            return 0

        margin = 8
        expanded = rect.adjusted(-margin, -margin, margin, margin)
        if not expanded.contains(widget_pos):
            return 0

        edges = 0
        if abs(widget_pos.x() - rect.left()) <= margin:
            edges |= EDGE_LEFT
        if abs(widget_pos.x() - rect.right()) <= margin:
            edges |= EDGE_RIGHT
        if abs(widget_pos.y() - rect.top()) <= margin:
            edges |= EDGE_TOP
        if abs(widget_pos.y() - rect.bottom()) <= margin:
            edges |= EDGE_BOTTOM
        return edges

    def _cursor_for_edges(self, edges: int):
        """Return the correct resize cursor for a hovered edge combination."""
        if edges in (EDGE_LEFT | EDGE_TOP, EDGE_RIGHT | EDGE_BOTTOM):
            return Qt.SizeFDiagCursor
        if edges in (EDGE_RIGHT | EDGE_TOP, EDGE_LEFT | EDGE_BOTTOM):
            return Qt.SizeBDiagCursor
        if edges & (EDGE_LEFT | EDGE_RIGHT):
            return Qt.SizeHorCursor
        if edges & (EDGE_TOP | EDGE_BOTTOM):
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def _update_hover_cursor(self, widget_pos: QPoint) -> None:
        """Update the cursor shape for hover feedback."""
        if self.pixmap is None:
            self.setCursor(Qt.ArrowCursor)
            return

        edges = self._resize_edges_at_widget_pos(widget_pos)
        if edges:
            self.setCursor(self._cursor_for_edges(edges))
            return

        image_pos = self._widget_to_image(widget_pos)
        if self.selection_rect.contains(image_pos):
            self.setCursor(Qt.OpenHandCursor)
            return

        self.setCursor(Qt.ArrowCursor)

    def _resize_selection_rect(self, start_rect: QRect, delta_x: int, delta_y: int, edges: int) -> QRect:
        """Resize the crop rectangle from the stored starting geometry."""
        if self.pixmap is None:
            return start_rect

        min_size = 4
        x0 = start_rect.x()
        y0 = start_rect.y()
        x1 = start_rect.x() + start_rect.width()
        y1 = start_rect.y() + start_rect.height()

        if edges & EDGE_LEFT:
            x0 = int(np.clip(x0 + delta_x, 0, x1 - min_size))
        if edges & EDGE_RIGHT:
            x1 = int(np.clip(x1 + delta_x, x0 + min_size, self.pixmap.width()))
        if edges & EDGE_TOP:
            y0 = int(np.clip(y0 + delta_y, 0, y1 - min_size))
        if edges & EDGE_BOTTOM:
            y1 = int(np.clip(y1 + delta_y, y0 + min_size, self.pixmap.height()))

        return QRect(x0, y0, x1 - x0, y1 - y0)

    def paintEvent(self, event) -> None:
        """Draw the preview image, crop rectangle, and patch grid."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        if self.pixmap is None:
            painter.setPen(Qt.white)
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No image loaded\n\nScroll to zoom, right-drag to pan, left-drag to crop",
            )
            return

        origin = self._image_origin()
        scaled_w = int(round(self.pixmap.width() * self.zoom_level))
        scaled_h = int(round(self.pixmap.height() * self.zoom_level))
        target_rect = QRect(origin.x(), origin.y(), scaled_w, scaled_h)
        painter.drawPixmap(target_rect, self.pixmap)

        if self.selection_rect.isNull():
            return

        selection_widget_rect = self._selection_widget_rect()

        painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.SolidLine))
        painter.drawRect(selection_widget_rect)

        if selection_widget_rect.width() < 20 or selection_widget_rect.height() < 20:
            return

        painter.setPen(QPen(QColor(255, 220, 80), 1, Qt.DashLine))
        for col in range(1, self.grid_cols):
            x = selection_widget_rect.left() + int(round(selection_widget_rect.width() * col / self.grid_cols))
            painter.drawLine(x, selection_widget_rect.top(), x, selection_widget_rect.bottom())
        for row in range(1, self.grid_rows):
            y = selection_widget_rect.top() + int(round(selection_widget_rect.height() * row / self.grid_rows))
            painter.drawLine(selection_widget_rect.left(), y, selection_widget_rect.right(), y)

        painter.setPen(QPen(QColor(255, 140, 0), 1, Qt.SolidLine))
        cell_width = selection_widget_rect.width() / float(self.grid_cols)
        cell_height = selection_widget_rect.height() / float(self.grid_rows)
        inset_fraction = patch_inset_fraction(self.patch_sample_fraction)
        inset_x = cell_width * inset_fraction
        inset_y = cell_height * inset_fraction
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x0 = selection_widget_rect.left() + cell_width * col + inset_x
                y0 = selection_widget_rect.top() + cell_height * row + inset_y
                x1 = selection_widget_rect.left() + cell_width * (col + 1) - inset_x
                y1 = selection_widget_rect.top() + cell_height * (row + 1) - inset_y
                painter.drawRect(QRect(int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))))

        label_font = painter.font()
        label_font.setBold(True)
        label_font.setPointSize(max(8, label_font.pointSize() - 1))
        painter.setFont(label_font)

        label_specs = [
            ("White", QColor(255, 255, 255, 220), QColor(20, 20, 20), rotate_grid_position(0, 3, 6, 4, self.orientation_value)),
            ("Black", QColor(20, 20, 20, 220), QColor(255, 255, 255), rotate_grid_position(5, 3, 6, 4, self.orientation_value)),
        ]
        for label, background, foreground, (col, row) in label_specs:
            cell_rect = QRect(
                int(round(selection_widget_rect.left() + cell_width * col)),
                int(round(selection_widget_rect.top() + cell_height * row)),
                int(round(cell_width)),
                int(round(cell_height)),
            )
            text_rect = cell_rect.adjusted(4, 4, -4, -4)
            metrics = painter.fontMetrics()
            box_width = min(text_rect.width(), metrics.horizontalAdvance(label) + 12)
            box_height = min(text_rect.height(), metrics.height() + 8)
            label_box = QRect(text_rect.left(), text_rect.top(), box_width, box_height)
            painter.fillRect(label_box, background)
            painter.setPen(foreground)
            painter.drawText(label_box, Qt.AlignCenter, label)

    def mousePressEvent(self, event) -> None:
        """Left-click starts crop/resize, right-click moves the crop box."""
        if self.pixmap is None:
            return

        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            image_pos = self._widget_to_image(event.pos())
            if self.selection_rect.contains(image_pos):
                self.is_panning = True
                self.pan_start = event.pos()
                self.pan_start_selection = QRect(self.selection_rect)
                self._grab_drag_mouse()
                self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            resize_edges = self._resize_edges_at_widget_pos(event.pos())
            if resize_edges:
                self.is_resizing = True
                self.resize_edges = resize_edges
                self.resize_start_image = self._widget_to_image(event.pos())
                self.resize_start_selection = QRect(self.selection_rect)
                self._grab_drag_mouse()
                self.setCursor(self._cursor_for_edges(resize_edges))
                return

            image_pos = self._widget_to_image(event.pos())
            if 0 <= image_pos.x() < self.pixmap.width() and 0 <= image_pos.y() < self.pixmap.height():
                self.selection_start = image_pos
                self.selection_rect = QRect(image_pos, image_pos)
                self.is_selecting = True
                self._grab_drag_mouse()
                self.update()

    def mouseMoveEvent(self, event) -> None:
        """Update crop selection or move the crop box while dragging."""
        if self.is_panning:
            if not (event.buttons() & (Qt.RightButton | Qt.MiddleButton)):
                self.is_panning = False
                self._release_drag_mouse()
                self._update_hover_cursor(event.pos())
                self.selection_changed.emit(self.selection_rect)
                return
            delta = event.pos() - self.pan_start
            dx = int(round(delta.x() / max(self.zoom_level, EPSILON)))
            dy = int(round(delta.y() / max(self.zoom_level, EPSILON)))
            translated = self.pan_start_selection.translated(dx, dy)
            self.selection_rect = self._clamp_selection_rect(translated)
            self.update()
            return

        if self.is_resizing:
            if not (event.buttons() & Qt.LeftButton):
                self.is_resizing = False
                self.resize_edges = 0
                self._release_drag_mouse()
                self._update_hover_cursor(event.pos())
                self.selection_changed.emit(self.selection_rect)
                return
            image_pos = self._widget_to_image(event.pos())
            delta_x = image_pos.x() - self.resize_start_image.x()
            delta_y = image_pos.y() - self.resize_start_image.y()
            self.selection_rect = self._resize_selection_rect(self.resize_start_selection, delta_x, delta_y, self.resize_edges)
            self.update()
            return

        if self.is_selecting and self.selection_start is not None and self.pixmap is not None:
            if not (event.buttons() & Qt.LeftButton):
                self.is_selecting = False
                self.selection_start = None
                self._release_drag_mouse()
                if self.selection_rect.width() < 2 or self.selection_rect.height() < 2:
                    self.reset_selection()
                else:
                    self.selection_changed.emit(self.selection_rect)
                self._update_hover_cursor(event.pos())
                return
            image_pos = self._widget_to_image(event.pos())
            image_pos.setX(max(0, min(image_pos.x(), self.pixmap.width() - 1)))
            image_pos.setY(max(0, min(image_pos.y(), self.pixmap.height() - 1)))
            self.selection_rect = QRect(self.selection_start, image_pos).normalized()
            self.update()
            return

        self._update_hover_cursor(event.pos())

    def mouseReleaseEvent(self, event) -> None:
        """Finish moving or cropping."""
        if event.button() in (Qt.RightButton, Qt.MiddleButton) and self.is_panning:
            self.is_panning = False
            self._release_drag_mouse()
            self._update_hover_cursor(event.pos())
            self.selection_changed.emit(self.selection_rect)
            return

        if event.button() == Qt.LeftButton and self.is_resizing:
            self.is_resizing = False
            self.resize_edges = 0
            self._release_drag_mouse()
            self._update_hover_cursor(event.pos())
            self.selection_changed.emit(self.selection_rect)
            return

        if event.button() == Qt.LeftButton and self.is_selecting:
            self.is_selecting = False
            self.selection_start = None
            self._release_drag_mouse()
            if self.selection_rect.width() < 2 or self.selection_rect.height() < 2:
                self.reset_selection()
            else:
                self.selection_changed.emit(self.selection_rect)
            self._update_hover_cursor(event.pos())

    def wheelEvent(self, event) -> None:
        """Zoom in or out around the mouse pointer."""
        if self.pixmap is None:
            return

        old_zoom = self.zoom_level
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.zoom_level *= zoom_factor
        else:
            self.zoom_level /= zoom_factor

        min_zoom = min(
            0.1,
            min(self.width() / max(1, self.pixmap.width()), self.height() / max(1, self.pixmap.height())) * 0.5,
        )
        self.zoom_level = max(min_zoom, min(self.zoom_level, 20.0))

        mouse_pos = event.pos()
        center_before = QPoint(
            self.width() // 2 + self.pan_offset.x(),
            self.height() // 2 + self.pan_offset.y(),
        )
        scale_ratio = self.zoom_level / max(old_zoom, EPSILON)
        new_pan_x = int(mouse_pos.x() - (mouse_pos.x() - center_before.x()) * scale_ratio) - self.width() // 2
        new_pan_y = int(mouse_pos.y() - (mouse_pos.y() - center_before.y()) * scale_ratio) - self.height() // 2
        self.pan_offset = QPoint(new_pan_x, new_pan_y)
        self.update()

    def resizeEvent(self, event) -> None:
        """Keep the full image visible on initial resizes when zoom is unchanged."""
        if self.pixmap is not None and self.zoom_level <= 1.0:
            self._fit_to_widget()
        super().resizeEvent(event)

    def leaveEvent(self, event) -> None:
        """Restore the default cursor when leaving the preview."""
        if not self.is_panning and not self.is_resizing:
            self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)


class ForwardMatrixGUI(QMainWindow):
    """Main application window."""

    def __init__(self, siril_instance=None):
        super().__init__()
        self.siril = siril_instance
        self.siril_wd: Optional[Path] = None
        self.loaded_image: Optional[LoadedImage] = None
        self.last_path: Optional[Path] = None

        self.init_ui()
        self.detect_siril_working_directory()
        self.load_settings()
        self.update_grid_shape()

    def init_ui(self) -> None:
        """Build the UI."""
        self.setWindowTitle("Forward Matrix Builder for Siril")
        self.setGeometry(100, 100, 1220, 980)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        input_group = QGroupBox("Input Image")
        input_layout = QGridLayout()

        self.file_label = QLabel("No FITS or TIFF image selected")
        self.file_label.setWordWrap(True)
        browse_button = QPushButton("Select Image")
        browse_button.clicked.connect(self.select_image)
        input_layout.addWidget(QLabel("Image:"), 0, 0)
        input_layout.addWidget(self.file_label, 0, 1)
        input_layout.addWidget(browse_button, 0, 2)

        self.image_info_label = QLabel("Load a ColorChecker image to begin.")
        self.image_info_label.setWordWrap(True)
        input_layout.addWidget(QLabel("Info:"), 1, 0)
        input_layout.addWidget(self.image_info_label, 1, 1, 1, 2)

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        options_group = QGroupBox("Calibration Options")
        options_layout = QGridLayout()

        self.reference_combo = QComboBox()
        self.reference_combo.addItems(["After Nov 2014", "Before Nov 2014"])
        options_layout.addWidget(QLabel("Reference Data:"), 0, 0)
        options_layout.addWidget(self.reference_combo, 0, 1)

        self.orientation_combo = QComboBox()
        for label, _value in ORIENTATION_OPTIONS:
            self.orientation_combo.addItem(label)
        self.orientation_combo.currentIndexChanged.connect(self.update_grid_shape)
        options_layout.addWidget(QLabel("Chart Orientation:"), 0, 2)
        options_layout.addWidget(self.orientation_combo, 0, 3)

        self.wb_patch_combo = QComboBox()
        self.wb_patch_combo.addItems(WB_PATCH_LABELS)
        self.wb_patch_combo.setCurrentText("Neutral 6.5")
        options_layout.addWidget(QLabel("WB Patch:"), 1, 0)
        options_layout.addWidget(self.wb_patch_combo, 1, 1)

        self.wb_reference_combo = QComboBox()
        self.wb_reference_combo.addItems(list(WB_REFERENCE_CHANNELS.keys()))
        self.wb_reference_combo.setCurrentText("G")
        options_layout.addWidget(QLabel("WB Reference:"), 2, 0)
        options_layout.addWidget(self.wb_reference_combo, 2, 1)

        self.patch_sample_spin = QDoubleSpinBox()
        self.patch_sample_spin.setRange(10.0, 95.0)
        self.patch_sample_spin.setDecimals(1)
        self.patch_sample_spin.setSingleStep(2.0)
        self.patch_sample_spin.setValue(DEFAULT_PATCH_SAMPLE_FRACTION * 100.0)
        self.patch_sample_spin.setSuffix("%")
        self.patch_sample_spin.valueChanged.connect(self.on_patch_sample_changed)
        options_layout.addWidget(QLabel("Patch Sample Size:"), 1, 2)
        patch_sample_row = QHBoxLayout()
        patch_smaller_button = QPushButton("-")
        patch_smaller_button.clicked.connect(lambda: self.patch_sample_spin.setValue(self.patch_sample_spin.value() - self.patch_sample_spin.singleStep()))
        patch_sample_row.addWidget(patch_smaller_button)
        patch_sample_row.addWidget(self.patch_sample_spin, 1)
        patch_larger_button = QPushButton("+")
        patch_larger_button.clicked.connect(lambda: self.patch_sample_spin.setValue(self.patch_sample_spin.value() + self.patch_sample_spin.singleStep()))
        patch_sample_row.addWidget(patch_larger_button)
        options_layout.addLayout(patch_sample_row, 1, 3)

        self.summary_label = QLabel("Crop the chart, then compute the matrix.")
        self.summary_label.setStyleSheet("font-weight: bold;")
        self.summary_label.setWordWrap(True)
        options_layout.addWidget(self.summary_label, 3, 0, 1, 4)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        preview_group = QGroupBox("Preview and Crop")
        preview_layout = QVBoxLayout()

        help_label = QLabel(
            "Left-drag to define the crop around the ColorChecker. "
            "Hover over a crop edge to resize it, then left-drag. "
            "Right-drag inside the crop to move the patch grid. Scroll to zoom. "
            "The yellow/orange overlay shows the patch grid and sampling regions."
            " This tool assumes the classic 24-patch ColorChecker / Passport layout."
        )
        help_label.setWordWrap(True)
        preview_layout.addWidget(help_label)

        self.preview_widget = CropImageViewer()
        self.preview_widget.selection_changed.connect(self.on_selection_changed)
        preview_layout.addWidget(self.preview_widget, 1)

        crop_row = QHBoxLayout()
        self.crop_label = QLabel("Crop: not set")
        self.crop_label.setWordWrap(True)
        crop_row.addWidget(self.crop_label, 1)

        fit_button = QPushButton("Fit Preview")
        fit_button.clicked.connect(self.preview_widget.fit_to_window)
        crop_row.addWidget(fit_button)

        reset_crop_button = QPushButton("Reset Crop")
        reset_crop_button.clicked.connect(self.preview_widget.reset_selection)
        crop_row.addWidget(reset_crop_button)

        preview_layout.addLayout(crop_row)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, 1)

        action_row = QHBoxLayout()
        self.compute_button = QPushButton("Compute Forward Matrix")
        self.compute_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.compute_button.clicked.connect(self.compute_forward_matrix)
        self.compute_button.setEnabled(False)
        action_row.addWidget(self.compute_button)
        action_row.addStretch(1)
        main_layout.addLayout(action_row)

        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Monospace", 9))
        self.output_text.setPlaceholderText("The markdown tables will appear here.")
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group, 1)

    def detect_siril_working_directory(self) -> None:
        """Capture Siril's working directory when available."""
        if not self.siril:
            return
        try:
            wd = self.siril.get_siril_wd()
            if wd:
                path = Path(wd)
                if path.exists():
                    self.siril_wd = path
        except Exception:
            self.siril_wd = None

    def select_image(self) -> None:
        """Choose a FITS or TIFF image to analyze."""
        start_dir_path = self.siril_wd
        if self.last_path is not None and self.last_path.exists():
            start_dir_path = self.last_path.parent if self.last_path.is_file() else self.last_path
        start_dir = str(start_dir_path or Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ColorChecker Image",
            start_dir,
            "Images (*.fit *.fits *.fts *.tif *.tiff);;All Files (*)",
        )
        if not file_path:
            return

        try:
            self.loaded_image = read_image_file(Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self.file_label.setText(file_path)
        self.last_path = Path(file_path)
        height, width = self.loaded_image.rgb.shape[:2]
        scale_note = f"preview downsample {self.loaded_image.preview_scale}x" if self.loaded_image.preview_scale > 1 else "preview at full resolution"
        color_note = "color image detected" if self.loaded_image.is_color else "grayscale image detected"
        self.image_info_label.setText(f"{width} x {height} pixels, {color_note}, {scale_note}")
        self.preview_widget.set_image(self.loaded_image.preview_rgb, self.loaded_image.preview_scale)
        self.output_text.clear()
        self.compute_button.setEnabled(True)
        self.summary_label.setText("Crop the chart, check orientation/reference, then compute the matrix.")
        self.save_settings()

    def orientation_value(self) -> int:
        """Return the orientation correction setting."""
        return ORIENTATION_OPTIONS[self.orientation_combo.currentIndex()][1]

    def update_grid_shape(self) -> None:
        """Match the preview grid to the apparent chart orientation."""
        orientation = self.orientation_value()
        self.preview_widget.set_orientation(orientation)
        if orientation in (0, 2):
            self.preview_widget.set_grid_shape(6, 4)
        else:
            self.preview_widget.set_grid_shape(4, 6)

    def on_selection_changed(self, selection_rect: QRect) -> None:
        """Update the crop label when the user changes the selection."""
        full_selection = self.preview_widget.current_full_resolution_selection()
        if full_selection is None:
            self.crop_label.setText("Crop: not set")
            return
        x, y, width, height = full_selection
        self.crop_label.setText(f"Crop: x={x}, y={y}, width={width}, height={height}")

    def selected_reference_lab(self) -> np.ndarray:
        """Return the selected reference Lab data."""
        return REFERENCE_LAB[self.reference_combo.currentText()]

    def selected_wb_patch_index(self) -> int:
        """Return the selected neutral patch index."""
        return WB_PATCH_INDEX[self.wb_patch_combo.currentText()]

    def selected_wb_reference_channel(self) -> str:
        """Return the selected white-balance reference channel."""
        return self.wb_reference_combo.currentText()

    def patch_sample_fraction(self) -> float:
        """Return the fraction of each patch cell used for color sampling."""
        return clamp_patch_sample_fraction(self.patch_sample_spin.value() / 100.0)

    def on_patch_sample_changed(self) -> None:
        """Refresh the overlay after changing the patch sample size."""
        self.preview_widget.set_patch_sample_fraction(self.patch_sample_fraction())
        self.save_settings()

    def compute_forward_matrix(self) -> None:
        """Sample the crop and compute the calibration result."""
        if self.loaded_image is None:
            QMessageBox.warning(self, "Missing Image", "Please select a FITS or TIFF image first.")
            return

        if not self.loaded_image.is_color:
            QMessageBox.warning(
                self,
                "Missing Color Data",
                "The selected image appears to be grayscale. Use a color image or a Bayer FITS file with BAYERPAT metadata.",
            )
            return

        crop_rect = self.preview_widget.current_full_resolution_selection()
        if crop_rect is None:
            QMessageBox.warning(self, "Missing Crop", "Please draw a crop over the ColorChecker.")
            return

        try:
            crop_rgb = crop_array(self.loaded_image.rgb, crop_rect)
            sampled_rgb = sample_colorchecker_patches(crop_rgb, self.orientation_value(), self.patch_sample_fraction())
            wb_patch_index = self.selected_wb_patch_index()
            wb_reference_channel = self.selected_wb_reference_channel()
            wb_reference_index = WB_REFERENCE_CHANNELS[wb_reference_channel]
            wb_vector = white_balance_from_patch(sampled_rgb[wb_patch_index], wb_reference_channel)
            wb_reference_rgb = sampled_rgb[wb_patch_index] * wb_vector
            siril_output_scale = float(wb_reference_rgb[wb_reference_index])
            result = solve_forward_matrix(sampled_rgb, self.selected_reference_lab(), wb_vector, siril_output_scale)
        except Exception as exc:
            QMessageBox.critical(self, "Computation Error", str(exc))
            return

        markdown = format_output_markdown(
            self.reference_combo.currentText(),
            self.wb_patch_combo.currentText(),
            self.selected_wb_reference_channel(),
            result,
        )
        self.output_text.setPlainText(markdown)
        self.summary_label.setText(f"Done. Mean dE00 {result.mean_delta_e:.3f} | Max dE00 {result.max_delta_e:.3f}")
        self.save_settings()

    def config_path(self) -> Path:
        """Return the per-user config file path."""
        return Path.home() / ".make_forward_matrix_config.json"

    def save_settings(self) -> None:
        """Persist lightweight UI state."""
        settings = {
            "last_path": self.file_label.text() if self.loaded_image is not None else "",
            "reference": self.reference_combo.currentText(),
            "orientation_index": self.orientation_combo.currentIndex(),
            "wb_patch": self.wb_patch_combo.currentText(),
            "wb_reference_channel": self.wb_reference_combo.currentText(),
            "patch_sample_percent": self.patch_sample_spin.value(),
        }
        try:
            self.config_path().write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_settings(self) -> None:
        """Restore lightweight UI state."""
        config_path = self.config_path()
        if not config_path.exists():
            return
        try:
            settings = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        reference = settings.get("reference")
        if reference in REFERENCE_LAB:
            self.reference_combo.setCurrentText(reference)

        orientation_index = int(settings.get("orientation_index", 0))
        if 0 <= orientation_index < self.orientation_combo.count():
            self.orientation_combo.setCurrentIndex(orientation_index)

        wb_patch = settings.get("wb_patch")
        if wb_patch in WB_PATCH_INDEX:
            self.wb_patch_combo.setCurrentText(wb_patch)

        wb_reference_channel = settings.get("wb_reference_channel")
        if wb_reference_channel in WB_REFERENCE_CHANNELS:
            self.wb_reference_combo.setCurrentText(wb_reference_channel)

        patch_sample_percent = settings.get("patch_sample_percent")
        if patch_sample_percent is not None:
            self.patch_sample_spin.setValue(float(patch_sample_percent))

        last_path = settings.get("last_path")
        if last_path:
            path = Path(last_path)
            if path.exists():
                self.last_path = path


def main() -> int:
    """Main entry point."""
    if not SIRILPY_AVAILABLE:
        print("Error: sirilpy is not available. Run this script from Siril's Scripts menu.")
        return 1

    try:
        siril = sirilpy.SirilInterface()
        siril.connect()
    except Exception as exc:
        print(f"Error connecting to Siril: {exc}")
        print("Make sure this script is run from Siril's Scripts menu.")
        return 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ForwardMatrixGUI(siril_instance=siril)
    window.show()
    result = app.exec_()
    return 0 if result == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
