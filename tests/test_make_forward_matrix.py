import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

import make_forward_matrix as fm


def synthetic_chart(cell_height=20, cell_width=24):
    chart = np.zeros((4 * cell_height, 6 * cell_width, 3), dtype=np.float32)
    expected = []
    for row in range(4):
        for col in range(6):
            value = float(row * 6 + col + 1)
            rgb = np.array([value, value + 0.25, value + 0.5], dtype=np.float32)
            chart[
                row * cell_height : (row + 1) * cell_height,
                col * cell_width : (col + 1) * cell_width,
            ] = rgb
            expected.append(rgb)
    return chart, np.asarray(expected)


class ImageLoadingTests(unittest.TestCase):
    def test_16_bit_tiff_values_are_preserved(self):
        data = np.array([[[0, 32768, 65535], [1234, 50000, 60000]]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sixteen-bit.tif"
            tifffile.imwrite(path, data, photometric="rgb")
            rgb, is_color = fm.read_tiff_rgb(path)

        self.assertTrue(is_color)
        self.assertEqual(rgb.dtype, np.float32)
        np.testing.assert_array_equal(rgb, data.astype(np.float32))
        self.assertGreater(float(rgb[0, 1, 1]), 255.0)


class PatchSamplingTests(unittest.TestCase):
    def test_all_rotations_and_mirror_orientations(self):
        upright, expected = synthetic_chart()
        for orientation in range(8):
            with self.subTest(orientation=orientation):
                displayed = fm.orient_upright_chart(upright, orientation)
                sampled = fm.sample_colorchecker_patches(displayed, orientation, 0.6)
                np.testing.assert_allclose(sampled, expected)

    def test_patch_with_more_than_one_percent_clipped_pixels_is_excluded(self):
        chart, _ = synthetic_chart(cell_height=40, cell_width=40)
        chart *= 1000.0
        # Fill a central block in patch 0, safely inside the sampled region.
        chart[14:20, 14:20, 0] = 65535.0
        details = fm.sample_colorchecker_patches(
            chart,
            0,
            0.6,
            clipping_value=65535.0,
            return_details=True,
        )
        self.assertIsInstance(details, fm.PatchSamplingResult)
        self.assertGreater(details.clipped_fractions[0], 0.01)
        self.assertFalse(details.included_mask[0])
        self.assertTrue(np.all(details.included_mask[1:]))


class MatrixNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reference_xyz = fm.lab_to_xyz(fm.REFERENCE_LAB[fm.DEFAULT_REFERENCE_NAME])
        neutral_y = reference_xyz[fm.DEFAULT_NORMALIZATION_PATCH_INDEX, 1]
        normalized_xyz = reference_xyz / neutral_y
        camera_to_xyz = np.array(
            [[0.72, 0.19, 0.06], [0.24, 0.68, 0.08], [0.03, 0.15, 0.82]],
            dtype=np.float64,
        )
        normalized_camera = normalized_xyz @ np.linalg.inv(camera_to_xyz).T
        normalized_camera -= min(0.0, float(np.min(normalized_camera))) - 0.05
        cls.samples = normalized_camera * 12000.0
        cls.wb = fm.white_balance_from_patch(
            cls.samples[fm.DEFAULT_NORMALIZATION_PATCH_INDEX], "G"
        )
        cls.result = fm.solve_forward_matrix(
            cls.samples,
            fm.REFERENCE_LAB[fm.DEFAULT_REFERENCE_NAME],
            cls.wb,
        )

    def test_neutral_5_normalization_makes_fit_exposure_independent(self):
        scaled = fm.solve_forward_matrix(
            self.samples * 17.0,
            fm.REFERENCE_LAB[fm.DEFAULT_REFERENCE_NAME],
            self.wb,
        )
        np.testing.assert_allclose(
            scaled.siril_matrix_one_step,
            self.result.siril_matrix_one_step,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_forward_matrix_luminance_row_sums_to_unity(self):
        self.assertAlmostEqual(float(np.sum(self.result.forward_matrix[1])), 1.0, places=12)
        self.assertTrue(np.all(np.isfinite(self.result.patch_delta_e)))

    def test_one_step_equals_white_balance_then_ccm(self):
        pixels = np.array([[0.12, 0.24, 0.36], [0.8, 0.4, 0.1]], dtype=np.float64)
        separate = (pixels * self.result.wb_vector) @ self.result.siril_matrix_after_wb.T
        one_step = pixels @ self.result.siril_matrix_one_step.T
        np.testing.assert_allclose(one_step, separate, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
