import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rnc_color_stretch as rnc


def default_params(**overrides):
    values = dict(
        tone_curve=False,
        sky_region_mode="full",
        sky_x=0,
        sky_y=0,
        sky_width=32,
        sky_height=32,
        sky_step_fraction=10,
        skylevelfactor=0.06,
        zerosky_r=4096,
        zerosky_g=4096,
        zerosky_b=4096,
        stretch_type="root",
        rootpower=5,
        rootpower2=5,
        rootiter=2,
        asinh_k1=100,
        asinh_k2=5,
        asinh_iter=1,
        log_k1=100,
        log_k2=5,
        log_iter=1,
        s_curve=0,
        setmin=False,
        setmin_r=0,
        setmin_g=0,
        setmin_b=0,
        color_correction_mode="ratio",
        colorenhance=1.0,
        color_gamma=5.0,
        hsv_adjust=False,
        hue_adjust=0.0,
        sat_adjust=1.0,
        val_adjust=1.0,
        vib_adjust=0.0,
        white_balance="none",
        temp=1.0,
        tint=1.0,
        ca_correct=False,
        vignette_correct=False,
        vignette_strength=30,
        gradient_correct=False,
        gradient_strength=100,
        star_reduction=False,
        star_reduction_strength=1.0,
        rl_deconvolve=False,
        rl_iterations=15,
        rl_sigma=0.8,
    )
    values.update(overrides)
    return rnc.StretchParameters(**values)


def synthetic_image(seed=4):
    rng = np.random.default_rng(seed)
    base = rng.normal(9000.0, 1400.0, size=(48, 64, 1))
    colors = np.array([0.93, 1.0, 1.08]).reshape(1, 1, 3)
    return np.clip(base * colors, 500.0, 30000.0).astype(np.float32)


def loaded_source(dtype="float32", shape=(3, 4, 5)):
    return rnc.LoadedImage(
        path=Path("active.fit"),
        rgb=np.zeros((4, 5, 3), dtype=np.float32),
        header=rnc.fits.Header(),
        source_is_current=True,
        is_color=True,
        original_shape=shape,
        original_dtype=dtype,
        preview_input=np.zeros((4, 5, 3), dtype=np.float32),
        preview_scale=1,
        display_limits=(0.0, 1.0),
        siril_data_shape=shape,
        siril_data_dtype=dtype,
    )


class HistogramTests(unittest.TestCase):
    def test_exact_rgb_counts_and_smoothing(self):
        image = np.zeros((10, 12, 3), dtype=np.float32)
        image[:, :, 0] = 1000
        image[:, :, 1] = 2000
        image[:, :, 2] = 3000
        snapshot = rnc.make_histogram_snapshot(image, default_params(rootiter=1), "test", "Test")
        self.assertEqual(snapshot.raw_rgb.shape, (3, 65536))
        self.assertEqual(snapshot.smoothed_rgb.shape, (3, 65536))
        self.assertEqual(snapshot.raw_rgb[0, 1000], 120)
        self.assertEqual(snapshot.raw_rgb[1, 2000], 120)
        self.assertEqual(snapshot.raw_rgb[2, 3000], 120)
        self.assertAlmostEqual(snapshot.smoothed_rgb[0, 1000], 120 / 601)
        self.assertEqual(snapshot.peaks, (700, 1700, 2700))

    def test_stage_order_for_two_rootpower_sky_iterations(self):
        result = rnc.apply_rnc_stretch(synthetic_image(), default_params(rootpower=5, rootpower2=1))
        keys = [snapshot.key for snapshot in result.histograms]
        self.assertEqual(
            keys[0:5],
            [
                "input",
                "initial_sky_1_analysis",
                "initial_sky_1_adjusted",
                "initial_sky_2_analysis",
                "initial_sky_2_adjusted",
            ],
        )
        self.assertIn("root_2_before_sky", keys)
        self.assertLess(keys.index("root_1_before_sky"), keys.index("root_2_before_sky"))
        self.assertEqual(keys[-2:], ["color_recovery", "final"])
        self.assertIn("Root stretch pass 2: power 5", result.log)

    def test_histogram_capture_does_not_change_pixels_or_log(self):
        image = synthetic_image()
        with_histograms = rnc.apply_rnc_stretch(image, default_params())
        without_histograms = rnc.apply_rnc_stretch(image, default_params(), capture_histograms=False)
        np.testing.assert_array_equal(with_histograms.rgb, without_histograms.rgb)
        self.assertEqual(with_histograms.log, without_histograms.log)
        self.assertEqual(without_histograms.histograms, ())

    def test_alternative_stretches_and_s_curve_emit_named_stages(self):
        image = synthetic_image()
        for stretch_type, prefix in (("asinh", "asinh_1"), ("log", "log_1")):
            with self.subTest(stretch_type=stretch_type):
                result = rnc.apply_rnc_stretch(
                    image,
                    default_params(stretch_type=stretch_type, rootiter=1, asinh_iter=1, log_iter=1),
                )
                keys = [snapshot.key for snapshot in result.histograms]
                self.assertIn(f"{prefix}_before_sky", keys)

        result = rnc.apply_rnc_stretch(image, default_params(rootiter=1, s_curve=2))
        keys = [snapshot.key for snapshot in result.histograms]
        self.assertIn("s_curve_before_sky", keys)
        self.assertIn("s-curve_sky_1_analysis", keys)


class SirilApplyTests(unittest.TestCase):
    def test_apply_target_requires_same_active_pixels_and_current_source(self):
        preview = loaded_source()
        active = loaded_source()
        self.assertTrue(rnc.images_match_for_apply(preview, active))
        active.preview_input[0, 0, 0] = 0.2
        self.assertFalse(rnc.images_match_for_apply(preview, active))
        local_preview = loaded_source()
        local_preview.source_is_current = False
        self.assertFalse(rnc.images_match_for_apply(local_preview, loaded_source()))

    def test_float_siril_output_normalizes_dn_scale_instead_of_clipping(self):
        source = loaded_source("float32")
        rgb = np.full((4, 5, 3), 32767.5, dtype=np.float32)
        converted = rnc.result_rgb_to_siril_data(rgb, source)
        self.assertEqual(converted.shape, (3, 4, 5))
        self.assertEqual(converted.dtype, np.float32)
        np.testing.assert_allclose(converted, 0.5, atol=1e-6)

    def test_uint16_siril_output_is_planar_and_preserves_dn(self):
        source = loaded_source("uint16")
        rgb = np.full((4, 5, 3), 12345.0, dtype=np.float32)
        converted = rnc.result_rgb_to_siril_data(rgb, source)
        self.assertEqual(converted.shape, (3, 4, 5))
        self.assertEqual(converted.dtype, np.uint16)
        self.assertTrue(np.all(converted == 12345))

    def test_commit_orders_lock_undo_write_and_readback(self):
        source = loaded_source("float32")

        class FakeSiril:
            def __init__(self):
                self.events = []
                self.data = None

            @contextlib.contextmanager
            def image_lock(self):
                self.events.append("lock-enter")
                yield
                self.events.append("lock-exit")

            def undo_save_state(self, label):
                self.events.append(("undo", label))

            def set_image_pixeldata(self, data):
                self.events.append("write")
                self.data = data.copy()

            def get_image(self, with_pixels=True, preview=False):
                self.events.append("readback")
                return SimpleNamespace(data=self.data.copy())

        fake = FakeSiril()
        rgb = np.full((4, 5, 3), 0.25, dtype=np.float32)
        rnc.write_result_to_siril(fake, source, rgb)
        self.assertEqual(
            fake.events,
            ["lock-enter", ("undo", "RNC color stretch"), "write", "lock-exit", "readback"],
        )

    def test_readback_mismatch_is_detected(self):
        expected = np.zeros((3, 4, 5), dtype=np.float32)
        actual = expected.copy()
        actual.reshape(-1)[0] = 1.0
        with self.assertRaisesRegex(ValueError, "do not match"):
            rnc.verify_siril_pixeldata(expected, actual)


if __name__ == "__main__":
    unittest.main()
