# Clarkvision-style natural color in Siril

This is the validated linear color-calibration workflow for the **Player One Poseidon-C Pro (IMX571) with the Astronomik L-3 filter**. It is Clark-inspired, not a claim of scientifically exact color: daylight white balance precedes a fitted ColorChecker correction, additive sky glow is removed while the data are linear, and a linked display stretch preserves the resulting channel balance.

The fixed workflow ends at `result_natural_color_linear.fit`. Stretch that image manually for the final presentation. Siril's linked autostretch is the acceptance preview, not the prescribed final nonlinear stretch.

## Provisional camera/filter calibration

The tested one-step matrix combines daylight white balance and the full 3×3 sensor correction:

```text
ccm 2.2755289 -0.2831013 -0.2330648 \
   -0.4281588  1.6776502 -0.7212311 \
    0.0501242 -0.5005715  2.5519763
```

Siril requires the nine coefficients on one command line. The backslashes above only make the matrix easier to read; do not include them when pasting into Siril.

This matrix is provisional. The calibration chart's white patch clips, and the upper neutral patches show illumination or glare variation. A lower-exposure, evenly illuminated, glare-free daylight recapture would provide a higher-confidence final matrix. Do not reuse these coefficients for a different camera sensor, filter, or meaningfully different optical spectral response.

## Linear Siril sequence

Run the following after normal calibration, registration, and stacking. The input must still be linear and must not have undergone channel equalization or background neutralization.

```text
set32bits
load result.fit

# Daylight WB and full 3×3 sensor correction in linear data
ccm 2.2755289 -0.2831013 -0.2330648 -0.4281588 1.6776502 -0.7212311 0.0501242 -0.5005715 2.5519763
icc_assign sRGBlinear

# Remove additive sky glow after CCM so Siril leaves a neutral pedestal
subsky -rbf -samples=20 -tolerance=1.0 -smooth=0.5

save result_natural_color_linear
autostretch -linked
```

`icc_assign sRGBlinear` labels the corrected linear RGB values with the appropriate profile; it does not stretch them. Save before `autostretch` so the durable result remains linear.

The order of `ccm` and `subsky` is important. Siril's `subsky` leaves an equal RGB pedestal. If the one-step matrix is applied afterward, its daylight-WB gains transform that neutral pedestal into a colored one. Applying the CCM first lets `subsky` estimate the sky in corrected color and leave its pedestal neutral.

### Why the image can look blue immediately after the CCM

The CCM corrects the camera/filter's **multiplicative** spectral response; it does not remove the **additive** sky foreground already present in every pixel. The one-step matrix includes daylight white-balance gains and channel mixing, so it also transforms that foreground. In this dataset the transformed sky term is blue-heavy and, because it dominates the faint linear image, a linked display stretch makes the whole frame appear blue.

RBF background extraction estimates that slowly varying additive term independently in the corrected RGB data and subtracts it. Siril then retains an equal-channel safety pedestal, so the linked preview becomes neutral while the calibrated color differences in stars and nebulae remain. The disappearance of the blue wash therefore does not mean that background extraction has undone the CCM: the two operations address different components of the signal. If blue nebulosity or blue stars were included in the background model, however, extraction could remove real color, so inspect the samples/model and keep them off astronomical structure.

On the supplied `test_img/result.fit`, Siril 1.4.4 completed this sequence with finite output. After RBF subtraction, the linear channel medians were 0.019152, 0.019178, and 0.019178—agreement comfortably within 1%. The linked preview showed a neutral dark sky, a warm stellar field, pink M8/M20 emission, and blue reflection nebulosity around M20.

## Operations to avoid during linear calibration

- Do not use unlinked autostretch to judge the result. Siril warns that unlinked autostretch changes a calibrated white balance; use `autostretch -linked`.
- Do not stack with `-rgb_equal`.
- Do not align the R, G, and B histograms independently.
- Do not neutralize the background before the CCM.
- Do not apply SCNR or saturation enhancement during this linear calibration stage.

These operations can force the channels toward a visually convenient result while undoing or obscuring the daylight/ColorChecker calibration. They may be deliberate creative choices later, after the calibrated linear master has been saved.

See the [Siril 1.4.4 command reference](https://siril.readthedocs.io/en/stable/Commands.html) for the command behavior and autostretch warning.

## Rebuilding the matrix from a chart

Capture the post-November-2014 24-patch ColorChecker in direct daylight with the same camera and filter. Keep the chart linear, lower the exposure enough that every patch is unclipped, illuminate it evenly, and avoid specular glare. Prepare the camera frame in Siril with:

```text
set32bits
calibrate_single <chart.fits> -bias="=8*$OFFSET" -debayer -prefix=<output-prefix>
```

Open the resulting color FITS in `make_forward_matrix.py`, crop precisely to the 6×4 patch grid, select the chart orientation, and use these defaults:

- Reference data: **After Nov 2014**
- WB patch: **Neutral 5**
- WB reference: **G**

The builder reads TIFF data through `tifffile`, so 16-bit samples are not reduced to 8 bits. It handles all rotations and mirror/transpose orientations, reports each patch's dE00, and automatically excludes a patch when more than 1% of its sampled pixels clip. It white-balances first, normalizes the samples against Neutral 5, and normalizes the fitted forward matrix so its luminance (Y) row sums to one. Its output separates:

1. the daylight white-balance vector;
2. the CCM to apply after that WB; and
3. the combined one-step Siril `ccm` matrix.

Use either the separate WB plus after-WB CCM or the one-step matrix, never both. No gray-patch brightness scaling is embedded in these matrices.

The conceptual ordering follows Clark's [sensor calibration and color sequence](https://clarkvision.com/articles/sensor-calibration-and-color/) and his [M8/M20 natural-color example](https://clarkvision.com/articles/astrophotography-rnc-color-stretch-m8%2Bm20/). Clark's broader workflow and Siril's tools are not identical, which is why this document calls the result Clarkvision-style rather than an exact reproduction.
