"""
FFT Frequency Domain Analysis — Project GroundTruth
=====================================================
AI generators (GANs, diffusion models) often introduce periodic artifacts
in the frequency domain caused by:

  • Transposed convolution "checkerboard" (stride-2 upsampling in GANs)
  • Pixel-shuffle periodic patterns (ESRGAN, RealESRGAN upscaling)
  • Bilinear/bicubic upsampling followed by convolution → aliasing spurs
  • Diffusion model noise schedules leaving harmonic signatures

These appear as bright, periodic dots or starburst arms in the 2D FFT
magnitude spectrum — especially in mid-to-high frequency bands away from
the central DC component.

Pipeline:
  1. Convert to grayscale float.
  2. Apply Hann window (reduces spectral leakage).
  3. numpy.fft.fft2 + fftshift to centre DC.
  4. log(1 + |magnitude|) for display.
  5. Radial bandpass: exclude DC blob (r < 8px) and outer noise ring.
  6. Detect local maxima in the bandpass band.
  7. Count peaks → compute FFT anomaly score.
"""

import numpy as np
from scipy.ndimage import maximum_filter, label
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# Radial exclusion zones (in fraction of half-image size)
DC_EXCLUDE_FRAC      = 0.04   # ignore central DC blob within this radius
OUTER_EXCLUDE_FRAC   = 0.90   # ignore very high-freq outer ring
PEAK_NEIGHBORHOOD    = 11     # pixels — local max search window
PEAK_THRESHOLD_STD   = 3.5    # σ above mean to count as a peak
ORGANIC_PEAK_LIMIT   = 12     # lower from 20 to 12
ANGULAR_BINS         = 16     # divide 360° into bins
ANGULAR_REG_THRESH   = 0.50   # lower from 0.55 to 0.50


def analyze_fft(img_array: np.ndarray) -> dict:
    """
    Run FFT2 frequency analysis on an RGB image.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict with keys:
        spectrum        : np.ndarray (H, W) — log magnitude spectrum
        peak_count      : int               — anomalous peaks detected
        peak_coords     : list[tuple]       — (y, x) of detected peaks
        fft_score       : float             — 0–100, higher = more organic
        dominant_freq   : float             — dominant periodic frequency (px)
        image_b64       : str               — base64 PNG of annotated spectrum
    """
    h, w = img_array.shape[:2]

    # ── 1. Grayscale ──────────────────────────────────────────────
    gray = (
        0.2126 * img_array[:, :, 0]
        + 0.7152 * img_array[:, :, 1]
        + 0.0722 * img_array[:, :, 2]
    ).astype(np.float64)

    # ── 2. Hann window to reduce leakage ─────────────────────────
    hann_y = np.hanning(h)
    hann_x = np.hanning(w)
    window = np.outer(hann_y, hann_x)
    gray_windowed = gray * window

    # ── 3. FFT2 ───────────────────────────────────────────────────
    fft = np.fft.fft2(gray_windowed)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    spectrum = np.log1p(magnitude)           # log scale for display

    # ── 4. Radial mask ────────────────────────────────────────────
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    radii = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    r_max = min(h, w) / 2

    dc_r     = DC_EXCLUDE_FRAC    * r_max
    outer_r  = OUTER_EXCLUDE_FRAC * r_max

    bandpass_mask = (radii > dc_r) & (radii < outer_r)
    spectrum_bp   = spectrum * bandpass_mask   # zero outside band

    # ── 5. Local peak detection ───────────────────────────────────
    local_max = maximum_filter(spectrum_bp, size=PEAK_NEIGHBORHOOD)
    is_local_max = (spectrum_bp == local_max) & bandpass_mask

    bp_values  = spectrum_bp[bandpass_mask]
    mean_bp    = np.mean(bp_values)
    std_bp     = np.std(bp_values) + 1e-8
    threshold  = mean_bp + PEAK_THRESHOLD_STD * std_bp

    peak_mask  = is_local_max & (spectrum_bp > threshold)
    peak_ys, peak_xs = np.where(peak_mask)
    peak_coords = list(zip(peak_ys.tolist(), peak_xs.tolist()))

    # Compute dominant spatial frequency from brightest peak
    dominant_freq = 0.0
    if peak_coords:
        strengths  = [spectrum[py, px] for py, px in peak_coords]
        top_idx    = int(np.argmax(strengths))
        top_py, top_px = peak_coords[top_idx]
        r_peak     = float(radii[top_py, top_px])
        if r_peak > 0:
            dominant_freq = round(min(h, w) / r_peak, 1)

    peak_count = len(peak_coords)
    
    # Calculate peak energy ratio: sum of peak amplitudes compared to total amplitude in the bandpass band
    peak_energy_ratio = 0.0
    if peak_count > 0:
        peak_sum = sum(spectrum[py, px] for py, px in peak_coords)
        bp_sum = np.sum(spectrum_bp)
        peak_energy_ratio = float(peak_sum / (bp_sum + 1e-8))

    # ── 6. Angular regularity analysis ───────────────────────────
    angular_regularity = 0.0
    if peak_count >= 4:
        angles = []
        cy_f, cx_f = h / 2.0, w / 2.0
        for py, px in peak_coords:
            dy = py - cy_f
            dx = px - cx_f
            angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
            angles.append(angle_deg)
        bin_size = 360.0 / ANGULAR_BINS
        occupied_bins = set(int(a / bin_size) % ANGULAR_BINS for a in angles)
        angular_regularity = len(occupied_bins) / ANGULAR_BINS

    # ── 7. Score ──────────────────────────────────────────────────
    # If the peak energy ratio is high, the periodic spikes are concentrated and dominant (typical of structured scene texture).
    # AI upsampling grids tend to produce many lower-energy peaks dispersed across the bandpass zone (low peak energy ratio).
    if peak_count <= ORGANIC_PEAK_LIMIT:
        count_score = 100.0 - (peak_count / max(1, ORGANIC_PEAK_LIMIT)) * 15.0
    else:
        excess = peak_count - ORGANIC_PEAK_LIMIT
        # If peak energy ratio is high (> 0.05), it is more likely organic pattern/texture, so mitigate penalty
        penalty_factor = 1.0
        if peak_energy_ratio > 0.05:
            penalty_factor = 0.25
        elif peak_energy_ratio > 0.02:
            penalty_factor = 0.50
        count_score = max(0.0, 90.0 - excess * 3.5 * penalty_factor)

    # Angular regularity penalty
    if angular_regularity > ANGULAR_REG_THRESH:
        # Similarly mitigate regularity penalty if peak energy ratio is high
        reg_penalty_factor = 0.3 if peak_energy_ratio > 0.04 else 1.0
        reg_penalty = ((angular_regularity - ANGULAR_REG_THRESH) / (1.0 - ANGULAR_REG_THRESH) * 50.0) * reg_penalty_factor
    else:
        reg_penalty = 0.0

    fft_score = max(0.0, min(100.0, count_score - reg_penalty))

    # ── 7. Visualisation ──────────────────────────────────────────
    image_b64 = _render_spectrum(spectrum, peak_coords, peak_count,
                                 dc_r, outer_r, h, w, cx, cy)

    return {
        "spectrum":         spectrum,
        "peak_count":       peak_count,
        "peak_coords":      peak_coords[:50],   # cap payload
        "fft_score":        round(fft_score, 2),
        "dominant_freq":    dominant_freq,
        "angular_regularity": round(angular_regularity, 4),
        "peak_energy_ratio": round(peak_energy_ratio, 6),
        "image_b64":        image_b64,
    }


def _render_spectrum(
    spectrum: np.ndarray,
    peaks: list,
    peak_count: int,
    dc_r: float, outer_r: float,
    h: int, w: int,
    cx: int, cy: int,
) -> str:
    """Render annotated FFT spectrum PNG → base64."""
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#0a0d11")
    ax.set_facecolor("#0a0d11")

    ax.imshow(
        spectrum,
        cmap="inferno",
        interpolation="nearest",
        aspect="auto",
    )

    # Draw exclusion rings
    for r, ls, color in [
        (dc_r,    "--", "#00FF88"),
        (outer_r, "--", "#FF6A00"),
    ]:
        circle = Circle((cx, cy), r, fill=False, linestyle=ls,
                         edgecolor=color, linewidth=0.7, alpha=0.6)
        ax.add_patch(circle)

    # Mark detected peaks
    if peaks:
        pys, pxs = zip(*peaks)
        ax.scatter(pxs, pys, c="#FF0032", s=12, marker="x",
                   linewidths=0.8, zorder=5, label=f"Peaks: {peak_count}")

    ax.axis("off")
    ax.set_title(
        f"FFT MAGNITUDE SPECTRUM  //  ANOMALOUS PEAKS: {peak_count}",
        color="#FF6A00" if peak_count > ORGANIC_PEAK_LIMIT else "#00FF88",
        fontsize=7, pad=4, fontfamily="monospace",
    )

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
