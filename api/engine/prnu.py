"""
PRNU Noise Extraction — Project GroundTruth
============================================
Photo Response Non-Uniformity (PRNU) analysis.

Real camera sensors have a unique, stochastic per-pixel sensitivity pattern
(the PRNU fingerprint) caused by manufacturing variations. AI-generated images
either lack this entirely or exhibit structured/periodic noise instead of
the expected random Gaussian-like distribution.

Pipeline:
  1. Apply Wiener filter (adaptive least-squares) per RGB channel to estimate
     the "clean" denoised image.
  2. Subtract denoised from original → noise residual (PRNU-like signal).
  3. Compute statistical metrics on the residual to score authenticity.
"""

import numpy as np
from scipy.signal import wiener
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


WIENER_KERNEL = 5   # 5×5 Wiener filter window


def extract_prnu(img_array: np.ndarray) -> dict:
    """
    Extract the PRNU noise residual from an RGB image array.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict with keys:
        noise_map   : np.ndarray (H, W, 3) — noise residual per channel
        gray_noise  : np.ndarray (H, W)    — luminance noise residual
        prnu_score  : float                — 0–100, higher = more organic
        noise_power : float                — RMS of noise residual
        noise_var   : float                — variance of noise residual
        spatial_corr: float                — measure of spatial correlation
                                             (high = structured/AI noise)
        image_b64   : str                  — base64 PNG of the noise map vis
    """
    h, w = img_array.shape[:2]
    img_f = img_array.astype(np.float32)

    # ── 1. Wiener filter per channel ──────────────────────────────
    denoised = np.zeros_like(img_f)
    for c in range(3):
        denoised[:, :, c] = wiener(img_f[:, :, c], mysize=WIENER_KERNEL)

    # ── 2. Noise residual ─────────────────────────────────────────
    noise_map = img_f - denoised  # shape (H, W, 3)

    # ── 3. Luminance noise ────────────────────────────────────────
    gray_noise = (
        0.2126 * noise_map[:, :, 0]
        + 0.7152 * noise_map[:, :, 1]
        + 0.0722 * noise_map[:, :, 2]
    )

    # ── 4. Metrics ────────────────────────────────────────────────
    noise_power = float(np.sqrt(np.mean(gray_noise ** 2)))       # RMS
    noise_var   = float(np.var(gray_noise))
    noise_std   = float(np.std(gray_noise))

    # Spatial correlation: compare local patch means
    # A real camera gives white-noise-like residual → low spatial correlation.
    # AI images tend to have structured residual → higher spatial correlation.
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(gray_noise, size=15)
    spatial_corr = float(np.std(smoothed) / (noise_std + 1e-8))
    # spatial_corr near 0 → random (organic), near 1 → structured (synthetic)

    # Kurtosis of noise distribution (real sensors → near-Gaussian, kurtosis ~3)
    flat = gray_noise.flatten()
    mean_n = np.mean(flat)
    std_n  = np.std(flat) + 1e-8
    kurtosis = float(np.mean(((flat - mean_n) / std_n) ** 4))

    # Score: lower spatial_corr and kurtosis near 3 → organic
    # spatial_corr score: 0.0 → 100 (organic), 0.5+ → 0 (synthetic)
    sc_score = max(0.0, 100.0 - spatial_corr * 250.0)

    # Kurtosis score: penalise deviation from ~3
    kurt_dev = abs(kurtosis - 3.0)
    kurt_score = max(0.0, 100.0 - kurt_dev * 8.0)

    # Combined PRNU score
    prnu_score = float(0.6 * sc_score + 0.4 * kurt_score)
    prnu_score = min(100.0, max(0.0, prnu_score))

    # ── 5. Visualisation ──────────────────────────────────────────
    image_b64 = _render_noise_map(gray_noise, noise_power)

    return {
        "noise_map":    noise_map,
        "gray_noise":   gray_noise,
        "prnu_score":   round(prnu_score, 2),
        "noise_power":  round(noise_power, 4),
        "noise_var":    round(noise_var, 4),
        "spatial_corr": round(spatial_corr, 4),
        "kurtosis":     round(kurtosis, 4),
        "image_b64":    image_b64,
    }


def _render_noise_map(gray_noise: np.ndarray, noise_power: float) -> str:
    """Render the noise residual as a diverging colormap PNG → base64."""
    h, w = gray_noise.shape

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#0a0d11")
    ax.set_facecolor("#0a0d11")

    # Symmetric clamp around ±3σ for display
    sigma = np.std(gray_noise) + 1e-8
    vmax  = 3.0 * sigma
    im = ax.imshow(
        gray_noise,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.axis("off")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#5a6a7e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#5a6a7e", fontsize=6)
    cbar.set_label("Noise Residual (DN)", color="#5a6a7e", fontsize=7)

    # Overlay stats text
    ax.set_title(
        f"PRNU NOISE RESIDUAL  //  RMS={noise_power:.3f}  //  ±3σ={3*sigma:.2f}",
        color="#00FF88", fontsize=7, pad=4,
        fontfamily="monospace",
    )

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
