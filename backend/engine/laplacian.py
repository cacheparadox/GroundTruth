"""
Laplacian High-Pass + Anomaly Mask — Project GroundTruth
=========================================================
The Laplacian filter is a second-order derivative operator that isolates
high-frequency components (edges, texture, noise) from an image.

For forensic purposes:
  • OVER-SHARPENED pixels (|Lap| >> threshold): indicate aggressive post-
    processing, typical of AI generators that apply sharpening to compensate
    for diffusion blurring, or JPEG over-sharpening artifacts at edit seams.
  • SUSPICIOUSLY FLAT regions (|Lap| << threshold AND local variance ≈ 0):
    indicate AI-generated smooth sky/background regions that lack the natural
    micro-texture noise found in real sensor captures.

Pipeline:
  1. Convert to grayscale float.
  2. Apply 3×3 Laplacian kernel via scipy.ndimage.convolve.
  3. Compute 9×9 local variance of the Laplacian response.
  4. Flag pixels: over-sharpened OR suspiciously flat.
  5. Generate RGBA overlay mask (neon red on desaturated original).
"""

import numpy as np
from scipy.ndimage import convolve, uniform_filter
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 3×3 Laplacian kernel (second-order Laplacian of Gaussian approximation)
LAPLACIAN_KERNEL = np.array([
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0],
], dtype=np.float32)

# Raised thresholds to reduce false positives on real images:
# • THRESH_HIGH raised: real sharp edges (furniture, fabric, strong JPEG) can
#   easily exceed 25; AI-specific over-sharpening is at much higher values.
# • THRESH_LOW lowered: only flag truly zero-response flat regions.
# • LUMA_GATE: genuinely dark pixels (shadows, vignetting, night scenes) are
#   NOT suspicious flat regions — they are just dark. Skip them entirely.
THRESH_HIGH = 55.0   # |Lap| above this → over-sharpened (was 45)
THRESH_LOW  = 1.0    # |Lap| below this → candidate flat region (was 2)
VAR_THRESH  = 0.6    # local variance below this → confirmed flat (was 0.8)
LUMA_GATE   = 18.0   # pixels darker than this are legitimately dark, not AI flat
VAR_WINDOW  = 9      # local variance window size


def analyze_laplacian(img_array: np.ndarray) -> dict:
    """
    Run Laplacian analysis and build anomaly mask.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict with keys:
        lap_response    : np.ndarray (H, W) — |Laplacian| response
        anomaly_mask    : np.ndarray (H, W) — bool, True = flagged pixel
        lap_score       : float             — 0–100, higher = more organic
        anomaly_ratio   : float             — fraction of image flagged
        mean_response   : float             — mean |Lap| response
        pixel_variance  : float             — global pixel intensity variance
        edge_divergence : float             — mean edge strength (Lap mean)
        image_b64       : str               — base64 PNG of annotated overlay
    """
    h, w = img_array.shape[:2]

    # ── 1. Grayscale ──────────────────────────────────────────────
    gray = (
        0.2126 * img_array[:, :, 0]
        + 0.7152 * img_array[:, :, 1]
        + 0.0722 * img_array[:, :, 2]
    ).astype(np.float32)

    # ── 2. Laplacian convolution ──────────────────────────────────
    lap = convolve(gray, LAPLACIAN_KERNEL, mode="reflect")
    lap_abs = np.abs(lap)                          # (H, W)

    # ── 3. Local variance of Laplacian response ───────────────────
    lap_sq_mean = uniform_filter(lap_abs ** 2, size=VAR_WINDOW)
    lap_mean    = uniform_filter(lap_abs,      size=VAR_WINDOW)
    local_var   = lap_sq_mean - lap_mean ** 2    # E[X²] - E[X]²
    local_var   = np.maximum(local_var, 0.0)

    # ── 4. Anomaly flagging ────────────────────────────────────────────────
    over_sharp = lap_abs > THRESH_HIGH

    # Flat-region detection with luminance gate:
    # Only flag as suspiciously flat if the pixel is actually bright enough
    # to be expected to contain texture. Dark pixels (shadow, vignetting,
    # low-light scenes) legitimately have low Laplacian response.
    luma_ok    = gray > LUMA_GATE          # pixel is bright enough to matter
    flat_region = (lap_abs < THRESH_LOW) & (local_var < VAR_THRESH) & luma_ok

    anomaly_mask = over_sharp | flat_region      # (H, W) bool

    # ── 5. Metrics ────────────────────────────────────────────────
    total_px      = h * w
    flagged       = int(np.sum(anomaly_mask))
    anomaly_ratio = flagged / total_px

    mean_response   = float(np.mean(lap_abs))
    pixel_variance  = float(np.var(gray))         # global intensity variance
    edge_divergence = float(np.mean(lap_abs[lap_abs > 1.0])) if np.any(lap_abs > 1.0) else 0.0

    # Score: moderate mean response + low anomaly ratio → organic
    # Very low or very high response, or high anomaly ratio → synthetic
    response_score = max(0.0, min(100.0,
        100.0 - abs(mean_response - 12.0) * 1.5))   # ideal ~12, gentler slope

    # Halved penalty vs original (was ×500): real images can have 10–20% flagged
    # due to legitimate sharp edges and JPEG noise at high ISO.
    anomaly_score = max(0.0, 100.0 - anomaly_ratio * 250.0)

    lap_score = float(0.40 * response_score + 0.60 * anomaly_score)
    lap_score = min(100.0, max(0.0, lap_score))

    # ── 6. Visualisation ─────────────────────────────────────────
    image_b64 = _render_anomaly_overlay(img_array, anomaly_mask, lap_abs,
                                        flagged, total_px)

    return {
        "lap_response":   lap_abs,
        "anomaly_mask":   anomaly_mask,
        "lap_score":      round(lap_score, 2),
        "anomaly_ratio":  round(anomaly_ratio, 5),
        "flagged_pixels": flagged,
        "total_pixels":   total_px,
        "mean_response":  round(mean_response, 3),
        "pixel_variance": round(pixel_variance, 2),
        "edge_divergence":round(edge_divergence, 3),
        "image_b64":      image_b64,
    }


def _render_anomaly_overlay(
    img_array: np.ndarray,
    anomaly_mask: np.ndarray,
    lap_abs: np.ndarray,
    flagged: int,
    total: int,
) -> str:
    """Render neon-red anomaly mask overlaid on desaturated original → base64 PNG."""
    h, w = img_array.shape[:2]

    # Desaturate original
    gray_rgb = np.stack([
        0.2126*img_array[:,:,0] + 0.7152*img_array[:,:,1] + 0.0722*img_array[:,:,2]
    ] * 3, axis=-1).astype(np.uint8)

    # Blend: desaturate + slight darkening
    base = (gray_rgb * 0.55).astype(np.uint8)

    # RGBA output
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = base
    rgba[:, :, 3]  = 255

    # Neon-red mask overlay
    # Alpha proportional to Laplacian response for smooth gradient
    strength = np.clip(lap_abs / 60.0, 0, 1)
    alpha    = (anomaly_mask * (100 + 120 * strength)).astype(np.uint8)

    rgba[anomaly_mask, 0] = 255
    rgba[anomaly_mask, 1] = 0
    rgba[anomaly_mask, 2] = 50
    rgba[anomaly_mask, 3] = alpha[anomaly_mask]

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#0a0d11")
    ax.set_facecolor("#0a0d11")

    ax.imshow(rgba, interpolation="nearest", aspect="auto")
    ax.axis("off")

    ratio_pct = flagged / total * 100
    color = "#FF0032" if ratio_pct > 5.0 else "#00FF88"
    ax.set_title(
        f"ANOMALY MASK  //  FLAGGED: {flagged:,} px  //  {ratio_pct:.2f}% of image",
        color=color, fontsize=7, pad=4, fontfamily="monospace",
    )

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
