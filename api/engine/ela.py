"""
Error Level Analysis (ELA) — Project GroundTruth
==================================================
ELA exploits the fact that JPEG compression is lossy and iterative.

When a real camera captures an image, every region of the scene has been
compressed exactly once from its raw sensor data. The compression artifacts
are spatially uniform and correlated with scene content.

When an image is manipulated (or AI-generated), regions that came from
different sources will have different compression histories. Re-encoding
the image at a known quality reveals:
  • Untouched regions: low ELA signal (already near compression floor)
  • Manipulated/AI regions: HIGH ELA signal (re-compression changes them more)

AI-generated images often show unusually UNIFORM ELA (flat signal everywhere)
because the generation process doesn't produce real JPEG compression history,
OR they show high ELA in AI-blended seams.

Pipeline:
  1. Convert working image to RGB PIL.
  2. Save to in-memory BytesIO at JPEG quality=75.
  3. Reload → compute absolute pixel difference (×amplification factor).
  4. Analyse ELA map statistics.
"""

import numpy as np
from PIL import Image
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ELA_QUALITY       = 75    # re-encode quality (lower → stronger signal)
ELA_AMPLIFY       = 15    # amplification factor for display
ELA_SCALE         = 255.0 / ELA_AMPLIFY  # normalisation


def analyze_ela(img_array: np.ndarray) -> dict:
    """
    Run Error Level Analysis on an RGB image array.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict with keys:
        ela_map         : np.ndarray (H, W, 3) — amplified difference map
        ela_gray        : np.ndarray (H, W)    — grayscale ELA
        ela_score       : float                — 0–100, higher = more organic
        ela_mean        : float                — mean ELA signal
        ela_var         : float                — variance of ELA
        ela_uniformity  : float                — 1 = perfectly uniform (AI flag)
        image_b64       : str                  — base64 PNG heatmap
    """
    h, w = img_array.shape[:2]

    # ── 1. Round-trip JPEG ────────────────────────────────────────
    pil_orig = Image.fromarray(img_array.astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    pil_orig.save(buf, format="JPEG", quality=ELA_QUALITY, subsampling=0)
    buf.seek(0)
    pil_reenc = Image.open(buf).convert("RGB")

    orig_arr = np.array(pil_orig, dtype=np.float32)
    reenc_arr = np.array(pil_reenc, dtype=np.float32)

    # ── 2. Difference map ────────────────────────────────────────
    diff = np.abs(orig_arr - reenc_arr)          # (H, W, 3)
    ela_map = np.clip(diff * ELA_AMPLIFY, 0, 255).astype(np.uint8)

    # Grayscale ELA (max across channels — most informative for manipulation)
    ela_gray = np.max(diff, axis=2)              # (H, W)

    # ── 3. Metrics ────────────────────────────────────────────────
    ela_mean  = float(np.mean(ela_gray))
    ela_std   = float(np.std(ela_gray))
    ela_var   = float(np.var(ela_gray))

    # Uniformity: coefficient of variation (std/mean). Very low → uniform → AI
    ela_uniformity = ela_std / (ela_mean + 1e-8)

    # ELA score logic:
    # - Real images: moderate mean ELA (5–25) with HIGH variation (uniformity > 0.8)
    # - AI images: either near-zero mean ELA (uniformity < 0.4) or very high
    #   uniform ELA across the whole image

    # Score based on uniformity: penalise suspiciously low variation
    uniformity_score = min(100.0, ela_uniformity * 80.0)

    # Also penalise near-zero ELA (AI images from diffusion models)
    if ela_mean < 1.5:
        mean_score = 20.0  # very suspicious
    elif ela_mean < 4.0:
        mean_score = 50.0
    else:
        mean_score = min(100.0, ela_mean * 3.0)

    ela_score = float(0.5 * uniformity_score + 0.5 * mean_score)
    ela_score = min(100.0, max(0.0, ela_score))

    # ── 4. Visualisation ─────────────────────────────────────────
    image_b64 = _render_ela_map(ela_gray, ela_mean, ela_uniformity)

    return {
        "ela_map":        ela_map,
        "ela_gray":       ela_gray,
        "ela_score":      round(ela_score, 2),
        "ela_mean":       round(ela_mean, 3),
        "ela_var":        round(ela_var, 3),
        "ela_uniformity": round(ela_uniformity, 4),
        "image_b64":      image_b64,
    }


def _render_ela_map(
    ela_gray: np.ndarray,
    ela_mean: float,
    ela_uniformity: float,
) -> str:
    """Render ELA grayscale map as hot colormap PNG → base64."""
    h, w = ela_gray.shape

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("#0a0d11")
    ax.set_facecolor("#0a0d11")

    # Display with amplification for visibility
    display = np.clip(ela_gray * ELA_AMPLIFY, 0, 255)
    ax.imshow(
        display,
        cmap="hot",
        vmin=0, vmax=255,
        interpolation="nearest",
        aspect="auto",
    )
    ax.axis("off")

    flag = ela_uniformity < 0.5
    color = "#FF0032" if flag else "#00FF88"
    ax.set_title(
        f"ELA MAP (Q{ELA_QUALITY})  //  MEAN={ela_mean:.2f}  //  UNIFORMITY={ela_uniformity:.3f}",
        color=color, fontsize=7, pad=4, fontfamily="monospace",
    )

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
