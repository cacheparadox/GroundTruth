"""
Color Physics & Channel Correlation Analysis — Project GroundTruth
===================================================================
Analyzes the inter-channel physical correlation between the Red, Green, and Blue channels,
and evaluates color distribution entropy by converting the image to CIE L*a*b* space.
Real camera sensors have distinct inter-channel correlation signatures due to color filter arrays
(CFAs) and physical light distribution. Synthetic generators often exhibit unnaturally high
correlations (due to neural smoothing) or pixel-level chromatic decoupling.
"""

import numpy as np
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze_color(img_array: np.ndarray) -> dict:
    """
    Analyze color channel correlation and CIE L*a*b* chroma entropy.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 RGB array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict with color score, Pearson correlations, chroma entropy, and heatmap visual.
    """
    # 1. Flatten channels
    r_flat = img_array[:, :, 0].flatten()
    g_flat = img_array[:, :, 1].flatten()
    b_flat = img_array[:, :, 2].flatten()

    # 2. Pearson Correlation Coefficients
    corr_matrix = np.corrcoef([r_flat, g_flat, b_flat])
    r_g = float(corr_matrix[0, 1])
    g_b = float(corr_matrix[1, 2])
    r_b = float(corr_matrix[0, 2])
    mean_corr = (r_g + g_b + r_b) / 3.0

    # 3. Convert to CIE L*a*b* Space
    # A. Linearize sRGB and convert to XYZ
    rgb_norm = img_array / 255.0
    mask = rgb_norm > 0.04045
    rgb_lin = np.zeros_like(rgb_norm)
    rgb_lin[mask] = ((rgb_norm[mask] + 0.055) / 1.055) ** 2.4
    rgb_lin[~mask] = rgb_norm[~mask] / 12.92

    # XYZ Conversion matrix (D65 standard)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)
    xyz = np.dot(rgb_lin, M.T)

    # B. XYZ to Lab
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x_t = xyz[:, :, 0] / xn
    y_t = xyz[:, :, 1] / yn
    z_t = xyz[:, :, 2] / zn

    def f_t(t):
        mask_t = t > 0.008856
        res = np.zeros_like(t)
        res[mask_t] = t[mask_t] ** (1.0 / 3.0)
        res[~mask_t] = 7.787 * t[~mask_t] + 16.0 / 116.0
        return res

    fx = f_t(x_t)
    fy = f_t(y_t)
    fz = f_t(z_t)

    # We extract the chromatic channels: a (red-green axis), b (blue-yellow axis)
    a_chan = 500.0 * (fx - fy)
    b_chan = 200.0 * (fy - fz)

    # 4. Compute Chroma Entropy (64 bins histogram)
    hist_a, _ = np.histogram(a_chan, bins=64, density=True)
    hist_b, _ = np.histogram(b_chan, bins=64, density=True)
    
    entropy_a = float(-np.sum(hist_a * np.log2(hist_a + 1e-8)))
    entropy_b = float(-np.sum(hist_b * np.log2(hist_b + 1e-8)))
    chroma_entropy = (entropy_a + entropy_b) / 2.0

    # 5. Calibration & Scoring
    # Natural images: channel correlation is high but not identical, typically 0.88 - 0.98.
    # If correlation is near 1.0 (e.g. 0.999), it is highly regular (AI / filtered).
    # If correlation is low (< 0.82), it indicates color misalignment or noisy synthesis.
    corr_penalty = 0.0
    if mean_corr < 0.86:
        corr_penalty = (0.86 - mean_corr) * 250.0
    elif mean_corr > 0.997:
        corr_penalty = (mean_corr - 0.997) * 1200.0

    # Natural chroma entropy: typically 2.2 - 4.6
    entropy_penalty = 0.0
    if chroma_entropy < 1.8:
        entropy_penalty = (1.8 - chroma_entropy) * 40.0
    elif chroma_entropy > 4.8:
        entropy_penalty = (chroma_entropy - 4.8) * 30.0

    color_score = 100.0 - (corr_penalty + entropy_penalty)
    color_score = float(min(100.0, max(0.0, color_score)))

    # 6. Generate base64 visualization (Pearson Correlation Matrix Map)
    image_b64 = _render_correlation_matrix(corr_matrix, mean_corr, chroma_entropy)

    return {
        "color_score": round(color_score, 2),
        "r_g_corr": round(r_g, 4),
        "g_b_corr": round(g_b, 4),
        "r_b_corr": round(r_b, 4),
        "chroma_entropy": round(chroma_entropy, 4),
        "image_b64": image_b64,
    }

def _render_correlation_matrix(matrix: np.ndarray, mean_corr: float, entropy: float) -> str:
    """Render a nice 3x3 heatmap of the RGB channel correlations."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("#0a0d11")
    ax.set_facecolor("#0a0d11")

    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=0.8, vmax=1.0)
    
    # Grid ticks and labels
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Red", "Green", "Blue"], color="#5a6a7e", fontsize=8)
    ax.set_yticklabels(["Red", "Green", "Blue"], color="#5a6a7e", fontsize=8)
    
    ax.tick_params(color="#5a6a7e")
    for spine in ax.spines.values():
        spine.set_color("#5a6a7e")

    # Add numeric labels to cells
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", 
                    color="black" if matrix[i, j] > 0.9 else "#5a6a7e", 
                    fontsize=8, fontfamily="monospace")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#5a6a7e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#5a6a7e", fontsize=7)
    cbar.set_label("Correlation Coefficient", color="#5a6a7e", fontsize=7)

    ax.set_title(
        f"RGB INTER-CHANNEL CORRELATION\nMean Corr: {mean_corr:.4f} | Lab Entropy: {entropy:.3f}",
        color="#00FF88", fontsize=8, pad=8,
        fontfamily="monospace",
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
