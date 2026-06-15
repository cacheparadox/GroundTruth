"""
Discrete Cosine Transform (DCT) Block Analysis — Project GroundTruth
==================================================================
Performs block-wise 2D DCT across 8x8 image blocks using a fully vectorized NumPy pipeline.
Computes frequency-domain statistics and measures the compliance of AC coefficients
with Benford's Law (first digit distribution), which is an industry-standard indicator
of JPEG double-compression, pixel-level manipulation, or synthetic generation.
"""

import numpy as np
from scipy.fftpack import dct
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze_dct(img_array: np.ndarray) -> dict:
    """
    Perform vectorized 8x8 block DCT analysis.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict with DCT score, variance statistics, Benford's Law deviation, and visualization.
    """
    # 1. Convert to Luminance (grayscale)
    if len(img_array.shape) == 3:
        gray = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
    else:
        gray = img_array.copy()

    h, w = gray.shape
    # Drop edge pixels that don't fit into a clean 8x8 grid
    gray = gray[:h - h % 8, :w - w % 8]
    h, w = gray.shape

    if h < 8 or w < 8:
        return {
            "dct_score": 100.0,
            "benford_deviation": 0.0,
            "ac_high_low_ratio": 0.0,
            "image_b64": "",
        }

    # 2. Vectorized block reshaping (no copy in memory)
    # Shape: (h//8, 8, w//8, 8) -> transpose to (h//8, w//8, 8, 8)
    blocks = gray.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)

    # 3. 2D DCT across the last two axes (8x8 blocks)
    # type=2 is the standard DCT-II
    dct_blocks = dct(dct(blocks, axis=-1, type=2, norm='ortho'), axis=-2, type=2, norm='ortho')

    # 4. Variance map of coefficients across all blocks
    # var_map shape is (8, 8), representing the variance of each coefficient frequency.
    var_map = np.var(dct_blocks, axis=(0, 1))

    # Calculate high vs low AC variance ratio
    # Natural images decay rapidly; synthetic or manipulated images often have flatter energy.
    low_freq_mask = np.zeros((8, 8), dtype=bool)
    high_freq_mask = np.zeros((8, 8), dtype=bool)
    for u in range(8):
        for v in range(8):
            if 0 < (u + v) < 4:
                low_freq_mask[u, v] = True
            elif (u + v) >= 5:
                high_freq_mask[u, v] = True

    mean_low_var = np.mean(var_map[low_freq_mask])
    mean_high_var = np.mean(var_map[high_freq_mask])
    ac_ratio = mean_high_var / (mean_low_var + 1e-8)

    # 5. Benford's Law analysis of first significant digits of AC coefficients
    # Select all AC coefficients (ignore the DC coefficient at (0,0))
    ac_coeffs = dct_blocks[:, :, :, :]
    # Make a mask to extract all AC elements
    ac_mask = np.ones((8, 8), dtype=bool)
    ac_mask[0, 0] = False
    ac_values = ac_coeffs[:, :, ac_mask].flatten()

    # Filter out zero coefficients and tiny noise floor values
    ac_values_abs = np.abs(ac_values)
    ac_values_abs = ac_values_abs[ac_values_abs >= 1e-3]

    benford_deviation = 0.0
    dct_score = 100.0

    if len(ac_values_abs) > 100:
        # Extract the first significant digit: e.g. 0.035 -> 3, 42.1 -> 4
        log_ac = np.log10(ac_values_abs)
        first_digits = np.floor(ac_values_abs / (10 ** np.floor(log_ac))).astype(int)
        # Handle rounding/float issues keeping digits between 1 and 9
        first_digits = np.clip(first_digits, 1, 9)

        # Count occurrences of each digit 1-9
        counts = np.bincount(first_digits, minlength=10)[1:10]
        empirical_dist = counts / (np.sum(counts) + 1e-8)

        # Standard Benford's distribution
        benford_dist = np.log10(1.0 + 1.0 / np.arange(1, 10))

        # Chi-square statistic/divergence metric
        benford_deviation = float(np.sum((empirical_dist - benford_dist) ** 2 / benford_dist))

        # Calibrate DCT score based on Benford deviation and AC ratio
        # Organic Benford deviation is usually < 0.02. Manipulated/synthetic goes to 0.06 - 0.15+.
        # High high-to-low AC ratio is also penalized.
        benford_penalty = max(0.0, (benford_deviation - 0.015) * 600.0)
        ac_ratio_penalty = max(0.0, (ac_ratio - 0.04) * 800.0)
        dct_score = 100.0 - (benford_penalty + ac_ratio_penalty)
        dct_score = float(min(100.0, max(0.0, dct_score)))

    # 6. Generate base64 visualization
    image_b64 = _render_dct_visualization(var_map, benford_deviation, ac_ratio)

    return {
        "dct_score": round(dct_score, 2),
        "benford_deviation": round(benford_deviation, 5),
        "ac_high_low_ratio": round(ac_ratio, 5),
        "image_b64": image_b64,
    }

def _render_dct_visualization(var_map: np.ndarray, benford_dev: float, ac_ratio: float) -> str:
    """Render the log-variance of the 8x8 DCT coefficients as a heat map."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("#0a0d11")
    ax.set_facecolor("#0a0d11")

    # Add log scaling to prevent DC variance from saturating the scale
    log_var_map = np.log1p(var_map)

    im = ax.imshow(
        log_var_map,
        cmap="magma",
        interpolation="nearest"
    )
    ax.axis("on")
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.tick_params(color="#5a6a7e", labelcolor="#5a6a7e", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#5a6a7e")

    # Labels for axes
    ax.set_xlabel("Horizontal Frequency", color="#5a6a7e", fontsize=7)
    ax.set_ylabel("Vertical Frequency", color="#5a6a7e", fontsize=7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#5a6a7e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#5a6a7e", fontsize=6)
    cbar.set_label("Log Coefficient Variance", color="#5a6a7e", fontsize=7)

    ax.set_title(
        f"8x8 DCT VARIANCE GRID\nBenford Dev: {benford_dev:.4f} | AC Ratio: {ac_ratio:.4f}",
        color="#00FF88", fontsize=8, pad=8,
        fontfamily="monospace",
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
