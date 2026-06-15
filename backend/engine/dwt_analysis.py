"""
Discrete Wavelet Transform (DWT) Analysis — Project GroundTruth
==============================================================
Analyzes high-frequency sub-band energy distributions using a multi-level 2D DWT.
Organic images exhibit structured orientation (edges align horizontally/vertically),
while synthetic images generated via GAN or Diffusion upsampling grids leave uniform,
high-frequency checkerboard artifacts in the diagonal (HH) sub-bands.
"""

import numpy as np
import pywt
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def analyze_dwt(img_array: np.ndarray) -> dict:
    """
    Perform 2D DWT wavelet analysis on the luminance channel.

    Parameters
    ----------
    img_array : np.ndarray
        Float32 array, shape (H, W, 3), values in [0, 255].

    Returns
    -------
    dict containing DWT metrics, sub-band energies, and a base64 visual.
    """
    # 1. Convert to Luminance (grayscale)
    if len(img_array.shape) == 3:
        gray = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
    else:
        gray = img_array.copy()

    h, w = gray.shape

    # 2. Compute 3-level discrete wavelet decomposition with Daubechies 4 ('db4')
    # db4 is highly sensitive to periodic high-frequency patterns.
    try:
        coeffs = pywt.wavedec2(gray, 'db4', level=3)
        cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    except Exception as e:
        # Fallback in case of tiny image sizes where level 3 fails
        try:
            coeffs = pywt.wavedec2(gray, 'db4', level=1)
            cA1, (cH1, cV1, cD1) = coeffs
            cH3, cV3, cD3 = cH1, cV1, cD1
            cH2, cV2, cD2 = cH1, cV1, cD1
        except Exception:
            # Complete dummy fallback
            cH1 = cV1 = cD1 = cH2 = cV2 = cD2 = cH3 = cV3 = cD3 = np.zeros((8, 8))

    # 3. Calculate sub-band energy (mean of squared coefficients)
    def subband_energy(sub):
        return float(np.mean(sub ** 2)) if sub.size > 0 else 0.0

    e_h1, e_v1, e_d1 = subband_energy(cH1), subband_energy(cV1), subband_energy(cD1)
    e_h2, e_v2, e_d2 = subband_energy(cH2), subband_energy(cV2), subband_energy(cD2)
    e_h3, e_v3, e_d3 = subband_energy(cH3), subband_energy(cV3), subband_energy(cD3)

    # 4. Diagonal-to-Orthogonal Energy Ratios (HH / (HL + LH))
    # Synthetic/Upsampled images show isotropic noise distribution -> high HH energy.
    ratio_l1 = e_d1 / (e_h1 + e_v1 + 1e-8)
    ratio_l2 = e_d2 / (e_h2 + e_v2 + 1e-8)
    ratio_l3 = e_d3 / (e_h3 + e_v3 + 1e-8)

    # 5. Scoring calibration
    # Natural scene stats: ratio_l1 is typically 0.03 to 0.08.
    # AI upsampling grid artifacts drive it higher (often 0.12 - 0.50+).
    penalty_l1 = max(0.0, (ratio_l1 - 0.09) * 400.0)
    penalty_l2 = max(0.0, (ratio_l2 - 0.11) * 300.0)
    dwt_score = 100.0 - (penalty_l1 + penalty_l2)
    dwt_score = float(min(100.0, max(0.0, dwt_score)))

    # 6. Visualization
    image_b64 = _render_dwt_visualization(cH1, cV1, cD1, ratio_l1)

    return {
        "dwt_score": round(dwt_score, 2),
        "hh_ratio_l1": round(ratio_l1, 4),
        "hh_ratio_l2": round(ratio_l2, 4),
        "hh_ratio_l3": round(ratio_l3, 4),
        "l1_energy_h": round(e_h1, 4),
        "l1_energy_v": round(e_v1, 4),
        "l1_energy_d": round(e_d1, 4),
        "image_b64": image_b64,
    }

def _render_dwt_visualization(cH1: np.ndarray, cV1: np.ndarray, cD1: np.ndarray, ratio_l1: float) -> str:
    """Generate a clean side-by-side plot of the Level 1 wavelet sub-bands."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
    fig.patch.set_facecolor("#0a0d11")

    subbands = [cH1, cV1, cD1]
    titles = ["Horizontal (LH)", "Vertical (HL)", "Diagonal (HH)"]
    # Terminal-green colormap for the details
    cmap = "viridis"

    for ax, sub, title in zip(axes, subbands, titles):
        ax.set_facecolor("#0a0d11")
        if sub.size > 0:
            # Emphasize details by showing log scale of absolute values
            vis = np.log1p(np.abs(sub))
            # Symmetric clip
            vmax = np.percentile(vis, 99.5)
            ax.imshow(vis, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.axis("off")
        ax.set_title(title, color="#5a6a7e", fontsize=7, fontfamily="monospace")

    fig.suptitle(
        f"DWT LEVEL 1 SUB-BANDS  //  DIAGONAL ENERGY RATIO: {ratio_l1:.4f}",
        color="#00FF88", fontsize=8, fontfamily="monospace", y=0.98
    )

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a0d11")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
