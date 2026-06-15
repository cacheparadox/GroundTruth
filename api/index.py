"""
Vercel Serverless Function — Project GroundTruth Backend
=================================================
Single endpoint: POST /analyze (now routed under same-origin /api/analyze)
"""

import sys
import os
import time
import io
import base64
import traceback

# Inject the current directory (api/) into system path to resolve the engine package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

from engine.prnu              import extract_prnu
from engine.fft_analysis      import analyze_fft
from engine.ela                import analyze_ela
from engine.laplacian          import analyze_laplacian
from engine.dwt_analysis      import analyze_dwt
from engine.dct_analysis      import analyze_dct
from engine.srm_analysis      import analyze_srm
from engine.color_correlation import analyze_color
from engine.metadata_analysis import analyze_metadata
from engine.verdict            import compute_verdict

import os
os.environ['MPLCONFIGDIR'] = '/tmp'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Flask setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow cross-origin requests in development

# Defensive memory gating to prevent server OOM container restarts
MAX_DIM   = 2048   
MAX_BYTES = 20 * 1024 * 1024   # 20 MB upload limit

def compute_patch_consistency(map_array: np.ndarray, num_patches: int = 4) -> float:
    """
    Split a 2D map into a grid of patches (e.g. 4x4) and calculate the standard
    deviation of patch means. High SD indicates high natural variation; low SD
    indicates uniform synthetic noise.
    """
    h, w = map_array.shape[:2]
    patch_h = h // num_patches
    patch_w = w // num_patches
    if patch_h < 2 or patch_w < 2:
        return 0.0
    
    means = []
    for i in range(num_patches):
        for j in range(num_patches):
            patch = map_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            means.append(np.mean(patch))
    return float(np.std(means))

# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
def health():
    return jsonify({"status": "online", "service": "GroundTruth Forensics API (Vercel Serverless)"})

# ── Main analysis endpoint ─────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
def analyze():
    t_start = time.time()
    logs = []

    def log(tag: str, msg: str):
        logs.append(f"[{tag}] {msg}")

    try:
        # ── 1. Receive image ──────────────────────────────────────────────
        if "image" not in request.files:
            return jsonify({"error": "No image field in request."}), 400
 
        file = request.files["image"]
        raw_bytes = file.read()
        if len(raw_bytes) > MAX_BYTES:
            return jsonify({"error": "Image exceeds 20 MB limit."}), 413

        log("SYS", f"Image received — {len(raw_bytes) // 1024} KB — filename: {file.filename}")

        # ── 2. Decode original image & parse EXIF ──────────────────────────
        try:
            # Open first to extract EXIF data before any colorspace conversion or resizing
            pil_orig = Image.open(io.BytesIO(raw_bytes))
        except Exception as e:
            return jsonify({"error": f"Cannot decode image: {e}"}), 422

        # Run EXIF metadata analysis
        log("SYS", "Parsing image metadata and EXIF tags...")
        metadata_result = analyze_metadata(pil_orig)
        if metadata_result.get("neutral"):
            log("WARN", "No EXIF data found. Weight will be redistributed dynamically.")
        else:
            log("OK", f"Metadata complete — trust score: {metadata_result.get('metadata_score')}/100")

        # Convert to RGB for matrix operations
        pil_img = pil_orig.convert("RGB")
        orig_w, orig_h = pil_img.size
        log("SYS", f"Source dimensions: {orig_w}×{orig_h} px")

        # Downscale to MAX_DIM if needed using Resampling.LANCZOS
        if orig_w > MAX_DIM or orig_h > MAX_DIM:
            scale   = MAX_DIM / max(orig_w, orig_h)
            work_w  = max(1, round(orig_w * scale))
            work_h  = max(1, round(orig_h * scale))
            pil_work = pil_img.resize((work_w, work_h), Image.LANCZOS)
            log("SYS", f"Downscaled to working resolution: {work_w}×{work_h}")
        else:
            pil_work = pil_img
            work_w, work_h = orig_w, orig_h
            log("SYS", f"Working at native resolution: {work_w}×{work_h}")

        img_array = np.array(pil_work, dtype=np.float32)   # (H, W, 3) [0–255]
        total_px  = work_w * work_h

        # ── 3. PRNU noise extraction ──────────────────────────────────────
        log("SYS",  "Initialising PRNU extraction pipeline...")
        log("MATH", f"Applying Wiener filter (5×5 kernel) across {total_px:,} pixels...")
        prnu_result = extract_prnu(img_array)
        log("OK",   f"PRNU complete — RMS noise power: {prnu_result['noise_power']:.4f}")
        log("MATH", f"Spatial correlation: {prnu_result['spatial_corr']:.4f} | Kurtosis: {prnu_result['kurtosis']:.3f}")
        log("SYS",  f"PRNU score: {prnu_result['prnu_score']:.1f}/100")

        # ── 4. FFT frequency analysis ─────────────────────────────────────
        log("SYS",  "Initialising FFT2 frequency domain analysis...")
        log("MATH", "Applying Hann window + numpy.fft.fft2 ...")
        fft_result = analyze_fft(img_array)
        log("MATH", "Log-magnitude spectrum computed. Running bandpass peak detection...")
        log("OK",   f"FFT complete — anomalous peaks: {fft_result['peak_count']}")
        if fft_result["dominant_freq"] > 0:
            log("MATH", f"Dominant periodic frequency: {fft_result['dominant_freq']} px stride")
        log("SYS",  f"FFT score: {fft_result['fft_score']:.1f}/100")

        # ── 5. Error Level Analysis ───────────────────────────────────────
        log("SYS",  "Initialising Error Level Analysis (ELA)...")
        log("MATH", "Re-encoding JPEG at Q75 in memory...")
        ela_result = analyze_ela(img_array)
        log("MATH", f"ELA mean: {ela_result['ela_mean']:.3f} | uniformity: {ela_result['ela_uniformity']:.4f}")
        log("OK",   f"ELA complete — score: {ela_result['ela_score']:.1f}/100")

        # ── 6. Laplacian analysis ─────────────────────────────────────────
        log("SYS",  "Initialising Laplacian high-pass convolution...")
        log("MATH", "Applying 3×3 Laplacian kernel...")
        lap_result = analyze_laplacian(img_array)
        log("MATH", "Local variance map computed (9×9 windows)")
        log("MATH", f"Flagged {lap_result['flagged_pixels']:,} px ({lap_result['anomaly_ratio']*100:.2f}%) as anomalous")
        log("OK",   f"Laplacian complete — score: {lap_result['lap_score']:.1f}/100")

        # ── 7. Discrete Wavelet Transform (DWT) ───────────────────────────
        log("SYS",  "Initialising 3-level DWT analysis (db4 wavelet)...")
        dwt_result = analyze_dwt(img_array)
        log("MATH", f"DWT Level 1 subband energies: H={dwt_result['l1_energy_h']:.2f}, V={dwt_result['l1_energy_v']:.2f}, D={dwt_result['l1_energy_d']:.2f}")
        log("OK",   f"DWT complete — diagonal energy ratio: {dwt_result['hh_ratio_l1']:.4f}")
        log("SYS",  f"DWT score: {dwt_result['dwt_score']:.1f}/100")

        # ── 8. Vectorized Block DCT ───────────────────────────────────────
        log("SYS",  "Initialising vectorized 8x8 block DCT analysis...")
        dct_result = analyze_dct(img_array)
        log("MATH", f"Benford Law Chi-Square deviation: {dct_result['benford_deviation']:.5f}")
        log("OK",   f"DCT complete — high/low AC variance ratio: {dct_result['ac_high_low_ratio']:.4f}")
        log("SYS",  f"DCT score: {dct_result['dct_score']:.1f}/100")

        # ── 9. SRM & LBP Micro-Texture ────────────────────────────────────
        log("SYS",  "Initialising Spatial Rich Model (SRM) & Local Binary Patterns (LBP)...")
        srm_result = analyze_srm(img_array)
        log("MATH", f"SRM residual mean amplitude: {srm_result['srm_mean']:.3f} | LBP uniform pattern entropy: {srm_result['lbp_entropy']:.3f}")
        log("OK",   f"SRM & LBP complete — score: {srm_result['srm_score']:.1f}/100")

        # ── 10. Color Physics & Channel Correlation ──────────────────────
        log("SYS",  "Initialising color channel correlation check...")
        color_result = analyze_color(img_array)
        log("MATH", f"RGB correlations: R-G={color_result['r_g_corr']:.4f}, G-B={color_result['g_b_corr']:.4f}")
        log("OK",   f"Color physics complete — score: {color_result['color_score']:.1f}/100")

        # Compute patch consistency statistics
        prnu_patch_sd = compute_patch_consistency(prnu_result["gray_noise"])
        srm_patch_sd  = compute_patch_consistency(srm_result["srm_composite"])
        lap_patch_sd  = compute_patch_consistency(lap_result["lap_response"])

        log("MATH", f"Patch Consistency SD: PRNU={prnu_patch_sd:.4f}, SRM={srm_patch_sd:.4f}, Laplacian={lap_patch_sd:.4f}")

        # ── 11. Verdict aggregation ───────────────────────────────────────
        log("SYS",  "Computing weighted Ground Truth Index (GTI)...")
        verdict_result = compute_verdict(
            prnu_result, fft_result, ela_result, lap_result,
            dwt_result, dct_result, srm_result, color_result, metadata_result,
            dims=[work_w, work_h],
            patch_stats={
                "prnu_sd": prnu_patch_sd,
                "srm_sd": srm_patch_sd,
                "lap_sd": lap_patch_sd
            }
        )
        gti     = verdict_result["gti"]
        verdict = verdict_result["verdict"]

        log("SYS",  f"GTI = {gti:.1f}% (threshold: 45.0%)")
        for comp, val in verdict_result["component_scores"].items():
            log("MATH", f"  {comp.upper()} weight-adjusted contribution: {val:.1f}/100")

        if verdict == "SYNTHETIC_DETECTED":
            log("ERR", f"VERDICT: SYNTHETIC PATTERN DETECTED (GTI={gti:.1f}%)")
            for flag in verdict_result["explanation"]:
                log("WARN", flag)
        else:
            log("OK",  f"VERDICT: ORGANIC SENSOR MATCH (GTI={gti:.1f}%)")
            for note in verdict_result["explanation"]:
                log("OK", note)

        # ── 12. Original thumbnail ────────────────────────────────────────
        log("SYS",  "Rendering original image thumbnail...")
        orig_b64 = _pil_to_b64(pil_work)

        # ── 13. Build response payload ────────────────────────────────────
        elapsed_ms = int((time.time() - t_start) * 1000)
        log("SYS",  f"Analysis complete in {elapsed_ms} ms.")

        return jsonify({
            "verdict":      verdict,
            "gti":          gti,
            "confidence":   verdict_result["confidence"],
            "metrics": {
                # PRNU
                "prnu_score":      prnu_result["prnu_score"],
                "noise_power":     prnu_result["noise_power"],
                "spatial_corr":    prnu_result["spatial_corr"],
                "kurtosis":        prnu_result["kurtosis"],
                # FFT
                "fft_score":       fft_result["fft_score"],
                "fft_peak_count":  fft_result["peak_count"],
                "dominant_freq":   fft_result["dominant_freq"],
                # ELA
                "ela_score":       ela_result["ela_score"],
                "ela_mean":        ela_result["ela_mean"],
                "ela_uniformity":  ela_result["ela_uniformity"],
                # Laplacian
                "lap_score":       lap_result["lap_score"],
                "anomaly_ratio":   round(lap_result["anomaly_ratio"] * 100, 3),
                "flagged_pixels":  lap_result["flagged_pixels"],
                "total_pixels":    lap_result["total_pixels"],
                "pixel_variance":  lap_result["pixel_variance"],
                "edge_divergence": lap_result["edge_divergence"],
                # DWT
                "dwt_score":       dwt_result["dwt_score"],
                "hh_ratio_l1":     dwt_result["hh_ratio_l1"],
                "hh_ratio_l2":     dwt_result["hh_ratio_l2"],
                # DCT
                "dct_score":       dct_result["dct_score"],
                "benford_dev":     dct_result["benford_deviation"],
                "ac_ratio":        dct_result["ac_high_low_ratio"],
                # SRM/LBP
                "srm_score":       srm_result["srm_score"],
                "srm_mean":        srm_result["srm_mean"],
                "lbp_entropy":     srm_result["lbp_entropy"],
                # Color
                "color_score":     color_result["color_score"],
                "r_g_corr":        color_result["r_g_corr"],
                "chroma_entropy":  color_result["chroma_entropy"],
                # Patch Consistency SDs
                "prnu_sd":         round(prnu_patch_sd, 5),
                "srm_sd":          round(srm_patch_sd, 5),
                "lap_sd":          round(lap_patch_sd, 5),
                # Metadata
                "metadata_score":  metadata_result.get("metadata_score"),
                "metadata_neutral":metadata_result.get("neutral", False),
                "metadata_desc":   metadata_result.get("explanation", ""),
                "metadata_details":metadata_result.get("details", {}),
                # UI compatibility
                "grid_coherence":  verdict_result["grid_coherence"],
                "anomaly_density": verdict_result["anomaly_density"],
            },
            "component_scores": verdict_result["component_scores"],
            "explanation":      verdict_result["explanation"],
            "images": {
                "original":     orig_b64,
                "ela_map":      ela_result["image_b64"],
                "fft_spectrum": fft_result["image_b64"],
                "prnu_map":     prnu_result["image_b64"],
                "anomaly_mask": lap_result["image_b64"],
                "dwt_map":      dwt_result["image_b64"],
                "dct_map":      dct_result["image_b64"],
                "texture_map":  srm_result["image_b64"],
                "color_map":    color_result["image_b64"],
            },
            "dims":         [work_w, work_h],
            "orig_dims":    [orig_w, orig_h],
            "scan_time_ms": elapsed_ms,
            "logs":         logs,
        })

    except Exception:
        tb = traceback.format_exc()
        log("ERR", f"Unhandled exception:\n{tb}")
        return jsonify({"error": "Internal server error.", "logs": logs, "trace": tb}), 500


def _pil_to_b64(pil_img: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI PNG."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
