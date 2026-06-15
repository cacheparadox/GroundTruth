"""
Verdict Engine — Project GroundTruth
======================================
Aggregates signals from all nine forensic modules into a unified
Ground Truth Index (GTI) score and issues a binary verdict.
Implements dynamic weight redistribution if any module returns neutral or missing data.

Weights Scheme (Proportionally tuned against camera vs synthetic image distributions):
  DWT (Discrete Wavelet Transform)   : 18%
  DCT (Discrete Cosine Transform)    : 15%
  FFT (Fast Fourier Transform)       : 12%
  PRNU (Sensor Noise Fingerprint)    : 12%
  SRM & LBP (Micro-Texture Analysis) : 12%
  Color Correlation (Color Physics)  : 10%
  ELA (Error Level Analysis)         : 8%
  Laplacian (Edge Anomaly Analysis)  : 8%
  Metadata (EXIF Investigation)      : 5%
"""

# Default weight allocation (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "dwt": 0.18,
    "dct": 0.15,
    "fft": 0.12,
    "prnu": 0.12,
    "srm": 0.12,
    "color": 0.10,
    "ela": 0.08,
    "lap": 0.08,
    "metadata": 0.05,
}

# GTI below this threshold → SYNTHETIC_DETECTED
VERDICT_THRESHOLD = 45.0

def compute_verdict(
    prnu_result: dict,
    fft_result: dict,
    ela_result: dict,
    lap_result: dict,
    dwt_result: dict,
    dct_result: dict,
    srm_result: dict,
    color_result: dict,
    metadata_result: dict,
    dims: list = None,
    patch_stats: dict = None
) -> dict:
    """
    Aggregate forensic stage results into a final verdict with dynamic weight redistribution.

    Returns
    -------
    dict containing gti, verdict, confidence, component_scores, and explanation.
    """
    # 1. Collect scores and filter active modules
    raw_scores = {
        "prnu": prnu_result.get("prnu_score"),
        "fft": fft_result.get("fft_score"),
        "ela": ela_result.get("ela_score"),
        "lap": lap_result.get("lap_score"),
        "dwt": dwt_result.get("dwt_score"),
        "dct": dct_result.get("dct_score"),
        "srm": srm_result.get("srm_score"),
        "color": color_result.get("color_score"),
    }

    # Metadata can be neutral (if stripped)
    if not metadata_result.get("neutral", False) and metadata_result.get("metadata_score") is not None:
        raw_scores["metadata"] = metadata_result["metadata_score"]

    # Filter out modules that returned None or are missing from raw_scores
    active_modules = {k: v for k, v in raw_scores.items() if v is not None}

    # 2. Dynamic Weight Redistribution
    total_active_weight = sum(DEFAULT_WEIGHTS[m] for m in active_modules)
    
    if total_active_weight <= 0:
        gti = 50.0
        component_scores = {}
    else:
        # Scale active weights proportionally so they sum to 100% (1.0)
        base_gti = 0.0
        component_scores = {}
        for m in active_modules:
            adjusted_weight = DEFAULT_WEIGHTS[m] / total_active_weight
            base_gti += adjusted_weight * active_modules[m]
            component_scores[m] = round(active_modules[m], 2)

        # 3. Disagreement check and Non-Linear Global Penalty
        high_disagreement = False
        disagreement_msg = ""
        
        fft_score = active_modules.get("fft")
        ela_score = active_modules.get("ela")
        
        if fft_score is not None and fft_score < 25.0:
            # Check other core AI indicators to make sure FFT is a true outlier (disagreement)
            dwt_val = active_modules.get("dwt", 100.0)
            color_val = active_modules.get("color", 100.0)
            dct_val = active_modules.get("dct", 100.0)
            srm_val = active_modules.get("srm", 100.0)
            ela_val = active_modules.get("ela", 100.0)
            
            if (ela_val >= 55.0 and
                dwt_val >= 75.0 and
                color_val >= 70.0 and
                dct_val >= 80.0 and
                srm_val >= 60.0):
                
                other_scores = [v for k, v in active_modules.items() if k != "fft"]
                if other_scores:
                    other_avg = sum(other_scores) / len(other_scores)
                    if other_avg > 70.0:
                        # Grant exemption if:
                        # 1) It has upscaling signatures (lap < 55 or prnu < 70)
                        # OR 2) All other physical sensors are highly organic/clean (other_avg > 80.0)
                        lap_val = active_modules.get("lap", 100.0)
                        prnu_val = active_modules.get("prnu", 100.0)
                        if lap_val < 55.0 or prnu_val < 70.0 or other_avg > 80.0:
                            high_disagreement = True
                            disagreement_msg = (
                                "High disagreement: FFT indicates synthetic periodic patterns, "
                                "but all other physical sensors confirm organic capture (indicative "
                                "of real upscaling/re-sampling or natural texture)."
                            )

        # Calculate base global penalty
        low_score_deviations = sum(40.0 - val for val in active_modules.values() if val < 40.0)
        global_penalty = low_score_deviations * 1.20

        # Scale down global penalty if there is high disagreement, and recalculate base_gti excluding the outlier
        if high_disagreement:
            global_penalty *= 0.25
            # Recompute base_gti excluding any module scored < 25.0 to prevent average-masking/outlier-dragging
            clean_active = {k: v for k, v in active_modules.items() if v >= 25.0}
            clean_weight = sum(DEFAULT_WEIGHTS[m] for m in clean_active)
            if clean_weight > 0:
                base_gti = sum((DEFAULT_WEIGHTS[m] / clean_weight) * clean_active[m] for m in clean_active)

        gti = base_gti - global_penalty

        # Genuinely low ELA (< 55.0) indicates synthetic textures/compression, penalize aggressively
        if ela_score is not None and ela_score < 55.0:
            gti -= (55.0 - ela_score) * 0.80

        # 4. Patch Consistency Fusion
        if patch_stats:
            prnu_sd = patch_stats.get("prnu_sd")
            srm_sd = patch_stats.get("srm_sd")
            lap_sd = patch_stats.get("lap_sd")

            # Apply penalties for abnormally low (uniform) noise/texture variation
            if srm_sd is not None and srm_sd < 2.0:
                gti -= (2.0 - srm_sd) * 25.0
            if prnu_sd is not None and prnu_sd < 0.012:
                gti -= (0.012 - prnu_sd) * 2000.0

        # 5. Low Resolution Feature Amplification
        if dims:
            w, h = dims
            min_dim = min(w, h)
            if min_dim < 512:
                low_res_multiplier = 512.0 / min_dim
                for m in ["prnu", "dwt", "dct", "ela"]:
                    if m in active_modules and active_modules[m] < 92.0:
                        gti -= (92.0 - active_modules[m]) * 0.45 * low_res_multiplier

        gti = round(min(100.0, max(0.0, gti)), 2)

    # 3. Verdict
    verdict = "ORGANIC_MATCH" if gti >= VERDICT_THRESHOLD else "SYNTHETIC_DETECTED"

    # 4. Confidence: distance from the threshold, normalized
    distance = abs(gti - VERDICT_THRESHOLD)
    confidence = round(min(1.0, 0.5 + distance / 50.0), 3)

    if high_disagreement:
        confidence = round(confidence * 0.5, 3)

    # 5. Diagnostic explanations
    explanation = []
    
    if high_disagreement and disagreement_msg:
        explanation.append(disagreement_msg)

    if patch_stats:
        srm_sd = patch_stats.get("srm_sd")
        prnu_sd = patch_stats.get("prnu_sd")
        if srm_sd is not None and srm_sd < 2.0:
            explanation.append(f"Texture patch uniformity is abnormally high (SD: {srm_sd:.3f}) — indicative of uniform synthetic textures.")
        if prnu_sd is not None and prnu_sd < 0.012:
            explanation.append(f"PRNU patch uniformity is abnormally high (SD: {prnu_sd:.4f}) — indicative of simulated sensor noise.")

    # 6. Low-resolution verification check
    if dims:
        w, h = dims
        min_dim = min(w, h)
        if min_dim < 512:
            factor = max(0.2, min_dim / 512.0)
            confidence = round(confidence * factor, 3)
            explanation.append(
                f"Warning: Low working image resolution ({w}x{h} px). Downscaling/low resolution destroys "
                f"high-frequency sensor noise, upsampling grids, and VAE textures, which can cause "
                f"synthetic images to falsely appear organic. Analysis confidence reduced."
            )
    
    # Check PRNU
    if prnu_result.get("prnu_score", 100) < 40:
        explanation.append("Low PRNU score: noise residual lacks authentic, stochastic camera sensor fingerprint.")

    # Check FFT
    if fft_result.get("peak_count", 0) > 25:
        explanation.append(
            f"FFT detected {fft_result['peak_count']} periodic frequency peaks "
            f"with angular regularity {fft_result.get('angular_regularity', 0):.2f} "
            f"— typical of AI upsampling grids."
        )
    elif fft_result.get("angular_regularity", 0) > 0.55:
        explanation.append(
            f"FFT frequency spectrum shows abnormally uniform angular regularity ({fft_result['angular_regularity']:.2f})."
        )

    # Check ELA
    if ela_result.get("ela_uniformity", 1.0) < 0.4:
        explanation.append(
            f"ELA uniformity {ela_result['ela_uniformity']:.3f} is abnormally low, suggesting uniform synthetic generation."
        )

    # Check Laplacian
    if lap_result.get("anomaly_ratio", 0) > 0.15:
        explanation.append(
            f"Laplacian anomaly ratio ({lap_result['anomaly_ratio']*100:.2f}%) exceeds threshold. "
            f"Suspect flat luminance patches or over-sharpened local structures detected."
        )

    # Check DWT
    if dwt_result.get("hh_ratio_l1", 0) > 0.20:
        explanation.append(
            f"DWT Level 1 Diagonal sub-band energy ratio ({dwt_result['hh_ratio_l1']:.4f}) is elevated, "
            f"indicating isotropic high-frequency patterns from generative models."
        )

    # Check DCT
    if dct_result.get("benford_deviation", 0) > 0.05:
        explanation.append(
            f"DCT coefficients deviate from Benford's Law (dev: {dct_result['benford_deviation']:.4f}), "
            f"indicating double compression or synthetic grid structures."
        )
    if dct_result.get("ac_high_low_ratio", 0) > 0.06:
        explanation.append(
            f"DCT AC high-to-low frequency variance ratio ({dct_result['ac_high_low_ratio']:.4f}) is abnormally high."
        )

    # Check SRM / LBP
    if srm_result.get("lbp_entropy", 4.0) < 3.2:
        explanation.append(
            f"LBP texture entropy ({srm_result['lbp_entropy']:.3f}) is abnormally low — indicative of artificial smoothing."
        )
    elif srm_result.get("lbp_entropy", 4.0) > 4.3:
        explanation.append(
            f"LBP texture entropy ({srm_result['lbp_entropy']:.3f}) is abnormally high — consistent with random high-pass synthetic noise."
        )
    if srm_result.get("srm_mean", 1.0) < 0.6:
        explanation.append(
            f"SRM noise mean residual ({srm_result['srm_mean']:.3f}) is abnormally flat."
        )

    # Check Color Correlation
    if color_result.get("color_score", 100) < 70:
        explanation.append(
            f"Color channel analysis detected abnormal inter-channel correlation or L*a*b* chroma entropy ({color_result.get('chroma_entropy', 0.0):.3f})."
        )

    # Check Metadata
    if "metadata" in active_modules:
        meta_details = metadata_result.get("details", {})
        if meta_details.get("ai_flagged"):
            explanation.append("Metadata check flagged explicit generative AI signature in EXIF tags.")
        elif metadata_result["metadata_score"] < 70:
            explanation.append(f"EXIF Metadata alert: {metadata_result['explanation']}")

    if not explanation and verdict == "ORGANIC_MATCH":
        explanation.append("All active forensic indicators returned values consistent with physical camera captures.")

    # Grid coherence & anomaly density (for backward compatibility / UI display)
    grid_coherence = round(max(0.0, 100.0 - lap_result.get("anomaly_ratio", 0) * 800.0), 2)
    anomaly_density = round(min(100.0, lap_result.get("anomaly_ratio", 0) * 500.0), 2)

    return {
        "gti": gti,
        "verdict": verdict,
        "confidence": confidence,
        "component_scores": component_scores,
        "explanation": explanation,
        "grid_coherence": grid_coherence,
        "anomaly_density": anomaly_density,
    }
