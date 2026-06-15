"""
EXIF Metadata Analysis — Project GroundTruth
=============================================
Extracts and parses EXIF metadata using Pillow's image extraction.
Checks for signatures of AI generation software (e.g., Midjourney, Stable Diffusion,
DALL-E, Adobe Firefly, ComfyUI) and validates typical physical camera properties.
If no EXIF is present, it returns a neutral state so that the verdict engine
can dynamically redistribute its weight.
"""

from PIL import Image
from PIL.ExifTags import TAGS

# Substrings linked to AI synthesis software or tools
AI_SIGNATURES = [
    "stable diffusion", "stable-diffusion", "midjourney", "dall-e", "dalle",
    "firefly", "adobe firefly", "comfyui", "automatic1111", "novelai",
    "generative", "ai generator", "ai generated", "sdxl", "flux.1"
]

# Substrings indicating editing software (suspicious but not outright AI)
EDIT_SIGNATURES = [
    "photoshop", "gimp", "lightroom", "pixelmator", "canva", "figma", "affinity"
]

def analyze_metadata(pil_img: Image.Image) -> dict:
    """
    Analyze the metadata of the original PIL image.

    Parameters
    ----------
    pil_img : Image.Image
        The original PIL image object before any conversion/resizing.

    Returns
    -------
    dict with keys:
        neutral         : bool  — True if no EXIF is present, triggering weight redistribution
        metadata_score  : float — 0-100 score (None if neutral)
        details         : dict  — resolved EXIF details
        explanation     : str   — summary of findings
    """
    try:
        exif_raw = pil_img._getexif()
    except Exception:
        exif_raw = None

    if not exif_raw:
        return {
            "neutral": True,
            "metadata_score": None,
            "details": {},
            "explanation": "EXIF metadata is entirely absent or has been stripped."
        }

    # Resolve raw tags
    exif = {}
    for tag, value in exif_raw.items():
        tag_name = TAGS.get(tag, tag)
        exif[tag_name] = value

    details = {
        "make": str(exif.get("Make", "")).strip(),
        "model": str(exif.get("Model", "")).strip(),
        "software": str(exif.get("Software", "")).strip(),
        "datetime": str(exif.get("DateTimeOriginal", exif.get("DateTime", ""))).strip(),
        "exposure_time": str(exif.get("ExposureTime", "")),
        "f_number": str(exif.get("FNumber", "")),
        "iso": str(exif.get("ISOSpeedRatings", "")),
        "lens_model": str(exif.get("LensModel", "")).strip(),
        "ai_flagged": False,
        "editor_flagged": False,
        "has_camera_metadata": False
    }

    # Search all text fields for AI signatures
    ai_found = False
    ai_software = ""
    for tag_name, val in exif.items():
        val_str = str(val).lower()
        for sig in AI_SIGNATURES:
            if sig in val_str:
                ai_found = True
                ai_software = sig
                break
        if ai_found:
            break

    details["ai_flagged"] = ai_found

    # Search software/description for editor signatures
    editor_found = False
    software_val = details["software"].lower()
    for sig in EDIT_SIGNATURES:
        if sig in software_val:
            editor_found = True
            break
    details["editor_flagged"] = editor_found

    # Check for presence of physical camera metrics (Make, Model, ExposureTime, FNumber, ISO)
    camera_fields = ["Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings"]
    has_cam_fields = sum(1 for field in camera_fields if field in exif)
    if has_cam_fields >= 3:
        details["has_camera_metadata"] = True

    # ── Scoring logic ──
    if ai_found:
        metadata_score = 0.0
        explanation = f"Metadata explicitly flags AI generation: '{ai_software}' signature detected in EXIF fields."
    elif details["has_camera_metadata"]:
        metadata_score = 100.0
        if editor_found:
            metadata_score = 85.0
            explanation = f"Captured via physical hardware ({details['make']} {details['model']}), but edited with '{details['software']}'."
        else:
            explanation = f"Authentic physical camera capture confirmed ({details['make']} {details['model']} EXIF profile intact)."
    else:
        # Has some EXIF but lacks hardware markers
        if editor_found:
            metadata_score = 60.0
            explanation = f"No physical camera tags found. Software metadata references editor '{details['software']}'."
        else:
            metadata_score = 75.0
            explanation = "Metadata present but lacks physical sensor properties (Make/Model/Lens)."

    return {
        "neutral": False,
        "metadata_score": round(metadata_score, 2),
        "details": details,
        "explanation": explanation
    }
