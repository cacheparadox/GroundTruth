"""
Benchmark Simulation & Verification Script — Project GroundTruth
================================================================
Iterates over all test images in the brain artifacts folder,
sends them to the local Flask backend analyze endpoint, and
verifies that they match their expected classification.
"""

import os
import glob
import urllib.request
import urllib.error
import json
import io

# Define expected classifications
EXPECTED_VERDICTS = {
    # Synthetic / AI-generated images
    "groundtruth_main_page_1781526656380.png": "SYNTHETIC_DETECTED",
    "groundtruth_preview_1781526631298.webp": "SYNTHETIC_DETECTED",
    "groundtruth_ui_1781527789551.png": "SYNTHETIC_DETECTED",
    "groundtruth_v2_preview_1781527772755.webp": "SYNTHETIC_DETECTED",
    "media__1781528887084.jpg": "SYNTHETIC_DETECTED",
    "media__1781528890709.png": "SYNTHETIC_DETECTED",
    "media__1781528897401.png": "SYNTHETIC_DETECTED",
    "media__1781530504173.png": "SYNTHETIC_DETECTED",
    "media__1781530505408.png": "SYNTHETIC_DETECTED",
    "media__1781530513448.png": "SYNTHETIC_DETECTED",
    "media__1781530597415.png": "SYNTHETIC_DETECTED",
    "media__1781530598793.png": "SYNTHETIC_DETECTED",
    "media__1781530614466.jpg": "SYNTHETIC_DETECTED",
    "media__1781530917136.png": "SYNTHETIC_DETECTED",
    "media__1781531162775.png": "SYNTHETIC_DETECTED",  # Low-res ballerina

    # Organic / Physical camera images
    "media__1781528877689.png": "ORGANIC_MATCH",
    "media__1781530502858.jpg": "ORGANIC_MATCH",
    "media__1781530599946.jpg": "ORGANIC_MATCH",
    "media__1781530601103.jpg": "ORGANIC_MATCH",
    "media__1781530902099.png": "ORGANIC_MATCH",
    "media__1781533045420.jpg": "ORGANIC_MATCH",  # Newly uploaded real image
}

def analyze_image(file_path):
    filename = os.path.basename(file_path)
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    # Form boundary for multipart/form-data
    boundary = b"----WebKitFormBoundaryGroundTruthDebug"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="image"; filename="' + filename.encode() + b'"\r\n'
        b"Content-Type: image/octet-stream\r\n\r\n"
        + image_bytes + b"\r\n"
        b"--" + boundary + b"--\r\n"
    )

    url = "http://localhost:5000/analyze"
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
    req.add_header("Content-Length", len(body))

    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code} for {filename}: {e.read().decode('utf-8')}")
        return None
    except urllib.error.URLError as e:
        print(f"URLError for {filename}: {e.reason}")
        return None

def run_benchmark():
    brain_dir = r"C:\Users\SaSha\.gemini\antigravity-ide\brain\d27f452d-a60b-4ca4-9c50-9cb38b855e77"
    
    # Gather matching images
    image_patterns = ["media__*", "groundtruth_*"]
    all_files = []
    for pattern in image_patterns:
        all_files.extend(glob.glob(os.path.join(brain_dir, pattern)))

    # Sort files by name to ensure consistent output order
    all_files.sort(key=lambda x: os.path.basename(x))

    print(f"Starting Benchmark Simulation over {len(all_files)} images...")
    print("-" * 105)
    print(f"{'Filename':<45} | {'Expected':<18} | {'Got':<18} | {'GTI':<6} | {'Conf':<5} | {'Result':<5}")
    print("-" * 105)

    passed = 0
    total = 0

    for file_path in all_files:
        filename = os.path.basename(file_path)
        if filename not in EXPECTED_VERDICTS:
            continue

        expected = EXPECTED_VERDICTS[filename]
        result = analyze_image(file_path)

        if result is None:
            print(f"{filename:<45} | {expected:<18} | {'FAILED TO ANALYZE':<18} | {'-':<6} | {'-':<5} | FAIL")
            total += 1
            continue

        got_verdict = result.get("verdict")
        gti = result.get("gti")
        conf = result.get("confidence")
        
        status = "PASS" if got_verdict == expected else "FAIL"
        if status == "PASS":
            passed += 1
        total += 1

        print(f"{filename:<45} | {expected:<18} | {got_verdict:<18} | {gti:>5}% | {conf:<5} | {status}")

    print("-" * 105)
    accuracy = (passed / total) * 100 if total > 0 else 0
    print(f"Benchmark Completed: {passed}/{total} Passed ({accuracy:.1f}% Accuracy)")
    print("-" * 105)

    if accuracy == 100.0:
        print("[SUCCESS] All calibrated engine thresholds are 100% correct!")
    else:
        print("[WARNING] Verification failed. Some images did not match their expected classes.")

if __name__ == "__main__":
    run_benchmark()
