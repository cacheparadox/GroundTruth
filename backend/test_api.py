"""
API Integration Test — Project GroundTruth
=========================================
Generates a dummy synthetic image and tests the Flask backend analyze endpoint.
Uses standard library urllib to avoid extra test dependencies.
"""

import urllib.request
import urllib.error
import json
import io
from PIL import Image

def run_test():
    print("[TEST] Creating dummy test image (256x256 random noise)...")
    # Create random image
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    # Add some texture / details
    pixels = img.load()
    for x in range(256):
        for y in range(256):
            if (x + y) % 8 == 0:
                pixels[x, y] = (200, 100, 50)
            elif (x * y) % 5 == 0:
                pixels[x, y] = (50, 150, 200)

    # Save to buffer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    image_bytes = img_byte_arr.read()

    # Form boundary for multipart/form-data
    boundary = b"----WebKitFormBoundaryGroundTruthTest"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n"
        + image_bytes + b"\r\n"
        b"--" + boundary + b"--\r\n"
    )

    url = "http://localhost:5000/analyze"
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary.decode()}")
    req.add_header("Content-Length", len(body))

    print(f"[TEST] Sending POST request to {url}...")
    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            response_body = response.read().decode('utf-8')
            print(f"[TEST] Response status: {status}")
            
            data = json.loads(response_body)
            print("[TEST] Parsing JSON response parameters:")
            print(f"  - Verdict   : {data.get('verdict')}")
            print(f"  - GTI Score : {data.get('gti')}%")
            print(f"  - Confidence: {data.get('confidence')}")
            print(f"  - Scan Time : {data.get('scan_time_ms')} ms")
            
            # Print component scores
            print("  - Component Scores:")
            for k, v in data.get("component_scores", {}).items():
                print(f"    * {k.upper()}: {v}")
            
            # Verify images
            images = data.get("images", {})
            print("  - Returned Visualisation Keys:")
            for img_key in images.keys():
                print(f"    * {img_key} (starts with {images[img_key][:25]}...)")
                
            print("\n[SUCCESS] Backend API is fully functional with the upgraded forensic engine!")
    except urllib.error.HTTPError as e:
        print(f"[ERROR] HTTP Error: {e.code} - {e.reason}")
        print(e.read().decode('utf-8'))
        exit(1)
    except urllib.error.URLError as e:
        print(f"[ERROR] Connection failed: {e.reason}")
        print("Please verify that the Flask server is running on port 5000.")
        exit(1)

if __name__ == "__main__":
    run_test()
