# Project GroundTruth

Project GroundTruth is a defense-grade image forensics suite designed to detect synthetic patterns (generative AI, GANs, diffusion models) vs organic camera sensor captures. The suite uses a hybrid verification pipeline that aggregates signal-level physical features, frequency artifacts, and texture statistics to compute a unified **Ground Truth Index (GTI)** score.

## Forensic Features & Core Engines

The forensic engine calculates the GTI (0% = Synthetic, 100% = Organic) by evaluating 9 active analysis sensors:

1. **DWT Subbands (Discrete Wavelet Transform):** Computes a 3-level decomposition using the `db4` wavelet to measure diagonal-to-orthogonal high-frequency energy ratios, flagging upsampling grids.
2. **DCT Variance (Discrete Cosine Transform):** Runs block-level $8\times 8$ DCT and evaluates first-digit Benford's Law compliance to identify double-compression artifacts and synthetic grids.
3. **FFT Spectrum (Fast Fourier Transform):** Detects periodic frequencies, peaks, and angular regularity with bandpass energy scaling to identify upsampling strides.
4. **PRNU Noise (Photo Response Non-Uniformity):** Extracts the camera sensor noise fingerprint using a 5x5 spatial Wiener filter to evaluate noise amplitude and kurtosis.
5. **Micro-Texture & Residuals (SRM & LBP):** Convolves 10 high-pass Spatial Rich Model kernels and computes Local Binary Pattern (LBP) entropy to detect artificial texture smoothing.
6. **Color Physics:** Pairwise RGB channel correlation and CIE L\*a\*b\* chroma entropy checking.
7. **ELA (Error Level Analysis):** Re-compresses the image at JPEG Q75 to highlight compression mismatch seams or uniform synthetic compression profiles.
8. **Laplacian Edge Anomaly:** Detects localized sharpening and structural boundaries (the basis for the **Multizone Anomaly Mask**).
9. **Metadata Forensics:** Inspects original EXIF tags for software/model signatures of popular AI generators (Stable Diffusion, Midjourney, ComfyUI).

## Tech Stack

* **Backend:** Python, Flask, Flask-CORS, NumPy, SciPy, Scikit-Image, Matplotlib, Pillow, Gunicorn.
* **Frontend:** Vanilla HTML5, TailwindCSS, HSL custom CSS system, Canvas API, and interactive visual dashboards.

## Installation & Deployment

### Backend
1. Navigate to the `backend` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development Flask server:
   ```bash
   python app.py
   ```

### Frontend
1. Open the root `index.html` directly in a browser or serve it using any static web server (e.g., Live Server).
2. Configure your API Endpoint URL in the left-hand panel of the visual interface.
