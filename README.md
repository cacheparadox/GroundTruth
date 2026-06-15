# Project GroundTruth

Project GroundTruth is an advanced image forensics suite combining 9 analytical engines—including DWT wavelets, block DCT, FFT, and PRNU noise extraction—to verify physical camera capture vs synthetic AI generation (GTI index).

---

## Forensic Features & Core Engines

The forensic engine calculates the Ground Truth Index (GTI) (0% = Synthetic, 100% = Organic) by evaluating 9 active analysis sensors:

1. **DWT Subbands (Discrete Wavelet Transform):** Computes a 3-level decomposition using the `db4` wavelet to measure diagonal-to-orthogonal high-frequency energy ratios, flagging upsampling grids.
2. **DCT Variance (Discrete Cosine Transform):** Runs block-level $8\times 8$ DCT and evaluates first-digit Benford's Law compliance to identify double-compression artifacts and synthetic grids.
3. **FFT Spectrum (Fast Fourier Transform):** Detects periodic frequencies, peaks, and angular regularity with bandpass energy scaling to identify upsampling strides.
4. **PRNU Noise (Photo Response Non-Uniformity):** Extracts the camera sensor noise fingerprint using a 5x5 spatial Wiener filter to evaluate noise amplitude and kurtosis.
5. **Micro-Texture & Residuals (SRM & LBP):** Convolves 10 high-pass Spatial Rich Model kernels and computes Local Binary Pattern (LBP) entropy to detect artificial texture smoothing.
6. **Color Physics:** Pairwise RGB channel correlation and CIE L\*a\*b\* chroma entropy checking.
7. **ELA (Error Level Analysis):** Re-compresses the image at JPEG Q75 to highlight compression mismatch seams or uniform synthetic compression profiles.
8. **Laplacian Edge Anomaly:** Detects localized sharpening and structural boundaries (the basis for the **Multizone Anomaly Mask**).
9. **Metadata Forensics:** Inspects original EXIF tags for software/model signatures of popular AI generators (Stable Diffusion, Midjourney, ComfyUI).

---

## Tech Stack

* **Backend:** Python Serverless (Flask, NumPy, SciPy, Scikit-Image, Matplotlib, Pillow, PyWavelets) hosted on Vercel.
* **Frontend:** Vanilla HTML5, TailwindCSS, custom HSL CSS theme, Canvas API, and interactive visual dashboards.

---

## Local Development & Deployment

### Local Development
To run both the frontend and backend locally in a unified environment:
1. Install the [Vercel CLI](https://vercel.com/cli) globally:
   ```bash
   npm install -g vercel
   ```
2. Start the local serverless development environment:
   ```bash
   vercel dev
   ```
   This serves the frontend at `http://localhost:3000` and proxies the Python serverless backend functions under `http://localhost:3000/api`.

### Deploying to Vercel
1. Push this clean codebase to a GitHub repository.
2. In Vercel, import the repository.
3. Keep the **Application Preset** (Framework Preset) as **Other** (do NOT select "Services" as Vercel Serverless automatically routes `/api` python functions).
4. Deploy!
