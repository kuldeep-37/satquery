# SatQuery AI — Professional Geospatial Frontend

Modern, tech-aesthetic web interface for satellite visual question answering and land-cover captioning, replacing the prototype Streamlit interface.

Built for the **Smart India Hackathon (SIH)** / ISRO problem statement demonstration.

---

## Key Features

1. **Header & Live Health Monitor**:
   - Real-time pipeline status polled from `GET /health` with pulsing emerald LED indicator when online.
   - Shows device architecture (`CPU` or `CUDA`) and model variant (`BLIP + LoRA`).
   - Scoped feature badges for **SAR Multimodal Fusion** (architecture slot placeholder) and **Multilingual Output** (IndicTrans2 hook placeholder).

2. **Satellite Imagery & Interactive ROI Canvas**:
   - Drag & drop or click-to-upload zone supporting standard imagery (PNG, JPG, TIFF).
   - **1-Click Demo Imagery Selector**: Pre-loaded real satellite samples (`Forest`, `Urban`, `Water`, `Agricultural`) from the UC Merced and EuroSAT datasets.
   - Interactive HTML5 canvas with crosshair cursor, rectangular bounding box drawing, corner anchors, dimming overlay, and coordinate badge tracking intrinsic pixel coordinates.
   - Quick canvas tools: "Full ROI", "Clear ROI", and "Change Image".

3. **Natural-Language Query Input & Demo Presets**:
   - Free-form text input with Enter-key submission.
   - Clickable demo query chips to accelerate live camera presentations:
     - *"Describe the land cover shown in this satellite image."*
     - *"What land cover is shown in this region?"*
     - *"Identify built-up urban structures or buildings."*
     - *"Is there water or river visible in this region?"*

4. **Results & Intelligence Stream**:
   - Cropped ROI thumbnail preview.
   - High-contrast prediction answer card.
   - **Dynamic Visual Confidence Indicator**: Animated progress gauge with three-tier color coding:
     - **≥ 70%**: Emerald Green (High Confidence)
     - **55% – 69%**: Amber Gold (Moderate Confidence)
     - **< 55%**: Ruby Crimson (Low Confidence Flag)
   - **Safety Banner ("Flagged for Human Review")**: Displays when confidence drops below 55%, illustrating autonomous safety governance for disaster-management workflows.
   - **One-Click Exports**:
     - `satquery_roi.geojson`: RFC 7946 GeoJSON Polygon feature with embedded predictions and confidence.
     - `satquery_audit_report.json`: Formatted audit trail record with ISO 8601 timestamp, ROI pixel box, model name, confidence, and inference latency.
   - **Live JSON Inspector**: Tabbed in-browser inspector for GeoJSON and Audit payloads.

---

## How to Run

### Option 1: Via FastAPI Unified Server (Recommended)
The backend mounts the `frontend/` directory at the root URL.
Start the backend from the workspace root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern browser.

### Option 2: Standalone Static Server / Browser
Open `frontend/index.html` directly in Google Chrome, Edge, or Firefox, or run any static server:

```bash
python -m http.server 3000 --directory frontend
```
*(The frontend automatically connects to the backend at `http://127.0.0.1:8000`)*

---

## Real vs. Stubbed System Architecture (Judges' Technical Q&A)

| Component | Status | Technical Details |
|---|---|---|
| **VLM Backbone** | **Real** | `Salesforce/blip-image-captioning-base` (~247M parameters) |
| **LoRA Domain Adaptation** | **Real** | Adapter fine-tuned across 3 epochs on unified UC Merced (256×256 aerial) + EuroSAT (64×64 Sentinel-2) dataset; rank $r=8$, alpha $\alpha=16$ on `query` and `value` projection layers (0.26% trainable parameters) |
| **Monotonic Loss Curve** | **Real** | Training loss decreased monotonically: Epoch 1 (8.27) $\to$ Epoch 2 (7.14) $\to$ Epoch 3 (6.37); artifact stored in `outputs/loss_curve.png` |
| **Confidence Scoring** | **Real** | Mean of max softmax probabilities over generated output tokens |
| **Interactive ROI Bounding** | **Real** | Client-side canvas math converts display resolution to intrinsic image pixels and sends to backend for PIL server-side cropping |
| **GeoJSON & Audit Trail** | **Real** | Standard GeoJSON Polygon geometry coordinates + JSON audit trail |
| **Optical-SAR Fusion** | **Stub** | Architecture placeholder slot in backend API schema and frontend toggle; model is optical-only |
| **Multi-Language Output** | **Stub / Extensible** | Language parameter accepted; translation model slot reserved |
