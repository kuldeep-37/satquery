# SatQuery AI — Backend API Service

FastAPI-powered backend service for satellite image Visual Question Answering (VQA) and land-cover captioning, powered by a fine-tuned LoRA adapter on `Salesforce/blip-image-captioning-base`.

---

## Getting Started

### 1. Requirements & Installation

Dependencies are listed in `backend/requirements.txt`:

```bash
pip install -r backend/requirements.txt
```

### 2. Starting the Backend Server

Run Uvicorn from the project root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The model and LoRA adapter (`outputs/blip-lora-satellite-adapter`) are loaded **once at startup** using the FastAPI `lifespan` handler.

---

## API Endpoints

### 1. `GET /health`
Returns pipeline readiness, loaded device, and feature flag statuses.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "has_adapter": true,
  "device": "cpu",
  "pipeline_features": {
    "sar_fusion": "stub (untrained architecture slot)",
    "multilingual": ["en"]
  }
}
```

---

### 2. `POST /analyze`
Crops an uploaded satellite image server-side to the user-specified Region of Interest (ROI), executes LoRA-adapted BLIP inference, computes token-level confidence, and serializes GeoJSON and audit records.

**Input (Multipart Form):**
- `file`: Image binary (PNG, JPG, TIFF)
- `x`: ROI top-left X coordinate (float)
- `y`: ROI top-left Y coordinate (float)
- `width`: ROI box width (float)
- `height`: ROI box height (float)
- `roi`: *(Alternative)* JSON string with `{"x": 10, "y": 20, "width": 100, "height": 80}`
- `query`: Optional natural-language question
- `enable_sar`: Boolean toggle for optical-SAR fusion stub
- `language`: Target language code (default: `"en"`)

**Test with cURL:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tests/sample_satellite.jpg" \
  -F "x=20" \
  -F "y=20" \
  -F "width=150" \
  -F "height=150" \
  -F "query=Describe the land cover in this region"
```

**Response (JSON):**
```json
{
  "caption": "this satellite image shows a forested area with dense tree cover",
  "confidence": 0.8421,
  "low_confidence_flag": false,
  "geojson": {
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[[20, 20], [170, 20], [170, 170], [20, 170], [20, 20]]]
    },
    "properties": {
      "query": "Describe the land cover in this region",
      "answer": "this satellite image shows a forested area with dense tree cover",
      "confidence": 0.8421,
      "model_variant": "lora"
    }
  },
  "audit": {
    "timestamp": "2026-09-26T16:00:00.000000Z",
    "query": "Describe the land cover in this region",
    "roi": {"x": 20, "y": 20, "width": 150, "height": 150},
    "model": "BLIP + LoRA (UC Merced + EuroSAT adapted)",
    "output": "this satellite image shows a forested area with dense tree cover",
    "confidence": 0.8421,
    "latency_sec": 0.312,
    "image_filename": "sample_satellite.jpg"
  },
  "pipeline_metadata": {
    "sar_fusion_enabled": false,
    "sar_status": "stub active (architecture placeholder)",
    "language": "en",
    "device": "cpu"
  }
}
```

---

## What is Real vs. What is Stubbed (Q&A Reference for Judges)

| Component | Status | Details |
|---|---|---|
| **VLM Backbone** | **Real** | `Salesforce/blip-image-captioning-base` conditional generator |
| **LoRA Domain Adaptation** | **Real** | Trained on unified UC Merced + EuroSAT dataset across 4 unified land-cover classes (`forest`, `water`, `agricultural`, `urban`) across 2 sensor resolutions |
| **Confidence Scoring** | **Real** | Computed dynamically from average token generation softmax probabilities |
| **Server-Side ROI Cropping** | **Real** | Pixel coordinates bounding box cropped server-side via PIL before model forward pass |
| **GeoJSON & Audit Generation** | **Real** | Standards-compliant GeoJSON Polygon feature & JSON audit log |
| **Optical-SAR Fusion** | **Stub** | Architecture placeholder slot in backend schema and frontend toggle; model is optical-only |
| **Multi-Language Output** | **Stub / Extensible** | Language parameter accepted; translation model slot reserved |
