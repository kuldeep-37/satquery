# Satellite Imagery VQA — SatQuery AI (SIH Prototype)

An agentic VLM system for satellite image captioning, Visual Question Answering (VQA), and land-cover intelligence, adapted to remote sensing via LoRA fine-tuning, with a FastAPI backend and a custom geospatial web frontend. Built toward ISRO's satellite imagery analysis problem statement.

---

## Project Structure

```
satellite-vqa-project/
├── requirements.txt                   # Local & training dependencies
├── data_prep/
│   ├── prepare_dataset.py             # UC Merced dataset loader & balancer
│   └── prepare_combined_dataset.py    # UC Merced + EuroSAT unified taxonomy (4 classes)
├── training/
│   ├── train_lora.py                  # LoRA BLIP training loop, dynamic device support & artifacts
│   ├── run_phase1_training.py         # End-to-end training pipeline orchestrator
│   └── original_colab_notebook.py     # Initial exploratory Colab session (reference)
├── backend/
│   ├── main.py                        # FastAPI service (/health, /analyze, static frontend mount)
│   ├── requirements.txt               # Backend specific dependencies
│   └── README.md                      # Backend API documentation & cURL examples
├── frontend/
│   ├── index.html                     # Semantic geospatial dashboard UI
│   ├── style.css                      # Modern dark navy / cyan design system
│   ├── app.js                         # Canvas coordinate mapping & API caller
│   ├── samples/                       # 1-click demo satellite images (forest, urban, water, agri)
│   └── README.md                      # Frontend architecture & usage guide
├── outputs/                           # Verified artifacts:
│   ├── blip-lora-satellite-adapter/   # Fine-tuned LoRA weights & tokenizer
│   ├── sample_grid.png                # Dataset sanity check grid
│   ├── training_loss.json             # Monotonic loss history across 3 epochs
│   ├── loss_curve.png                 # Rendered training loss plot
│   ├── before_after_comparisons.json  # Base BLIP vs LoRA validation comparisons
│   └── before_after_comparisons.txt   # Human-readable comparison evidence
├── tests/
│   ├── test_backend.py                # SatQueryModel & GeoJSON unit tests
│   └── test_api.py                    # Live FastAPI /health and /analyze integration tests
└── app/                               # Legacy Streamlit prototype (kept for reference)
    ├── app.py
    └── backend.py
```

---

## Quickstart: Run the System

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Launch the Unified Server (Backend + Professional Frontend)
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

- The model and LoRA adapter (`outputs/blip-lora-satellite-adapter`) load once at server startup.
- The UI automatically checks pipeline health via `GET /health`.
- Click any sample chip (**Forest Cover**, **Urban & Housing**, **Water / River**, or **Agricultural Land**) for instant 1-click demoing.
- Drag to draw an ROI bounding box, enter/select a query, and click **Run VLM Analysis**.
- Download **GeoJSON** and **Audit Reports** with 1 click.

### 3. Run Automated Tests
```bash
pytest -v
```
All 19 unit and integration tests run in ~20 seconds.

---

## Domain Adaptation & Training Evidence

Trained on a unified dataset mapping **UC Merced** (256×256 aerial photography) and **EuroSAT** (64×64 Sentinel-2 imagery) into 4 shared land-cover classes (`forest`, `water`, `agricultural`, `urban`).

- **Base Model**: `Salesforce/blip-image-captioning-base` (~247M parameters)
- **Trainable Parameters**: **0.2627%** (LoRA $r=8, \alpha=16$ on `query` and `value` attention projections)
- **Loss Progression (Monotonically Decreasing)**:
  - **Epoch 1**: `8.2728`
  - **Epoch 2**: `7.1413`
  - **Epoch 3**: `6.3698`
- **Saved Evidence**:
  - Sample distribution grid: `outputs/sample_grid.png`
  - Loss curve: `outputs/loss_curve.png`
  - Side-by-side validation comparisons: `outputs/before_after_comparisons.txt`

---

## What's Real vs. What's a Stub (Judges' Technical Q&A)

| Component | Status | Technical Details |
|---|---|---|
| **VLM Backbone** | **Real** | `Salesforce/blip-image-captioning-base` conditional generation |
| **LoRA Domain Adaptation** | **Real** | Adapter fine-tuned across 3 epochs on unified UC Merced + EuroSAT dataset across 4 unified land-cover classes across 2 sensor resolutions |
| **Confidence Scoring** | **Real** | Mean of max softmax probabilities over generated output tokens |
| **Interactive ROI Bounding** | **Real** | Client-side canvas math converts display resolution to intrinsic image pixels and crops server-side with PIL |
| **GeoJSON & Audit Trail** | **Real** | Standard RFC 7946 GeoJSON Polygon feature & ISO 8601 audit record |
| **Optical–SAR Fusion** | **Stub** | Architecture placeholder slot in backend API schema and frontend toggle; model is optical-only |
| **Multi-Language Output** | **Stub / Extensible** | Language parameter accepted; translation model slot reserved |
