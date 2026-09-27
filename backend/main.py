"""FastAPI Backend Service for Satellite Imagery VQA.

Provides endpoints:
- GET /health: Status indicator and model readiness.
- POST /analyze: Multipart image upload + ROI bounding box + query analysis.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.backend import SatQueryModel, export_geojson
from backend.agentic_pipeline import run_agentic_pipeline, compute_optical_sar_fusion
from backend.translator import get_supported_languages, translate_text
from backend.pdf_report import generate_audit_pdf
from backend.qa_memory import (
    save_qa_record,
    get_all_records,
    get_memory_stats,
    get_relevant_memory,
    clear_memory,
    update_qa_record,
    DEFAULT_CROPS_DIR,
)
from backend.continuous_learner import train_lora_on_memory

# Ensure UTF-8 output on Windows for Indian language logging
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("satquery-api")

# Config constants
DEFAULT_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "blip-lora-satellite-adapter")
CONFIDENCE_THRESHOLD = 0.55
MODEL_INSTANCE: Optional[SatQueryModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load model once at startup, clean up on shutdown."""
    global MODEL_INSTANCE
    adapter_path = os.environ.get("LORA_ADAPTER_PATH", DEFAULT_ADAPTER_DIR)
    
    if os.path.isdir(adapter_path):
        logger.info(f"Loading SatQuery model with LoRA adapter from: {adapter_path}")
        MODEL_INSTANCE = SatQueryModel(adapter_path=adapter_path, confidence_threshold=CONFIDENCE_THRESHOLD)
    else:
        logger.warning(
            f"Adapter directory '{adapter_path}' not found on disk. Falling back to base BLIP model."
        )
        MODEL_INSTANCE = SatQueryModel(adapter_path=None, confidence_threshold=CONFIDENCE_THRESHOLD)

    try:
        MODEL_INSTANCE.load_model()
        logger.info("Model loaded successfully into memory. API ready for inference.")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}", exc_info=True)
        # Keep MODEL_INSTANCE as-is so /health can report status accurately

    yield

    logger.info("Shutting down API server...")


app = FastAPI(
    title="SatQuery AI API",
    description="Agentic Satellite Imagery VQA Backend with LoRA Domain Adaptation",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable permissive CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health and Model Readiness Check")
async def health_check() -> Dict[str, Any]:
    """Returns service health status and model readiness for client UI indicators."""
    is_loaded = MODEL_INSTANCE is not None and MODEL_INSTANCE.is_loaded
    has_adapter = MODEL_INSTANCE is not None and MODEL_INSTANCE.has_adapter
    device = MODEL_INSTANCE.device if MODEL_INSTANCE else "unknown"

    return {
        "status": "ok",
        "model_loaded": is_loaded,
        "has_adapter": has_adapter,
        "device": device,
        "pipeline_features": {
            "sar_fusion": "stub (untrained architecture slot)",
            "multilingual": ["en"],
        },
    }


def _parse_roi_coords(
    x: Optional[float],
    y: Optional[float],
    width: Optional[float],
    height: Optional[float],
    roi: Optional[str],
) -> tuple[float, float, float, float]:
    """Parse ROI coordinates from either explicit form fields or JSON roi string."""
    if roi:
        try:
            roi_obj = json.loads(roi) if isinstance(roi, str) else roi
            x = float(roi_obj.get("x", roi_obj.get("left", 0)))
            y = float(roi_obj.get("y", roi_obj.get("top", 0)))
            width = float(roi_obj.get("width", 0))
            height = float(roi_obj.get("height", 0))
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid ROI JSON string: {e}",
            )

    if x is None or y is None or width is None or height is None:
        raise HTTPException(
            status_code=422,
            detail="Missing ROI coordinates. Provide 'x', 'y', 'width', 'height' form fields or a 'roi' JSON string.",
        )

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid ROI dimensions: width ({width}) and height ({height}) must be strictly positive.",
        )

    return float(x), float(y), float(width), float(height)


@app.post("/analyze", summary="Analyze Satellite ROI with VLM")
async def analyze_image(
    file: UploadFile = File(..., description="Satellite image file (PNG, JPG, TIFF)"),
    x: Optional[float] = Form(None, description="ROI top-left X coordinate"),
    y: Optional[float] = Form(None, description="ROI top-left Y coordinate"),
    width: Optional[float] = Form(None, description="ROI box width"),
    height: Optional[float] = Form(None, description="ROI box height"),
    roi: Optional[str] = Form(None, description="Optional ROI JSON string e.g. {'x':0,'y':0,'width':100,'height':100}"),
    query: Optional[str] = Form(None, description="Natural language question or prompt"),
    enable_sar: Optional[bool] = Form(False, description="Placeholder toggle for Optical-SAR fusion"),
    language: Optional[str] = Form("en", description="Target output language code"),
) -> Dict[str, Any]:
    """Crop uploaded image to the requested ROI, run LoRA BLIP inference, and compute confidence."""
    # Ensure model is ready
    if MODEL_INSTANCE is None or not MODEL_INSTANCE.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded or initialized on the server.",
        )

    # 1. Decode Image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.load()  # verify image data is readable
    except Exception as e:
        logger.warning(f"Failed to decode uploaded image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot decode image file. Ensure a valid PNG, JPG, or TIFF format: {e}",
        )

    # 2. Parse and Validate ROI
    rx, ry, rw, rh = _parse_roi_coords(x, y, width, height, roi)
    img_w, img_h = image.size

    # Clamp coordinates safely to image boundary
    crop_x1 = max(0.0, min(rx, img_w - 1))
    crop_y1 = max(0.0, min(ry, img_h - 1))
    crop_x2 = max(crop_x1 + 1.0, min(rx + rw, float(img_w)))
    crop_y2 = max(crop_y1 + 1.0, min(ry + rh, float(img_h)))

    if (crop_x2 - crop_x1) < 2 or (crop_y2 - crop_y1) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"ROI [{rx}, {ry}, {rw}, {rh}] lies entirely outside image dimensions ({img_w}x{img_h}).",
        )

    # 3. Crop server-side
    cropped_roi = image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))

    # 4. Input Query Preprocessing (Multi-language Translation if Indian script/language selected)
    target_lang = (language or "en").lower().strip()
    clean_query = query.strip() if query and query.strip() else None
    query_for_model = clean_query

    if clean_query:
        has_non_ascii = any(ord(c) > 127 for c in clean_query)
        if has_non_ascii:
            try:
                translated_q = translate_text(clean_query, target_lang="en", source_lang=target_lang)
                if translated_q and translated_q.strip():
                    query_for_model = translated_q.strip()
                    logger.info(f"Translated regional query '{clean_query}' [{target_lang}] -> '{query_for_model}' [en]")
            except Exception as e:
                logger.warning(f"Regional query translation to English skipped: {e}")

    # 5. Model Inference
    try:
        result = MODEL_INSTANCE.infer(image=cropped_roi, prompt=query_for_model, use_lora=True)
    except Exception as e:
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(e)}",
        )

    # 6. Retrieve In-Context Memory (Relevant previous Q&As for immediate learning)
    recalled_memories = get_relevant_memory(clean_query, top_k=2) if clean_query else []

    # 7. Agentic Reasoning & Contextual Paragraph Synthesis
    agentic_result = run_agentic_pipeline(
        image=cropped_roi,
        query=query_for_model,
        vlm_caption=result.caption,
        confidence=result.confidence,
        enable_sar=bool(enable_sar),
        language=target_lang,
        prior_memories=recalled_memories,
    )

    paragraph_answer = agentic_result["detailed_paragraph"]

    # 8. Store Question & Answer Interaction into Continuous Learning Memory
    memory_record = save_qa_record(
        cropped_image=cropped_roi,
        question=clean_query,
        answer=paragraph_answer,
        short_caption=agentic_result["short_caption"],
        roi={"x": rx, "y": ry, "width": rw, "height": rh},
        confidence=result.confidence,
        intent=agentic_result["task_intent"],
        language=target_lang,
        image_filename=file.filename or "satellite_capture.png",
    )
    mem_stats = get_memory_stats()

    # 9. Format GeoJSON & Audit
    geojson_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [round(rx, 2), round(ry, 2)],
                [round(rx + rw, 2), round(ry, 2)],
                [round(rx + rw, 2), round(ry + rh, 2)],
                [round(rx, 2), round(ry + rh, 2)],
                [round(rx, 2), round(ry, 2)],
            ]],
        },
        "properties": {
            "query": clean_query or "Land cover captioning",
            "answer": paragraph_answer,
            "short_caption": agentic_result["short_caption"],
            "confidence": round(result.confidence, 4),
            "model_variant": result.model_variant,
            "task_intent": agentic_result["task_intent"],
            "language": target_lang,
        },
    }

    audit_entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": clean_query or "Land cover captioning",
        "roi": {"x": round(rx, 2), "y": round(ry, 2), "width": round(rw, 2), "height": round(rh, 2)},
        "model": "BLIP + LoRA (UC Merced + EuroSAT adapted)",
        "output": paragraph_answer,
        "short_output": agentic_result["short_caption"],
        "confidence": round(result.confidence, 4),
        "latency_sec": round(result.latency_sec, 3),
        "image_filename": file.filename,
        "language": target_lang,
        "task_intent": agentic_result["task_intent"],
    }

    # Server-side logging per request
    logger.info(
        f"Analyzed ROI: ({rx:.1f}, {ry:.1f}, {rw:.1f}x{rh:.1f}) | "
        f"Query: '{clean_query}' | Lang: {target_lang} | Conf: {result.confidence:.1%} | "
        f"Intent: {agentic_result['task_intent']} | Stored Memory ID: #{memory_record['id']} | Latency: {result.latency_sec:.3f}s"
    )

    crop_basename = os.path.basename(memory_record["crop_path"]) if memory_record.get("crop_path") else None

    return {
        "caption": paragraph_answer,
        "short_caption": agentic_result["short_caption"],
        "confidence": round(result.confidence, 4),
        "low_confidence_flag": result.low_confidence,
        "geojson": geojson_feature,
        "audit": audit_entry,
        "agentic_intel": {
            "task_intent": agentic_result["task_intent"],
            "optical_metrics": agentic_result["optical_metrics"],
            "sar_fusion": agentic_result["sar_fusion"],
            "language": target_lang,
            "english_paragraph": agentic_result.get("english_paragraph", ""),
            "recalled_memories": recalled_memories,
        },
        "memory": {
            "record_id": memory_record["id"],
            "stored": True,
            "crop_url": f"/memory/crops/{crop_basename}" if crop_basename else None,
            "total_records": mem_stats["total_records"],
            "trained_records": mem_stats["trained_records"],
            "pending_training": mem_stats["pending_training"],
            "recalled_count": len(recalled_memories),
        },
        "pipeline_metadata": {
            "sar_fusion_enabled": bool(enable_sar),
            "sar_status": "Active (Simulated Sentinel-1 Dual-Pol C-Band)" if enable_sar else "Inactive (Optical Only)",
            "language": target_lang,
            "device": MODEL_INSTANCE.device,
        },
    }


@app.get("/memory", summary="Get Stored Q&A Memory Records and Learning Stats")
async def get_memory_history(limit: Optional[int] = 50):
    """Retrieve historical questions, answers, and training readiness."""
    stats = get_memory_stats()
    records = get_all_records(limit=int(limit or 50))
    return {
        "status": "ok",
        "stats": stats,
        "records": records,
    }


@app.post("/train/memory", summary="Fine-Tune Model Weights on Stored Previous Q&As")
async def train_on_memory_endpoint(
    epochs: Optional[int] = Form(2),
    learning_rate: Optional[float] = Form(5e-5),
    train_all: Optional[bool] = Form(True),
):
    """Execute continuous LoRA adaptation on accumulated question-answer pairs and reload weights."""
    if MODEL_INSTANCE is None or not MODEL_INSTANCE.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or ready for training.",
        )

    try:
        report = train_lora_on_memory(
            model_instance=MODEL_INSTANCE,
            epochs=int(epochs or 2),
            lr=float(learning_rate or 5e-5),
            train_all=bool(train_all),
        )
        return report
    except Exception as e:
        logger.error(f"Continuous training execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Continuous training failed: {str(e)}",
        )


@app.post("/memory/clear", summary="Clear Stored Memory Records")
async def clear_memory_endpoint():
    """Clear all stored questions, answers, and cached image crops."""
    clear_memory()
    return {"status": "ok", "message": "Memory store cleared successfully.", "stats": get_memory_stats()}


@app.post("/memory/update", summary="Update or Correct a Stored Answer (Human-in-the-Loop)")
async def update_memory_answer_endpoint(
    record_id: int = Form(...),
    answer: str = Form(...),
    short_caption: Optional[str] = Form(None),
):
    """Correct an answer in memory before training so the model learns from verified human feedback."""
    success = update_qa_record(record_id=int(record_id), answer=answer, short_caption=short_caption)
    if not success:
        raise HTTPException(status_code=404, detail=f"Memory record #{record_id} not found.")
    return {"status": "ok", "message": f"Record #{record_id} updated with verified feedback.", "stats": get_memory_stats()}


@app.get("/memory/export", summary="Export Q&A Memory Dataset as JSONL")
async def export_memory_dataset():
    """Export all stored questions, answers, and image crop paths as a JSONL training dataset."""
    from fastapi.responses import Response
    records = get_all_records(limit=2000)
    lines = [json.dumps(r) for r in records]
    content = "\n".join(lines)
    return Response(
        content=content,
        media_type="application/x-jsonlines",
        headers={"Content-Disposition": "attachment; filename=satquery_qa_dataset.jsonl"},
    )


@app.get("/languages", summary="Supported Indian Regional Languages")
async def get_languages():
    """Return dictionary of supported Indian languages with scripts."""
    return {"languages": get_supported_languages()}


@app.post("/export/pdf", summary="Download Official Audit PDF Report")
async def export_pdf_report(
    file: UploadFile = File(...),
    x: Optional[float] = Form(0),
    y: Optional[float] = Form(0),
    width: Optional[float] = Form(100),
    height: Optional[float] = Form(100),
    query: Optional[str] = Form(""),
    paragraph: Optional[str] = Form(""),
    confidence: Optional[float] = Form(0.75),
    latency_sec: Optional[float] = Form(0.25),
    language: Optional[str] = Form("en"),
    enable_sar: Optional[bool] = Form(False),
    english_paragraph: Optional[str] = Form(None),
):
    """Generate and return official ISRO/SIH auditable PDF report with embedded ROI image."""
    from fastapi.responses import Response
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img_w, img_h = image.size
        cx1 = max(0.0, min(float(x or 0), img_w - 1))
        cy1 = max(0.0, min(float(y or 0), img_h - 1))
        cx2 = max(cx1 + 1.0, min(cx1 + float(width or img_w), float(img_w)))
        cy2 = max(cy1 + 1.0, min(cy1 + float(height or img_h), float(img_h)))
        cropped_roi = image.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
    except Exception:
        cropped_roi = None

    sar_data = compute_optical_sar_fusion(cropped_roi or image, paragraph or "land") if enable_sar else None

    audit_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query or "Autonomous Land Cover Analysis",
        "roi": {"x": round(float(x or 0), 1), "y": round(float(y or 0), 1), "width": round(float(width or 100), 1), "height": round(float(height or 100), 1)},
        "model": "BLIP + LoRA (UC Merced + EuroSAT adapted)",
        "confidence": float(confidence or 0.75),
        "latency_sec": float(latency_sec or 0.25),
        "language": language or "en",
        "image_filename": file.filename or "satellite_capture.png",
        "low_confidence_flag": float(confidence or 0.75) < CONFIDENCE_THRESHOLD,
    }

    pdf_bytes = generate_audit_pdf(
        audit_data=audit_payload,
        paragraph_text=paragraph or "Detailed satellite imagery analysis.",
        roi_image=cropped_roi,
        sar_data=sar_data,
        english_paragraph=english_paragraph,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=satquery_audit_report.pdf"},
    )


# Mount memory crops and frontend directory for seamless web access
from fastapi.staticfiles import StaticFiles
os.makedirs(DEFAULT_CROPS_DIR, exist_ok=True)
app.mount("/memory/crops", StaticFiles(directory=DEFAULT_CROPS_DIR), name="memory_crops")

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")



