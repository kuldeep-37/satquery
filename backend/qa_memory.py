"""SatQuery AI — Question-Answer Memory & Persistent Dataset Store.

Persists all user-submitted questions, model answers, ROI crops, and metadata
to an SQLite database and disk storage. Serves as:
1. Long-term interaction memory
2. In-context memory retrieval (continuous learning across queries)
3. Active fine-tuning dataset for continuous LoRA model adaptation.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional
from PIL import Image

logger = logging.getLogger("satquery-memory")

DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs", "qa_memory.db")
)
DEFAULT_CROPS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "outputs", "memory_crops")
)


def init_memory_db(db_path: str = DEFAULT_DB_PATH, crops_dir: str = DEFAULT_CROPS_DIR) -> None:
    """Initialize SQLite database schema and crop storage directory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                crop_path TEXT NOT NULL,
                roi_x REAL NOT NULL,
                roi_y REAL NOT NULL,
                roi_w REAL NOT NULL,
                roi_h REAL NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                short_caption TEXT NOT NULL,
                confidence REAL NOT NULL,
                intent TEXT NOT NULL,
                language TEXT NOT NULL,
                trained INTEGER DEFAULT 0,
                trained_at TEXT
            )
            """
        )
        conn.commit()


def save_qa_record(
    cropped_image: Image.Image,
    question: Optional[str],
    answer: str,
    short_caption: str,
    roi: Dict[str, float],
    confidence: float,
    intent: str,
    language: str = "en",
    image_filename: str = "satellite_image.png",
    db_path: str = DEFAULT_DB_PATH,
    crops_dir: str = DEFAULT_CROPS_DIR,
) -> Dict[str, Any]:
    """Save an interaction (image crop + Q&A metadata) to disk and database."""
    init_memory_db(db_path, crops_dir)

    timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    timestamp_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    # Save cropped image file to disk
    crop_filename = f"crop_{timestamp_slug}.png"
    crop_filepath = os.path.join(crops_dir, crop_filename)
    try:
        cropped_image.convert("RGB").save(crop_filepath, format="PNG")
    except Exception as e:
        logger.warning(f"Could not save cropped image to disk: {e}")
        crop_filepath = ""

    clean_question = (question or "Autonomous Land Cover Captioning").strip()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO qa_memory (
                timestamp, image_filename, crop_path,
                roi_x, roi_y, roi_w, roi_h,
                question, answer, short_caption,
                confidence, intent, language, trained
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                timestamp_iso,
                image_filename,
                crop_filepath,
                float(roi.get("x", 0)),
                float(roi.get("y", 0)),
                float(roi.get("width", 0)),
                float(roi.get("height", 0)),
                clean_question,
                answer,
                short_caption,
                float(confidence),
                intent,
                language,
            ),
        )
        record_id = cursor.lastrowid
        conn.commit()

    logger.info(f"Stored Q&A record #{record_id} in memory: '{clean_question[:40]}...'")

    return {
        "id": record_id,
        "timestamp": timestamp_iso,
        "image_filename": image_filename,
        "crop_path": crop_filepath,
        "question": clean_question,
        "answer": answer,
        "short_caption": short_caption,
        "confidence": confidence,
        "intent": intent,
        "language": language,
        "trained": 0,
    }


def get_all_records(
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 100,
    trained_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Retrieve stored Q&A memory records."""
    init_memory_db(db_path)

    query = "SELECT id, timestamp, image_filename, crop_path, roi_x, roi_y, roi_w, roi_h, question, answer, short_caption, confidence, intent, language, trained, trained_at FROM qa_memory"
    params: List[Any] = []

    if trained_only is not None:
        query += " WHERE trained = ?"
        params.append(1 if trained_only else 0)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    records = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        for row in cursor.fetchall():
            records.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "image_filename": row["image_filename"],
                "crop_path": row["crop_path"],
                "roi": {
                    "x": row["roi_x"],
                    "y": row["roi_y"],
                    "width": row["roi_w"],
                    "height": row["roi_h"],
                },
                "question": row["question"],
                "answer": row["answer"],
                "short_caption": row["short_caption"],
                "confidence": row["confidence"],
                "intent": row["intent"],
                "language": row["language"],
                "trained": bool(row["trained"]),
                "trained_at": row["trained_at"],
            })

    return records


def get_memory_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Return memory counts: total stored, trained in LoRA, and pending training."""
    init_memory_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM qa_memory")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM qa_memory WHERE trained = 1")
        trained = cursor.fetchone()[0]

        cursor.execute("SELECT timestamp, trained_at FROM qa_memory ORDER BY id DESC LIMIT 1")
        last_row = cursor.fetchone()
        last_updated = last_row[0] if last_row else None
        last_trained = last_row[1] if (last_row and last_row[1]) else None

    return {
        "total_records": total,
        "trained_records": trained,
        "pending_training": total - trained,
        "last_updated": last_updated,
        "last_trained": last_trained,
    }


def mark_as_trained(
    record_ids: List[int],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Flag records as successfully incorporated into model training."""
    if not record_ids:
        return

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    init_memory_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        placeholders = ",".join("?" for _ in record_ids)
        cursor.execute(
            f"UPDATE qa_memory SET trained = 1, trained_at = ? WHERE id IN ({placeholders})",
            [now_iso] + record_ids,
        )
        conn.commit()


def get_relevant_memory(
    query: Optional[str],
    db_path: str = DEFAULT_DB_PATH,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """Retrieve previous questions and answers that semantically or lexically match the query.

    Used by the agentic pipeline for immediate in-context continuous learning.
    """
    if not query or not query.strip():
        return []

    records = get_all_records(db_path=db_path, limit=50)
    if not records:
        return []

    q_words = set(re_tokenize(query.lower()))
    scored_records = []

    for r in records:
        prev_q = r["question"].lower()
        if prev_q == "autonomous land cover captioning":
            continue
        prev_words = set(re_tokenize(prev_q))
        overlap = len(q_words.intersection(prev_words))
        if overlap > 0:
            scored_records.append((overlap, r))

    scored_records.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored_records[:top_k]]


def re_tokenize(text: str) -> List[str]:
    """Helper tokenizer for keyword matching."""
    import re
    return [w for w in re.findall(r"\w+", text) if len(w) > 2]


def clear_memory(
    db_path: str = DEFAULT_DB_PATH,
    crops_dir: str = DEFAULT_CROPS_DIR,
) -> None:
    """Clear memory records and remove stored image crops."""
    init_memory_db(db_path, crops_dir)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM qa_memory")
        conn.commit()

    if os.path.exists(crops_dir):
        for f in os.listdir(crops_dir):
            try:
                os.remove(os.path.join(crops_dir, f))
            except Exception:
                pass


def update_qa_record(
    record_id: int,
    answer: str,
    short_caption: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """Update or correct a memorized answer before model adaptation (Human-in-the-loop)."""
    init_memory_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if short_caption:
            cursor.execute(
                "UPDATE qa_memory SET answer = ?, short_caption = ?, trained = 0 WHERE id = ?",
                (answer.strip(), short_caption.strip(), int(record_id)),
            )
        else:
            cursor.execute(
                "UPDATE qa_memory SET answer = ?, trained = 0 WHERE id = ?",
                (answer.strip(), int(record_id)),
            )
        conn.commit()
        return cursor.rowcount > 0

