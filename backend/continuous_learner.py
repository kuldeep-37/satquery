"""SatQuery AI — Continuous LoRA Learner.

Trains and adapts the BLIP LoRA weights directly on accumulated previous questions,
answers, and satellite ROI crops stored in the Q&A memory database.
Provides:
1. Online / One-Click Continuous LoRA Training
2. Hot-Reloading of updated adapter weights into the live API model
3. Monitored training loss tracking and audit persistence.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.qa_memory import (
    DEFAULT_DB_PATH,
    get_all_records,
    get_memory_stats,
    mark_as_trained,
)
try:
    from training.train_lora import attach_lora, ADAPTER_SAVE_PATH
except ImportError:
    ADAPTER_SAVE_PATH = "outputs/blip-lora-satellite-adapter"

    def attach_lora(model):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
        )
        return get_peft_model(model, lora_config)

logger = logging.getLogger("satquery-learner")



class MemoryQADataset(Dataset):
    """PyTorch Dataset wrapping stored satellite ROI images and Q&A targets."""

    def __init__(self, samples: List[Dict[str, Any]], processor: Any, max_length: int = 40):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        image = item["image"].convert("RGB")
        text = item["text"]

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def prepare_training_samples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load images and format target conditioning strings from DB records."""
    valid_samples = []

    for r in records:
        crop_path = r.get("crop_path")
        if not crop_path or not os.path.exists(crop_path):
            continue

        try:
            img = Image.open(crop_path)
            img.load()
        except Exception as e:
            logger.warning(f"Could not load crop image '{crop_path}': {e}")
            continue

        q = r.get("question", "").strip()
        ans = r.get("short_caption", "").strip() or r.get("answer", "").strip()
        
        # If the question was autonomous captioning, target is just the domain caption
        if not q or q.lower() == "autonomous land cover captioning":
            target_text = ans
        else:
            # Multi-turn VQA target conditioning: question -> answer
            target_text = f"question: {q} answer: {ans}"

        valid_samples.append({
            "id": r["id"],
            "image": img,
            "text": target_text,
            "raw_record": r,
        })

    return valid_samples


def train_lora_on_memory(
    model_instance: Any,
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 5e-5,
    db_path: str = DEFAULT_DB_PATH,
    adapter_save_path: str = ADAPTER_SAVE_PATH,
    train_all: bool = True,
) -> Dict[str, Any]:
    """Execute continuous LoRA fine-tuning on stored Q&A pairs and reload live model.

    Args:
        model_instance: The active SatQueryModel instance loaded in backend/main.py.
        epochs: Number of training epochs over accumulated memory.
        batch_size: DataLoader mini-batch size.
        lr: AdamW learning rate for LoRA parameters.
        db_path: Path to qa_memory.db.
        adapter_save_path: Directory where LoRA adapter weights are persisted.
        train_all: If True, trains on all accumulated records; if False, pending only.

    Returns:
        Dictionary with training telemetry: samples trained, initial/final loss, duration.
    """
    start_time = time.perf_counter()

    # 1. Fetch records
    records = get_all_records(
        db_path=db_path,
        limit=200,
        trained_only=None if train_all else False,
    )

    if not records and not train_all:
        # Fallback to all records if no pending records
        records = get_all_records(db_path=db_path, limit=200)

    if not records:
        return {
            "status": "no_data",
            "message": "No stored question-answer records found in memory to train on. Submit queries first.",
            "samples_trained": 0,
        }

    # 2. Prepare dataset
    samples = prepare_training_samples(records)
    if not samples:
        return {
            "status": "no_images",
            "message": "Stored records exist but crop images are missing or unreadable on disk.",
            "samples_trained": 0,
        }

    logger.info(f"Starting continuous LoRA adaptation on {len(samples)} stored Q&A records...")

    # 3. Ensure base model and LoRA adapter are ready
    model_instance._ensure_loaded()
    processor = model_instance._processor
    device = model_instance.device
    dtype = model_instance.torch_dtype

    # Ensure LoRA adapter exists on model
    if model_instance._lora_model is None:
        logger.info("LoRA adapter not currently attached; creating fresh LoRA adapter...")
        model_instance._lora_model = attach_lora(model_instance._base_model)

    trainable_model = model_instance._lora_model
    trainable_model.train()

    # Ensure LoRA parameters have requires_grad=True
    for name, param in trainable_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Filter optimizer parameters to trainable weights
    trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.info("No parameters with 'lora' in name found requiring grad; enabling all parameters...")
        for param in trainable_model.parameters():
            param.requires_grad = True
        trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable_params, lr=lr)

    # 4. Construct DataLoader
    dataset = MemoryQADataset(samples, processor=processor)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
    )

    loss_history: List[float] = []

    # 5. Training Loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        step_count = 0

        for batch in loader:
            batch_tensors = {
                k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
                for k, v in batch.items()
            }

            outputs = trainable_model(**batch_tensors, labels=batch_tensors["input_ids"])
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        loss_history.append(round(avg_loss, 4))
        logger.info(f"Continuous Learning Epoch {epoch + 1}/{epochs} — Avg Loss: {avg_loss:.4f}")

    # Set model back to eval mode
    trainable_model.eval()

    # 6. Save updated adapter to disk
    os.makedirs(adapter_save_path, exist_ok=True)
    trainable_model.save_pretrained(adapter_save_path)
    processor.save_pretrained(adapter_save_path)
    logger.info(f"Updated LoRA adapter weights saved to '{adapter_save_path}'.")

    # 7. Mark trained records in SQLite DB
    trained_ids = [s["id"] for s in samples]
    mark_as_trained(trained_ids, db_path=db_path)

    # 8. Save telemetry report
    telemetry_path = os.path.join(adapter_save_path, "continuous_training_telemetry.json")
    duration_sec = round(time.perf_counter() - start_time, 2)
    report = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_trained": len(samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "duration_sec": duration_sec,
        "loss_history": loss_history,
        "initial_loss": loss_history[0] if loss_history else None,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_reduction": round(loss_history[0] - loss_history[-1], 4) if len(loss_history) > 1 else 0.0,
    }

    try:
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write telemetry JSON: {e}")

    logger.info(
        f"Continuous LoRA Training Complete: {len(samples)} samples | "
        f"Loss: {report['initial_loss']} -> {report['final_loss']} | {duration_sec}s"
    )

    return report
