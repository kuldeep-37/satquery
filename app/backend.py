"""SatQuery AI — Satellite Image VQA Assistant Backend.

This module contains model loading, inference logic, confidence scoring,
and GeoJSON serialization. It is completely independent of Streamlit
and can be unit-tested standalone.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from peft import PeftModel

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_ID = "Salesforce/blip-image-captioning-base"
DEFAULT_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class InferenceResult:
    """Dataclass holding inference output and metadata."""

    caption: str
    confidence: float
    low_confidence: bool
    latency_sec: float
    model_variant: str  # "base" or "lora"
    prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Container for side-by-side comparison between base and LoRA models."""

    base: InferenceResult
    lora: Optional[InferenceResult] = None

    def __iter__(self) -> Iterator[Optional[InferenceResult]]:
        """Allow tuple unpacking: base_res, lora_res = compare_before_after(...)"""
        yield self.base
        yield self.lora

    def __getitem__(self, key: Union[int, str]) -> Optional[InferenceResult]:
        """Allow indexing by index (0, 1) or key ('base', 'lora')."""
        if key == 0 or key == "base":
            return self.base
        elif key == 1 or key == "lora":
            return self.lora
        raise KeyError(f"Invalid key '{key}'. Use 0/'base' or 1/'lora'.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison to dictionary."""
        return {
            "base": self.base.to_dict(),
            "lora": self.lora.to_dict() if self.lora is not None else None,
        }


def export_geojson(
    image_name: str,
    result: InferenceResult,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    indent: Optional[int] = 2,
) -> str:
    """Build a GeoJSON Feature string from an InferenceResult.

    Properties include caption, confidence, low-confidence flag,
    image_name, latency_sec, and model_variant.
    Geometry is a Point [lon, lat] if coordinates are given, else null.

    Args:
        image_name: Identifier or filename of the analyzed image.
        result: InferenceResult dataclass containing model predictions.
        lat: Optional latitude float (-90 to 90).
        lon: Optional longitude float (-180 to 180).
        indent: JSON indentation spaces (default 2).

    Returns:
        GeoJSON Feature as formatted JSON string.
    """
    geometry = None
    if lat is not None and lon is not None:
        try:
            geometry = {
                "type": "Point",
                "coordinates": [float(lon), float(lat)],
            }
        except (ValueError, TypeError):
            geometry = None

    feature = {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "image_name": str(image_name),
            "caption": result.caption,
            "confidence": round(result.confidence, 4) if isinstance(result.confidence, float) else result.confidence,
            "low_confidence": bool(result.low_confidence),
            "latency_sec": round(result.latency_sec, 3) if isinstance(result.latency_sec, float) else result.latency_sec,
            "model_variant": result.model_variant,
            "prompt": result.prompt,
        },
    }
    return json.dumps(feature, indent=indent)


class SatQueryModel:
    """Satellite image captioning and VQA inference engine.

    Lazily loads a base BLIP conditional generation model, and optionally
    attaches a fine-tuned LoRA adapter if an adapter path is provided.
    Runs on CUDA if available, falling back cleanly to CPU.
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
    ):
        clean_path = str(adapter_path).strip() if adapter_path is not None else ""
        self.adapter_path: Optional[str] = clean_path if clean_path else None
        self.confidence_threshold: float = float(confidence_threshold)
        self.base_model_id: str = base_model_id

        # Determine compute device and precision
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Model and processor state (lazy loaded)
        self._processor: Optional[BlipProcessor] = None
        self._base_model: Optional[BlipForConditionalGeneration] = None
        self._lora_model: Optional[PeftModel] = None
        self._is_loaded: bool = False

    @property
    def has_adapter(self) -> bool:
        """True if an adapter path is configured or a LoRA adapter is loaded."""
        if self._lora_model is not None:
            return True
        if self._is_loaded and self._lora_model is None:
            return False
        return self.adapter_path is not None

    @property
    def is_loaded(self) -> bool:
        """True if weights have been loaded into memory."""
        return self._is_loaded

    def load_model(self) -> SatQueryModel:
        """Explicitly load processor, base model, and optional LoRA adapter."""
        if self._is_loaded:
            return self

        logger.info(f"Loading processor for {self.base_model_id}...")
        self._processor = BlipProcessor.from_pretrained(self.base_model_id)

        logger.info(f"Loading base BLIP model ({self.base_model_id}) on {self.device}...")
        self._base_model = BlipForConditionalGeneration.from_pretrained(
            self.base_model_id,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self._base_model.eval()

        if self.adapter_path:
            if not os.path.exists(self.adapter_path):
                raise FileNotFoundError(
                    f"LoRA adapter path '{self.adapter_path}' does not exist on disk."
                )
            logger.info(f"Attaching LoRA adapter from {self.adapter_path}...")
            self._lora_model = PeftModel.from_pretrained(
                self._base_model,
                self.adapter_path,
            )
            self._lora_model.eval()
            logger.info("LoRA adapter attached successfully.")
        else:
            self._lora_model = None
            logger.info("No adapter path specified; running in base model mode.")

        self._is_loaded = True
        return self

    def _ensure_loaded(self) -> None:
        """Ensure model components are loaded before inference."""
        if not self._is_loaded:
            self.load_model()

    def _generate(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 30,
    ) -> Any:
        """Internal helper to execute model.generate with score outputs.

        Extracted to facilitate unit testing with stubbed/mocked generations.
        """
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

    def infer(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        use_lora: bool = True,
        max_new_tokens: int = 30,
    ) -> InferenceResult:
        """Run captioning or VQA inference on a satellite image.

        Args:
            image: PIL.Image instance.
            prompt: Optional text prompt or question.
            use_lora: If True and adapter is available, uses LoRA weights;
                      otherwise falls back to base model.
            max_new_tokens: Maximum new tokens to generate (default 30).

        Returns:
            InferenceResult dataclass with caption, confidence, latency, etc.
        """
        self._ensure_loaded()

        start_time = time.perf_counter()

        # Ensure image is in RGB mode (handles RGBA, greyscale, TIFFs, etc.)
        rgb_image = image.convert("RGB") if image.mode != "RGB" else image

        # Prepare inputs with optional text prompt
        clean_prompt = prompt.strip() if (prompt and prompt.strip()) else None

        # Check if the prompt is an instruction/question rather than a continuation prefix.
        # Questions/instructions typically ask "describe...", "what is...", "tell me...",
        # end with "?", or contain directive words. BLIP is a conditional caption generator,
        # so passing full questions as a text prefix causes it to emit an immediate EOS (echoing the prompt).
        is_question_or_instruction = False
        if clean_prompt:
            lowered = clean_prompt.lower()
            question_words = (
                "describe", "what", "how", "which", "where", "is there", "are there",
                "identify", "tell", "can you", "classify", "explain"
            )
            if lowered.endswith("?") or any(lowered.startswith(w) for w in question_words):
                is_question_or_instruction = True

        if clean_prompt and not is_question_or_instruction:
            inputs = self._processor(images=rgb_image, text=clean_prompt, return_tensors="pt")
        else:
            # For instructions/questions or empty prompt: run unconditional captioning
            # This generates the fine-tuned satellite domain description directly!
            inputs = self._processor(images=rgb_image, return_tensors="pt")

        # Move tensors to target device and precision (only floating tensors to torch_dtype)
        inputs = {
            k: v.to(device=self.device, dtype=self.torch_dtype)
            if v.is_floating_point()
            else v.to(device=self.device)
            for k, v in inputs.items()
        }

        # Select model and LoRA adapter context
        if use_lora and self._lora_model is not None:
            active_model = self._lora_model
            model_variant = "lora"
            adapter_context = contextlib.nullcontext()
        else:
            model_variant = "base"
            if self._lora_model is not None:
                active_model = self._lora_model
                adapter_context = self._lora_model.disable_adapter()
            else:
                active_model = self._base_model
                adapter_context = contextlib.nullcontext()

        # Run generation
        with adapter_context:
            output = self._generate(active_model, inputs, max_new_tokens=max_new_tokens)

        # Decode output caption
        caption = self._processor.decode(output.sequences[0], skip_special_tokens=True).strip()
        if clean_prompt and caption.lower().startswith(clean_prompt.lower()):
            trimmed = caption[len(clean_prompt):].lstrip(" :.-")
            if trimmed:
                caption = trimmed
            else:
                # If model generated nothing new beyond echoing the prompt, fall back to unconditional captioning
                uncond_inputs = self._processor(images=rgb_image, return_tensors="pt")
                uncond_inputs = {
                    k: (v.to(device=self.device, dtype=self.torch_dtype) if v.is_floating_point() else v.to(device=self.device))
                    for k, v in uncond_inputs.items()
                }
                with adapter_context:
                    uncond_output = self._generate(active_model, uncond_inputs, max_new_tokens=max_new_tokens)
                caption = self._processor.decode(uncond_output.sequences[0], skip_special_tokens=True).strip()
                output = uncond_output


        # Compute confidence score from mean max-softmax probabilities over generated tokens
        if hasattr(output, "scores") and output.scores:
            token_max_probs = [
                float(torch.softmax(s[0], dim=-1).max().item())
                for s in output.scores
            ]
            confidence = sum(token_max_probs) / len(token_max_probs) if token_max_probs else 0.0
        else:
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))
        low_confidence = bool(confidence < self.confidence_threshold)
        latency_sec = time.perf_counter() - start_time

        return InferenceResult(
            caption=caption,
            confidence=confidence,
            low_confidence=low_confidence,
            latency_sec=latency_sec,
            model_variant=model_variant,
            prompt=clean_prompt,
        )

    def compare_before_after(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 30,
    ) -> ComparisonResult:
        """Run both base and LoRA models on the same image for side-by-side comparison.

        If no adapter is loaded, lora result is None.
        """
        self._ensure_loaded()

        base_result = self.infer(
            image=image,
            prompt=prompt,
            use_lora=False,
            max_new_tokens=max_new_tokens,
        )

        lora_result: Optional[InferenceResult] = None
        if self.has_adapter:
            lora_result = self.infer(
                image=image,
                prompt=prompt,
                use_lora=True,
                max_new_tokens=max_new_tokens,
            )

        return ComparisonResult(base=base_result, lora=lora_result)
