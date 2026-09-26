"""High-Speed Multi-Language Translation Engine for Indian Regional Languages.

Features:
1. Persistent Disk Cache (`backend/translation_cache.json`) for sub-millisecond instant responses.
2. Concurrent Parallel ThreadPoolExecutor for simultaneous multi-sentence translation.
3. Pre-seeded geospatial domain vocabulary across 10 Indian regional languages.
4. Graceful offline/timeout resilience with zero UI blocking.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple
from deep_translator import MyMemoryTranslator

logger = logging.getLogger("satquery-translator")

# Language code mapping to ISO-locale for MyMemory / translators
INDIAN_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "native": "English", "locale": "en-GB"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "locale": "hi-IN"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "locale": "ta-IN"},
    "te": {"name": "Telugu", "native": "తెలుగు", "locale": "te-IN"},
    "bn": {"name": "Bengali", "native": "বাংলা", "locale": "bn-IN"},
    "mr": {"name": "Marathi", "native": "मराठी", "locale": "mr-IN"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "locale": "kn-IN"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "locale": "ml-IN"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "locale": "gu-IN"},
    "pa": {"name": "Punjabi", "native": "ਪੰਜਾਬੀ", "locale": "pa-IN"},
}

CACHE_FILE = os.path.join(os.path.dirname(__file__), "translation_cache.json")
_TRANSLATION_CACHE: Dict[Tuple[str, str], str] = {}


def _load_cache() -> None:
    """Load persistent translation cache from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    src = entry.get("text", "")
                    lang = entry.get("lang", "")
                    tr = entry.get("translated", "")
                    if src and lang and tr:
                        _TRANSLATION_CACHE[(src, lang)] = tr
            logger.info(f"Loaded {len(_TRANSLATION_CACHE)} translations from disk cache.")
        except Exception as e:
            logger.warning(f"Could not read translation cache file: {e}")


def _save_cache() -> None:
    """Persist translation cache to disk in background-safe manner."""
    try:
        entries = [
            {"text": k[0], "lang": k[1], "translated": v}
            for k, v in _TRANSLATION_CACHE.items()
        ]
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"Failed to save translation cache to disk: {e}")


# Initialize cache at module load
_load_cache()


def get_supported_languages() -> Dict[str, Dict[str, str]]:
    """Return dictionary of supported Indian languages with native scripts."""
    return INDIAN_LANGUAGES


def _translate_single_clause(clause: str, target_lang: str, source_lang: str = "en") -> str:
    """Translate a single sentence or phrase with fast caching and timeout."""
    clean = clause.strip()
    if not clean or target_lang == source_lang:
        return clean

    cache_key = (clean, target_lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    # Check instant dictionary from multilingual_engine
    try:
        from backend.multilingual_engine import translate_caption_instant
        fast_tr = translate_caption_instant(clean, target_lang)
        if fast_tr and fast_tr != clean:
            _TRANSLATION_CACHE[cache_key] = fast_tr
            return fast_tr
    except Exception:
        pass

    target_info = INDIAN_LANGUAGES.get(target_lang.lower())
    if not target_info:
        return clean

    source_info = INDIAN_LANGUAGES.get(source_lang.lower(), {"locale": "en-GB"})

    def _fetch_external() -> Optional[str]:
        # 1. MyMemory Translator
        try:
            translator = MyMemoryTranslator(
                source=source_info["locale"],
                target=target_info["locale"],
            )
            res = translator.translate(clean)
            if res and not res.startswith("MYMEMORY WARNING:") and not res.startswith("QUERY LENGTH"):
                return res
        except Exception:
            pass

        # 2. Google Translator Fallback
        try:
            from deep_translator import GoogleTranslator
            res = GoogleTranslator(source="auto", target=target_lang).translate(clean)
            if res:
                return res
        except Exception:
            pass
        return None

    # Execute network lookup with strict 1.2-second timeout to prevent lag
    try:
        with ThreadPoolExecutor(max_workers=1) as single_exec:
            fut = single_exec.submit(_fetch_external)
            translated = fut.result(timeout=1.2)
            if translated:
                _TRANSLATION_CACHE[cache_key] = translated
                _save_cache()
                return translated
    except Exception as e:
        logger.debug(f"External translation timed out or failed for clause: {e}")

    # Return original clause if network unavailable or timed out
    return clean


def translate_text(text: str, target_lang: str, source_lang: str = "en") -> str:
    """High-speed translation with parallel sentence dispatch and persistent cache.
    
    1. Checks if entire text is in cache -> 0ms.
    2. Splits multi-sentence paragraphs into clauses.
    3. Translates uncached sentences concurrently using ThreadPoolExecutor.
    4. Automatically updates persistent cache on disk.
    """
    clean_text = text.strip() if text else ""
    if not clean_text or target_lang == source_lang:
        return clean_text

    cache_key = (clean_text, target_lang)
    if cache_key in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[cache_key]

    # If multi-sentence paragraph, translate sentences in parallel
    if len(clean_text) > 120 or ". " in clean_text or "। " in clean_text:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?।])\s+', clean_text) if s.strip()]
        if len(sentences) > 1:
            translated_sentences = [None] * len(sentences)
            missing_indices = []

            # 1. Check cache for each individual sentence
            for i, s in enumerate(sentences):
                ck = (s, target_lang)
                if ck in _TRANSLATION_CACHE:
                    translated_sentences[i] = _TRANSLATION_CACHE[ck]
                else:
                    missing_indices.append(i)

            # 2. Concurrently fetch all uncached sentences in parallel
            if missing_indices:
                max_workers = min(len(missing_indices), 6)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(_translate_single_clause, sentences[i], target_lang, source_lang): i
                        for i in missing_indices
                    }
                    for fut in as_completed(future_to_idx):
                        idx = future_to_idx[fut]
                        try:
                            translated_sentences[idx] = fut.result()
                        except Exception as e:
                            logger.debug(f"Thread translation error for idx {idx}: {e}")
                            translated_sentences[idx] = sentences[idx]

            # 3. Assemble full paragraph in original order
            combined = " ".join([ts for ts in translated_sentences if ts])
            _TRANSLATION_CACHE[cache_key] = combined
            _save_cache()
            return combined

    # Single clause translation
    result = _translate_single_clause(clean_text, target_lang=target_lang, source_lang=source_lang)
    _TRANSLATION_CACHE[cache_key] = result
    _save_cache()
    return result
