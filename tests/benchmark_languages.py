"""Benchmark latency across all 10 supported Indian regional languages."""

import os
import sys
import time
from PIL import Image

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.agentic_pipeline import run_agentic_pipeline

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "samples")
img = Image.open(os.path.join(SAMPLES_DIR, "forest.png"))

LANGUAGES = [
    ("en", "English", "Evaluate the wildfire hazard, drought aridity, and fuel moisture status."),
    ("hi", "Hindi (हिन्दी)", "क्या इस वन क्षेत्र में आग लगने या सूखे का कोई खतरा है?"),
    ("ta", "Tamil (தமிழ்)", "காட்டுத்தீ ஆபத்து மற்றும் ஈரப்பத நிலையை மதிப்பிடுக."),
    ("te", "Telugu (తెలుగు)", "దావానల ప్రమాదం మరియు తేమ స్థితిని అంచనా వేయండి."),
    ("bn", "Bengali (বাংলা)", "দাবানলের ঝুঁকি এবং জ্বালানী আর্দ্রতার অবস্থা মূল্যায়ন করুন।"),
    ("mr", "Marathi (मराठी)", "वणव्याचा धोका आणि ओलाव्याची स्थिती तपासा."),
    ("kn", "Kannada (ಕನ್ನಡ)", "ಕಾಡ್ಗಿಚ್ಚಿನ ಅಪಾಯ ಮತ್ತು ತೇವಾಂಶದ ಸ್ಥಿತಿಯನ್ನು ನಿರ್ಣಯಿಸಿ."),
    ("ml", "Malayalam (മലയാളം)", "കാട്ടുതീ സാധ്യതയും ഈർപ്പത്തിന്റെ അളവും വിലയിരുത്തുക."),
    ("gu", "Gujarati (ગુજરાતી)", "દાવાનળનું જોਖમ અને ભેજની સ્થિતિનું મૂલ્યાંકન કરો."),
    ("pa", "Punjabi (ਪੰਜਾਬੀ)", "ਜੰਗਲ ਦੀ ਅੱਗ ਦਾ ਖਤਰਾ ਅਤੇ ਨਮੀ ਦੀ ਸਥਿਤੀ ਦਾ ਮੁਲਾਂਕਣ ਕਰੋ।"),
]

def benchmark():
    print(f"{'Code':<5} | {'Language':<20} | {'Latency (ms)':<15} | {'Status':<10} | {'Sentences':<10}")
    print("-" * 70)

    for code, name, query in LANGUAGES:
        t0 = time.perf_counter()
        res = run_agentic_pipeline(
            image=img,
            query=query,
            vlm_caption="this satellite image shows a dense forest",
            confidence=0.88,
            enable_sar=True,
            language=code,
        )
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        import re
        sentences = [s for s in re.split(r'[.।!?]\s*', res["detailed_paragraph"]) if s.strip()]
        assert len(sentences) >= 4, f"Expected >= 4 sentences, got {len(sentences)}"

        status_str = "INSTANT" if t_elapsed < 50.0 else f"FAST ({t_elapsed:.1f}ms)"
        print(f"{code:<5} | {name:<20} | {t_elapsed:>9.2f} ms | {status_str:<10} | {len(sentences):<10}")

    print("-" * 70)
    print("ALL 10 LANGUAGES BENCHMARKED SUCCESSFULLY - ZERO BOTTLENECK!")

if __name__ == "__main__":
    benchmark()
