import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from backend.agentic_pipeline import run_agentic_pipeline, extract_geospatial_metrics

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "samples")

COMPLEX_QUERIES = [
    # 1. Deforestation & Forestry
    ("forest.png", "Is there any evidence of deforestation, logging, or canopy loss in this area?"),
    # 2. Wildfire & Fuel Moisture
    ("forest.png", "Evaluate the wildfire hazard, drought aridity, and fuel moisture level."),
    # 3. Water Quality & Hydrology
    ("water.png", "What is the water quality of this river and does it show sediment load or algal blooms?"),
    # 4. Urban Density & Road Connectivity
    ("urban.png", "Assess the urban building density, zoning characteristics, and road connectivity."),
    # 5. Agricultural Health & Irrigation
    ("agricultural.png", "Analyze the crop vitality, irrigation status, and parcel organization."),
    # 6. Aviation / Helicopter Landing Safety
    ("urban.png", "Is this ground suitable as an emergency helicopter landing zone?"),
    # 7. Solar Energy Potential
    ("urban.png", "What is the rooftop solar photovoltaic generation potential here?"),
    # 8. Comparative Spatial Analysis
    ("agricultural.png", "Compare the vegetation coverage against built-up impervious surfaces."),
    # 9. Military / Tactical Assessment
    ("forest.png", "Are there any strategic military bunkers or perimeter fortifications?"),
    # 10. Open Complex Question in Hindi
    ("forest.png", "क्या इस वन क्षेत्र में आग लगने या सूखे का कोई खतरा है?", "hi"),
]

def run_tests():
    for idx, item in enumerate(COMPLEX_QUERIES, 1):
        filename = item[0]
        query = item[1]
        lang = item[2] if len(item) > 2 else "en"

        img_path = os.path.join(SAMPLES_DIR, filename)
        image = Image.open(img_path)

        res = run_agentic_pipeline(
            image=image,
            query=query,
            vlm_caption="this satellite image shows a natural landscape",
            confidence=0.84,
            enable_sar=True,
            language=lang,
        )

        print(f"\n==================================================")
        print(f"TEST {idx}: [{lang.upper()}] Query: '{query}'")
        print(f"Task Intent: {res['task_intent']}")
        print(f"Output Paragraph ({len(res['detailed_paragraph'].split('. '))} sentences):")
        print(res["detailed_paragraph"].encode("utf-8", errors="replace").decode("utf-8"))
        import re
        sentences = [s for s in re.split(r'[.।!?]\s*', res["detailed_paragraph"]) if s.strip()]
        assert len(sentences) >= 4, f"Should be at least 4 sentences, got {len(sentences)}"
        assert res["sar_fusion"]["sar_enabled"] is True

    print("\n>>> ALL 10 COMPLEX QUESTION TESTS PASSED! <<<")

if __name__ == "__main__":
    run_tests()
