"""Live end-to-end verification of innovations:
1. Paragraph output relating directly to questions
2. Indian regional languages (Input & Output)
3. Optical-SAR Fusion telemetry
4. PDF Audit Report generation
"""

import os
import requests

BASE_URL = "http://127.0.0.1:8000"
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "samples")


def test_paragraph_english_sar():
    img_path = os.path.join(SAMPLES_DIR, "forest.png")
    with open(img_path, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/analyze",
            files={"file": ("forest.png", f, "image/png")},
            data={
                "x": "10",
                "y": "10",
                "width": "200",
                "height": "200",
                "query": "Is there dense forest canopy or vegetation in this sector?",
                "language": "en",
                "enable_sar": "true",
            },
        )
    assert res.status_code == 200, res.text
    data = res.json()
    print("\n--- TEST 1: Forest + SAR + Paragraph (EN) ---")
    print("Caption / Paragraph:\n", data["caption"])
    print("Sentences in output:", len(data["caption"].split(". ")))
    print("Confidence:", data["confidence"])
    print("SAR Fusion Info:", data["agentic_intel"]["sar_fusion"])
    assert len(data["caption"].split(". ")) >= 3, "Output should be a coherent multi-sentence paragraph"
    assert data["agentic_intel"]["sar_fusion"]["sar_enabled"] is True
    assert "sigma0_vv_db" in data["agentic_intel"]["sar_fusion"]


def test_hindi_input_and_output():
    img_path = os.path.join(SAMPLES_DIR, "water.png")
    with open(img_path, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/analyze",
            files={"file": ("water.png", f, "image/png")},
            data={
                "x": "20",
                "y": "20",
                "width": "180",
                "height": "180",
                "query": "क्या इस उपग्रह क्षेत्र में जल निकाय या नदी दिखाई दे रही है?",
                "language": "hi",
                "enable_sar": "true",
            },
        )
    assert res.status_code == 200, res.text
    data = res.json()
    print("\n--- TEST 2: Water in Hindi (Input & Output) ---")
    print("Language:", data["agentic_intel"]["language"])
    print("Task Intent:", data["agentic_intel"]["task_intent"])
    print("Hindi Paragraph (UTF-8 bytes):\n", data["caption"].encode('utf-8'))
    assert data["agentic_intel"]["language"] == "hi"
    assert data["agentic_intel"]["task_intent"] == "water_hydrology"


def test_pdf_export():
    img_path = os.path.join(SAMPLES_DIR, "urban.png")
    with open(img_path, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/export/pdf",
            files={"file": ("urban.png", f, "image/png")},
            data={
                "x": "0",
                "y": "0",
                "width": "256",
                "height": "256",
                "query": "Identify urban infrastructure and residential housing.",
                "paragraph": "Comprehensive geospatial analysis confirms dense residential urban structures.",
                "confidence": "0.88",
                "latency_sec": "0.18",
                "language": "en",
                "enable_sar": "true",
            },
        )
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    assert len(res.content) > 1000
    print("\n--- TEST 3: Official Audit PDF Export ---")
    print("PDF Size:", len(res.content), "bytes")
    print("PDF Header:", res.content[:5])
    assert res.content.startswith(b"%PDF")


if __name__ == "__main__":
    test_paragraph_english_sar()
    test_hindi_input_and_output()
    test_pdf_export()
    print("\n>>> ALL INNOVATION TESTS PASSED SUCCESSFULLY! <<<")
