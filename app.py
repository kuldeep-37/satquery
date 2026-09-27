"""Hugging Face Spaces Entrypoint: Mounts SatQuery FastAPI backend into Gradio.

Allows running the entire FastAPI + PyTorch + LoRA backend on Hugging Face Spaces
using the 100% FREE Gradio SDK (16 GB RAM, no credit card required).
"""

from __future__ import annotations

import os
import sys
import gradio as gr

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.main import app as fastapi_app

# Create a clean Gradio landing page for status & documentation
with gr.Blocks(title="SatQuery AI — Backend Microservice") as demo:
    gr.Markdown("# 🛰️ SatQuery AI — Backend Microservice Running Live")
    gr.Markdown(
        """
        ### Pipeline Status: Online & Ready for Inference
        This Hugging Face Space hosts the **FastAPI + PyTorch + LoRA** backend microservice for the Vercel frontend.

        - **Health Check Endpoint:** [`/health`](/health)
        - **VQA Analysis Endpoint:** `/analyze`
        - **Continuous Q&A Memory:** `/memory`
        - **Active LoRA Fine-Tuning:** `/train/memory`
        - **ISRO/SIH Audit PDF Export:** `/export/pdf`
        """
    )

# Mount the complete FastAPI REST API onto the Gradio application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
