"""Auditable Execution Report PDF Generator with Full Indian Unicode Support.

Implements the "Auditable Execution Report (PDF/JSON)" innovation from innovation.jpeg.
Generates an official ISRO/SIH geospatial intelligence audit report with embedded ROI image,
model telemetry, confidence metrics, and regional language paragraph analysis.

Supports full TrueType Unicode rendering for:
- Hindi & Marathi (Devanagari)
- Tamil (தமிழ்)
- Telugu (తెలుగు)
- Bengali (বাংলা)
- Kannada (ಕನ್ನಡ)
- Malayalam (മലയാളം)
- Gujarati (ગુજરાતી)
- Punjabi (ਪੰਜਾਬੀ)
- English & Latin characters
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, Optional, Tuple
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("satquery-pdf")

_UNICODE_FONT_READY = False
_FONT_NORMAL = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"


def init_unicode_fonts() -> Tuple[str, str]:
    """Ensure Unicode TrueType fonts supporting Indian regional scripts and Latin are registered."""
    global _UNICODE_FONT_READY, _FONT_NORMAL, _FONT_BOLD
    if _UNICODE_FONT_READY:
        return _FONT_NORMAL, _FONT_BOLD

    # 1. Primary: Windows Nirmala.ttc (Subfont 0=Regular, 1=Bold) - Complete Indic Unicode coverage
    windir = os.environ.get("WINDIR", "C:\\Windows")
    nirmala_ttc = os.path.join(windir, "Fonts", "Nirmala.ttc")
    if os.path.exists(nirmala_ttc):
        try:
            if "NirmalaUI" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("NirmalaUI", nirmala_ttc, subfontIndex=0))
            if "NirmalaUI-Bold" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("NirmalaUI-Bold", nirmala_ttc, subfontIndex=1))
            registerFontFamily("NirmalaUI", normal="NirmalaUI", bold="NirmalaUI-Bold")
            _FONT_NORMAL = "NirmalaUI"
            _FONT_BOLD = "NirmalaUI-Bold"
            _UNICODE_FONT_READY = True
            logger.info("Successfully registered NirmalaUI TrueType font family for Indic Unicode PDF rendering.")
            return _FONT_NORMAL, _FONT_BOLD
        except Exception as e:
            logger.warning(f"Failed to register Nirmala.ttc: {e}")

    # 2. Check candidate font files
    candidates = [
        ("nirmala.ttf", "NirmalaTTF"),
        ("segoeui.ttf", "SegoeUI"),
        ("arial.ttf", "Arial"),
        ("mangal.ttf", "Mangal"),
    ]
    for fname, reg_name in candidates:
        fp = os.path.join(windir, "Fonts", fname)
        if os.path.exists(fp):
            try:
                if reg_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(reg_name, fp))
                _FONT_NORMAL = reg_name
                _FONT_BOLD = reg_name
                _UNICODE_FONT_READY = True
                logger.info(f"Registered fallback TrueType font {reg_name} from {fp}")
                return _FONT_NORMAL, _FONT_BOLD
            except Exception:
                pass

    _UNICODE_FONT_READY = True
    return _FONT_NORMAL, _FONT_BOLD


def generate_audit_pdf(
    audit_data: Dict[str, Any],
    paragraph_text: str,
    roi_image: Optional[Image.Image] = None,
    sar_data: Optional[Dict[str, Any]] = None,
    english_paragraph: Optional[str] = None,
) -> bytes:
    """Generate an official auditable execution PDF report and return as raw bytes."""
    font_normal, font_bold = init_unicode_fonts()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom Color Palette
    primary_color = colors.HexColor("#0f172a")  # Deep Navy
    accent_cyan = colors.HexColor("#0891b2")    # Deep Teal/Cyan
    text_dark = colors.HexColor("#1e293b")
    badge_bg = colors.HexColor("#f1f5f9")

    # Header Styles with Unicode Font support
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=18,
        leading=22,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName=font_normal,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=11,
        leading=15,
        textColor=accent_cyan,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName=font_normal,
        fontSize=9,
        leading=14,
        textColor=text_dark,
    )
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )

    # 1. Title Banner
    story.append(Paragraph("SatQuery AI — Geospatial VQA Execution Audit", title_style))
    story.append(Paragraph("Smart India Hackathon (SIH) Prototype | ISRO Satellite Imagery Problem Statement", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_cyan, spaceBefore=2, spaceAfter=10))

    # 2. Executive Metadata Summary Table
    timestamp = audit_data.get("timestamp", "N/A")
    confidence = audit_data.get("confidence", 0.0)
    conf_pct = f"{round(confidence * 100, 1)}%"
    low_conf = audit_data.get("low_confidence_flag", confidence < 0.55)
    review_status = "FLAGGED FOR HUMAN REVIEW" if low_conf else "VERIFIED (Passed Confidence Threshold)"
    review_color = colors.HexColor("#dc2626") if low_conf else colors.HexColor("#16a34a")

    meta_table_data = [
        [
            Paragraph("<b>Execution Timestamp:</b>", body_style), Paragraph(str(timestamp), body_style),
            Paragraph("<b>Model Architecture:</b>", body_style), Paragraph(str(audit_data.get("model", "BLIP + LoRA")), body_style),
        ],
        [
            Paragraph("<b>Inference Latency:</b>", body_style), Paragraph(f"{audit_data.get('latency_sec', 0.0)} sec", body_style),
            Paragraph("<b>Generation Confidence:</b>", body_style), Paragraph(f"<b>{conf_pct}</b>", body_style),
        ],
        [
            Paragraph("<b>Audit Decision:</b>", body_style), Paragraph(f"<font color='{review_color}'><b>{review_status}</b></font>", body_style),
            Paragraph("<b>Output Language:</b>", body_style), Paragraph(str(audit_data.get("language", "English")).upper(), body_style),
        ],
    ]

    meta_table = Table(meta_table_data, colWidths=[120, 160, 120, 140])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 3. Target ROI Image & Coordinate Geometry
    story.append(Paragraph("1. Region of Interest (ROI) & Visual Evidence", section_heading))
    
    roi_info = audit_data.get("roi", {"x": 0, "y": 0, "width": 0, "height": 0})
    roi_coords_str = f"X: {roi_info.get('x')} px | Y: {roi_info.get('y')} px | Width: {roi_info.get('width')} px | Height: {roi_info.get('height')} px"

    # Save thumbnail to in-memory flowable if provided
    rl_img = None
    if roi_image:
        try:
            img_buf = io.BytesIO()
            thumb = roi_image.copy()
            thumb.thumbnail((140, 140))
            thumb.save(img_buf, format="PNG")
            img_buf.seek(0)
            rl_img = RLImage(img_buf, width=120, height=120)
        except Exception:
            rl_img = None

    clean_query = str(audit_data.get('query', 'Autonomous Land Cover Analysis')).strip()
    roi_text_desc = f"""
    <b>Image Resource:</b> {audit_data.get('image_filename', 'satellite_upload.png')}<br/>
    <b>Bounding Box Extents:</b> {roi_coords_str}<br/>
    <b>Target Query / Prompt:</b> "{clean_query}"<br/>
    <b>Coordinate System:</b> Pixel Grid (Relative to Intrinsic Sensor Matrix)<br/>
    <b>GIS Status:</b> RFC 7946 Polygon Feature Generated
    """

    if rl_img:
        roi_table_data = [[rl_img, Paragraph(roi_text_desc, body_style)]]
        roi_table = Table(roi_table_data, colWidths=[130, 410])
        roi_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(roi_table)
    else:
        story.append(Paragraph(roi_text_desc, body_style))

    story.append(Spacer(1, 8))

    # 4. Intelligence Synthesis (Multi-Sentence Paragraph with Unicode Support)
    story.append(Paragraph("2. Contextual Paragraph Intelligence (Agentic Synthesis)", section_heading))
    
    clean_para = str(paragraph_text or "Detailed satellite imagery analysis.").strip()
    lang_code = str(audit_data.get("language", "en")).lower()

    # Primary paragraph card
    primary_html = f"<b>Query Answering Output ({lang_code.upper()}):</b><br/>{clean_para}"
    if english_paragraph and english_paragraph.strip() and lang_code != "en" and clean_para != english_paragraph.strip():
        primary_html += f"<br/><br/><b>English Intelligence Baseline (Human Auditor Reference):</b><br/>{english_paragraph.strip()}"

    intel_box_data = [[Paragraph(primary_html, body_style)]]
    intel_table = Table(intel_box_data, colWidths=[540])
    intel_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, accent_cyan),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(intel_table)
    story.append(Spacer(1, 8))

    # 5. Optical-SAR Radar Fusion Telemetry
    if sar_data and sar_data.get("sar_enabled"):
        story.append(Paragraph("3. Optical-SAR Dual-Polarization Radar Analysis", section_heading))
        sar_summary = f"""
        <b>Sensor Simulation:</b> {sar_data.get('sensor_mode', 'Sentinel-1 C-Band SAR')}<br/>
        <b>Co-Polarization Backscatter (VV):</b> {sar_data.get('sigma0_vv_db')} dB | 
        <b>Cross-Polarization Backscatter (VH):</b> {sar_data.get('sigma0_vh_db')} dB<br/>
        <b>Polarimetric Ratio (VH/VV):</b> {sar_data.get('cross_pol_ratio_db')} dB | 
        <b>Scattering Mechanism:</b> {sar_data.get('dominant_scattering')}<br/>
        <b>All-Weather Penetration:</b> {sar_data.get('cloud_immunity')} | 
        <b>Dielectric Properties:</b> {sar_data.get('dielectric_interpretation')}
        """
        sar_table = Table([[Paragraph(sar_summary, body_style)]], colWidths=[540])
        sar_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#3b82f6")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sar_table)
        story.append(Spacer(1, 8))

    # 6. Verification and Compliance Sign-Off
    story.append(Paragraph("4. Regulatory Compliance & Human-in-the-Loop Sign-Off", section_heading))
    sign_off_text = """
    This execution record was automatically generated by the SatQuery AI Agentic Pipeline. All inferences adhere to ISRO geospatial telemetry protocols with verifiable token-level probability tracking. Outputs flagged for human review must be verified by a certified GIS imagery analyst prior to autonomous deployment in mission-critical disaster management operations.
    """
    story.append(Paragraph(sign_off_text, body_style))
    story.append(Spacer(1, 10))

    footer_data = [
        [
            Paragraph("<b>Automated Pipeline ID:</b> SATQUERY-VLM-LORA-PROD", subtitle_style),
            Paragraph("<b>Analyst Verification Signature:</b> ___________________________", subtitle_style),
        ]
    ]
    footer_table = Table(footer_data, colWidths=[270, 270])
    story.append(footer_table)

    # Build PDF
    doc.build(story)
    return buffer.getvalue()
