"""
BioVision AI — PDF Report Generator
Generates professional PDF reports using ReportLab.
Each report includes: image, prediction, confidence, severity, treatment, charts, timestamp.
"""

import os
import io
import logging
from datetime import datetime

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, HRFlowable, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False
    logger.warning("reportlab not installed — PDF generation disabled.")

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Colour constants (ReportLab) ──────────────────────────────────────────────
C_PRIMARY   = colors.HexColor("#1e3a5f")
C_ACCENT    = colors.HexColor("#4cc9f0")
C_SUCCESS   = colors.HexColor("#28a745")
C_WARNING   = colors.HexColor("#fd7e14")
C_DANGER    = colors.HexColor("#dc3545")
C_LIGHT     = colors.HexColor("#f8f9fa")
C_DARK      = colors.HexColor("#212529")

SEVERITY_COLOUR = {"LOW": C_SUCCESS, "MEDIUM": C_WARNING, "HIGH": C_DANGER}


def _pil_to_rl_image(pil_img, max_width=12 * cm, max_height=9 * cm):
    """Convert a PIL Image to a ReportLab Image flowable."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    aspect = h / w
    width  = min(max_width, max_width)
    height = width * aspect
    if height > max_height:
        height = max_height
        width  = height / aspect
    return RLImage(buf, width=width, height=height)


def _bytes_to_rl_image(img_bytes, max_width=14 * cm, max_height=6 * cm):
    """Convert raw PNG bytes to a ReportLab Image flowable."""
    buf = io.BytesIO(img_bytes)
    return RLImage(buf, width=max_width, height=max_height)


def generate_report(
    disease:    str,
    confidence: float,
    severity:   str,
    category:   str,
    treatment_info: dict,
    pil_image=None,
    annotated_image=None,
    chart_png_bytes: bytes = b"",
    infected_pct: float = 0.0,
) -> str:
    """
    Generate a PDF report and save it to the reports/ directory.

    Returns:
        Path to the saved PDF file, or empty string on failure.
    """
    if not _RL_AVAILABLE:
        logger.error("reportlab is not installed. Cannot generate PDF.")
        return ""

    try:
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name  = disease.replace(" ", "_").replace("/", "-")
        filename   = f"report_{category}_{safe_name}_{timestamp}.pdf"
        filepath   = os.path.join(REPORTS_DIR, filename)

        doc    = SimpleDocTemplate(filepath, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Header ──────────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontSize=22,
            textColor=C_PRIMARY,
            spaceAfter=4,
            alignment=TA_CENTER,
        )
        sub_style = ParagraphStyle(
            "Sub",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        story.append(Paragraph("BioVision AI — Disease Prediction Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
            sub_style,
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=C_ACCENT))
        story.append(Spacer(1, 0.4 * cm))

        # ── Summary table ───────────────────────────────────────────────────
        sev_colour = SEVERITY_COLOUR.get(severity, C_DARK)
        summary_data = [
            ["Field",       "Value"],
            ["Category",    category.capitalize()],
            ["Disease",     disease],
            ["Confidence",  f"{confidence * 100:.1f}%"],
            ["Severity",    severity],
            ["Infected Area", f"{infected_pct:.1f}%"],
        ]
        tbl = Table(summary_data, colWidths=[5 * cm, 11 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), C_PRIMARY),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 11),
            ("BACKGROUND",   (0, 1), (-1, -1), C_LIGHT),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, C_LIGHT]),
            ("FONTNAME",     (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 1), (-1, -1), 10),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            # Colour severity cell
            ("TEXTCOLOR",    (1, 4), (1, 4), sev_colour),
            ("FONTNAME",     (1, 4), (1, 4), "Helvetica-Bold"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

        # ── Images side-by-side ─────────────────────────────────────────────
        if pil_image or annotated_image:
            img_cells = []
            captions  = []
            if pil_image:
                img_cells.append(_pil_to_rl_image(pil_image))
                captions.append(Paragraph("Original Image", styles["Normal"]))
            if annotated_image:
                if isinstance(annotated_image, np.ndarray):
                    ann_pil = Image.fromarray(annotated_image.astype(np.uint8))
                else:
                    ann_pil = annotated_image
                img_cells.append(_pil_to_rl_image(ann_pil))
                captions.append(Paragraph("Severity Analysis", styles["Normal"]))

            if img_cells:
                img_tbl = Table([img_cells, captions])
                img_tbl.setStyle(TableStyle([
                    ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING",  (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ]))
                story.append(img_tbl)
                story.append(Spacer(1, 0.4 * cm))

        # ── Distribution chart ───────────────────────────────────────────────
        if chart_png_bytes:
            story.append(Paragraph("Disease Analytics", ParagraphStyle(
                "H2", parent=styles["Heading2"], textColor=C_PRIMARY, spaceBefore=6)))
            story.append(_bytes_to_rl_image(chart_png_bytes))
            story.append(Spacer(1, 0.4 * cm))

        # ── Treatment section ────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("About This Condition", ParagraphStyle(
            "H2", parent=styles["Heading2"], textColor=C_PRIMARY)))
        story.append(Paragraph(treatment_info.get("description", "N/A"), styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Recommended Treatments", ParagraphStyle(
            "H2", parent=styles["Heading2"], textColor=C_PRIMARY)))
        for step in treatment_info.get("treatments", []):
            story.append(Paragraph(f"• {step}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))

        story.append(Paragraph("Prevention", ParagraphStyle(
            "H2", parent=styles["Heading2"], textColor=C_PRIMARY)))
        story.append(Paragraph(treatment_info.get("prevention", "N/A"), styles["Normal"]))
        story.append(Spacer(1, 0.5 * cm))

        # ── Footer disclaimer ────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(
            "<i>Disclaimer: This report is generated by an AI system and is for informational "
            "purposes only. It does not replace professional medical, veterinary, or agricultural "
            "advice. Always consult a qualified specialist for diagnosis and treatment.</i>",
            ParagraphStyle("Disclaimer", parent=styles["Normal"],
                           fontSize=8, textColor=colors.grey),
        ))

        doc.build(story)
        logger.info(f"Report saved: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return ""


def list_reports():
    """Return a list of existing report file paths, newest first."""
    try:
        files = [
            os.path.join(REPORTS_DIR, f)
            for f in os.listdir(REPORTS_DIR)
            if f.endswith(".pdf")
        ]
        return sorted(files, key=os.path.getmtime, reverse=True)
    except Exception:
        return []
