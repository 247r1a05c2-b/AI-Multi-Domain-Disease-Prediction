"""
BioVision AI — Main Streamlit Application
Advanced Multi-Domain Disease Prediction System

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import json
import logging
import time
import datetime

# ── Streamlit (must be first) ──────────────────────────────────────────────────
import streamlit as st

# ── Standard scientific stack ──────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    st.error("numpy not found. Run:  pip install numpy"); st.stop()

try:
    import pandas as pd
except ImportError:
    st.error("pandas not found. Run:  pip install pandas"); st.stop()

try:
    from PIL import Image
except ImportError:
    st.error("Pillow not found. Run:  pip install Pillow"); st.stop()

# ── OpenCV (optional for live camera) ─────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# ── Plotly ─────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.database       import initialize_database, save_prediction, get_all_predictions, get_analytics_summary, clear_all_predictions
from utils.preprocessing  import preprocess_image, scan_dataset
from utils.severity       import analyze_severity, get_treatment, severity_color
from utils.model_loader   import predict, model_exists, clear_cache

from utils.charts import (
    disease_distribution_bar, category_pie, severity_gauge,
    confidence_bar, trend_line, severity_bar, disease_bar_png,
)
from utils.report_generator import generate_report, list_reports
from utils.voice import get_assistant

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Detect Streamlit version for compatibility ─────────────────────────────────
import streamlit as _st_version_check
_ST_VERSION = tuple(int(x) for x in _st_version_check.__version__.split(".")[:2])

def _st_image(img, caption="", width=None):
    """Wrapper for st.image that works on all Streamlit versions."""
    try:
        if _ST_VERSION >= (1, 20):
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.image(img, caption=caption, use_column_width=True)
    except TypeError:
        # fallback for very old versions
        st.image(img, caption=caption)

def _st_dataframe(df, **kwargs):
    """Wrapper for st.dataframe compatible with all versions."""
    try:
        if _ST_VERSION >= (1, 20):
            st.dataframe(df, use_container_width=True, **kwargs)
        else:
            st.dataframe(df, **kwargs)
    except Exception:
        st.dataframe(df)

def _st_plotly(fig, **kwargs):
    """Wrapper for st.plotly_chart compatible with all versions."""
    if not PLOTLY_OK:
        st.info("Plotly not installed. Run: pip install plotly")
        return
    try:
        st.plotly_chart(fig, use_container_width=True, **kwargs)
    except Exception:
        st.plotly_chart(fig)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioVision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark theme ────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #181825;
    border-right: 1px solid #313244;
}
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2940 100%);
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    color: #cdd6f4;
}
.metric-card h2 { font-size: 2.2rem; margin: 0; color: #89dceb; }
.metric-card p  { margin: 0; font-size: 0.85rem; color: #a6adc8; }
.pred-box {
    background: #181825;
    border: 2px solid #4cc9f0;
    border-radius: 12px;
    padding: 20px;
    margin-top: 12px;
}
.pred-title { font-size: 1.5rem; font-weight: 700; color: #89dceb; }
.conf-text  { font-size: 1.1rem; color: #cba6f7; }
.sev-low    { background:#1a3a22; color:#28a745; border-radius:6px; padding:4px 12px; }
.sev-medium { background:#3a2a0f; color:#fd7e14; border-radius:6px; padding:4px 12px; }
.sev-high   { background:#3a0f0f; color:#dc3545; border-radius:6px; padding:4px 12px; }
.stButton > button {
    background: linear-gradient(135deg, #1e3a5f, #0f2940);
    color: #cdd6f4;
    border: 1px solid #4cc9f0;
    border-radius: 8px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #4cc9f0;
    color: #1e1e2e;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "voice_on":        False,
        "voice_lang":      "en",
        "last_prediction": None,
        "camera_active":   False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Safe DB init
try:
    initialize_database()
except Exception as e:
    st.warning(f"Database init warning: {e}")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 BioVision AI")
    st.markdown("*Multi-Domain Disease Prediction*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Predict", "📷 Live Camera",
         "📊 Analytics", "📋 History", "📥 Reports",
         "⚙️ Settings", "❓ Help"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Domain**")
    category = st.selectbox(
        "Category",
        ["plant", "human", "animal"],
        format_func=lambda x: {"plant": "🌿 Plant", "human": "👤 Human", "animal": "🐾 Animal"}[x],
    )

    st.divider()
    voice_label = "🔊 Voice ON" if st.session_state.voice_on else "🔇 Voice OFF"
    if st.button(voice_label):
        st.session_state.voice_on = not st.session_state.voice_on
        try:
            get_assistant().enabled = st.session_state.voice_on
        except Exception:
            pass

    st.divider()
    st.markdown("**Model Status**")
    for cat in ["plant", "human", "animal"]:
        icon = "✅" if model_exists(cat) else "⚠️ Demo"
        st.markdown(f"{icon} {cat.capitalize()}")

    st.divider()
    st.caption(f"v1.0.0  |  Streamlit {_st_version_check.__version__}")


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🔬 BioVision AI")
    st.subheader("Advanced Multi-Domain Disease Prediction Platform")
    st.markdown(
        "Detect diseases in **plants**, **humans**, and **animals** using "
        "deep learning, computer vision, and real-time analysis."
    )
    st.divider()

    # KPIs
    try:
        summary = get_analytics_summary()
    except Exception:
        summary = {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{summary.get("total",0)}</h2><p>Total Predictions</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h2>{summary.get("by_category",{}).get("plant",0)}</h2><p>Plant Analyses</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{summary.get("by_category",{}).get("human",0)}</h2><p>Human Analyses</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h2>{summary.get("by_category",{}).get("animal",0)}</h2><p>Animal Analyses</p></div>', unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚀 Features")
        for f in [
            "🌿 **Plant Disease** — Blight, Mildew, Rust",
            "👤 **Human Skin** — Dermatitis, Psoriasis, Acne",
            "🐾 **Animal Health** — Mange, Ringworm, Infections",
            "📷 **Live Camera Detection** with real-time FPS",
            "📊 **Analytics Dashboard** with interactive charts",
            "📋 **PDF Report** with image & treatment plan",
            "🔊 **Voice Assistant** for spoken predictions",
            "⚡ **Severity Analysis** with contour mapping",
        ]:
            st.markdown(f)

    with col2:
        st.markdown("### 📘 Quick Start")
        st.code("""# 1. Generate sample data
python setup_datasets.py

# 2. (Optional) Train models
python training/train_plants.py

# 3. Launch app
streamlit run app.py""", language="bash")

    recent = get_all_predictions(limit=5)
    if recent:
        st.divider()
        st.markdown("### 🕒 Recent Predictions")
        df = pd.DataFrame(recent)[["timestamp","category","disease","confidence","severity"]]
        df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        _st_dataframe(df, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.title(f"🔍 Disease Prediction — {category.capitalize()}")

    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Capture from Camera"])

    # ── Upload Tab ─────────────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an image (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded is not None:
            try:
                pil_img = Image.open(uploaded).convert("RGB")
            except Exception as e:
                st.error(f"Could not open image: {e}")
                pil_img = None

            if pil_img is not None:
                col_img, col_result = st.columns([1, 1])

                with col_img:
                    _st_image(pil_img, caption="Uploaded Image")

                with col_result:
                    with st.spinner("Analysing image..."):
                        try:
                            arr = preprocess_image(pil_img)
                        except Exception as e:
                            st.error(f"Preprocessing error: {e}")
                            arr = None

                        if arr is None:
                            st.error("Could not preprocess the image. Please try a different file.")
                        else:
                            # ── Prediction ────────────────────────────────
                            try:
                                results   = predict(category, arr)
                                top_label = results[0][0].replace("_", " ")
                                top_conf  = results[0][1]
                            except Exception as e:
                                st.error(f"Prediction error: {e}")
                                results   = [("Unknown", 1.0)]
                                top_label = "Unknown"
                                top_conf  = 1.0

                            # ── Severity ──────────────────────────────────
                            try:
                                sev_data = analyze_severity(pil_img)
                                severity = sev_data["severity"]
                                inf_pct  = sev_data["infected_pct"]
                            except Exception as e:
                                st.warning(f"Severity analysis skipped: {e}")
                                sev_data = {"severity":"LOW","infected_pct":0.0,"annotated_img":None,"contour_count":0}
                                severity = "LOW"
                                inf_pct  = 0.0

                            # ── Treatment ─────────────────────────────────
                            try:
                                treatment = get_treatment(top_label)
                            except Exception:
                                treatment = {"description":"N/A","treatments":[],"prevention":"N/A"}

                            # ── Save to DB ────────────────────────────────
                            try:
                                save_prediction(
                                    category=category,
                                    disease=top_label,
                                    confidence=top_conf,
                                    severity=severity,
                                    image_path=uploaded.name,
                                    treatment="; ".join(treatment.get("treatments", [])),
                                )
                            except Exception as e:
                                logger.warning(f"DB save failed: {e}")

                            # ── Voice ─────────────────────────────────────
                            if st.session_state.voice_on:
                                try:
                                    get_assistant().announce_prediction(top_label, top_conf, severity, category)
                                except Exception:
                                    pass

                            # ── Result Card ───────────────────────────────
                            sev_class = f"sev-{severity.lower()}"
                            st.markdown(f"""
<div class="pred-box">
  <div class="pred-title">🦠 {top_label}</div>
  <div class="conf-text">Confidence: <b>{top_conf*100:.1f}%</b></div>
  <br/>
  <span class="{sev_class}"><b>Severity: {severity}</b></span>
  &nbsp;&nbsp;
  <span style="color:#a6adc8;">Infected Area: {inf_pct:.1f}%</span>
</div>
""", unsafe_allow_html=True)

                # ── Confidence Chart ───────────────────────────────────────
                st.divider()
                st.markdown("### 📊 Confidence Per Class")
                try:
                    labels = [r[0].replace("_"," ") for r in results]
                    confs  = [r[1] for r in results]
                    _st_plotly(confidence_bar(labels, confs))
                except Exception as e:
                    st.warning(f"Chart unavailable: {e}")

                # ── Severity Annotated Image ───────────────────────────────
                if sev_data.get("annotated_img") is not None:
                    st.divider()
                    st.markdown("### 🔬 Severity Map — Infected Regions Highlighted")
                    try:
                        _st_image(sev_data["annotated_img"], caption="Contour Analysis")
                    except Exception as e:
                        st.warning(f"Could not display severity map: {e}")

                # ── Severity Gauge ────────────────────────────────────────
                try:
                    _st_plotly(severity_gauge(inf_pct, severity))
                except Exception as e:
                    st.info(f"Gauge unavailable: {e}")

                # ── Disease Comparison Table ───────────────────────────────
                st.divider()
                st.markdown("### 📋 Disease Comparison")
                try:
                    comp_df = pd.DataFrame({
                        "Disease":      [r[0].replace("_"," ") for r in results],
                        "Confidence %": [round(r[1]*100, 2)     for r in results],
                        "Rank":         list(range(1, len(results)+1)),
                    })
                    _st_dataframe(comp_df, hide_index=True)
                except Exception as e:
                    st.warning(f"Table unavailable: {e}")

                # ── Treatment Section ──────────────────────────────────────
                st.divider()
                st.markdown("### 💊 Treatment & Advice")
                st.info(f"**About:** {treatment.get('description','N/A')}")
                with st.expander("📋 Recommended Treatments", expanded=True):
                    for step in treatment.get("treatments", []):
                        st.markdown(f"• {step}")
                with st.expander("🛡️ Prevention"):
                    st.write(treatment.get("prevention","N/A"))

                # ── PDF Report ─────────────────────────────────────────────
                st.divider()
                if st.button("📥 Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            chart_png = disease_bar_png(
                                get_analytics_summary().get("by_disease", {})
                            )
                            ann_img = None
                            if sev_data.get("annotated_img") is not None:
                                ann_img = Image.fromarray(
                                    sev_data["annotated_img"].astype("uint8")
                                )
                            report_path = generate_report(
                                disease=top_label,
                                confidence=top_conf,
                                severity=severity,
                                category=category,
                                treatment_info=treatment,
                                pil_image=pil_img,
                                annotated_image=ann_img,
                                chart_png_bytes=chart_png,
                                infected_pct=inf_pct,
                            )
                        except Exception as e:
                            st.error(f"Report generation error: {e}")
                            report_path = ""

                    if report_path and os.path.exists(report_path):
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=f.read(),
                                file_name=os.path.basename(report_path),
                                mime="application/pdf",
                            )
                        st.success(f"Report ready: {os.path.basename(report_path)}")
                    elif not report_path:
                        st.error("Report generation failed. Ensure reportlab is installed: pip install reportlab")

    # ── Camera Tab ─────────────────────────────────────────────────────────────
    with tab_camera:
        st.markdown("### 📷 Capture Image from Webcam")
        st.info("Allow browser camera access, then click the shutter button.")

        try:
            cam_img = st.camera_input("Take a photo")
        except Exception:
            st.warning("Camera input is not supported in your Streamlit version. Please use the Upload tab.")
            cam_img = None

        if cam_img is not None:
            try:
                pil_img = Image.open(cam_img).convert("RGB")
            except Exception as e:
                st.error(f"Could not read captured image: {e}")
                pil_img = None

            if pil_img is not None:
                _st_image(pil_img, caption="Captured Frame")

                with st.spinner("Analysing..."):
                    arr      = preprocess_image(pil_img)
                    results  = predict(category, arr) if arr is not None else [("Unknown",1.0)]
                    top_label= results[0][0].replace("_"," ")
                    top_conf = results[0][1]

                    try:
                        sev_data = analyze_severity(pil_img)
                        severity = sev_data["severity"]
                        inf_pct  = sev_data["infected_pct"]
                    except Exception:
                        severity, inf_pct = "LOW", 0.0
                        sev_data = {"annotated_img": None}

                    treatment = get_treatment(top_label)

                    try:
                        save_prediction(category, top_label, top_conf, severity,
                                        "webcam_capture",
                                        "; ".join(treatment.get("treatments",[])))
                    except Exception:
                        pass

                sev_class = f"sev-{severity.lower()}"
                st.markdown(f"""
<div class="pred-box">
  <div class="pred-title">🦠 {top_label}</div>
  <div class="conf-text">Confidence: <b>{top_conf*100:.1f}%</b></div><br/>
  <span class="{sev_class}"><b>Severity: {severity}</b></span>
  &nbsp;&nbsp;<span style="color:#a6adc8;">Infected: {inf_pct:.1f}%</span>
</div>
""", unsafe_allow_html=True)

                if sev_data.get("annotated_img") is not None:
                    _st_image(sev_data["annotated_img"], caption="Severity Map")

                _st_plotly(severity_gauge(inf_pct, severity))
                _st_plotly(confidence_bar(
                    [r[0].replace("_"," ") for r in results],
                    [r[1] for r in results]
                ))

                if st.session_state.voice_on:
                    try:
                        get_assistant().announce_prediction(top_label, top_conf, severity, category)
                    except Exception:
                        pass

                st.divider()
                st.markdown("### 💊 Treatment")
                st.info(treatment.get("description","N/A"))
                for step in treatment.get("treatments",[]):
                    st.markdown(f"• {step}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE CAMERA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📷 Live Camera":
    st.title("📷 Live Camera Detection")
    st.markdown("Real-time frame-by-frame detection via OpenCV.")

    if not CV2_OK:
        st.error("OpenCV is not installed. Run:  pip install opencv-python")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button("▶ Start Camera", type="primary")
    with col2:
        stop_btn = st.button("⏹ Stop Camera")
    with col3:
        st.markdown(f"**Domain:** {category.capitalize()}")

    if start_btn:
        st.session_state.camera_active = True
    if stop_btn:
        st.session_state.camera_active = False

    frame_placeholder = st.empty()
    fps_placeholder   = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. It may be in use or not connected.")
            st.session_state.camera_active = False
        else:
            frame_count = 0
            last_label, last_conf, last_severity = "Analysing...", 0.0, "LOW"
            predict_every = 10

            while st.session_state.camera_active:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera.")
                    break

                frame_count += 1

                if frame_count % predict_every == 0:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(rgb_frame)
                        arr       = preprocess_image(pil_frame)
                        if arr is not None:
                            res          = predict(category, arr)
                            last_label   = res[0][0].replace("_"," ")
                            last_conf    = res[0][1]
                            sd           = analyze_severity(pil_frame)
                            last_severity= sd["severity"]
                    except Exception:
                        pass

                colour_map = {"LOW":(0,255,0),"MEDIUM":(0,165,255),"HIGH":(0,0,255)}
                colour = colour_map.get(last_severity, (255,255,255))
                cv2.putText(frame, f"{last_label} ({last_conf*100:.0f}%) [{last_severity}]",
                            (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
                fps = 1.0 / max(time.time() - t0, 0.001)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10,65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                fps_placeholder.caption(f"FPS: {fps:.1f}  |  Frame: {frame_count}")

            cap.release()
            frame_placeholder.empty()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")

    try:
        summary = get_analytics_summary()
    except Exception as e:
        st.error(f"Could not load analytics: {e}")
        summary = {}

    total       = summary.get("total", 0)
    by_disease  = summary.get("by_disease",  {})
    most_common = max(by_disease, key=by_disease.get) if by_disease else "N/A"
    high_sev    = summary.get("by_severity", {}).get("HIGH", 0)
    low_sev     = summary.get("by_severity", {}).get("LOW",  0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions",  total)
    c2.metric("Most Common Disease", most_common)
    c3.metric("High Severity Cases", high_sev)
    c4.metric("Low Severity Cases",  low_sev)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if by_disease:
            _st_plotly(disease_distribution_bar(by_disease))
        else:
            st.info("No data yet. Run some predictions first.")
    with col2:
        if summary.get("by_category"):
            _st_plotly(category_pie(summary["by_category"]))
        else:
            st.info("No category data yet.")

    col3, col4 = st.columns(2)
    with col3:
        if summary.get("by_severity"):
            _st_plotly(severity_bar(summary["by_severity"]))
        else:
            st.info("No severity data yet.")
    with col4:
        if summary.get("recent_trend"):
            _st_plotly(trend_line(summary["recent_trend"]))
        else:
            st.info("Not enough data for trend chart.")

    st.divider()
    st.markdown("### 📋 All Predictions")
    try:
        all_preds = get_all_predictions(limit=200)
        if all_preds:
            df = pd.DataFrame(all_preds)
            df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
            _st_dataframe(df, hide_index=True)
            csv = pd.DataFrame(get_all_predictions(limit=10000)).to_csv(index=False)
            st.download_button("⬇️ Export CSV", data=csv,
                               file_name="biovision_predictions.csv", mime="text/csv")
        else:
            st.info("No predictions yet.")
    except Exception as e:
        st.error(f"Could not load predictions: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 History":
    st.title("📋 Prediction History")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_cat = st.selectbox("Category", ["All","plant","human","animal"])
    with col2:
        filter_sev = st.selectbox("Severity",  ["All","LOW","MEDIUM","HIGH"])
    with col3:
        search_q = st.text_input("Search Disease", placeholder="e.g. Blight")

    try:
        data = get_all_predictions(
            category=None if filter_cat == "All" else filter_cat,
            limit=1000,
        )
        if filter_sev != "All":
            data = [d for d in data if d["severity"] == filter_sev]
        if search_q:
            data = [d for d in data if search_q.lower() in d["disease"].lower()]
    except Exception as e:
        st.error(f"Could not load history: {e}")
        data = []

    if data:
        df = pd.DataFrame(data)
        df["confidence"] = df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        _st_dataframe(
            df[["id","timestamp","category","disease","confidence","severity"]],
            hide_index=True,
        )
        st.caption(f"Showing {len(data)} records")

        st.divider()
        if st.button("🗑️ Clear All History"):
            try:
                clear_all_predictions()
                st.success("History cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear: {e}")
    else:
        st.info("No matching records found.")


# ══════════════════════════════════════════════════════════════════════════════
# REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Reports":
    st.title("📥 Reports")
    st.markdown("Previously generated PDF reports are listed below.")

    try:
        reports = list_reports()
    except Exception:
        reports = []

    if reports:
        for rp in reports:
            fname = os.path.basename(rp)
            size  = os.path.getsize(rp)
            mtime = datetime.datetime.fromtimestamp(
                os.path.getmtime(rp)).strftime("%Y-%m-%d %H:%M")
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"📄 **{fname}**  \n*{mtime} · {size/1024:.1f} KB*")
            with c2:
                try:
                    with open(rp, "rb") as f:
                        st.download_button(
                            "⬇️ Download", data=f.read(),
                            file_name=fname, mime="application/pdf",
                            key=f"dl_{fname}",
                        )
                except Exception as e:
                    st.error(f"Cannot read file: {e}")
            st.divider()
    else:
        st.info("No reports yet. Generate one from the **Predict** page.")


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    st.markdown("### 🔊 Voice Assistant")
    col1, col2 = st.columns(2)
    with col1:
        voice_on = st.checkbox("Enable Voice", value=st.session_state.voice_on)
        st.session_state.voice_on = voice_on
        try:
            get_assistant().enabled = voice_on
        except Exception:
            pass
    with col2:
        lang = st.selectbox("Language", ["en — English", "hi — Hindi"])
        st.session_state.voice_lang = lang.split(" ")[0]
        try:
            get_assistant().set_language(st.session_state.voice_lang)
        except Exception:
            pass

    rate = st.slider("Speech Rate (WPM)", 80, 220, 145)
    try:
        get_assistant().set_rate(rate)
    except Exception:
        pass

    st.divider()
    st.markdown("### 🧠 Model Management")
    for cat in ["plant", "human", "animal"]:
        exists = model_exists(cat)
        status = "✅ Trained model" if exists else "⚠️ Demo mode (no model)"
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"**{cat.capitalize()}** — {status}")
        if exists:
            with c2:
                if st.button("Unload", key=f"unload_{cat}"):
                    clear_cache()
                    st.success("Cache cleared.")

    st.divider()
    st.markdown("### 📁 Dataset Status")
    for domain in ["plants","humans","animals"]:
        ddir  = os.path.join(os.path.dirname(__file__), "datasets", domain)
        try:
            stats = scan_dataset(ddir)
            st.markdown(f"**{domain.capitalize()}** — {stats['total']} images · {len(stats['classes'])} classes")
            if stats.get("corrupted"):
                st.warning(f"⚠️ {stats['corrupted']} corrupted images.")
        except Exception as e:
            st.warning(f"Could not scan {domain}: {e}")

    st.divider()
    st.markdown("### 🗄️ Database")
    if st.button("⚠️ Clear ALL prediction data"):
        try:
            clear_all_predictions()
            st.success("Cleared.")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "❓ Help":
    st.title("❓ Help & About")

    with st.expander("📖 What is BioVision AI?", expanded=True):
        st.markdown("""
**BioVision AI** is a fully local, offline AI disease detection platform for:
- 🌿 **Plants** — Tomato Blight, Powdery Mildew, Rust, Healthy
- 👤 **Humans** — Dermatitis, Psoriasis, Acne
- 🐾 **Animals** — Skin Infection, Mange, Ringworm

No internet or paid API keys required. Everything runs on your machine.
        """)

    with st.expander("🚀 Setup Steps"):
        st.markdown("""
```bash
# 1. Create conda environment
conda create -n biovision python=3.10 -y
conda activate biovision

# 2. Install packages
pip install -r requirements.txt

# 3. Generate sample data
python setup_datasets.py

# 4. (Optional) Train models
python training/train_plants.py

# 5. Run
streamlit run app.py
```
        """)

    with st.expander("📊 Severity Levels"):
        st.markdown("""
| Level  | Infected Area | Action |
|--------|--------------|--------|
| 🟢 LOW | 0–20%  | Monitor, apply prevention |
| 🟠 MEDIUM | 20–50% | Consult specialist soon |
| 🔴 HIGH | >50%  | Seek immediate expert advice |
        """)

    with st.expander("🐛 Common Errors & Fixes"):
        st.markdown("""
| Error | Fix |
|-------|-----|
| `use_container_width` error | Update Streamlit: `pip install --upgrade streamlit` |
| `pyttsx3` error on Windows | `pip install pywin32` |
| TensorFlow error | `pip install tensorflow-cpu==2.13.0` |
| `reportlab` missing | `pip install reportlab` |
| Webcam not found | Use Upload Image tab |
| `cv2` missing | `pip install opencv-python` |
        """)

    st.divider()
    st.markdown("""
**BioVision AI v1.0.0**
Built with Streamlit · TensorFlow · OpenCV · ReportLab
Open-source · No paid APIs · Fully offline
    """)
