"""
BioVision AI — Charts & Visualisation Utilities
Generates Plotly and Matplotlib charts for the analytics dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import logging

logger = logging.getLogger(__name__)

# ─── Colour palette ───────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "plant":  "#28a745",
    "human":  "#007bff",
    "animal": "#fd7e14",
}
SEVERITY_COLORS = {
    "LOW":    "#28a745",
    "MEDIUM": "#fd7e14",
    "HIGH":   "#dc3545",
}


# ─────────────────────────────────────────────────────────────────────────────
# Plotly charts (returned as Plotly Figure objects for st.plotly_chart)
# ─────────────────────────────────────────────────────────────────────────────

def disease_distribution_bar(by_disease: dict, title="Disease Distribution"):
    """
    Horizontal bar chart of prediction counts per disease.
    """
    try:
        diseases = list(by_disease.keys())
        counts   = list(by_disease.values())
        fig = px.bar(
            x=counts, y=diseases, orientation="h",
            labels={"x": "Predictions", "y": "Disease"},
            title=title, color=counts,
            color_continuous_scale="Teal",
        )
        fig.update_layout(
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            coloraxis_showscale=False,
        )
        return fig
    except Exception as e:
        logger.error(f"disease_distribution_bar error: {e}")
        return go.Figure()


def category_pie(by_category: dict, title="Predictions by Category"):
    """
    Pie chart of predictions per domain (plant / human / animal).
    """
    try:
        labels = list(by_category.keys())
        values = list(by_category.values())
        colors = [CATEGORY_COLORS.get(l, "#888") for l in labels]
        fig = go.Figure(data=[go.Pie(
            labels=[l.capitalize() for l in labels],
            values=values,
            marker_colors=colors,
            hole=0.4,
            textinfo="label+percent",
        )])
        fig.update_layout(
            title=title,
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
        )
        return fig
    except Exception as e:
        logger.error(f"category_pie error: {e}")
        return go.Figure()


def severity_gauge(infected_pct: float, severity: str):
    """
    Gauge chart showing the infected area percentage.
    """
    try:
        colour = SEVERITY_COLORS.get(severity, "#888")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=infected_pct,
            number={"suffix": "%", "font": {"color": colour}},
            title={"text": f"Severity: {severity}", "font": {"color": colour}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#cdd6f4"},
                "bar":  {"color": colour},
                "bgcolor": "#313244",
                "steps": [
                    {"range": [0, 20],  "color": "#1e1e2e"},
                    {"range": [20, 50], "color": "#45475a"},
                    {"range": [50, 100],"color": "#585b70"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": infected_pct,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            height=300,
        )
        return fig
    except Exception as e:
        logger.error(f"severity_gauge error: {e}")
        return go.Figure()


def confidence_bar(classes: list, confidences: list):
    """
    Horizontal confidence bar chart for top-N class predictions.
    """
    try:
        fig = px.bar(
            x=[c * 100 for c in confidences],
            y=classes,
            orientation="h",
            labels={"x": "Confidence (%)", "y": "Disease"},
            color=[c * 100 for c in confidences],
            color_continuous_scale="Blues",
            range_x=[0, 100],
        )
        fig.update_layout(
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            coloraxis_showscale=False,
            height=300,
        )
        return fig
    except Exception as e:
        logger.error(f"confidence_bar error: {e}")
        return go.Figure()


def trend_line(trend_data: dict, title="Prediction Trend (Last 30 Days)"):
    """
    Line chart showing daily prediction counts over time.
    """
    try:
        dates  = list(trend_data.keys())
        counts = list(trend_data.values())
        fig = px.line(
            x=dates, y=counts,
            labels={"x": "Date", "y": "Predictions"},
            title=title, markers=True,
        )
        fig.update_traces(line_color="#89dceb", marker=dict(color="#cba6f7", size=8))
        fig.update_layout(
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
        )
        return fig
    except Exception as e:
        logger.error(f"trend_line error: {e}")
        return go.Figure()


def severity_bar(by_severity: dict, title="Severity Distribution"):
    """
    Vertical bar chart for LOW / MEDIUM / HIGH counts.
    """
    try:
        labels = list(by_severity.keys())
        values = list(by_severity.values())
        colors = [SEVERITY_COLORS.get(l, "#888") for l in labels]
        fig = go.Figure([go.Bar(
            x=[l.capitalize() for l in labels],
            y=values,
            marker_color=colors,
            text=values,
            textposition="outside",
        )])
        fig.update_layout(
            title=title,
            xaxis_title="Severity",
            yaxis_title="Count",
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
        )
        return fig
    except Exception as e:
        logger.error(f"severity_bar error: {e}")
        return go.Figure()


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib chart → PNG bytes (used for embedding in PDF reports)
# ─────────────────────────────────────────────────────────────────────────────

def disease_bar_png(by_disease: dict) -> bytes:
    """
    Generate a simple Matplotlib bar chart and return it as PNG bytes.
    Used by the PDF report generator.
    """
    try:
        fig, ax = plt.subplots(figsize=(7, max(3, len(by_disease) * 0.5)))
        diseases = list(by_disease.keys())
        counts   = list(by_disease.values())
        colours  = plt.cm.Set3(np.linspace(0, 1, len(diseases)))
        bars = ax.barh(diseases, counts, color=colours)
        ax.bar_label(bars, padding=3)
        ax.set_xlabel("Predictions")
        ax.set_title("Disease Distribution")
        ax.invert_yaxis()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"disease_bar_png error: {e}")
        return b""


def confusion_matrix_png(cm: np.ndarray, class_names: list) -> bytes:
    """
    Render a confusion matrix as PNG bytes.
    """
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"confusion_matrix_png error: {e}")
        return b""


def training_history_png(history_dict: dict) -> bytes:
    """
    Plot training/validation accuracy and loss from a Keras history dict.
    Returns PNG bytes.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        if "accuracy" in history_dict:
            ax1.plot(history_dict["accuracy"],     label="Train Accuracy", color="#4cc9f0")
        if "val_accuracy" in history_dict:
            ax1.plot(history_dict["val_accuracy"], label="Val Accuracy",   color="#f72585")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Loss
        if "loss" in history_dict:
            ax2.plot(history_dict["loss"],     label="Train Loss", color="#4cc9f0")
        if "val_loss" in history_dict:
            ax2.plot(history_dict["val_loss"], label="Val Loss",   color="#f72585")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.error(f"training_history_png error: {e}")
        return b""
