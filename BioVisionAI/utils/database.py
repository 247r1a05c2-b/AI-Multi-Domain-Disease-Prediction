"""
BioVision AI — Database Manager
Handles all SQLite database operations for storing and retrieving predictions.
"""

import sqlite3
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "predictions.db")


def get_connection():
    """Create and return a database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def initialize_database():
    """Create all required tables if they don't exist."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                category    TEXT NOT NULL,
                disease     TEXT NOT NULL,
                confidence  REAL NOT NULL,
                severity    TEXT NOT NULL,
                image_path  TEXT,
                treatment   TEXT
            )
        """)

        # Users table for authentication
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email         TEXT,
                created_at    TEXT NOT NULL
            )
        """)

        # Settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def save_prediction(category, disease, confidence, severity, image_path="", treatment=""):
    """
    Save a prediction record to the database.

    Args:
        category    : 'plant', 'human', or 'animal'
        disease     : Predicted disease name
        confidence  : Confidence score (0.0 – 1.0)
        severity    : 'LOW', 'MEDIUM', or 'HIGH'
        image_path  : Path to the image that was analyzed
        treatment   : Suggested treatment string
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (timestamp, category, disease, confidence, severity, image_path, treatment)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            category,
            disease,
            round(float(confidence), 4),
            severity,
            image_path,
            treatment,
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        logger.info(f"Prediction saved with id={record_id}.")
        return record_id
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
        return None


def get_all_predictions(category=None, limit=500):
    """
    Retrieve predictions from the database.

    Args:
        category : Optional filter ('plant', 'human', 'animal')
        limit    : Maximum number of rows to return
    Returns:
        List of dicts
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        if category:
            cursor.execute(
                "SELECT * FROM predictions WHERE category=? ORDER BY id DESC LIMIT ?",
                (category, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return []


def get_analytics_summary():
    """
    Return aggregated statistics for the analytics dashboard.

    Returns:
        Dict with keys: total, by_category, by_disease, by_severity, recent_trend
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) as total FROM predictions")
        total = cursor.fetchone()["total"]

        # By category
        cursor.execute(
            "SELECT category, COUNT(*) as count FROM predictions GROUP BY category"
        )
        by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

        # By disease (top 10)
        cursor.execute(
            "SELECT disease, COUNT(*) as count FROM predictions GROUP BY disease ORDER BY count DESC LIMIT 10"
        )
        by_disease = {row["disease"]: row["count"] for row in cursor.fetchall()}

        # By severity
        cursor.execute(
            "SELECT severity, COUNT(*) as count FROM predictions GROUP BY severity"
        )
        by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}

        # Recent trend (last 30 days daily counts)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM predictions
            WHERE timestamp >= DATE('now','-30 days')
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
        recent_trend = {row["date"]: row["count"] for row in cursor.fetchall()}

        conn.close()
        return {
            "total": total,
            "by_category": by_category,
            "by_disease": by_disease,
            "by_severity": by_severity,
            "recent_trend": recent_trend,
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return {}


def delete_prediction(record_id):
    """Delete a single prediction record by id."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id=?", (record_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting prediction {record_id}: {e}")
        return False


def clear_all_predictions():
    """Delete all prediction records (use with caution)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        return False


# Auto-initialize database when module is imported
initialize_database()
