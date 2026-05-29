"""
BioVision AI — Dataset Setup Script
Generates synthetic placeholder images for each disease class so the app
and training scripts can run immediately without real image data.

Run this ONCE before training:
    python setup_datasets.py

After running, replace the generated images with real disease images for
accurate predictions.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset structure ──────────────────────────────────────────────────────────
DATASET_STRUCTURE = {
    "plants": {
        "Tomato_Blight":   {"hue": (15, 25), "sat": (150, 220), "count": 60},
        "Powdery_Mildew":  {"hue": (0,   10), "sat": (10,  40),  "count": 60},
        "Rust":            {"hue": (20, 35),  "sat": (180, 255), "count": 60},
        "Healthy":         {"hue": (60, 90),  "sat": (100, 180), "count": 60},
    },
    "humans": {
        "Dermatitis":      {"hue": (5,  15),  "sat": (120, 200), "count": 60},
        "Psoriasis":       {"hue": (0,   8),  "sat": (80,  150), "count": 60},
        "Acne":            {"hue": (345,355),  "sat": (150, 220), "count": 60},
        "Healthy":         {"hue": (20, 35),  "sat": (60,  120), "count": 60},
    },
    "animals": {
        "Skin_Infection":  {"hue": (10, 20),  "sat": (100, 180), "count": 60},
        "Mange":           {"hue": (25, 40),  "sat": (80,  150), "count": 60},
        "Ringworm":        {"hue": (0,   5),  "sat": (50,  120), "count": 60},
        "Healthy":         {"hue": (75, 100), "sat": (80,  160), "count": 60},
    },
}


def hsv_to_rgb(h, s, v):
    """Convert HSV (0-360, 0-255, 0-255) to RGB tuple."""
    h = h / 360.0
    s = s / 255.0
    v = v / 255.0
    if s == 0:
        return (int(v * 255),) * 3
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return tuple(int(c * 255) for c in rgb)


def generate_disease_image(hue_range, sat_range, label, size=(224, 224)):
    """
    Generate a synthetic disease-like image with noise and spots.
    """
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)

    # Background colour derived from the disease hue
    h = random.uniform(*hue_range)
    s = random.uniform(*sat_range)
    v = random.uniform(100, 200)
    bg_color = hsv_to_rgb(h, s, v)
    draw.rectangle([0, 0, *size], fill=bg_color)

    # Add Gaussian noise via pixel manipulation
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 20, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Draw random "lesion" spots
    num_spots = random.randint(3, 12)
    for _ in range(num_spots):
        x = random.randint(10, size[0] - 10)
        y = random.randint(10, size[1] - 10)
        r = random.randint(5, 30)
        spot_h = (h + random.uniform(-20, 20)) % 360
        spot_s = min(255, s + random.uniform(20, 80))
        spot_v = random.uniform(50, 150)
        spot_color = hsv_to_rgb(spot_h, spot_s, spot_v)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=spot_color)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Add class label watermark (small, bottom-right)
    draw = ImageDraw.Draw(img)
    draw.text((4, size[1] - 14), label[:15], fill=(255, 255, 255, 180))

    return img


def create_datasets():
    """Generate all synthetic dataset images."""
    total = 0
    for domain, classes in DATASET_STRUCTURE.items():
        for cls_name, cfg in classes.items():
            cls_dir = os.path.join(BASE_DIR, "datasets", domain, cls_name)
            os.makedirs(cls_dir, exist_ok=True)

            existing = [f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png"))]
            if len(existing) >= cfg["count"]:
                logger.info(f"  Skipping {domain}/{cls_name} — already has {len(existing)} images.")
                continue

            logger.info(f"  Generating {cfg['count']} images for {domain}/{cls_name}...")
            for i in range(cfg["count"]):
                img = generate_disease_image(cfg["hue"], cfg["sat"], cls_name)
                path = os.path.join(cls_dir, f"{cls_name}_{i+1:04d}.jpg")
                img.save(path, "JPEG", quality=85)
                total += 1

    logger.info(f"\nDataset setup complete. Generated {total} synthetic images.")
    logger.info("Replace these images with real disease photos for accurate training.")


def print_dataset_stats():
    """Print a summary of the current dataset."""
    logger.info("\n=== Dataset Statistics ===")
    for domain in ["plants", "humans", "animals"]:
        domain_dir = os.path.join(BASE_DIR, "datasets", domain)
        if not os.path.isdir(domain_dir):
            continue
        logger.info(f"\n[{domain.upper()}]")
        total = 0
        for cls in sorted(os.listdir(domain_dir)):
            cls_path = os.path.join(domain_dir, cls)
            if os.path.isdir(cls_path):
                n = len([f for f in os.listdir(cls_path) if f.endswith((".jpg", ".png", ".jpeg"))])
                logger.info(f"  {cls:<20} {n:>4} images")
                total += n
        logger.info(f"  {'TOTAL':<20} {total:>4} images")


if __name__ == "__main__":
    logger.info("=== BioVision AI — Dataset Setup ===")
    create_datasets()
    print_dataset_stats()
    logger.info("\nDone! You can now run training scripts or launch the Streamlit app.")
