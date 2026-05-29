"""
BioVision AI — Image Preprocessing Utilities
Handles image loading, resizing, normalization, and augmentation.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)

# Standard input size for all models
IMAGE_SIZE = (224, 224)


def load_image(image_source):
    """
    Load an image from a file path, NumPy array, or PIL Image.

    Returns:
        PIL Image object, or None on failure.
    """
    try:
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
        elif isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")
        return img
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


def preprocess_image(image_source, target_size=IMAGE_SIZE):
    """
    Load, resize, and normalize an image for model inference.

    Returns:
        NumPy array of shape (1, H, W, 3) with values in [0, 1].
    """
    try:
        img = load_image(image_source)
        if img is None:
            return None

        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)

        # Convert to float32 array and normalize to [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


def augment_image(img_array):
    """
    Apply random augmentation to a single image array (H, W, 3) in [0,255].

    Returns augmented NumPy array.
    """
    try:
        img = Image.fromarray(img_array.astype(np.uint8))

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (-30° to +30°)
        angle = np.random.uniform(-30, 30)
        img = img.rotate(angle, fillcolor=(0, 0, 0))

        # Random brightness adjustment (0.7x – 1.3x)
        factor = np.random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

        # Random zoom (crop 80-100% then resize back)
        zoom = np.random.uniform(0.8, 1.0)
        w, h = img.size
        left = int((1 - zoom) / 2 * w)
        top = int((1 - zoom) / 2 * h)
        right = w - left
        bottom = h - top
        img = img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)

        return np.array(img)
    except Exception as e:
        logger.error(f"Augmentation error: {e}")
        return img_array


def scan_dataset(root_dir):
    """
    Scan a dataset directory and return statistics.

    Expected structure:
        root_dir/
            class_name_1/
                image1.jpg
                ...
            class_name_2/
                ...

    Returns:
        Dict with keys: classes, counts, total, corrupted
    """
    result = {"classes": [], "counts": {}, "total": 0, "corrupted": 0}
    if not os.path.isdir(root_dir):
        return result

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        count = 0
        corrupted = 0
        for fname in os.listdir(cls_path):
            if os.path.splitext(fname)[1].lower() not in valid_exts:
                continue
            fpath = os.path.join(cls_path, fname)
            try:
                with Image.open(fpath) as im:
                    im.verify()
                count += 1
            except Exception:
                corrupted += 1
                result["corrupted"] += 1

        result["classes"].append(cls)
        result["counts"][cls] = count
        result["total"] += count

    return result


def get_sample_images(class_dir, n=4):
    """
    Return up to n image file paths from a class directory.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    if not os.path.isdir(class_dir):
        return images
    for fname in os.listdir(class_dir):
        if os.path.splitext(fname)[1].lower() in valid_exts:
            images.append(os.path.join(class_dir, fname))
        if len(images) >= n:
            break
    return images


def train_val_test_split(file_list, train=0.7, val=0.15):
    """
    Split a list of file paths into train / val / test sets.

    Args:
        train : Fraction for training (default 70%)
        val   : Fraction for validation (default 15%)
                Remaining goes to test.
    Returns:
        (train_list, val_list, test_list)
    """
    np.random.shuffle(file_list)
    n = len(file_list)
    t = int(n * train)
    v = int(n * (train + val))
    return file_list[:t], file_list[t:v], file_list[v:]
