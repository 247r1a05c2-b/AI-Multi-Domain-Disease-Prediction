"""
BioVision AI — Model Loader
Handles loading saved Keras models, class maps, and running inference.
Falls back to a demo/mock predictor when no trained model is found.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
IMAGE_SIZE  = (224, 224)

# Default class labels (used when no .json class map exists)
DEFAULT_CLASSES = {
    "plant":  ["Healthy", "Powdery_Mildew", "Rust", "Tomato_Blight"],
    "human":  ["Acne", "Dermatitis", "Healthy", "Psoriasis"],
    "animal": ["Healthy", "Mange", "Ringworm", "Skin_Infection"],
}

_model_cache: dict = {}


def _load_keras_model(path: str):
    """Load a .h5 Keras model with error handling."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {path}: {e}")
        return None


def _load_class_map(model_path: str) -> dict:
    """
    Load the class index → label mapping saved alongside the model.
    Returns {index (int): label (str)}.
    """
    map_path = model_path.replace(".h5", "_classes.json")
    try:
        with open(map_path) as f:
            raw = json.load(f)  # {label: index}
        return {int(v): k for k, v in raw.items()}
    except Exception:
        logger.warning(f"No class map found at {map_path}. Using defaults.")
        return {}


def get_model(category: str):
    """
    Return (model, class_map) for the requested category.
    Uses an in-process cache to avoid reloading on every call.

    Args:
        category : 'plant', 'human', or 'animal'

    Returns:
        (model_or_None, {index: label})
    """
    category = category.lower()
    if category in _model_cache:
        return _model_cache[category]

    model_path = os.path.join(MODELS_DIR, f"{category}_model.h5")
    if not os.path.exists(model_path):
        logger.warning(f"No trained model found for '{category}'. Using demo predictor.")
        _model_cache[category] = (None, {i: c for i, c in enumerate(DEFAULT_CLASSES.get(category, []))})
        return _model_cache[category]

    model     = _load_keras_model(model_path)
    class_map = _load_class_map(model_path)
    if not class_map:
        defaults = DEFAULT_CLASSES.get(category, [])
        class_map = {i: c for i, c in enumerate(defaults)}

    _model_cache[category] = (model, class_map)
    return _model_cache[category]


def predict(category: str, preprocessed_image: np.ndarray):
    """
    Run inference on a preprocessed image array.

    Args:
        category          : 'plant', 'human', or 'animal'
        preprocessed_image: NumPy array of shape (1, 224, 224, 3) in [0, 1]

    Returns:
        List of (label, confidence) tuples sorted by confidence descending.
    """
    try:
        model, class_map = get_model(category)

        if model is None:
            # Demo mode: return random probabilities for UI testing
            n = len(class_map)
            raw = np.random.dirichlet(np.ones(n) * 0.5)
            results = [(class_map[i], float(raw[i])) for i in range(n)]
        else:
            preds  = model.predict(preprocessed_image, verbose=0)[0]
            results = [(class_map.get(i, f"Class_{i}"), float(preds[i])) for i in range(len(preds))]

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        defaults = DEFAULT_CLASSES.get(category, ["Unknown"])
        return [(defaults[0], 1.0)]


def clear_cache():
    """Clear the model cache (useful when models are retrained)."""
    global _model_cache
    _model_cache = {}
    logger.info("Model cache cleared.")


def model_exists(category: str) -> bool:
    """Return True if a trained model file exists for the category."""
    path = os.path.join(MODELS_DIR, f"{category}_model.h5")
    return os.path.exists(path)
