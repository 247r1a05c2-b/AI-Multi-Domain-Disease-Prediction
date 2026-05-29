"""
BioVision AI — Severity Analysis Engine
Detects infected regions, estimates infection percentage, and assigns severity level.
Uses image segmentation, contour detection, and thresholding.
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Severity thresholds (percentage of infected pixels)
SEVERITY_THRESHOLDS = {
    "LOW":    (0, 20),
    "MEDIUM": (20, 50),
    "HIGH":   (50, 100),
}

# Treatment database keyed by disease name (lower-case)
TREATMENT_DATABASE = {
    # ── Plants ──────────────────────────────────────────────────────────────
    "tomato blight": {
        "description": "Tomato Blight is a fungal disease causing dark lesions on leaves and fruit.",
        "treatments": [
            "Remove and destroy infected plant material immediately.",
            "Apply copper-based fungicide every 7–10 days.",
            "Avoid overhead irrigation; water at the base.",
            "Rotate crops — do not plant tomatoes in the same soil next season.",
            "Use resistant tomato varieties.",
        ],
        "prevention": "Ensure good air circulation and avoid wetting foliage.",
    },
    "powdery mildew": {
        "description": "Powdery Mildew is a fungal disease producing white powdery spots on leaves.",
        "treatments": [
            "Apply neem oil or potassium bicarbonate spray.",
            "Remove severely infected leaves and dispose of them.",
            "Use sulfur-based fungicide as a preventive measure.",
            "Improve air circulation around plants.",
        ],
        "prevention": "Plant in sunny areas with good air flow.",
    },
    "rust": {
        "description": "Rust is a fungal disease causing orange-brown pustules on leaf undersides.",
        "treatments": [
            "Apply sulfur or copper-based fungicide.",
            "Remove and destroy infected leaves.",
            "Avoid overhead watering.",
            "Plant rust-resistant varieties next season.",
        ],
        "prevention": "Keep foliage dry and plant in well-ventilated areas.",
    },
    "healthy": {
        "description": "No disease detected. The plant/subject appears healthy.",
        "treatments": ["Continue regular care and monitoring."],
        "prevention": "Maintain good hygiene and proper nutrition.",
    },
    # ── Humans ──────────────────────────────────────────────────────────────
    "dermatitis": {
        "description": "Dermatitis is skin inflammation causing redness, itching, and rashes.",
        "treatments": [
            "Apply topical corticosteroid cream as prescribed.",
            "Use fragrance-free moisturizers daily.",
            "Avoid known allergens and irritants.",
            "Take antihistamines to reduce itching.",
            "Consult a dermatologist for persistent cases.",
        ],
        "prevention": "Use gentle skin products and avoid harsh chemicals.",
    },
    "psoriasis": {
        "description": "Psoriasis is a chronic autoimmune condition causing scaly, red patches.",
        "treatments": [
            "Use topical corticosteroids or vitamin D analogues.",
            "Phototherapy (UVB light) for moderate-to-severe cases.",
            "Systemic medications (methotrexate, biologics) if needed.",
            "Keep skin well-moisturized.",
        ],
        "prevention": "Manage stress and avoid triggers such as infections and injuries.",
    },
    "acne": {
        "description": "Acne is a skin condition caused by clogged hair follicles.",
        "treatments": [
            "Cleanse face twice daily with a gentle cleanser.",
            "Apply benzoyl peroxide or salicylic acid topically.",
            "Use retinoids for persistent acne.",
            "Avoid touching or picking at pimples.",
            "Consult a dermatologist for severe acne.",
        ],
        "prevention": "Keep skin clean and change pillowcases regularly.",
    },
    "eczema": {
        "description": "Eczema causes dry, itchy, and inflamed skin patches.",
        "treatments": [
            "Apply emollient cream frequently.",
            "Use topical steroids during flare-ups.",
            "Avoid soap, detergents, and other irritants.",
            "Wear loose, breathable clothing.",
            "Identify and avoid personal triggers.",
        ],
        "prevention": "Moisturize skin daily and use fragrance-free products.",
    },
    # ── Animals ─────────────────────────────────────────────────────────────
    "skin infection": {
        "description": "Bacterial or fungal skin infection causing lesions, odour, and itching.",
        "treatments": [
            "Clean affected areas with antiseptic solution.",
            "Apply prescribed antibiotic or antifungal topical medication.",
            "Keep the animal's living area clean and dry.",
            "Complete the full course of any antibiotic prescribed by a vet.",
        ],
        "prevention": "Regular grooming and hygiene checks.",
    },
    "mange": {
        "description": "Mange is caused by parasitic mites, resulting in hair loss and skin crusting.",
        "treatments": [
            "Veterinary examination for mite identification.",
            "Apply prescribed miticide (ivermectin, selamectin).",
            "Treat all in-contact animals simultaneously.",
            "Thoroughly clean and disinfect bedding.",
        ],
        "prevention": "Quarantine new animals and maintain regular vet check-ups.",
    },
    "ringworm": {
        "description": "Ringworm is a contagious fungal infection causing circular lesions.",
        "treatments": [
            "Apply antifungal cream (clotrimazole, miconazole) to affected area.",
            "Give oral antifungal medication if widespread.",
            "Disinfect environment and bedding.",
            "Wear gloves when handling infected animals (zoonotic risk).",
        ],
        "prevention": "Regular veterinary visits and avoiding contact with infected animals.",
    },
}


def analyze_severity(image_source):
    """
    Analyse an image to estimate infected area percentage and severity level.

    Args:
        image_source : PIL Image, NumPy array (BGR), or file path.

    Returns:
        Dict with keys:
            infected_pct  – float [0, 100]
            severity      – str 'LOW' | 'MEDIUM' | 'HIGH'
            annotated_img – NumPy array (RGB) with contours drawn
            contour_count – int
    """
    try:
        # ── Convert to OpenCV BGR numpy array ──────────────────────────────
        if isinstance(image_source, str):
            bgr = cv2.imread(image_source)
        elif isinstance(image_source, Image.Image):
            bgr = cv2.cvtColor(np.array(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_source, np.ndarray):
            bgr = image_source if image_source.shape[-1] == 3 else cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported type: {type(image_source)}")

        if bgr is None:
            raise ValueError("Could not read image.")

        # ── Convert to HSV and detect anomalous (yellowed/browned) regions ─
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Combine masks: yellow-brown (disease spots) + very dark regions
        mask_yellow = cv2.inRange(hsv, (15, 30, 30), (40, 255, 255))
        mask_brown  = cv2.inRange(hsv, (5,  20, 20), (20, 255, 200))
        mask_dark   = cv2.inRange(hsv, (0,  0,  0),  (180, 255, 60))
        combined_mask = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_brown, mask_dark))

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # ── Calculate infection percentage ─────────────────────────────────
        total_pixels    = bgr.shape[0] * bgr.shape[1]
        infected_pixels = int(np.sum(combined_mask > 0))
        infected_pct    = round((infected_pixels / total_pixels) * 100, 2)

        # ── Determine severity ────────────────────────────────────────────
        severity = "LOW"
        for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
            if lo <= infected_pct < hi:
                severity = level
                break
        if infected_pct >= 50:
            severity = "HIGH"

        # ── Draw contours on a copy of the image ──────────────────────────
        annotated = bgr.copy()
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Colour per severity
        colours = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}
        colour = colours.get(severity, (0, 255, 0))
        cv2.drawContours(annotated, contours, -1, colour, 2)

        # Add text overlay
        label = f"Infected: {infected_pct:.1f}% | {severity}"
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, colour, 2, cv2.LINE_AA)

        # Return RGB for Streamlit / PIL display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        return {
            "infected_pct": infected_pct,
            "severity":     severity,
            "annotated_img": annotated_rgb,
            "contour_count": len(contours),
        }

    except Exception as e:
        logger.error(f"Severity analysis error: {e}")
        return {
            "infected_pct": 0.0,
            "severity":     "LOW",
            "annotated_img": None,
            "contour_count": 0,
        }


def get_treatment(disease_name):
    """
    Retrieve treatment information for a given disease name.

    Args:
        disease_name : str — predicted disease label

    Returns:
        Dict with keys: description, treatments (list), prevention
    """
    key = disease_name.lower().strip()
    return TREATMENT_DATABASE.get(key, {
        "description": f"Information for '{disease_name}' is not available in the local database.",
        "treatments": [
            "Consult a qualified specialist for accurate diagnosis and treatment.",
            "Isolate the affected subject to prevent spread.",
        ],
        "prevention": "Early detection and regular check-ups are key.",
    })


def severity_color(severity):
    """Return a hex color string matching the severity level."""
    return {"LOW": "#28a745", "MEDIUM": "#fd7e14", "HIGH": "#dc3545"}.get(severity, "#6c757d")
