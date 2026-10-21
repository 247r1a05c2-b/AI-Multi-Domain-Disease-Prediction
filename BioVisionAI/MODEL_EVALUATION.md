# Model evaluation and data safety

This document is a validation protocol, not evidence that any model is clinically or agriculturally validated.

## Before reporting performance

1. Identify the source, license, collection method, label quality, and limitations of each dataset.
2. Remove duplicate and near-duplicate images before splitting.
3. Split by source/person/plant/animal where possible, not only by random image, to reduce leakage.
4. Keep a held-out test set untouched during model selection.
5. Compare against a simple baseline and report per-class precision, recall, F1, confusion matrix, and class counts.
6. Report calibration and error examples; a softmax score is not automatically a calibrated probability.
7. Test distribution shifts such as lighting, camera, background, image quality, and underrepresented classes.
8. Record code version, dataset version, preprocessing, model architecture, hyperparameters, and random seed.

## Human-health boundary

The human-skin component is a student prototype, not a medical device or diagnostic service. Do not use it to diagnose, triage, recommend treatment, or replace a qualified clinician. Do not publish claims of medical accuracy without a suitable independently reviewed evaluation and appropriate approvals.

## Demo behavior

If a trained model is missing or fails to load, the application should display a non-diagnostic marker rather than fabricate a disease label or confidence. Synthetic placeholder images are for testing the interface only and must not be treated as training or validation evidence.

## Dataset and privacy requirements

- Use datasets with documented permission and license.
- Do not publish identifiable medical images, private patient information, or face images without explicit permission.
- Keep train/validation/test manifests and raw metric outputs for reproducibility.
