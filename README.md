# 🔬 BioVision AI — Advanced Multi-Domain Disease Prediction System

A fully local, open-source AI platform for detecting diseases in **plants**, **humans**, and **animals** using deep learning, computer vision, severity analysis, PDF reports, an analytics dashboard, and a voice assistant.

---

## ✅ Requirements

- **Python 3.9–3.11** (Python 3.10 recommended)
- **Anaconda** or **Miniconda**
- **Webcam** (optional — upload mode works without it)
- No internet or paid API keys required

---

## ⚡ Quick Start (Anaconda Prompt)

### Step 1 — Create and activate a Conda environment

```bash
conda create -n biovision python=3.10 -y
conda activate biovision
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation may take a few minutes.
> If you have a GPU, TensorFlow will use it automatically.

### Step 3 — Generate sample datasets

```bash
python setup_datasets.py
```

This creates synthetic placeholder images in `datasets/` so the app can run immediately.
**For accurate predictions**, replace these with real disease images.

### Step 4 — (Optional) Train models

```bash
python training/train_plants.py
python training/train_humans.py
python training/train_animals.py
```

Without trained models, the app runs in **demo mode** with random predictions — all UI features still work.

### Step 5 — Launch the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
BioVisionAI/
├── app.py                    ← Main Streamlit application
├── setup_datasets.py         ← Dataset generator (run first)
├── requirements.txt          ← Python dependencies
├── README.md
│
├── datasets/
│   ├── plants/
│   │   ├── Tomato_Blight/    ← Place plant disease images here
│   │   ├── Powdery_Mildew/
│   │   ├── Rust/
│   │   └── Healthy/
│   ├── humans/
│   │   ├── Dermatitis/
│   │   ├── Psoriasis/
│   │   ├── Acne/
│   │   └── Healthy/
│   └── animals/
│       ├── Skin_Infection/
│       ├── Mange/
│       ├── Ringworm/
│       └── Healthy/
│
├── saved_models/             ← Trained .h5 models saved here
│   ├── plant_model.h5
│   ├── plant_model_classes.json
│   ├── human_model.h5
│   └── animal_model.h5
│
├── reports/                  ← Generated PDF reports
│
├── database/
│   └── predictions.db        ← SQLite database (auto-created)
│
├── utils/
│   ├── database.py           ← SQLite CRUD operations
│   ├── preprocessing.py      ← Image loading, resizing, augmentation
│   ├── model_loader.py       ← Keras model loading & inference
│   ├── severity.py           ← Contour-based severity analysis + treatments
│   ├── voice.py              ← pyttsx3 voice assistant
│   ├── report_generator.py   ← ReportLab PDF generation
│   └── charts.py             ← Plotly & Matplotlib visualizations
│
├── training/
│   ├── train_plants.py       ← MobileNetV2 training — plant diseases
│   ├── train_humans.py       ← MobileNetV2 training — skin conditions
│   └── train_animals.py      ← MobileNetV2 training — animal diseases
│
├── assets/                   ← Saved chart PNGs from training
└── logs/                     ← Application logs
```

---

## 🌿 Supported Diseases

| Domain  | Classes |
|---------|---------|
| Plants  | Tomato Blight, Powdery Mildew, Rust, Healthy |
| Humans  | Dermatitis, Psoriasis, Acne, Healthy |
| Animals | Skin Infection, Mange, Ringworm, Healthy |

---

## 🖥️ App Features

| Feature | Description |
|---------|-------------|
| 🔍 **Predict** | Upload image or capture from webcam; get disease, confidence %, severity |
| 📷 **Live Camera** | Real-time frame-by-frame detection with OpenCV |
| 📊 **Analytics** | Dashboard with bar, pie, line, and gauge charts |
| 📋 **History** | Searchable prediction history with CSV export |
| 📥 **Reports** | Professional PDF reports with image, analysis, treatment, charts |
| 🔊 **Voice** | Local TTS via pyttsx3 — speaks prediction results aloud |
| ⚡ **Severity** | Contour detection → infected area % → LOW/MEDIUM/HIGH gauge |
| 💊 **Treatments** | Detailed treatment and prevention advice per disease |
| ⚙️ **Settings** | Voice toggle, language, model management, dataset status |

---

## 📊 Output Examples

- **Prediction**: "Tomato Blight — 92.3% confidence"
- **Severity**: "Infected area: 34.1% — MEDIUM"
- **Report**: PDF with image, charts, treatment plan — downloadable instantly
- **Analytics**: Disease distribution bar chart, category pie chart, trend line over 30 days
- **Comparison**: All disease classes ranked by confidence %

---

## 🛠️ Troubleshooting

### pyttsx3 voice not working on Linux
```bash
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
```

### Webcam not detected
The webcam tab requires a physical webcam. Use the **Upload Image** tab in cloud/VM environments.

### TensorFlow GPU support
Install CUDA 11.8 + cuDNN 8.6 for GPU acceleration, then TF will auto-detect it.

### PyAudio install fails
```bash
conda install -c conda-forge pyaudio
```

### reportlab missing
```bash
pip install reportlab
```

---

## 🔧 Training on Real Data

1. Download a public disease dataset (e.g. PlantVillage from Kaggle).
2. Organise into `datasets/plants/<ClassName>/` folders.
3. Run `python training/train_plants.py`.
4. The trained model is saved to `saved_models/plant_model.h5`.
5. Restart the app — it loads the new model automatically.

---
🚀 Live Demo: https://ai-multi-domain-disease-1st.streamlit.app/


## 📜 Disclaimer

This software is for **educational and research purposes only**.
It does **not** replace professional medical, veterinary, or agricultural diagnosis.
Always consult a qualified professional for treatment decisions.

---

*BioVision AI v1.0.0 — Fully offline · No paid APIs · Open source*
