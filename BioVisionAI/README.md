# 🔬 BioVision AI — Advanced Multi-Domain Disease Prediction System

A fully local, open-source AI platform for detecting diseases in **plants**, **humans**, and **animals** using deep learning, computer vision, severity analysis, PDF reports, an analytics dashboard, and a voice assistant.

---

## ✅ Requirements

- **Python 3.9–3.11** (Python 3.10 recommended)
- **Anaconda** or **Miniconda**
- **Webcam** (optional — upload mode works without it)
- Internet is required once to download datasets and ImageNet transfer-learning weights; inference can run locally afterward

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

### Step 3 — Configure Kaggle and download real images

Install the Kaggle CLI and configure your Kaggle API token (Kaggle account → Settings → API). Keep the token private; never commit it.

```bash
pip install kaggle
python training/prepare_real_datasets.py --domains plant
```

If you previously generated synthetic demo images or already have images in that dataset folder, the script stops rather than mixing them with real images. Use `--replace-existing` only if you intentionally want to replace the existing dataset.

This downloads the PlantVillage image dataset from Kaggle and organizes real images under `datasets/plants/<class>/`. To prepare all three domains, use `--domains plant human animal`; this downloads substantially more data. The human option prepares HAM10000 skin-lesion labels (7 categories), which is licensed for non-commercial use. Check each dataset's current license and terms before use.

### Step 4 — Train the selected model

```bash
python training/train_plants.py
```

Or prepare and train the plant model in one command:

```bash
python training/prepare_real_datasets.py --domains plant --train
```

The trained model and class map are saved under `saved_models/`. Human and animal training scripts are available too, but only train them after reviewing their source labels, licenses, and class balance. No model weights are bundled in this repository yet; the script downloads real datasets and trains weights in your environment. Without a successfully loaded trained model, the app shows a non-diagnostic unavailable message instead of inventing a disease label.

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
| Plants  | Classes found in the downloaded PlantVillage dataset (up to 38 disease/healthy categories) |
| Humans  | HAM10000 skin-lesion labels: akiec, bcc, bkl, df, mel, nv, vasc |
| Animals | Classes supplied by the selected animal dataset; inspect its labels before training |

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

## 🔧 Real dataset sources

- Plant disease: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Human skin lesions: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) — non-commercial license; not a diagnostic model.
- Animal disease: [Animal Disease Dataset on Kaggle](https://www.kaggle.com/datasets/muzammilwaseem/animal-disease-dataset) — verify the current dataset license and label definitions.

The preparation script never treats generated demo images as real training evidence. `setup_datasets.py` is only for UI demonstrations and must not be used to train a disease classifier. For model evaluation and health-safety limitations, see `MODEL_EVALUATION.md`.

---

## 📜 Disclaimer

This software is for **educational and research purposes only**.
It does **not** replace professional medical, veterinary, or agricultural diagnosis.
Always consult a qualified professional for treatment decisions.

---

*BioVision AI v1.0.0 — Fully offline · No paid APIs · Open source*
