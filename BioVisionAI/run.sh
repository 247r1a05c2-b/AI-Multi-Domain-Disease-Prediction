#!/bin/bash
echo "========================================"
echo "  BioVision AI — Startup Script"
echo "========================================"

# Activate or create conda environment
conda activate biovision 2>/dev/null || {
    echo "Creating conda environment..."
    conda create -n biovision python=3.10 -y
    conda activate biovision
}

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Generating sample datasets..."
python setup_datasets.py

echo "Launching BioVision AI..."
streamlit run app.py
