#!/usr/bin/env bash
# Build script for Render deployment
# Installs system packages + CPU-only Python deps to stay within 512MB RAM

set -o errexit

# Install Tesseract OCR as a system package (much lighter than EasyOCR)
apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch first (avoids pulling ~20 CUDA packages)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (no easyocr!)
pip install fastapi uvicorn python-multipart ultralytics pytesseract opencv-python-headless Pillow

echo "✅ Build complete - CPU-only deployment with Tesseract OCR ready"
