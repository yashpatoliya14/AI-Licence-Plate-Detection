#!/usr/bin/env bash
# Build script for Render deployment (native Python environment)
# Installs CPU-only PyTorch to stay within 512MB RAM limit

set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch FIRST (avoids pulling ~20 CUDA packages worth ~200MB+ RAM)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install fastapi uvicorn python-multipart ultralytics easyocr opencv-python-headless Pillow

echo "Build complete - CPU-only deployment ready"
