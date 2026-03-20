#!/usr/bin/env bash
# Build script for Render deployment
# This ensures CPU-only PyTorch is installed to stay within 512MB RAM limit

set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch first (avoids pulling CUDA packages)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install fastapi uvicorn python-multipart ultralytics easyocr opencv-python-headless Pillow

echo "✅ Build complete - CPU-only deployment ready"
