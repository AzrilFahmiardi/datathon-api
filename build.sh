#!/bin/bash
# Build script for Render.com with error handling
set -e

echo "==> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "==> Installing Python dependencies with timeout and retries..."
pip install --timeout=1000 --retries=5 -r requirements.txt

echo "==> Build completed successfully!"
