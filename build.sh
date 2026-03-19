#!/bin/bash
set -e

echo "Creating isolated venv with uv..."
uv venv .venv --python 3.11
source .venv/bin/activate

echo "Installing dependencies..."
uv pip install pyinstaller opencv-python numpy tkinterdnd2

echo "Building app..."
pyinstaller -y \
  --windowed \
  --onedir \
  --collect-all cv2 \
  --collect-all numpy \
  --collect-all tkinterdnd2 \
  --hidden-import=tkinterdnd2 \
  --name "Perforation Stabilizer" \
  src/perforation_stabilizer_app.py

echo ""
echo "Done! App is at: dist/Perforation Stabilizer.app"
echo "Zip it and send to the client."
