#!/bin/bash
set -e

echo "Installing PyInstaller..."
pip3 install pyinstaller

echo "Building app..."
pyinstaller \
  --windowed \
  --onedir \
  --hidden-import=tkinterdnd2 \
  --name "Perforation Stabilizer" \
  src/perforation_stabilizer_app.py

echo ""
echo "Done! App is at: dist/Perforation Stabilizer.app"
echo "Zip it and send to the client."
