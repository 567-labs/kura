#!/bin/bash
set -e

echo "Building UI..."
cd ui
npm install
npm run build
cd ..

echo "Building Python package..."
rm -rf dist build *.egg-info
uv build

echo "Build complete! Static files included:"
unzip -l dist/kura-*.whl | grep static || echo "No static files found in wheel"