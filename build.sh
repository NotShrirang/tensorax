#!/bin/bash
# Quick build script for development

set -e

echo "Building Tensorax..."

# Check if CUDA is available
if [ -d "/usr/local/cuda" ]; then
    echo "CUDA found, building with GPU support..."
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
else
    echo "CUDA not found, building CPU-only version..."
fi

# Clean previous builds
rm -rf build dist *.egg-info

# Build extension
python setup.py build_ext --inplace

echo "Build complete!"
echo "Run 'pip install -e .' to install in development mode"
