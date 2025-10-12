#!/bin/bash
set -e

# Setup script for the unified medical imaging pipeline
# This script installs dependencies and sets up the environment

echo "Setting up unified medical imaging pipeline..."

# Check Python version
echo "Checking Python version..."
python3 --version

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install TotalSegmentator (if not already installed)
echo "Installing TotalSegmentator..."
pip3 install TotalSegmentator

# Make scripts executable
echo "Making scripts executable..."
chmod +x unified_pipeline.sh
chmod +x setup_pipeline.sh

# Test installation
echo "Testing installation..."
python3 test_unified_pipeline.py

echo "Setup completed successfully!"
echo ""
echo "Usage examples:"
echo "  For vindr_ds: ./unified_pipeline.sh vindr_ds /path/to/vindr_ds"
echo "  For pars-ct:  ./unified_pipeline.sh pars-ct /path/to/pars-ct --phase-labels-csv /path/to/phase_labels.csv"
echo ""
echo "For more information, run: ./unified_pipeline.sh --help"

