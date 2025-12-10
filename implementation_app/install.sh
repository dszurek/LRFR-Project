#!/bin/bash
# Installation script for Raspberry Pi 5 LRFR Application
# UPDATED for Ubuntu 24.04

set -e # Exit on error

echo "======================================================================"
echo "Raspberry Pi 5 LRFR Application - Installation Script"
echo "======================================================================"
echo ""

# Check if running on ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "⚠️ Warning: Not running on ARM64 (detected: $ARCH)"
    echo "This script is designed for Raspberry Pi 5"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "Installing system dependencies for Ubuntu 24.04..."
sudo apt-get install -y \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    python3-tk \
    git-lfs \
    v4l-utils \
    libopencv-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libharfbuzz-dev \
    libwebp-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev

# Initialize Git LFS
echo ""
echo "Initializing Git LFS..."
git lfs install

# Create virtual environment
echo ""
echo "Creating Python 3.12 virtual environment..."
python3.12 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download models with Git LFS
echo ""
echo "Downloading quantized models..."
cd ..
git lfs pull
cd implementation_app

# Test setup
# echo ""
# echo "Running setup verification..."
# python test_setup.py

# Done
echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run application: python app.py"
echo ""
echo "For troubleshooting, see README.md"
echo ""
