#!/bin/bash
# Setup script for Vast.ai instance

echo "======================================"
echo "Setting up GPT-2 Training Environment"
echo "======================================"
echo ""

# Update package lists
echo "Updating system packages..."
apt-get update

# Install git if not present
echo "Installing git..."
apt-get install -y git wget

# Clone the repository
echo "Cloning repository..."
cd /workspace
git clone https://github.com/AugmentedCat/Power-LLM.git
cd Power-LLM

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install the GPT2 package
echo "Installing GPT2 package..."
cd GPT2
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p build checkpoints outputs

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Download/prepare your dataset (run download_c4_limited.py or download_wikitext.py)"
echo "2. Run training with: bash train.sh"
echo ""
