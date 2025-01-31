#!/bin/bash

# Set up a Python virtual environment and install dependencies

ENV_NAME="face_tracking_env"

echo "Creating Conda environment: $ENV_NAME"
conda create --name $ENV_NAME python=3.8 -y

# Activate the Conda environment
echo "Activating Conda environment..."
conda activate $ENV_NAME

# Install required Python packages
echo "Installing Python dependencies..."

pip install opencv-python mtcnn scikit-learn -y

# Confirm installation
echo "Installation complete!"
echo "To activate the Conda environment, run: conda activate $ENV_NAME"