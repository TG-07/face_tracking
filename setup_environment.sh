#!/bin/bash

# Set the name for the Conda environment
ENV_NAME="face_tracker"

# Set the Python version
PYTHON_VERSION="3.8"

# Create the Conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
source activate $ENV_NAME

# Install the required packages
conda install -c conda-forge opencv=4.5.3 -y
conda install -c conda-forge face_recognition=1.3.0 -y
conda install numpy=1.21.2 -y

# Install additional dependencies that might be required
conda install -c conda-forge dlib -y

# Print confirmation message
echo "Conda environment '$ENV_NAME' has been created and packages have been installed."
echo "To activate the environment, use: conda activate $ENV_NAME"

