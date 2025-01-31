#!/bin/bash

# Set the name of the Conda environment
ENV_NAME="face_tracker"

# Activate the Conda environment
source activate $ENV_NAME

# Set the paths for input and output
VIDEO_PATH="data/video_2.mp4"
REFERENCE_IMAGE="data/ref_image_1.png"
OUTPUT_DIR="output"
TRACKER="mosse"

# Run the main.py script
python main.py \
    --video_path "$VIDEO_PATH" \
    --reference_image "$REFERENCE_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --tracker "$TRACKER"

# Deactivate the Conda environment
conda deactivate

echo "Processing complete. Check the $OUTPUT_DIR directory for results."

