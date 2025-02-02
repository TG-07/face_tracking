#!/bin/bash

# Set the name of the Conda environment
ENV_NAME="face_tracker"

# Activate the Conda environment
conda activate $ENV_NAME

# Set the paths for input and output
VIDEO_PATH="data/1/video.mp4"
REFERENCE_IMAGE="data/1/ref_image.png"
OUTPUT_DIR="output/1/"
TRACKER="CSRT"   #available trackers - CSRT, MIL

# Run the main.py script
python main.py \
    --video_path "$VIDEO_PATH" \
    --reference_image "$REFERENCE_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --tracker "$TRACKER"

# Deactivate the Conda environment
conda deactivate

echo "Processing complete. Check the $OUTPUT_DIR directory for results."

