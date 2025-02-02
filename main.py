# main.py

import argparse
import os
import sys
from src import process_video

def main():
    parser = argparse.ArgumentParser(description="Face Tracker and Video Clipper")
    parser.add_argument("--video_path", required=True, help="Path to the input video file")
    parser.add_argument("--reference_image", required=True, help="Path to the reference image of the target face")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--tracker", choices=['CSRT', 'MIL'], 
                        default='CSRT', help="Choose the tracking algorithm")
    args = parser.parse_args()
    process_video(args.video_path, args.reference_image, args.output_dir, args.tracker)

if __name__ == "__main__":
    main()

