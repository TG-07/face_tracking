# Face Tracker and Video Clipper

This project detects and tracks a specific face in a video, creates clips containing only the target face, and generates metadata for the cropped clips.

## Features

- Identify the target face based on a reference image
- Track the target face across frames to create continuous clips
- Split clips when scene changes or full occlusions occur
- Extract video clips to include only the target face
- Save cropped videos as individual files
- Generate metadata (JSON) for each clip, including:
  - File name of the cropped video
  - Start and end timestamps of the clip
  - Frame-wise face coordinates

## Requirements

- Python 3.7+
- OpenCV
- face_recognition
- numpy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/face-tracker-clipper.git
cd face-tracker
```

2. Set up the conda environment (recommended):
```bash  
    chmod +x setup_environment.sh
    bash setup_environment.sh 
```
## Usage

You can run the face tracker and video clipper using the following command:

1. Make the script executable:
```bash
chmod +x run.sh
```

2. Run the script:
```bash 
bash run.sh
```

You can modify the variables in `run.sh` to change the input paths, output directory, or tracker type.

Alternatively, you can use this command to run it:

```bash 
python main.py --video_path <path_to_video> --reference_image <path_to_reference_image> --output_dir <output_directory> --tracker <tracker_type>
```

Arguments:
- `--video_path`: Path to the input video file
- `--reference_image`: Path to the reference image of the target face
- `--output_dir`: Directory to save output files
- `--tracker`: Choose the tracking algorithm (options: 'kcf', 'csrt', 'mosse', default: 'kcf')

Example:
```bash 
python main.py --video_path data/sample_video.mp4 --reference_image data/reference_face.jpg --output_dir output --tracker kcf
```
## Output

The script will create the following in the specified output directory:

1. Multiple video files, each containing a continuous clip of the target face
2. A `metadata.json` file containing information about each clip, including:
- File name
- Start and end timestamps
- Frame-wise face coordinates

## Tracker Types

- KCF (Kernelized Correlation Filters): Fast and accurate for most scenarios
- CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability): More accurate but slower than KCF
- MOSSE (Minimum Output Sum of Squared Error): Very fast, but may be less accurate in complex scenarios

## Limitations

- The face detection and recognition accuracy depends on the quality of the input video and reference image
- Extreme changes in face angle or lighting may affect tracking performance
- The script currently does not handle real-time video input

## Acknowledgments

- Face recognition is powered by the [face_recognition](https://github.com/ageitgey/face_recognition) library
- OpenCV is used for video processing and tracking algorithms

Stock footage provided by [Freepik](https://www.videvo.net/author/freepik/), downloaded from [videvo.net](https://www.videvo.net/)