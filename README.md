# Face Tracker and Video Clipper

This project detects and tracks a specific face in a video, creates clips containing only the target face, and generates metadata for the cropped clips.

## Features

- Identify the target face based on a reference image
- Track the target face across frames to create continuous clips
- Split clips when scene changes or full occlusions occur
- Crop video clips to include only the target face
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
git clone https://github.com/TG-07/face_tracking.git
cd face_tracking
```

2. Set up the conda environment (recommended):

Create and Activate Conda Environment
Ensure you have [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/install/) installed. Then, run:
```bash  
conda create --name face_tracker python=3.8 -y
conda activate face_tracker
```
Run the following command to install all required Python packages:
```bash
pip install -r requirements.txt
```

Check if all dependencies are installed properly:
```bash
pip list
```

## Project Structure
```bash
.
├── data/
│   ├── 1/
│   │   ├── video.mp4
│   │   ├── ref_image.png
│
├── output/
│   ├── 1/
│
├── src/
│   ├── track.py
│
├── main.py
├── requirements.txt
├── README.md
```
## Usage
You can run the face tracker and video clipper using the following command:

```bash 
python main.py --video_path <path_to_video> --reference_image <path_to_reference_image> --output_dir <output_directory> --tracker <tracker_type>
```

Arguments:
- `--video_path`: Path to the input video file
- `--reference_image`: Path to the reference image of the target face
- `--output_dir`: Directory to save output files
- `--tracker`: Choose the tracking algorithm (options: 'CSRT', 'MIL', default: 'CSRT')

Example:
```bash 
python main.py --video_path data/1/video.mp4 --reference_image data/1/ref_image.png --output_dir output/1 --tracker CSRT
```
## Output

The script will create the following in the specified output directory:

1. Multiple video files, each containing a continuous clip of the target face
2. A `metadata.json` file containing information about each clip, including:
- File name
- Start and end timestamps
- Frame-wise face coordinates

## Tracker Types

## 1. CSRT (Channel and Spatial Reliability Tracker)
- **Best for:** High accuracy tracking with slow-moving objects.
- **Pros:**
  - Works well with occlusions and abrupt motion.
  - Higher accuracy compared to other trackers.
- **Cons:**
  - Slower than other trackers, making it less suitable for real-time applications.

## 2. MIL (Multiple Instance Learning)
- **Best for:** Handling appearance changes of the object.
- **Pros:**
  - Works well with objects that change in appearance over time.
- **Cons:**
  - Less stable and can drift if misclassified initially.

## Results
The processed videos are saved in the output directory. Note that there is no output for data/2 because the face recognition algorithm failed to detect a face in the reference image due to its quality.

## Limitations

- The accuracy of face detection and recognition is influenced by the quality of the input video and reference image. Low-quality videos may result in missed face detections.
- There is a trade-off between performance and speed. Running facial recognition on every frame is computationally expensive. To optimize this, my approach applies face recognition initially to identify a match and then uses a tracking algorithm with periodic checks every 30 frames.
- Extreme changes in face angle or lighting affect tracking performance.
- The quality of the generated results depends on the effectiveness of the face recognition and matching algorithms.

## Acknowledgments

- Face recognition is powered by the [face_recognition](https://github.com/ageitgey/face_recognition) library
- OpenCV is used for video processing and tracking algorithms

Stock footage provided by [Freepik](https://www.videvo.net/author/freepik/), downloaded from [videvo.net](https://www.videvo.net/). 
Other input videos downloaded from Youtube