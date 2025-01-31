import cv2
import os
import json
import numpy as np
import face_recognition
import pandas as pd
from mtcnn import MTCNN
from moviepy.video.io.VideoFileClip import VideoFileClip

# Create output directories
OUTPUT_VIDEO_DIR = "output_videos"
OUTPUT_METADATA_FILE = "metadata.json"
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# Load MTCNN detector
detector = MTCNN()


def extract_faces(frame):
    """Detect faces in a frame using MTCNN."""
    faces = detector.detect_faces(frame)
    return [
        (face["box"][0], face["box"][1], face["box"][2], face["box"][3])
        for face in faces
    ]


def match_target_face(face_encodings, target_encoding):
    """Match detected faces with the reference image."""
    if not face_encodings:
        return None
    matches = face_recognition.compare_faces(face_encodings, target_encoding)
    if True in matches:
        return matches.index(True)
    return None


def process_video(video_path, reference_image_path):
    """Process video to detect, track, and extract target face clips."""
    
    # Load reference image and get face encoding
    ref_image = cv2.imread(reference_image_path)
    ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_encodings = face_recognition.face_encodings(ref_image_rgb)
    
    if len(ref_encodings) == 0:
        print("No face detected in reference image.")
        return
    
    target_encoding = ref_encodings[0]
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    metadata = []
    frame_id = 0
    current_clip = None
    clip_start_frame = None
    face_positions = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = extract_faces(frame_rgb)

        face_encodings = [
            face_recognition.face_encodings(frame_rgb, [face])[0]
            if face_recognition.face_encodings(frame_rgb, [face]) else None
            for face in faces
        ]
        
        # Match the detected faces with the reference face
        target_index = match_target_face(face_encodings, target_encoding)
        
        if target_index is not None:
            x, y, w, h = faces[target_index]
            
            # Start a new clip if not already in one
            if current_clip is None:
                clip_start_frame = frame_id
                current_clip = []

            current_clip.append(frame[y:y+h, x:x+w])
            face_positions.append({"frame": frame_id, "bbox": [x, y, w, h]})
        
        else:
            # If the target face is lost, save the clip
            if current_clip is not None:
                save_clip(video_path, current_clip, clip_start_frame, frame_id, fps, face_positions, metadata)
                current_clip = None
                face_positions = []

        frame_id += 1
    
    # Save any remaining clip
    if current_clip is not None:
        save_clip(video_path, current_clip, clip_start_frame, frame_id, fps, face_positions, metadata)
    
    # Release resources
    cap.release()

    # Save metadata
    with open(OUTPUT_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("Processing complete. Cropped videos and metadata saved.")


def save_clip(video_path, frames, start_frame, end_frame, fps, face_positions, metadata):
    """Save extracted face clip as a video file."""
    
    video_filename = os.path.join(OUTPUT_VIDEO_DIR, f"clip_{start_frame}_{end_frame}.mp4")
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    start_time = start_frame / fps
    end_time = end_frame / fps

    metadata.append({
        "file_name": video_filename,
        "start_time": start_time,
        "end_time": end_time,
        "face_positions": face_positions
    })


if __name__ == "__main__":
    video_path = "data/video_2.mp4"  # Change this to your video file path
    reference_image_path = "data/ref_image_1.png"  # Change this to your reference face image
    
    process_video(video_path, reference_image_path)