import cv2
import face_recognition
import numpy as np
import json
import os
from datetime import timedelta

# Function to extract face embeddings
def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

# Function to process video
def process_video(video_path, reference_image_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    ref_encoding = get_face_encoding(reference_image_path)
    
    if ref_encoding is None:
        print("Error: Could not extract face encoding from reference image.")
        return
    
    metadata = []
    clip_id, tracking_face, face_clip = 0, None, []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracker = cv2.TrackerCSRT_create()
    initialized = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not initialized:
            face_locations = face_recognition.face_locations(frame_rgb)
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            
            matched = False
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                match = face_recognition.compare_faces([ref_encoding], encoding, tolerance=0.5)
                if match[0]:
                    bbox = (left, top, right - left, bottom - top)
                    tracker.init(frame, bbox)
                    initialized = True
                    clip_id += 1
                    video_filename = os.path.join(output_folder, f'face_clip_{clip_id}.mp4')
                    out = cv2.VideoWriter(video_filename, fourcc, fps, (right - left, bottom - top))
                    tracking_face = video_filename
                    face_clip = []
                    break
        else:
            success, bbox = tracker.update(frame)
            if success:
                left, top, w, h = [int(v) for v in bbox]
                timestamp = str(timedelta(seconds=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                face_clip.append({"timestamp": timestamp, "bbox": [left, top, w, h]})
                face_crop = frame[top:top+h, left:left+w]
                out.write(cv2.resize(face_crop, (w, h)))
            else:
                out.release()
                metadata.append({"file": tracking_face, "frames": face_clip})
                tracking_face, face_clip, initialized = None, [], False
    
    cap.release()
    with open(os.path.join(output_folder, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Processing complete. Clips saved.")

if __name__ == "__main__":
    process_video("data/video_2.mp4", "ref_image_1.png", "output/")

