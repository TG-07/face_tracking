import cv2
import face_recognition
import numpy as np
import json
import os
from datetime import timedelta

def get_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

def create_tracker(tracker_type):
    trackers = {
        "CSRT": cv2.TrackerCSRT_create(),
        "MIL": cv2.TrackerMIL_create(),
    }
    return trackers[tracker_type]

def process_video(video_path, reference_image_path, output_folder, tracker_type):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ref_encoding = get_face_encoding(reference_image_path)
    
    if ref_encoding is None:
        print("Error: Could not extract face encoding from reference image.")
        return
    
    metadata = []
    clip_id, tracking_face, face_clip = 0, None, []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracker = create_tracker(tracker_type)
    initialized = False
    W, H = 0, 0
    frame_counter = 0
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        
        if not initialized:
            face_locations = face_recognition.face_locations(frame_rgb)
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                match = face_recognition.compare_faces([ref_encoding], encoding, tolerance=0.50)
                
                if match[0]:
                    bbox = (left, top, right - left, bottom - top)
                    W, H = right - left, bottom - top
                    tracker.init(frame, bbox)
                    initialized = True
                    clip_id += 1
                    video_filename = os.path.join(output_folder, f'clip{clip_id}.mp4')
                    out = cv2.VideoWriter(video_filename, fourcc, fps, (W, H))
                    tracking_face = video_filename
                    face_clip = []
                    start_time = current_time
                    break
        else:
            success, bbox = tracker.update(frame)
            if success:
                left, top, w, h = [int(v) for v in bbox]
                face_clip.append({
                    "timestamp": str(timedelta(seconds=current_time)),
                    "bbox": [left, top, w, h]
                })
                face_crop = frame[top:top+h, left:left+w]
                
                if face_crop.size != 0:
                    # Periodic face verification every 30 frames
                    if frame_counter % 30 == 0:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        encoding = face_recognition.face_encodings(face_crop_rgb)
                        if encoding:
                            match = face_recognition.compare_faces([ref_encoding], encoding[0], tolerance=0.50)
                            if not match[0]:
                                out.release()
                                metadata.append({
                                    "file": tracking_face,
                                    "start_time": start_time,
                                    "end_time": current_time,
                                    "frames": face_clip
                                })
                                tracking_face, face_clip, initialized = None, [], False
                                continue
                    
                    out.write(cv2.resize(face_crop, (W, H)))
            else:
                if tracking_face and face_clip:
                    out.release()
                    metadata.append({
                        "file": tracking_face,
                        "start_time": start_time,
                        "end_time": current_time,
                        "frames": face_clip
                    })
                tracking_face, face_clip, initialized = None, [], False
    
    if tracking_face and face_clip:
        out.release()
        metadata.append({
            "file": tracking_face,
            "start_time": start_time,
            "end_time": current_time,
            "frames": face_clip
        })
    
    cap.release()
    metadata_path = os.path.join(output_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Processing complete. Clips saved. Metadata written to {metadata_path}.")