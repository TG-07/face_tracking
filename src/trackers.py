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
        "CSRT": cv2.TrackerCSRT_create,
        "MIL": cv2.TrackerMIL_create,
    }
    return trackers.get(tracker_type, cv2.TrackerCSRT_create)()

def process_video(video_path, reference_image_path, output_folder, tracker_type="CSRT"):
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
    tracker = create_tracker(tracker_type)
    initialized = False
    out = None
    start_time = None
    frame_counter = 0

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
                match = face_recognition.compare_faces([ref_encoding], encoding, tolerance=0.95)
                
                if match[0]:
                    bbox = (left, top, right - left, bottom - top)
                    W, H = right - left, bottom - top
                    tracker.init(frame, bbox)
                    initialized = True
                    clip_id += 1
                    video_filename = os.path.join(output_folder, f"clip{clip_id}.mp4")
                    out = cv2.VideoWriter(video_filename, fourcc, fps, (right - left, bottom - top))
                    tracking_face = video_filename
                    face_clip = []
                    start_time = current_time
                    break
        else:
            success, bbox = tracker.update(frame)
            if success:
                left, top, w, h = [int(v) for v in bbox]
                face_clip.append({"timestamp": current_time, "bbox": [left, top, w, h]})
                face_crop = frame[top:top+h, left:left+w]
                
                if face_crop is not None and face_crop.size != 0:
                    if frame_counter % 10 == 0:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        encoding = face_recognition.face_encodings(face_crop_rgb)
                        if encoding:
                            match = face_recognition.compare_faces([ref_encoding], encoding[0], tolerance=0.95)
                            if not match[0]:
                                out.release()
                                metadata.append({"file": tracking_face, "start_time": start_time, "end_time": current_time, "frames": face_clip})
                                tracking_face, face_clip, initialized, out, start_time = None, [], False, None, None
                                continue

                    out.write(cv2.resize(face_crop, (W, H)))
            else:
                if tracking_face and face_clip and start_time is not None:
                    out.release()
                    metadata.append({"file": tracking_face, "start_time": start_time, "end_time": current_time, "frames": face_clip})
                tracking_face, face_clip, initialized, out, start_time = None, [], False, None, None
    
    if tracking_face and face_clip and start_time is not None:
        out.release()
        metadata.append({"file": tracking_face, "start_time": start_time, "end_time": current_time, "frames": face_clip})
    
    cap.release()
    metadata_path = os.path.join(output_folder, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Processing complete. Clips saved. Metadata written to {metadata_path}.")

if __name__ == "__main__":
    process_video("../data/4/video.mp4", "../data/4/ref_image.png", "../output/4/", tracker_type="CSRT")