# src/video_processor.py

import cv2
import json
import os
from .trackers import FaceTracker

class VideoProcessor:
    def __init__(self, video_path, reference_image_path, output_dir, tracker_type='kcf'):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = 32 # self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = 64 #int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.output_dir = output_dir
        self.tracker = FaceTracker(reference_image_path, tracker_type)
        self.clip_counter = 0
        self.current_clip = None
        self.current_writer = None
        self.metadata = []

        print(self.fps)
        print(self.frame_count)

    def process_video(self):
        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_number / self.fps
            target_face, all_faces = self.tracker.track_face(frame)

            if target_face:
                if self.current_clip is None:
                    self.start_new_clip(frame, timestamp)

                self.add_frame_to_clip(frame, target_face, timestamp)
            elif self.current_clip is not None:
                self.end_current_clip(timestamp - 1/self.fps)

            frame_number += 1

        if self.current_clip is not None:
            self.end_current_clip(self.frame_count / self.fps)

        self.save_metadata()

    def start_new_clip(self, frame, timestamp):
        self.clip_counter += 1
        clip_name = f"clip_{self.clip_counter}.mp4"
        output_path = os.path.join(self.output_dir, clip_name)
        height, width = frame.shape[:2]
        self.current_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
        self.current_clip = {
            "file_name": clip_name,
            "start_time": timestamp,
            "frames": []
        }

    def add_frame_to_clip(self, frame, face_coords, timestamp):
        x, y, w, h = face_coords
        cropped_frame = frame[y:y+h, x:x+w]
        self.current_writer.write(cropped_frame)
        self.current_clip["frames"].append({
            "timestamp": timestamp,
            "face_coordinates": list(face_coords)
        })

    def end_current_clip(self, end_timestamp):
        self.current_clip["end_time"] = end_timestamp
        self.metadata.append(self.current_clip)
        self.current_writer.release()
        self.current_clip = None
        self.current_writer = None

    def save_metadata(self):
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

def process_video(video_path, reference_image_path, output_dir, tracker_type='kcf'):
    os.makedirs(output_dir, exist_ok=True)
    processor = VideoProcessor(video_path, reference_image_path, output_dir, tracker_type)
    processor.process_video()
