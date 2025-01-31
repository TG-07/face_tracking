import cv2
import numpy as np
import json
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

class FaceTracker:
    def __init__(self, reference_path, similarity_threshold=0.7, 
                 scene_change_threshold=30, max_occlusion_frames=10):
        self.detector = MTCNN()
        self.reference_embed = self.get_face_embedding(cv2.imread(reference_path))
        self.sim_threshold = similarity_threshold
        self.scene_threshold = scene_change_threshold
        self.max_occlusion = max_occlusion_frames
        
        # Tracking variables
        self.current_clip = []
        self.metadata = []
        self.prev_frame = None
        self.missed_frames = 0
        self.frame_count = 0
        self.fps = 30  # Will be updated from video
        
    def get_face_embedding(self, frame):
        """Extract face embedding using MTCNN and simple feature vector"""
        result = self.detector.detect_faces(frame)
        if result:
            return result[0]['embedding']
        return None

    def is_same_face(self, frame_embed):
        """Compare face embeddings using cosine similarity"""
        if self.reference_embed is None or frame_embed is None:
            return False
        return cosine_similarity([self.reference_embed], [frame_embed])[0][0] > self.sim_threshold

    def scene_changed(self, current_frame):
        """Detect scene changes using histogram comparison"""
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return False
            
        hist_prev = cv2.calcHist([self.prev_frame], [0], None, [256], [0,256])
        hist_curr = cv2.calcHist([current_frame], [0], None, [256], [0,256])
        diff = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CHISQR)
        self.prev_frame = current_frame.copy()
        return diff > self.scene_threshold

    def process_video(self, video_path):
        """Main processing pipeline"""
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            print(self.frame_count)
            
            if self.scene_changed(frame):
                self._finalize_clip()
                
            faces = self.detector.detect_faces(frame)
            target_face = next((f for f in faces if self.is_same_face(f['embedding'])), None)
            
            if target_face:
                self._handle_detection(target_face, timestamp, frame)
            else:
                self._handle_occlusion()
                
        cap.release()
        self._finalize_clip()
        return self.metadata

    def _handle_detection(self, face, timestamp, frame):
        x, y, w, h = face['box']
        self.current_clip.append((frame, (x, y, w, h)))
        self.missed_frames = 0
        
    def _handle_occlusion(self):
        self.missed_frames += 1
        if self.missed_frames > self.max_occlusion:
            self._finalize_clip()

    def _finalize_clip(self):
        if len(self.current_clip) > 1:
            self._save_clip()
            self._save_metadata()
        self.current_clip = []
        self.missed_frames = 0

    def _save_clip(self):
        """Save cropped video clip"""
        if not self.current_clip:
            return
            
        first_frame = self.current_clip[0][0]
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'clip_{len(self.metadata)}.mp4', fourcc, 
                            self.fps, (width, height))
        
        for frame, _ in self.current_clip:
            out.write(frame)
        out.release()

    def _save_metadata(self):
        """Generate metadata entry for current clip"""
        start_time = self.current_clip[0][0] / self.fps
        end_time = self.current_clip[-1][0] / self.fps
        
        clip_meta = {
            "filename": f"clip_{len(self.metadata)}.mp4",
            "start_time": start_time,
            "end_time": end_time,
            "frames": [{
                "timestamp": (i/self.fps) + start_time,
                "bbox": list(bbox)
            } for i, (_, bbox) in enumerate(self.current_clip)]
        }
        self.metadata.append(clip_meta)

# Usage example
if __name__ == "__main__":
    tracker = FaceTracker("/Users/tanisha/Documents/Carnegie Mellon/intern/face_tracking/data/ref_image_1.png")
    metadata = tracker.process_video("/Users/tanisha/Documents/Carnegie Mellon/intern/face_tracking/data/video_2.mp4")
    
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
