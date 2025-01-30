import cv2
import face_recognition
import numpy as np

class HybridFaceTracker:
    def __init__(self, reference_image_path):
        # Initial face detection using face_recognition
        reference_image = face_recognition.load_image_file(reference_image_path)
        self.reference_encoding = face_recognition.face_encodings(reference_image)[0]
        
        # Initialize tracker
        self.tracker = cv2.TrackerKCF_create()
        self.tracked_face = None
        self.tracking_active = False

    def detect_initial_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match = face_recognition.compare_faces([self.reference_encoding], face_encoding)[0]
            
            if match:
                bbox = (left, top, right-left, bottom-top)
                self.tracker.init(frame, bbox)
                self.tracked_face = bbox
                self.tracking_active = True
                return bbox
        
        return None

    def track_face(self, frame):
        if not self.tracking_active:
            return self.detect_initial_face(frame)
        
        success, bbox = self.tracker.update(frame)
        
        if success:
            self.tracked_face = bbox
            return bbox
        else:
            self.tracking_active = False
            return self.detect_initial_face(frame)

def process_video(video_path, reference_image_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    face_tracker = HybridFaceTracker(reference_image_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        bbox = face_tracker.track_face(frame)
        
        if bbox:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

