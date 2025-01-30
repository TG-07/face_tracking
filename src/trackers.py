import cv2
import face_recognition
import numpy as np

def detect_faces(frame, reference_encoding):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    target_face = None
    all_faces = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        bbox = (left, top, right-left, bottom-top)
        all_faces.append(bbox)
        
        match = face_recognition.compare_faces([reference_encoding], face_encoding)[0]
        if match:
            target_face = bbox
    
    return target_face, all_faces

class FaceTracker:
    def __init__(self, reference_image_path, tracker_type='kcf'):
        reference_image = face_recognition.load_image_file(reference_image_path)
        self.reference_encoding = face_recognition.face_encodings(reference_image)[0]
        self.tracker_type = tracker_type
        self.tracker = None
        self.initialized = False

    def reset_tracker(self):
        if self.tracker_type == 'kcf':
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'csrt':
            self.tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == 'mosse':
            self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.initialized = False

    def track_face(self, frame):
        if not self.initialized:
            target_face, all_faces = detect_faces(frame, self.reference_encoding)
            if target_face:
                self.tracker.init(frame, target_face)
                self.initialized = True
            return target_face, all_faces
        
        success, bbox = self.tracker.update(frame)
        if success:
            return bbox, None
        else:
            self.reset_tracker()
            return self.track_face(frame)
