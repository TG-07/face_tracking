import cv2
import face_recognition
import numpy as np

class FaceTracker:
    def __init__(self, reference_image_path):
        self.reference_encoding = self._encode_reference_face(reference_image_path)
        self.face_locations = []
        self.face_encodings = []
        self.tracked_face = None

    def _encode_reference_face(self, image_path):
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            raise ValueError("No face found in the reference image")
        return encodings[0]

    def detect_and_track(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.face_locations = face_recognition.face_locations(rgb_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_frame, self.face_locations)

        for face_encoding, face_location in zip(self.face_encodings, self.face_locations):
            match = face_recognition.compare_faces([self.reference_encoding], face_encoding)[0]
            if match:
                self.tracked_face = face_location
                return face_location

        self.tracked_face = None
        return None

    def get_face_coordinates(self):
        if self.tracked_face:
            top, right, bottom, left = self.tracked_face
            return [left, top, right - left, bottom - top]
        return None
