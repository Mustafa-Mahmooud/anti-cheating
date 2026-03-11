import mediapipe as mp


class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def detect(self, frame):
        results = self.detector.process(frame)
        if results.detections:
            return len(results.detections)
        return 0