import mediapipe as mp
import numpy as np


class EyeGazeEstimator:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1
        )

        self.LEFT_EYE = [33, 133]
        self.RIGHT_EYE = [362, 263]

    def estimate(self, frame):
        h, w, _ = frame.shape
        results = self.mesh.process(frame)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        left_eye_x = np.mean([lm[i].x for i in self.LEFT_EYE])
        right_eye_x = np.mean([lm[i].x for i in self.RIGHT_EYE])

        gaze_ratio = (left_eye_x + right_eye_x) / 2

        if gaze_ratio < 0.45:
            return "left"
        elif gaze_ratio > 0.55:
            return "right"
        else:
            return "center"
