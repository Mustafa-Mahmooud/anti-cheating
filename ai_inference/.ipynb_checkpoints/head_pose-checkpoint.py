import cv2
import mediapipe as mp
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1
        )

    def estimate(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (landmarks[33].x * w, landmarks[33].y * h),    # Left eye
            (landmarks[263].x * w, landmarks[263].y * h),  # Right eye
            (landmarks[1].x * w, landmarks[1].y * h),      # Nose tip
            (landmarks[61].x * w, landmarks[61].y * h),    # Left mouth
            (landmarks[291].x * w, landmarks[291].y * h),  # Right mouth
            (landmarks[199].x * w, landmarks[199].y * h)   # Chin
        ], dtype="double")

        model_points = np.array([
            (-30.0, 0.0, -30.0),
            (30.0, 0.0, -30.0),
            (0.0, 0.0, 0.0),
            (-20.0, -40.0, -30.0),
            (20.0, -40.0, -30.0),
            (0.0, -70.0, -50.0)
        ])

        focal_length = w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, _ = cv2.solvePnP(
            model_points,
            image_points,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)

        pitch, yaw, roll = angles
        return pitch, yaw, roll
