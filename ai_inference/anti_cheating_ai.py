from ai_inference.face_detector import FaceDetector
from ai_inference.head_pose import HeadPoseEstimator
from ai_inference.phone_detector import PhoneDetector
from ai_inference.eye_gaze import EyeGazeEstimator
from ai_inference.temporal_logic import TemporalLogic


class AntiCheatingAI:
    def __init__(self):
        self.face = FaceDetector()
        self.pose = HeadPoseEstimator()
        self.phone = PhoneDetector()
        self.logic = TemporalLogic()
        self.eye_gaze = EyeGazeEstimator()

    def predict(self, frame):
        faces = self.face.detect(frame)

        if faces > 1:
            return {"cheating": True, "reason": "multiple_faces"}

        if self.logic.check_no_face(faces):
            return {"cheating": True, "reason": "no_face"}

        angles = self.pose.estimate(frame)
        gaze = self.eye_gaze.estimate(frame)
        if gaze in ["left", "right"]:
            return {"cheating": True, "reason": "eye_gaze_away"}

        if angles:
            pitch, yaw, _ = angles
            if abs(yaw) > 25 or pitch > 20:
                return {"cheating": True, "reason": "looking_away"}

        if self.phone.detect(frame):
            return {"cheating": True, "reason": "phone_detected"}

        return {"cheating": False}
