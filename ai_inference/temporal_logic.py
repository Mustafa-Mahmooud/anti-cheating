import time


class TemporalLogic:
    def __init__(self):
        self.no_face_start = None
        self.look_away_start = None

    def check_no_face(self, face_count, threshold_seconds=3):
        if face_count == 0:
            if self.no_face_start is None:
                self.no_face_start = time.time()
            elif time.time() - self.no_face_start >= threshold_seconds:
                return True
        else:
            self.no_face_start = None
        return False
