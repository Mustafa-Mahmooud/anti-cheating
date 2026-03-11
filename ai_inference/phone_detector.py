from ultralytics import YOLO


class PhoneDetector:
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class id for cell phone = 67
                if cls_id == 67 and conf > 0.7:
                    return True
        return False
