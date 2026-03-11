from ai_inference.anti_cheating_ai import AntiCheatingAI
import cv2

model = AntiCheatingAI()

def run_inference(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": "Could not read image"}
    
    result = model.predict(frame)
    return result
