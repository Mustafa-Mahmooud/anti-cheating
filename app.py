from fastapi import FastAPI, UploadFile, File
import shutil
import os
from ai_inference.inference import run_inference  

app = FastAPI(title="Anti-Cheating AI Service")

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "AI service is running"}


@app.post("/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_inference(file_path)

    os.remove(file_path)

    return {"analysis": result}