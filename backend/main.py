from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import List
import cv2
import numpy as np
from ai_models.src.inference.Manager import ModelManager
from ai_models.src.config import config
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Instantiated ModelManager 
model_manager = ModelManager(model_path=config["Prediction_Model"], global_search=False)

def read_image_as_numpy(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    np_array = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    prediction_results = []
    for file in files:
        try:
            file.file.seek(0)
            save_path = os.path.join("ai_models/src/data/validation", f"test.jpg")
            img_array = read_image_as_numpy(file)
            prediction = model_manager.local_prediction(img_array)
            print(prediction)
            person_name = prediction[0][1].split(".")[0]
            probability = prediction[0][0]
            if( prediction[0][0] < 0.89 ):
                person_name = "unknown"

            prediction_results.append({
                "name": person_name,
                "probability": probability
            })
        except Exception as e:
            prediction_results.append({"error": str(e)})
    
    return {"prediction_result": prediction_results}

@app.post("/register/")
async def register_criminal(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    try:
        img_array = read_image_as_numpy(file)

        save_path = os.path.join("ai_models/src/data/validation", f"{name}.jpg")
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)

        return {"message": f"Image saved as {name}.jpg in validation folder."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

