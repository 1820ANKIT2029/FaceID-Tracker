from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
import cv2
import numpy as np
from ai_models.src.inference.Manager import ModelManager
from ai_models.src.config import config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
model_manager = ModelManager(model_path=config["save_model_folder"], global_search=False)


app = FastAPI()

def read_image_as_numpy(file: UploadFile) -> np.ndarray:
    """Convert an uploaded image file to a NumPy array."""
    contents = file.file.read()
    np_array = np.frombuffer(contents, np.uint8) 
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img

@app.post("/predict/")  
async def predict(file: UploadFile = File(...)):
    img_array = read_image_as_numpy(file) 
    prediction_result = model_manager.local_prediction(img_array)

    return {
        "prediction_result": prediction_result
    }
