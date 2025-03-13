from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
import cv2
import numpy as np

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
    print(img_array)
    # Pass image to the model for prediction
    # prediction_result = model_manager.local_prediction(img_array)

    return {"filename": file.filename,}
