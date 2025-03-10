import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.face_recognition import process_image

router = APIRouter()

class ImageData(BaseModel):
    image_base64: str

@router.post("/verify-face")
async def verify_face(data: ImageData):
    try:
        # decode base64 to image
        img_data = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

        if img is None:
            return HTTPException(status_code=400, detail="Invalid image format")
        

        embedding = process_image(img)
        return {"embedding": embedding.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))