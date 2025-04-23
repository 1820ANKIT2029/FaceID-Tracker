import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch

def extract_face_region(parsing, image):
    face_parts = list(range(1, 14))
    face_mask = np.isin(parsing, face_parts).astype(np.uint8) * 255
    contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        face_crop = image[y:y+h, x:x+w]
        return face_crop, (x, y, w, h)
    return None, None

def fit_face_in_canvas(face_crop, canvas_size=(256, 256)):
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    face_resized = cv2.resize(face_crop, (256, 256))
    x_offset = (canvas_size[0] - 256) // 2
    y_offset = (canvas_size[1] - 256) // 2
    canvas[y_offset:y_offset+256, x_offset:x_offset+256] = face_resized
    return canvas

def get_parsing_with_model(image_path, model):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_resized = image.resize((512, 512))
    with torch.no_grad():
        tensor = transform(img_resized).unsqueeze(0)
        tensor = tensor.cuda() if torch.cuda.is_available() else tensor
        out = model(tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return parsing, np.array(img_resized)

def process_image_with_model(image_path, model):
    try:
        parsing, image_np = get_parsing_with_model(image_path, model)
        face_crop, _ = extract_face_region(parsing, image_np)
        if face_crop is not None:
            final_image = fit_face_in_canvas(face_crop)
            final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)  # Convert to BGR before saving
            cv2.imwrite(image_path, final_image_bgr)
            return final_image_bgr
    except Exception as e:
        print(f"Failed {image_path}: {e}")
        return None

