from os import path, remove
import tempfile
import torch
from PIL import Image
import cv2

from ..training.modelutils import load_saved_model_optimal, load_saved_model
from ..utils import get_image_names_and_img,get_image_names_and_paths, decode_img
from .model import BiSeNet
from .seg_model import process_image_with_model

from ..config import config

def delete_temp_image(img_path):
    if path.exists(img_path):
        remove(img_path)

class ModelManager:
    def __init__(self, model_path: str, global_search: bool = False):
        self.model_path = model_path
        self.global_search = global_search

        self.seg_model = None
        self.model = None
        self.person_embedding_vector_local = None  # image in validation directory
        self.person_embedding_vector_global = None # image in negative directory


        self.model = self._load_model_by_path(model_path)

        if self.global_search:
            self._load_person_embedding_vector_global()

        self._load_person_embedding_vector_local()

    def _load_person_embedding_vector_local(self):
        val_path = config["VAL_PATH"]

        self.person_embedding_vector_local = self._load_person_embedding_vector(val_path)
         

    def _load_person_embedding_vector_global(self):
        neg_path = config["NEG_PATH"]

        self.person_embedding_vector_local = self._load_person_embedding_vector(neg_path)

    def _load_person_embedding_vector(self, path):
        temp = get_image_names_and_paths(path)
        for _, path in temp:
            img = process_image_with_model(path, self.seg_model)
        data = get_image_names_and_img(temp)

        return self.model.get_embedding_vector(data)

    def _load_model_by_path(self, modelpath):
        if not modelpath:
            raise Exception("model path not provided")
        
        if not path.exists(modelpath):
            raise FileNotFoundError(f"No model at {modelpath}")
        
        try:
            self.seg_model = BiSeNet(n_classes=19)
            model_path = path.join(config["save_model_folder"], '79999_iter.pth')
            self.seg_model.load_state_dict(torch.load(model_path , map_location='cuda' if torch.cuda.is_available() else 'cpu'))
            self.seg_model.eval()
            self.seg_model = self.seg_model.cuda() if torch.cuda.is_available() else self.seg_model

            return load_saved_model_optimal(modelpath)
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
        
    def local_prediction(self, img):
        img = self.seg_predict(img)

        return self.model.custom_prediction(decode_img(img), self.person_embedding_vector_local)

    def global_prediction(self, img):
        img = self.seg_predict(img)

        return self.model.custom_prediction(decode_img(img), self.person_embedding_vector_global)
    
    def seg_predict(self, img):
        temp_file = tempfile.NamedTemporaryFile(suffix='.png')
        temp_file_path = temp_file.name
        temp_file.close()
        cv2.imwrite(temp_file_path, img)

        img = process_image_with_model(temp_file_path, self.seg_model)

        delete_temp_image(temp_file_path)
        return img

    def retrain(self, archor_img, positive_img):
        pass
        

