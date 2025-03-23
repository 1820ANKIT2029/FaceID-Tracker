from os import path

from ..training.modelutils import load_saved_model_optimal, load_saved_model
from ..utils import get_image_names_and_img,get_image_names_and_paths, decode_img

from ..config import config

class ModelManager:
    def __init__(self, model_path: str, global_search: bool = False):
        self.model_path = model_path
        self.global_search = global_search

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
        data = get_image_names_and_img(get_image_names_and_paths(path))

        return self.model.get_embedding_vector(data)

    def _load_model_by_path(self, modelpath):
        if not modelpath:
            raise Exception("model path not provided")
        
        if not path.exists(modelpath):
            raise FileNotFoundError(f"No model at {modelpath}")
        
        try:
            return load_saved_model_optimal(modelpath)
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
        
    def local_prediction(self, img):
        return self.model.custom_prediction(decode_img(img), self.person_embedding_vector_local)

    def global_prediction(self, img):
        return self.model.custom_prediction(decode_img(img), self.person_embedding_vector_global)

    def retrain(self, archor_img, positive_img):
        pass
        

