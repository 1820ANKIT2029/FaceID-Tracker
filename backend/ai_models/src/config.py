import os
import random

BASE_PATH = os.path.dirname(__file__)
# BASE_PATH = os.getcwd()    # for colab

config = {
    "data_folder": os.path.join(BASE_PATH, "data"),
    "POS_PATH": os.path.join(BASE_PATH, 'data' , 'positive'),
    "NEG_PATH": os.path.join(BASE_PATH, 'data', 'negative'),
    "ANC_PATH": os.path.join(BASE_PATH, 'data', 'anchor'),
    "VAL_PATH": os.path.join(BASE_PATH, 'data', 'validation'),
    "save_model_folder": os.path.join(BASE_PATH, "saved_models"),
    "Prediction_Model": os.path.join(BASE_PATH, "saved_models","model.keras"),

    # LFW Dataset
    "dataset_url": "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset",
    # VGGFace2
    "dataset_url2": "https://www.kaggle.com/api/v1/datasets/download/hearfool/vggface2",
    
    # DigiFace Dataset
    "P1": "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_0-1999_72_imgs.zip",
    "P2": "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_2000-3999_72_imgs.zip",
    "P3": "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_4000-5999_72_imgs.zip",
    "P4": "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_6000-7999_72_imgs.zip",
    "P5": "https://facesyntheticspubwedata.z6.web.core.windows.net/wacv-2023/subjects_8000-9999_72_imgs.zip",

    "combination_data_folder": [(i, j) for i in ["P1", "P2", "P3", "P4", "P5"] for j in ["P1", "P2", "P3", "P4", "P5"] if i != j],

    # Model HyperParameters
    "IM_SIZE": 100,
    "REGULARIZATION_RATE": 0.01,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 200,
    "BATCH_SIZE": 32,
    "TRAINING_RATIO": 0.8,
    "VALIDATION_RATIO": 0.1,
    "TESTING_RATIO": 0.1,
    "GENERATOR_ITER": 3,
    "DigiFace_gen_no": 5
}

random.shuffle(config["combination_data_folder"])
# print(config["combination_data_folder"])