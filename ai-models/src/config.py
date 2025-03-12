import os

BASE_PATH = os.path.dirname(__file__)

config = {
    "data_folder": os.path.join(BASE_PATH, "data"),
    "POS_PATH": os.path.join(BASE_PATH, 'data' , 'positive'),
    "NEG_PATH": os.path.join(BASE_PATH, 'data', 'negative'),
    "ANC_PATH": os.path.join(BASE_PATH, 'data', 'anchor'),
    "VAL_PATH": os.path.join(BASE_PATH, 'data', 'validation'),

    "dataset_url": "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset",
    "save_model_folder": os.path.join(BASE_PATH, "saved_models"),


    "IM_SIZE": 100,
    "LEARNING_RATE": 0.01,
    "EPOCHS": 1,
    "BATCH_SIZE": 32,
    "TRAINING_RATIO": 0.8,
    "VALIDATION_RATIO": 0.1,
    "TESTING_RATIO": 0.1,
    "GENERATOR_ITER": 1,
}