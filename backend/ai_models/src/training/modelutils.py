from tensorflow.keras.models import load_model
from tensorflow.random import normal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Precision


import os
from time import strftime

from .model import SiameseModel, custom_objects, SiameseModel
from ..utils import load_data_generator, load_data
from ..config import config

def training(saved_model_path=None):
    model = None

    # checkpoint_callback = ModelCheckpoint(
    #     'best_weights.keras',
    #     monitor = 'val_precision',
    #     mode='max',
    #     verbose=1,
    #     save_best_only = True
    # )

    if saved_model_path:
        print("Loading saved model")

        # Load the model with custom_objects
        model = load_model(saved_model_path, custom_objects=custom_objects)
    else:
        model = SiameseModel()
        dummy_input = normal((1, config["IM_SIZE"], config["IM_SIZE"], 3))
        model((dummy_input, dummy_input))  # triggers model build

        loss_function = BinaryCrossentropy()
        metrics = [BinaryAccuracy(name="accuracy"), Precision(name="precision")]

        model.compile(
            optimizer = Adam(learning_rate = config["LEARNING_RATE"]),
            loss = loss_function,
            metrics = metrics,
        )

    model.summary()

    data_gen = load_data_generator()
    combined_history = {}


    for train_data, validation_data, test_data in data_gen:

        history = model.fit(
            train_data,
            validation_data = validation_data,
            epochs = config["EPOCHS"],
            verbose = 1
        )

        for key, values in history.history.items():
            if key not in combined_history:
                combined_history[key] = []
            combined_history[key].extend(values)

        evaluation_metrics = model.evaluate(test_data)
        print("Test Evaluation:", dict(zip(model.metrics_names, evaluation_metrics)))

    if saved_model_path:
        model.save(saved_model_path)
    else:
        path = os.path.join(config["save_model_folder"], f'model-{strftime("%Y%m%d-%H%M%S")}.keras')
        model.save(path)

    return combined_history

def model_detail():
    model = SiameseModel()
    dummy_input = normal((1, config["IM_SIZE"], config["IM_SIZE"], 3))
    model((dummy_input, dummy_input))  # triggers model build
    model.summary()

def saved_model_detail():
    saved_models_folder = config["save_model_folder"]
    saved_model_path = None
    try:
        entries = os.listdir(saved_models_folder)
        for i, file in enumerate(entries):
            print(f"{i}. {file}")

        num = int(input("Enter file Number: "))

        saved_model_path = os.path.join(saved_models_folder, entries[num])

        model = load_model(saved_model_path, custom_objects=custom_objects)

        model.summary()

        return saved_model_path
    except FileNotFoundError:
        print(f"Error: Directory not found: {saved_models_folder}")
    except NotADirectoryError:
        print(f"Error: Not a directory: {saved_models_folder}")

def load_saved_model_optimal(path):
    model = SiameseModel()
    dummy_input = normal((1, config["IM_SIZE"], config["IM_SIZE"], 3))
    model((dummy_input, dummy_input))
    # model.load_weights(path, custom_objects=custom_objects)
    model.load_weights(path)
    
    return model

def load_saved_model(path):
    model = load_model(path, custom_objects=custom_objects)

    return model
    