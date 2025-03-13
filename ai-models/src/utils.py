import os
import shutil
import requests
import zipfile

import tensorflow as tf

from .config import config

IM_SIZE = config["IM_SIZE"]

def decode_img(img):
    # img = tf.image.decode_image(img)         # Decode to tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (IM_SIZE,IM_SIZE))
    # Scale image to be between 0 and 1
    img = img / 255.0

    return img

def get_image_names_and_paths(folder_path, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
    """
    Returns a list of tuples (image_name, full_image_path) for all images in the folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
        extensions (tuple): Tuple of image file extensions to look for.
    
    Returns:
        List[Tuple[str, str]]: List of (image_name, full_path) pairs.
    """
    image_data = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(extensions):
            full_path = os.path.join(folder_path, file)
            image_data.append((file, full_path))

    return image_data

def get_image_names_and_img(image_data):
    return list(map(preprocess_name_and_img, image_data))


def load_data(raw_data):
    BATCH_SIZE = config["BATCH_SIZE"]
    TRAINING_RATIO = config["TRAINING_RATIO"]
    VALIDATION_RATIO = config["VALIDATION_RATIO"]
    TESTING_RATIO = config["TESTING_RATIO"]

    DATA_SIZE = raw_data.cardinality().numpy()

    TRAINING_SIZE = round(DATA_SIZE*TRAINING_RATIO)
    VALIDATION_SIZE = round(DATA_SIZE*VALIDATION_RATIO)
    TESTING_SIZE = round(DATA_SIZE*TESTING_RATIO)

    data = raw_data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    # Training partition
    train_data = data.take(TRAINING_SIZE)
    train_data = train_data.batch(BATCH_SIZE)
    train_data = train_data.prefetch(8)

    # Validation partition
    validation_data = data.skip(TRAINING_SIZE)
    validation_data = validation_data.take(VALIDATION_SIZE)
    validation_data = validation_data.batch(BATCH_SIZE)
    validation_data = validation_data.prefetch(8)

    # Testing partition
    test_data = data.skip(TRAINING_SIZE + VALIDATION_SIZE)
    test_data = test_data.take(VALIDATION_SIZE)
    test_data = test_data.batch(BATCH_SIZE)
    test_data = test_data.prefetch(8)
    
    return train_data, validation_data, test_data

def preprocess_twin(input_img, validation_img, label):
    return ((preprocess(input_img), preprocess(validation_img)), label)

def preprocess_name_and_img(data):
    return (data[0], preprocess(data[1]))

def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (IM_SIZE,IM_SIZE))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img

def load_data_raw():
    """
    Data = [
        (filename1, filename2, 1 or 0),
        (filename1, filename2, 1 or 0), 
        ...
    ]
    """
    POS_PATH = config["POS_PATH"]
    NEG_PATH = config["NEG_PATH"]
    ANC_PATH = config["ANC_PATH"]

    POS_PATH_child_dir = None
    ANC_PATH_child_dir = None

    Data = None             # main data that will return

    try:
        POS_PATH_child_dir = os.listdir(POS_PATH)
        ANC_PATH_child_dir = os.listdir(ANC_PATH)
        for p in POS_PATH_child_dir:
            if p.startswith("."):
                POS_PATH_child_dir.remove(p)
        for p in ANC_PATH_child_dir:
            if p.startswith("."):
                ANC_PATH_child_dir.remove(p)
    except OSError as e:
        print(f"An error occurred: {e}")

    # negative raw data
    NEG_PATH_P = os.path.join(NEG_PATH, "")
    negative = tf.data.Dataset.list_files(NEG_PATH_P + '*.jpg')
    negative_size = negative.cardinality().numpy()

    archor_positive = []

    for p in POS_PATH_child_dir:
        try:
            POS_PATH_P = os.path.join(POS_PATH, p, "")
            ANC_PATH_P = os.path.join(ANC_PATH, p, "")

            anchor = tf.data.Dataset.list_files(ANC_PATH_P + '*.jpg')
            positive = tf.data.Dataset.list_files(POS_PATH_P + '*.jpg')

            archor_positive.append((anchor, positive))
        except Exception as e:
            print(f"Error processing class {p}: {e}")
            continue

    Data = None

    for anchor, positive in archor_positive:
        try:
            anchor = anchor.shuffle(buffer_size=10000)
            positive = positive.shuffle(buffer_size=10000)

            anchor_size = anchor.cardinality().numpy()
            positive_size = positive.cardinality().numpy()

            shuffled_negative = negative.shuffle(buffer_size=10000).take(anchor_size)

            positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(anchor_size))))
            shuffled_negative = tf.data.Dataset.zip((anchor, shuffled_negative, tf.data.Dataset.from_tensor_slices(tf.zeros(anchor_size))))

            if Data is None:
                Data = positives.concatenate(shuffled_negative)
            else:
                positives = positives.concatenate(shuffled_negative)
                Data = positives.concatenate(shuffled_negative)
        except GeneratorExit:
            return
        except Exception as e:
            print(f"Error processing class {p}: {e}")
            continue

    return load_data(Data)


def load_data_generator():
    """
    Data = [
        (filename1, filename2, 1 or 0),
        (filename1, filename2, 1 or 0),
        ...
    ]
    """
    POS_PATH = config["POS_PATH"]
    NEG_PATH = config["NEG_PATH"]
    ANC_PATH = config["ANC_PATH"]

    POS_PATH_child_dir = None
    ANC_PATH_child_dir = None

    Data = None             # main data that will return

    try:
        POS_PATH_child_dir = os.listdir(POS_PATH)
        ANC_PATH_child_dir = os.listdir(ANC_PATH)
        for p in POS_PATH_child_dir:
            if p.startswith("."):
                POS_PATH_child_dir.remove(p)
        for p in ANC_PATH_child_dir:
            if p.startswith("."):
                ANC_PATH_child_dir.remove(p)
    except OSError as e:
        print(f"An error occurred: {e}")

    # negative raw data
    NEG_PATH_P = os.path.join(NEG_PATH, "")
    negative = tf.data.Dataset.list_files(NEG_PATH_P + '*.jpg')
    negative_size = negative.cardinality().numpy()

    archor_positive = []

    for p in POS_PATH_child_dir:
        try:
            POS_PATH_P = os.path.join(POS_PATH, p, "")
            ANC_PATH_P = os.path.join(ANC_PATH, p, "")

            anchor = tf.data.Dataset.list_files(ANC_PATH_P + '*.jpg')
            positive = tf.data.Dataset.list_files(POS_PATH_P + '*.jpg')

            archor_positive.append((anchor, positive))
        except Exception as e:
            print(f"Error processing class {p}: {e}")
            continue

    for _ in range(config["GENERATOR_ITER"]):
        Data = None

        for anchor, positive in archor_positive:
            try:
                anchor = anchor.shuffle(buffer_size=10000)
                positive = positive.shuffle(buffer_size=10000)

                anchor_size = anchor.cardinality().numpy()
                positive_size = positive.cardinality().numpy()

                shuffled_negative = negative.shuffle(buffer_size=10000).take(anchor_size)

                positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(anchor_size))))
                shuffled_negative = tf.data.Dataset.zip((anchor, shuffled_negative, tf.data.Dataset.from_tensor_slices(tf.zeros(anchor_size))))

                if Data is None:
                    Data = positives.concatenate(shuffled_negative)
                else:
                    positives = positives.concatenate(shuffled_negative)
                    Data = positives.concatenate(shuffled_negative)
            except GeneratorExit:
                return
            except Exception as e:
                print(f"Error processing class {p}: {e}")
                continue

        yield load_data(Data)

def download_data(dataset_url, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    zip_path = os.path.join(target_dir, "data.zip")

    path = os.path.join(target_dir, 'lfw')
    path_image_dirs = os.path.join(target_dir, 'lfw/lfw-deepfunneled/lfw-deepfunneled/')

    if os.path.exists(path):
        print("Data exist...")
        return
    
    print("Downloading Dataset...")

    response = requests.get(dataset_url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path)

        # Remove the ZIP file after extraction
        os.remove(zip_path)

        print(f"Data downloaded and extracted to '{target_dir}'")
    else:
        print("Failed to download the file")

    POS_PATH = config["POS_PATH"]
    NEG_PATH = config["NEG_PATH"]
    ANC_PATH = config["ANC_PATH"]
    VAL_PATH = config["VAL_PATH"]

    if not os.path.exists(POS_PATH):
        os.makedirs(POS_PATH)
    if not os.path.exists(NEG_PATH):
        os.makedirs(NEG_PATH)
    if not os.path.exists(ANC_PATH):
        os.makedirs(ANC_PATH)
    if not os.path.exists(VAL_PATH):
        os.makedirs(VAL_PATH)

    for directory in os.listdir(path_image_dirs):
        for file in os.listdir(os.path.join(path_image_dirs, directory)):
            EX_PATH = os.path.join(path_image_dirs, directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)

    try:
        shutil.rmtree(path)
        print(f"Directory '{path}' and its contents removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{path}' not found.")
    except OSError as e:
        print(f"Error: {e}")
    print("Download data done!!")



def main():
    # dataset_url = "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/lfw-dataset"
    # data_folder = config["data_folder"]

    # download_data(dataset_url, data_folder)

    data = load_data_raw()
    for i in data.as_numpy_iterator():
        print(i)

if __name__ == "__main__":
    main()