import time
import os

from .config import config

def main():
    test5()

def test5():
    from .utils import load_data_generator

    print("hii")

    gen_data = load_data_generator()

    for train_data, validation_data, test_data in gen_data:
        print(train_data, validation_data, test_data)
        break

def test4():
    from .interence.Manager import ModelManager
    from .img_capture import get_video_frame

    path = os.path.join(config["save_model_folder"], "model-20250311-190130.keras")
    m = ModelManager(path, global_search=True)

    frame = get_video_frame()

    pre = m.local_prediction(frame)

    print(pre)


def test3():
    from .utils import get_image_names_and_paths, get_image_names_and_img

    val_path = config["VAL_PATH"]
    print(get_image_names_and_img(get_image_names_and_paths(val_path)))

def test2():
    import tensorflow as tf

    path = os.path.join(config["save_model_folder"], "model-20250311-142854.keras")
    model = tf.keras.models.load_model(path)

    model.summary()

def test1():
    gen_squ = gen_no()

    for i in gen_squ:
        print("main: ", i)

        time.sleep(2)


def gen_no():
    list = [1, 2, 3, 4, 5]
    for i in range(len(list)):
        y = list[i] * list[i]

        print("gen sq: ", y)
        yield y

if __name__ == "__main__":
    main()