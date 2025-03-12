import sys

from .config import config

flags = [
    "--help", "--download", "--model", "--saved-model", "--train-new-model", "--train-model",
    "--image-capture", "--run-tests"
]

tf_load = [
    "--model", "--saved-model", "--train-new-model", "--train-model"
]

for i in tf_load:
    if i in sys.argv:
        from .training.modelutils import model_detail, saved_model_detail, training
        break


def main():
    argv = sys.argv

    if len(argv) == 1:
        print("python -m src <flags>")
        print("flags: ", flags)
        return

    if "--download" in argv:
        from .utils import download_data

        dataset_url = config["dataset_url"]
        data_folder = config["data_folder"]

        download_data(dataset_url, data_folder)

    if "--model" in argv:
        model_detail()

    if "--saved-model" in argv:
        file = saved_model_detail()
        
    if "--train-new-model" in argv:
        historylist = training()

        # show_plot(historylist)

    if "--train-model" in argv:
        file = saved_model_detail()

        historylist = training(file)

        # show_plot(historylist)

    if "--image-capture" in argv:
        from .img_capture import save_archor_positive_image
        
        save_archor_positive_image()

    if "--run-test" in argv:
        from .testing import main

        main()


if __name__ == "__main__":
    main()