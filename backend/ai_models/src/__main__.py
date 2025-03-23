import sys, os

from .config import config

flags = [
    "--help", "--download", "--model", "--saved-model", "--train-new-model", "--train-model",
    "--image-capture", "--run-test"
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

    make_imp_file()

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
        from .testing import main as func1

        func1()

def make_imp_file():
    DATA_PATH = config["data_folder"]
    POS_PATH = config["POS_PATH"]
    NEG_PATH = config["NEG_PATH"]
    ANC_PATH = config["ANC_PATH"]
    VAL_PATH = config["VAL_PATH"]
    SAVED_MODEL_PATH = config["save_model_folder"]

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if not os.path.exists(POS_PATH):
        os.makedirs(POS_PATH)
    if not os.path.exists(NEG_PATH):
        os.makedirs(NEG_PATH)
    if not os.path.exists(ANC_PATH):
        os.makedirs(ANC_PATH)
    if not os.path.exists(VAL_PATH):
        os.makedirs(VAL_PATH)
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()