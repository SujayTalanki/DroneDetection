"""
* Modify the config file to change the parameters.
* Edit the weights_path and image_path in config file
* Adjust iou and conf hyperparameters for optimal
  bounding drawing/frequency

To run:
(Save necessary files)

>>> cd src/models_scripts
>>> python test_model.py

* Output with bounding boxes in:
  data -> test_results -> train_result<X>

* Look at estimated pole length calculation in terminal
"""

from ultralytics import YOLO
import yaml
import os


def test(config):
    """
    Generates bounding boxes for object given an image or video.
    NOTE: If you want to only run inference on a singular image,
    change the folder_or_file config variable to "file".
    If not, the model will run inference on every file in the
    folder_path specified in the config file.

    Args:
        config: file with editable parameters and config variables
    """

    def predict(model, source):
        """
        Runs inference on the source file with the model.

        Args:
            model: YOLOv8 model trained on aerial images
            source: filepath to a video or image
        """
        model.predict(
            # Configuration parameters
            source=source,
            save=config["save"],
            imgsz=config["imgsz"],
            project=config["test_project"],
            name=config["test_name"],
            # Hyperparameters for experimentation/fine-tuning
            iou=config["iou"],
            conf=config["conf"],
            show=config["show"],
        )

    # Loads in the weights for model you want to test
    model = YOLO(config["weights_path"])

    # Runs inference on all of the files in a specified folder
    if config["folder_or_file"] == "folder":
        for filename in os.listdir(config["folder_path"]):
            file = os.path.join(config["folder_path"], filename)
            predict(model, file)

    # Runs inference on singular example
    else:
        predict(model, config["file_path"])


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.loader.SafeLoader)
    test(config)
