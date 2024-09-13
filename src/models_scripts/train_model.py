"""
* Modify the config file to change the parameters.
* Trained model with weights already loaded in .pt file:
  root -> data -> train_results -> train_result<X> -> weights
* To modify dataset:
    If you have preprocessed images with annotations already,
    you can add preprocessed images with annotations to:
    root -> data -> processed
  1) If you have no where to start, sign up for a free
     Roboflow account: https://roboflow.com/ -> Get Started
  2) Create a workspace and project
  3) Clone all the images from the project to yours:
     **insert link**
  5) To add other images, you must annotate the Poles
  6) Add preprocessing and augmentations, follow directions
     from to replicate steps:
     root -> data -> processed -> README.dataset.txt
* To run:
(Save necessary files)

>>> cd src/models_scripts
>>> python train_model.py

* Monitor performance in the terminal and view results/weights in:
  data -> train_results -> train_result<X+1>
* Results on validation will be in:
  data -> train_results -> train_result<X+1>2
"""

from ultralytics import YOLO
import yaml
import os


def run(config):
    """
    Instatiates the model, runs the model on the specified data with
    the specified parameters (editable in config.yaml), and evaluates
    the trained model on the desired split.

    Args:
        config: yaml file where parameters are changed

    Returns:
        None. The model is trained and run on desired split
    """
    # load model to use (YOLOv8 already inputted)
    model = YOLO(config["model"])

    # Train the model
    model.train(
        # Configuration details
        data=config["data"],
        plots=config["plots"],
        amp=config["amp"],
        # Hyperparameters
        epochs=config["epochs"],
        patience=config["patience"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        fraction=config["fraction"],
        save_period=config["save_period"],
        device=config["device"],
        # Augmentation parameters
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        scale=config["scale"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        # Output folder specifications (data -> train_results)
        project=config["train_project"],
        name=config["train_name"],
    )

    # Test on desired split
    model.val(
        # Configurations
        data=config["data"],
        split=config["split"],
        plots=config["plots"],
        # Hyperparameters
        iou=config["iou"],
        conf=config["conf"],
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.loader.SafeLoader)
    run(config)
