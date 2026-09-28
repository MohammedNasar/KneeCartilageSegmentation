"""
This script is used for predicting segmentation results using a pre-trained model.
Functions:
    entry_func(args=None): Main entry function for the prediction script. It sets up the environment,
                           loads the dataset, builds the model, evaluates the model, and writes the results.
Modules:
    os: Provides a way of using operating system dependent functionality.
    logging: Provides a way to configure and use loggers.
    yaml: Provides a way to parse YAML files.
    sys: Provides access to some variables used or maintained by the interpreter.
    torch: Provides functions for tensor computation and deep learning.
    monai.config: Provides MONAI configuration utilities.
    monai.metrics: Provides MONAI metrics utilities.
    monai.transforms: Provides MONAI transformation utilities.
    monai.data: Provides MONAI data utilities.
    models: Custom module to build the model.
    utils: Custom module with utility functions.
    evaluate: Custom module with evaluation functions.
Usage:
    Run this script directly to execute the entry_func function.
"""
import os
import logging
import os
import yaml
import sys
import torch
from monai.metrics import DiceMetric

from monai.transforms import (
    AsDiscrete
)

from models.model_builder import *
from utils.data_utils import *
from utils.visualize import *
from trainer import *
from evaluate import *


def entry_func(args=None):
    """
    Main entry point for running 3D segmentation model prediction and evaluation.
    This function performs the following steps:
    1. Parses command-line arguments and loads training parameters from a YAML file.
    2. Sets up logging and CUDA device configuration.
    3. Prepares project, model, and output directories.
    4. Splits the dataset into training, validation, and test sets based on the specified split type.
    5. Sets up data loaders for the test set.
    6. Initializes the evaluation metric (Dice coefficient).
    7. Builds and loads the trained segmentation model.
    8. Evaluates the model on the test dataset and computes the Dice score.
    9. Writes the evaluation report to a CSV file.
    10. (Optional) Plots prediction images.
    Args:
        args (list or None): List of command-line arguments to parse. If None, uses sys.argv.
    Raises:
        ValueError: If an unknown data split type is specified in the training parameters.
    """
    
    # Get and check args
    args = get_argparser().parse_args(args)
    with open(os.path.join(args.project_dir, 'train_params.yaml')) as fp:
        train_params = yaml.safe_load(fp)
        train_params["mdname"]= args.mdname

    print(train_params)

    in_channels = train_params["in_channels"]
    out_channels = train_params["out_channels"]
     
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    use_cuda = torch.cuda.is_available()
    device_ids = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
    if use_cuda and torch.cuda.device_count() > 1:
        device_ids = [int(s) for s in args.device_ids.split(',')]
        print("Let's use", device_ids, "GPUs!")
    print(f"Using device: {device_ids}")

    # Check for project dir
    project_dir = args.project_dir
    model_dir = os.path.join(project_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    output_dir = os.path.join(project_dir, "output") 
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'pred_report.csv')
    os.chdir(project_dir)
        
    # Set data paths and set up readers.
    # Split dataset into training, validation, and test sets.
    if train_params["data_split_type"] == "fixed":
        images, segs = fixed_split_dataset(train_params)
    elif train_params["data_split_type"] == "kfold":
        images, segs = kfold_split_dataset(train_params)
    else:
        raise ValueError(f"Unknown data split type: {train_params['data_split_type']}")

    _, _, test_loader = augment_list_dataset(images, segs, device_ids, train_params, train_mode=False)

    # Set evaluation metric.
    dice_metric = DiceMetric(include_background=False, reduction="mean", num_classes=out_channels)

   # Build network model.
    model = build_model(args, in_channels=in_channels, out_channels=out_channels)
    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device_ids)

    # Evaluate the best model on the validation dataset.
    post_label = AsDiscrete(to_onehot=out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)
    model.load_state_dict(torch.load(os.path.join(model_dir, train_params['pred_model_file'])))
    model.eval()
    test_dice = eval_pred_images(output_dir, model, post_label, post_pred, test_loader, 
                                 segs['test'], dice_metric, train_params)
    write_report(report_path, train_params, images['test'], test_dice)

    # Plotting.
    # display_pred_images(model, test_loader, train_params)
    
if __name__ == "__main__":

    entry_func()
