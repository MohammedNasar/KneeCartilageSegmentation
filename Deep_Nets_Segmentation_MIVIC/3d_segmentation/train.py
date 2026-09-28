"""
This function implements the training loop for a deep learning model designed for segmentation tasks. It supports mixed 
precision training, periodic validation, and saving the best model based on validation performance.
This script provides a complete pipeline for training a deep learning model for segmentation tasks using PyTorch and MONAI. 
It includes functions for training, validation, and evaluation, as well as utilities for data augmentation, model building, 
and learning rate optimization.
Key Functions:
---------------
1. `entry_func`: Entry point for the script, handling argument parsing, data preparation, model setup, and training.
Features:
---------
- Mixed precision training using `torch.cuda.amp.GradScaler`.
- Multi-GPU support using `torch.nn.DataParallel`.
- Deterministic training for reproducibility.
- Learning rate optimization using MONAI's `LearningRateFinder`.
- TensorBoard integration for logging and visualization.
- Automatic saving of the best model based on validation Dice score.
Dependencies:
-------------
- PyTorch
- MONAI
- tqdm
- matplotlib
- PyYAML
- TensorBoard
Usage:
------
Run the script with appropriate command-line arguments to train a segmentation model. The training parameters are loaded 
from a YAML file (`train_params.yaml`), and the script supports various configurations such as multi-GPU training, 
learning rate optimization, and deterministic training.
--------
- Trained model saved to the specified project directory.
    and modules (`models`, `utils`, `evaluate`) in the project directory.
- Validation metrics and training loss logged to TensorBoard.
- Visualizations of predictions and training curves saved to the project directory.
------
- Ensure that the `train_params.yaml` file is correctly configured with the required training parameters.
- The script assumes the presence of utility functions (`load_dataset_kfolds`, `augment_list_dataset`, `build_model`, etc.) 
"""

import os
import matplotlib.pyplot as plt

import logging
import os
import yaml
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

from monai.metrics import DiceMetric
from monai.optimizers.lr_finder import LearningRateFinder
from monai.transforms import (
    AsDiscrete
)

from models.model_builder import *
from utils.data_utils import *
from utils.visualize import *
from trainer import *
from evaluate import *


def freeze_encoder(model):
    for name, param in model.named_parameters():
        if "encoder" in name:  
            param.requires_grad = False
        else:
            param.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
        

def entry_func(args=None):
    """
    Entry point function for training a deep learning model for segmentation.
    Args:
        args (Namespace, optional): Command-line arguments parsed by argparse. Defaults to None.
    This function performs the following steps:
    1. Parses command-line arguments and loads training parameters from a YAML file.
    2. Configures logging and sets up the computing device (CPU or GPU).
    3. Prepares the project directory and output paths.
    4. Loads the dataset and applies data augmentation for training and validation.
    5. Builds the neural network model and moves it to the appropriate device(s).
    6. Configures the loss function, optimizer, learning rate scheduler, and evaluation metrics.
    7. Optionally performs learning rate optimization using a learning rate finder.
    8. Trains the model using a training loop until the maximum number of iterations is reached.
    9. Saves the best model based on validation performance and evaluates it on the validation dataset.
    10. Writes a training report and generates visualizations for TensorBoard.
    Notes:
        - Mixed precision training is supported using `torch.cuda.amp.GradScaler`.
        - Multi-GPU training is supported using `torch.nn.DataParallel`.
        - Deterministic training can be enabled using the `--deterministic` flag.
        - Learning rate optimization can be enabled using the `--optimize_lr` flag.
    Outputs:
        - Trained model saved to the project directory.
        - Training report saved as a CSV file.
        - TensorBoard visualizations saved to the "figures" directory.
    Raises:
        FileNotFoundError: If the `train_params.yaml` file is not found.
        ValueError: If invalid arguments or parameters are provided.
    """

    # Get and check args and training parameters.
    args = get_argparser().parse_args(args)
    with open('train_params.yaml') as fp:
        train_params = yaml.safe_load(fp)
        train_params["mdname"]= args.mdname
    print(train_params)

    in_channels = train_params["in_channels"]
    out_channels = train_params["out_channels"]
    learning_rate = train_params["learning_rate"]

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    use_cuda = torch.cuda.is_available()
    device_ids = torch.device(torch.cuda.current_device() if use_cuda else "cpu")
    # if use_cuda and torch.cuda.device_count() > 1:
    #     device_ids = [f"cuda:{s}" for s in args.device_ids.split(',')]
    #     print("Let's use", device_ids, "GPUs!")
    # print(f"Using device: {device_ids}")
 
    # Check for project dir
    project_dir = args.project_dir
    model_dir = os.path.join(project_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    output_dir = os.path.join(project_dir, "output") 
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.csv")
    os.chdir(project_dir)
        
    # Set data paths and set up readers.
    # Split dataset into training, validation, and test sets.
    if train_params["data_split_type"] == "fixed":
        images, segs = fixed_split_dataset(train_params)
    elif train_params["data_split_type"] == "kfold":
        images, segs = kfold_split_dataset(train_params)
    else:
        raise ValueError(f"Unknown data split type: {train_params['data_split_type']}")

    # Set up data augmentation for training and validation datasets.
    train_loader, val_loader, _ = augment_list_dataset(images, segs, device_ids, 
                                                       train_params)

    # Build network model.
    model = build_model(args, in_channels=in_channels, out_channels=out_channels)
    if args.parallel and use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs!")
    else:
        print(f"Using device: {device_ids}")
    model.to(device_ids)
    
    # Train network.
    # Setting torch.backends.cudnn.benchmark = True can lead to non-deterministic results.
    # Use --deterministic flag to disable it for reproducibility.
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
    else:   
        torch.backends.cudnn.benchmark = True

    # Set loss function, optimizer, post-processing and evaluation metric.
    loss_function = set_loss_function(train_params["loss_name"])
    
    # Set optimizer and learning rate scheduler.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Set GradScaler for mixed precision training.
    scaler = torch.cuda.amp.GradScaler()

    # Set evaluation metric.
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Set up learning rate finder if specified.
    if args.optimize_lr:
        # Set up learning rate finder.
        lr_finder = LearningRateFinder(
            model = model,
            optimizer=optimizer,
            criterion=loss_function
        )
        lr_finder.range_test(train_loader,
                             end_lr=0.1, 
                             num_iter=500, 
                             step_mode="linear")
        lr_finder.plot()
        opt_learning_rate, opt_loss = lr_finder.get_steepest_gradient()
        print(f"Optimal learning rate: {opt_learning_rate}")
        print(f"Optimal loss: {opt_loss}")
        # Set the learning rate to the optimal value.
        learning_rate = opt_learning_rate
        train_params["learning_rate"] = opt_learning_rate
        optimizer.param_groups[0]['lr'] = opt_learning_rate
        # Set up TensorBoard writer.
        writer = SummaryWriter(log_dir=os.path.join(project_dir, "figures"))
        # Add model graph to TensorBoard.
        # Note: The model should be in eval mode for this to work properly.
        writer.add_figure('lr_plot', plt.gcf())
        writer.close()
    else:   
        # Set the learning rate to the specified value.
        optimizer.param_groups[0]['lr'] = learning_rate
        
    print(f"Using learning rate: {learning_rate}")
    


    # Set up early stopping.
    early_stopper = EarlyStopper(patience=train_params["early_stopping_patience"],
                                 min_delta=train_params["early_stopping_min_delta"])
    
    # Set up training loop.
    writer = SummaryWriter(log_dir=os.path.join(project_dir, "runs"))
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    early_stop_flag = False
    # Set up post-processing functions for labels and predictions.
    # Note: The post-processing functions are used to convert the model outputs and labels to the desired format.
    # For example, converting the model outputs to one-hot encoded format for multi-class segmentation.
    # The AsDiscrete transform is used to convert the model outputs to discrete labels.
    # The argmax transform is used to convert the model outputs to the predicted class labels.
    # The to_onehot argument specifies whether to convert the labels to one-hot encoded format.
    # The argmax argument specifies whether to take the argmax of the model outputs to get the predicted class labels.
    post_label = AsDiscrete(to_onehot=out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=out_channels)

    # Option to select 2-step training.
    # if 2-step training is set True, the encoder is frozen in the first step and only the decoder is trained.
    # In the second step, the entire model is trained.
    # if 2-step training is set False, then entire model is trained.
    if train_params["2step"]:
        print("STEP 1: Freeze encoder, train decoder only")
        freeze_encoder(model)

        # High LR for decoder training
        step1_lr = train_params.get("step1_lr", 1e-3)
        step1_iters = train_params.get("step1_iters", 3000)  # 3–5 recommended
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=step1_lr, weight_decay=1e-5)
        scheduler = None  # usually no scheduler for step1 warm-up

        global_step = 0
        while global_step < step1_iters:
            model, global_step, dice_val_best, global_step_best, early_stop_flag = \
                train_net(global_step, train_loader, dice_val_best,
                        global_step_best, model, post_label, post_pred,
                        dice_metric, loss_function, optimizer, scheduler,
                        scaler, train_params, val_loader,
                        early_stopper, epoch_loss_values,
                        metric_values, writer, model_dir)
            
        
        
        unfreeze_all(model)

        step2_lr = train_params.get("step2_lr", 1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=step2_lr, weight_decay=1e-5)

    # Set learning rate scheduler.
    if train_params["lr_scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=train_params["cawr_T_0"], T_mult=train_params["cawr_T_mult"],
            eta_min=float(train_params["cawr_eta_min"]))  # , last_epoch=train_params["cawr_last_epoch"]
    elif train_params["lr_scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=train_params["rlrop_factor"], 
                                    patience=train_params["rlrop_patience"])
    elif train_params["lr_scheduler"] == "StepLR":
        # Reduce learning rate by a factor every few epochs.
        scheduler = StepLR(optimizer, step_size=train_params["slr_step"], 
                           gamma=train_params["slr_gamma"])
    elif train_params["lr_scheduler"] == "CyclicLR":
        scheduler = CyclicLR(optimizer, base_lr=train_params["cyclic_base_lr"],
            max_lr=train_params["cyclic_max_lr"],step_size_up=train_params["cyclic_step_up"],
            mode=train_params.get("cyclic_mode", "triangular"),cycle_momentum=False)  # Set to True if using optimizers like SGD with momentum

    elif train_params["lr_scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_params["cosine_T_max"],
            eta_min=train_params.get("cosine_eta_min", 0))

    else:
        print("No learning rate scheduler specified.")
        scheduler = None


    # if train_params["lr_scheduler"] == "CosineAnnealingWarmRestarts":
    #     scheduler = CosineAnnealingWarmRestarts(
    #         optimizer, T_0=train_params.get("cawr_T_0", 10), T_mult=train_params.get("cawr_T_mult", 2),
    #         eta_min=float(train_params.get("cawr_eta_min", 1e-6)))  # , last_epoch=train_params.get("cawr_last_epoch"]
    # elif train_params["lr_scheduler"] == "ReduceLROnPlateau":
    #     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=train_params.get("rlrop_factor", 0.75), 
    #                                 patience=train_params.get("rlrop_patience", 5))
    # elif train_params["lr_scheduler"] == "StepLR":
    #     # Reduce learning rate by a factor every few epochs.
    #     scheduler = StepLR(optimizer, step_size=train_params.get("slr_step", 40), 
    #                        gamma=train_params.get("slr_gamma", 0.95))
    # elif train_params["lr_scheduler"] == "CyclicLR":
    #     scheduler = CyclicLR(optimizer, base_lr=train_params.get("cyclic_base_lr", 1e-5),
    #         max_lr=train_params.get("cyclic_max_lr", 1e-3),step_size_up=train_params.get("cyclic_step_up", 2000),
    #         mode=train_params.get("cyclic_mode", "triangular2"),cycle_momentum=False)  # Set to True if using optimizers like SGD with momentum

    # elif train_params["lr_scheduler"] == "CosineAnnealingLR":
    #     scheduler = CosineAnnealingLR(optimizer, T_max=train_params.get("cosine_T_max", 50),
    #         eta_min=train_params.get("cosine_eta_min", 1e-6))

    # else:
    #     print("No learning rate scheduler specified.")
    #     scheduler = None

    while global_step < train_params["max_iterations"] and not early_stop_flag:
        # Train the model for one epoch.
        # Note: The train_net function handles the training loop, validation, and logging.
        # It also handles the early stopping condition based on the validation loss.
        # The train_net function returns the updated global step, best Dice score, and best global step.
        # The global step is incremented after each training iteration.
        # The best Dice score is updated if the current Dice score is better than the previous best.
        # The best global step is updated if the current global step is better than the previous best.
        # The early stopping condition is checked after each training iteration.
        # The early stopping condition is met if the validation loss does not improve for a specified number of epochs.
        model, global_step, dice_val_best, global_step_best, early_stop_flag = \
        train_net(global_step, train_loader, dice_val_best, 
                  global_step_best, model, post_label,
                  post_pred, dice_metric, loss_function, 
                  optimizer, scheduler, scaler, train_params,
                  val_loader, early_stopper, epoch_loss_values, 
                  metric_values, writer, model_dir)

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), os.path.join(model_dir, "last_model.pth"))

    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")    

    # Evaluate the best model on the validation dataset.
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth")))
    model.eval()
    val_dice = eval_pred_images(output_dir, model, post_label, post_pred, 
                                val_loader, segs['val'], dice_metric,
                                train_params)
    write_report(report_path, train_params, images['val'], val_dice)

    # Plotting.
    # plot_training_curves(epoch_loss_values, metric_values, eval_num)
    writer = SummaryWriter(log_dir=os.path.join(project_dir, "figures"))
    writer.add_figure('predictions vs. actuals',
                      display_pred_images(model, val_loader, train_params))
    writer.close()
    

if __name__ == "__main__":
    entry_func()