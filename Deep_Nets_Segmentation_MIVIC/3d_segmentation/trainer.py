import torch
import os
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from tqdm import tqdm
from monai.losses import (DiceCELoss, 
                          DiceLoss, 
                          GeneralizedDiceLoss, 
                          FocalLoss,
                          DiceFocalLoss)


class EarlyStopper:
    """
    Early stopping utility to stop training when validation loss does not improve.
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.best_metric is None:
            self.best_metric = val_metric
        elif val_metric < (self.best_metric + self.min_delta):
            self.counter += 1
            print(f"Early stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_metric = val_metric
            self.counter = 0


def set_loss_function(loss_name):
    """
    Choose the loss function based on the given loss name.

    Args:
        loss_name (str): The name of the loss function to use.

    Returns:
        torch.nn.Module: The loss function.
    """

    if loss_name == "DiceLoss":
        loss_function = DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True)
    elif loss_name == "DiceCELoss":
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    elif loss_name == "GenDiceLoss":
        loss_function = GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "FocalLoss":
        loss_function = FocalLoss(to_onehot_y=True)
    elif loss_name == "DiceFocalLoss":
        loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif loss_name == "CrossEntropyLoss":
        loss_function = torch.nn.CrossEntropyLoss(
            # Optional parameters for segmentation
            ignore_index=-1,  # Ignore specific class labels
            reduction='mean'  # How to aggregate loss across batch
        )
    else:
        raise ValueError(f"Invalid loss function: {loss_name}")

    return loss_function


def validate_net(epoch_iterator_val, model, post_label, post_pred, global_step, dice_metric, roi_size, max_iterations):
    """
    Perform validation of a neural network model using a sliding window inference approach.
    Args:
        epoch_iterator_val (iterable): An iterator providing validation data batches. Each batch should contain
            "image" and "label" keys for input images and corresponding ground truth labels.
        model (torch.nn.Module): The neural network model to be validated.
        post_label (callable): A post-processing function to apply to the ground truth labels.
        post_pred (callable): A post-processing function to apply to the model predictions.
        global_step (int): The current global step in the training/validation process.
        dice_metric (monai.metrics.DiceMetric): A metric object to compute the Dice similarity coefficient.
        roi_size (tuple or list): The size of the region of interest (ROI) for sliding window inference.
        max_iterations (int): The maximum number of iterations for the validation process.
    Returns:
        float: The mean Dice similarity coefficient computed over the validation dataset.
    """

    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, roi_size, 1, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, max_iterations))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train_net(global_step, train_loader, dice_val_best, global_step_best, 
              model, post_label, post_pred, dice_metric,
              loss_function, optimizer, scheduler, scaler, 
              train_params, val_loader, early_stopper,
              epoch_loss_values, metric_values, writer,
              model_dir):
    """
    Train a neural network for segmentation tasks.
    Args:
        global_step (int): The current global step of training.
        train_loader (DataLoader): DataLoader for the training dataset.
        dice_val_best (float): The best Dice score achieved so far.
        global_step_best (int): The global step corresponding to the best Dice score.
        model (torch.nn.Module): The neural network model to be trained.
        post_label (Callable): Post-processing function for ground truth labels.
        post_pred (Callable): Post-processing function for model predictions.
        dice_metric (Callable): Metric function to compute the Dice score.
        loss_function (Callable): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        train_params (dict): Dictionary containing training parameters such as:
            - "roi_size" (tuple): Region of interest size for training.
            - "max_iterations" (int): Maximum number of training iterations.
            - "eval_num" (int): Frequency of evaluation during training.
        val_loader (DataLoader): DataLoader for the validation dataset.
        epoch_loss_values (list): List to store the average loss values for each epoch.
        metric_values (list): List to store the validation metric values.
        project_dir (str): Directory path for saving logs and model checkpoints.
    Returns:
        tuple: A tuple containing:
            - global_step (int): Updated global step after training.
            - dice_val_best (float): Updated best Dice score.
            - global_step_best (int): Updated global step corresponding to the best Dice score.
    Notes:
        - The function uses mixed precision training for efficiency.
        - The model is evaluated periodically during training, and the best model is saved based on the Dice score.
        - Training progress and metrics are logged using TensorBoard.
    """    

    roi_size = train_params["roi_size"]
    max_iterations = train_params["max_iterations"]
    eval_num = train_params["eval_num"]

    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X) (lr=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].long().cuda())
        # If the loss function is CrossEntropyLoss and y has only one channel, squeeze it.
        # This is a workaround for the issue where CrossEntropyLoss expects y to be of shape (N, C, D, H, W)
        # but y is of shape (N, 1, D, H, W).
        if isinstance(loss_function, torch.nn.modules.loss.CrossEntropyLoss) & y.shape[1] == 1:
            y = torch.squeeze(y, axis=1)
        # display_image_label_pair(x, y)
        # breakpoint()
        with torch.cuda.amp.autocast():
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
            scheduler.step()   # per iteration (step-wise decay)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            # CAWR expects a fractional epoch counter
            scheduler.step()  # global_step/ len(train_loader)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            # CAWR expects a fractional epoch counter
            scheduler.step()  # global_step/ len(train_loader)
        elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            # CAWR expects a fractional epoch counter
            scheduler.step()  # global_step/ len(train_loader)

        optimizer.zero_grad(set_to_none=True)
        epoch_iterator.set_description(  # noqa: B038
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f}) (lr={optimizer.param_groups[0]['lr']:.6f})"
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X) (lr=X.X)", dynamic_ncols=True)
            dice_val = validate_net(epoch_iterator_val, model, post_label, post_pred, 
                                    global_step, dice_metric, roi_size, max_iterations)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            # Log the average loss and validation metric to TensorBoard.
            writer.add_scalar("train_loss", epoch_loss, global_step)
            writer.add_scalar("val_dice", dice_val, global_step)
            # Log the learning rate to TensorBoard.
            writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
            # Update the learning rate in the optimizer.
            # scheduler.step(dice_val)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(dice_val)
            # optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
            # Log epoch loss into early_stopper.
            early_stopper(dice_val)
        global_step += 1
        early_stop_flag = early_stopper.early_stop   
    return model, global_step, dice_val_best, global_step_best, early_stop_flag
