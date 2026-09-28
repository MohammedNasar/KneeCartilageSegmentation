import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from monai.inferers import sliding_window_inference
import torch


def display_pred_images(model, my_data_loader, train_params):
    """
    Display predicted images, ground truth labels, and input images for a given model and validation data loader.
    Args:
        root_dir (str): The root directory where the images are stored.
        model (torch.nn.Module): The trained model used for making predictions.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    Returns:
        None: This function displays the images using matplotlib and does not return any value.
    """
    
    n_items = len(my_data_loader)
    rows, cols = n_items, 3
    figure = plt.figure("check", (18, 6))

    with torch.no_grad():
        for case_num, batch in enumerate(my_data_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # val_inputs = torch.unsqueeze(img, 1).cuda()
            # val_labels = torch.unsqueeze(label, 1).cuda()
            val_outputs = sliding_window_inference(val_inputs, train_params['roi_size'], 4, model, overlap=0.8)
            plt.subplot(rows, cols, case_num * cols + 1)
            plt.title("image")
            plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 31], cmap="gray")
            plt.subplot(rows, cols, case_num * cols + 2)
            plt.title("label")
            plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 31])
            plt.subplot(rows, cols, case_num * cols + 3)
            plt.title("output")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 31])
    return figure


def plot_training_curves(epoch_loss_values, metric_values, eval_num):
    """
    Plot training curves for iteration average loss and validation mean dice.
    Args:
        epoch_loss_values (list of float): List of average loss values for each epoch.
        metric_values (list of float): List of validation mean dice values for each epoch.
        eval_num (int): Evaluation number to scale the x-axis.
    Returns:
        None: This function only plots the training curves and does not return any value.
    """

    # Plot training curves. 
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Iteration Average Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("Iteration")
    plt.plot(x, y)
    plt.show()


def display_image_label_pair(x, y):
    """
    Display a pair of image and label slices.
    This function takes a 5D tensor for both the image and label, extracts the 
    middle slice along the z-axis (index 31), and displays them side by side 
    using matplotlib.
    Args:
        x (torch.Tensor): A 5D tensor representing the image with shape 
                          (batch_size, channels, depth, height, width).
        y (torch.Tensor): A 5D tensor representing the label with shape 
                          (batch_size, channels, depth, height, width).
    Returns:
        None
    """

    plt.figure("check", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(x.cpu().numpy()[0, 0, :, :, 31], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(y.cpu().numpy()[0, 0, :, :, 31])
    plt.show()


def display_multiple_slices(volumetric_data, num_slices_to_display = 9):
    """
    Displays multiple evenly spaced slices from a 5D volumetric data tensor using matplotlib.
    Parameters:
        volumetric_data (numpy.ndarray): A 5D tensor with shape (batch_size, channels, depth, height, width).
        num_slices_to_display (int, optional): Number of slices to display. Default is 4.
    Raises:
        ValueError: If the input volumetric_data is not a 5D tensor.
    Notes:
        - Only the first batch and first channel are visualized.
        - Slices are selected evenly across the width dimension.
        - The function displays the slices in a 2x2 grid using matplotlib.
    """
    
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten() # Flatten the axes array for easier iteration

    if volumetric_data.ndim != 5:
        raise ValueError("Input volumetric data must be a 5D tensor with shape (batch_size, channels, depth, height, width).")
    
    for i in range(num_slices_to_display):
        slice_index = i * (volumetric_data.shape[4] // num_slices_to_display) # Select evenly spaced slices
        axes[i].imshow(volumetric_data.cpu().numpy()[0, 0, :, :, slice_index], cmap='gray') # Display a slice
        axes[i].set_title(f'Slice {slice_index}')
        axes[i].axis('off') # Turn off axis labels and ticks for cleaner display       

    plt.tight_layout()
    plt.show()


def write_report(log_file, train_params, image_paths, metric_values):
    """
    Writes a report to a log file by combining training parameters, image names, 
    and metric values into a structured DataFrame.
    Args:
        log_file (str): Path to the log file where the report will be saved. 
                        If the file exists, the new data will be appended to it.
        train_params (dict): Dictionary containing training parameters. 
                             Keys are parameter names, and values are their corresponding values.
        image_paths (list of str): List of file paths to the images. 
                                   The base names of these paths will be used as image identifiers.
        metric_values (list of list of float): A 2D list where each sublist contains 
                                               metric values for a corresponding image.
    Returns:
        None: The function writes the report to the specified log file and prints the DataFrame.
    Notes:
        - The function ensures all rows and up to 25 columns of the DataFrame are displayed in the console.
        - If the log file already exists, the function appends the new data to the existing data.
        - The resulting DataFrame is saved to the log file in CSV format with an index column.
    """

    # breakpoint()
    image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_paths]
    params_df = pd.DataFrame.from_dict(train_params, orient='index').T
    result_columns=['muscle_'+str(i) for i in np.arange(len(metric_values[0]))+1]
    df = pd.DataFrame(metric_values, columns=result_columns)
    df.insert(loc=0, column='Image', value=image_names)
    df = pd.concat([df, params_df], axis=1, ignore_index=False)
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', 25)  # Show all columns
    print(df)
    if os.path.exists(log_file):
        df_ = pd.read_csv(log_file, index_col=0)
        df = pd.concat([df, df_], axis=0, ignore_index=False)
    df.to_csv(log_file, index=True)
    print('Log written!')
    