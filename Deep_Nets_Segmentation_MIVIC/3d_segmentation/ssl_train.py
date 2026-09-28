import os
import time
import yaml
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError

from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler 

from utils.ssl_data_utils import load_im_dataset_kfolds
from models.swinautoenc import SwinAutoEnc
from utils.ssl_data_utils import (ssl_set_train_val_transforms_vit,
                                ssl_set_train_val_transforms_swin)
from ssl_trainer import ssl_trainer_vit, ssl_trainer_swin, save_ckp
from losses.ssl_losses import Loss



def get_ssl_train_argparser():
    """Argument parser for SSL training script."""
    parser = ArgumentParser(description='SSL training script for 3D medical images using MONAI.')
    parser.add_argument("--project_path", type=str, default='./',
                        help="Path to project directory. "
                             "Defaults to the current directory.")
    parser.add_argument('--mdname', '-m', type=str, choices=['VIT_AUTOENC', 'SWIN_AUTOENC',
                                                             'SWIN_AUTOENCV2'],
                        default='SWIN_AUTOENC', help='Model name defaults to SWIN_AUTOENC')
    parser.add_argument('--n_folds', type=int, default=5, 
                        help='Number of folds for k-fold cross-validation defaults to 5')
    parser.add_argument('--fold_id', type=int, default=1,
                         help='Fold number for k-fold cross-validation defaults to 1')    
    return parser



def entry_func(args=None):
    """Entry point for the SSL training script."""
    if args is None:
        args = get_ssl_train_argparser().parse_args(args)

    project_path = os.path.normpath(args.project_path) #os.path.normpath("/home/users/sokratis/Workspace/BLSA_3D_Thigh_MRI_37/Nets_MONAI/SSL_Train")
    mdname = args.mdname # Model Name, can be SWIN_AUTOENC or VIT_AUTOENC

    if os.path.exists(project_path) is False:
        os.mkdir(project_path)

    with open('ssl_train_params.yaml') as fp:
        train_params = yaml.safe_load(fp)
        train_params["n_folds"] = args.n_folds
        train_params["fold_id"] = args.fold_id
        train_params["mdname"]= args.mdname

    print(train_params)


    # image_names = load_dataset_fixed_split(train_params)
    image_names = load_im_dataset_kfolds(train_params)

    train_data = [{"image": image_name} for image_name in image_names["train"]]
    val_data = [{"image": image_name} for image_name in image_names["val"]]

    print("Total Number of Training Data Samples: {}".format(len(train_data)))
    print(train_data)
    print("#" * 10)
    print("Total Number of Validation Data Samples: {}".format(len(val_data)))
    print(val_data)
    print("#" * 10)

    # Set Determinism
    set_determinism(seed=123)
    device = torch.device("cuda:0")

    # Define Training Transforms
    if mdname == "VIT_AUTOENC":
        train_transforms = ssl_set_train_val_transforms_vit(train_params)
        val_transforms = train_transforms
    elif mdname == "SWIN_AUTOENC" or mdname == "SWIN_AUTOENCV2":
        train_transforms, _ = ssl_set_train_val_transforms_swin(train_params)
        val_transforms, _ = ssl_set_train_val_transforms_swin(train_params)
    
    check_ds = Dataset(data=train_data, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image = check_data["image"][0][0]
    print(f"image shape: {image.shape}")

    # Define Network ViT backbone & Loss & Optimizer
    if mdname == "VIT_AUTOENC":
        model = ViTAutoEnc(
            train_params["in_channels"],
            img_size=train_params["roi_size"],
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
        )
    elif mdname == "SWIN_AUTOENC":
        model = SwinAutoEnc(train_params["in_channels"],
                            img_size=train_params["roi_size"],
                            use_v2=False)
    elif mdname == "SWIN_AUTOENCV2":
        model = SwinAutoEnc(train_params["in_channels"],
                            img_size=train_params["roi_size"],
                            use_v2=True)

    model = model.to(device)

    # Define Hyper-parameters for training loop
    batch_size = train_params["batch_size"]
    lr = train_params["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Define DataLoader using MONAI, CacheDataset needs to be used
    train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=None,
                              shuffle=True, num_workers=4, drop_last=True)

    val_ds = Dataset(data=val_data, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=4, drop_last=True)
    
    epoch = 0
    if mdname == "VIT_AUTOENC":
        recon_loss = L1Loss()
        contrastive_loss = ContrastiveLoss(temperature=0.05)
        model, epoch, = ssl_trainer_vit(train_params, model, train_loader, val_loader, optimizer, 
                                        recon_loss, contrastive_loss, train_ds, project_path, device)
    elif mdname == "SWIN_AUTOENC" or mdname == "SWIN_AUTOENCV2":
        # Initialize TensorBoard writer
        logdir = os.path.join(project_path, "logs", mdname, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
        # Set up training parameters
        args.local_rank = 0  # Set local rank for distributed training
        args.device = device   
        args.logdir = logdir
        args.amp = True  # Enable automatic mixed precision
        args.grad_clip = False  # Disable gradient clipping
        args.lrdecay = False  # Enable learning rate decay
        args.batch_size = batch_size
        train_params["upsample"] = "vae"  # Set upsample method for SwinAutoEnc
        args.sw_batch_size = train_params['num_samples']  # Swin batch size
        args.num_steps = train_params["max_iterations"]  # Total number of training steps
        args.eval_num = train_params["val_interval"]  # Evaluation frequency
        # Initialize model, optimizer, loss function, and scaler 
        loss_function = Loss(args.batch_size * args.sw_batch_size, args)
        scaler = GradScaler()
        scheduler = None
        # Initialize model 
        global_step = 0
        best_val = 1e8
        # Train the model
        print("Starting training...")
        while global_step < args.num_steps:
            epoch += 1
            print(f"Epoch {epoch}/{train_params['max_epochs']}")
            model, global_step, loss, best_val = ssl_trainer_swin(args, model, global_step, train_loader, 
                                                           best_val, scaler, optimizer, 
                                                           scheduler, loss_function, val_loader, 
                                                           logdir, writer)
        # Close TensorBoard writer
        writer.close()
    else:
        raise ValueError("Invalid model name. Choose from ['VIT_AUTOENC', 'SWIN_AUTOENC', 'SWIN_AUTOENCV2']")
    
    # Save the final model and checkpoint   
    checkpoint = {"epoch": epoch, "state_dict": model.state_dict(), 
                    "optimizer": optimizer.state_dict()}
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")
    torch.save(model.state_dict(), logdir + "final_model_vit_autoenc.pth")
    print("Final model saved at: {}".format(logdir + "final_model_vit_autoenc.pth"))


if __name__ == "__main__":
    entry_func()