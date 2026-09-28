import os
import glob
from sklearn.model_selection import KFold


from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    ScaleIntensityd,
    EnsureChannelFirstd,
    CopyItemsd,
    OneOf,
    RandCoarseDropoutd,
    RandCoarseShuffled,
)



def ssl_set_train_val_transforms_vit(train_params):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=train_params["pix_dim"], mode=("bilinear")),
            ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            CropForegroundd(keys=["image"], source_key="image"),
            SpatialPadd(keys=["image"], spatial_size=train_params["roi_size"]), 
            RandSpatialCropSamplesd(keys=["image"], roi_size=train_params["roi_size"], 
                                    random_size=False, num_samples=train_params["num_samples"]),
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image"], prob=0.8, holes=10, spatial_size=8),
            # Please note that that if image, image_2 are called via the same transform call because of the determinism
            # they will get augmented the exact same way which is not the required case here, hence two calls are made
            OneOf(
                transforms=[
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=5, dropout_holes=True, max_spatial_size=32
                    ),
                    RandCoarseDropoutd(
                        keys=["image_2"], prob=1.0, holes=6, spatial_size=20, dropout_holes=False, max_spatial_size=64
                    ),
                ]
            ),
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=10, spatial_size=8),
        ]
    )

    return train_transforms


def ssl_set_train_val_transforms_swin(train_params):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(
                keys=["image"],
                channel_dim="no_channel" if train_params.get('in_channels', 1) == 1 else None
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=train_params['pix_dim'],
                mode=("bilinear"),
            ),
            ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            SpatialPadd(keys=["image"], spatial_size=train_params['roi_size']),
            CropForegroundd(keys=["image"], source_key="image",
                            k_divisible= train_params['roi_size'],
                            allow_smaller=True),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=train_params['roi_size'],
                num_samples=train_params['num_samples'],
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(
                keys=["image"],
                channel_dim="no_channel" if train_params.get('in_channels', 1) == 1 else None
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=train_params['pix_dim'],
                mode=("bilinear"),
            ),
            ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            SpatialPadd(keys=["image"], spatial_size=train_params['roi_size']),
            CropForegroundd(keys=["image"], source_key="image",
                            k_divisible= train_params['roi_size'],
                            allow_smaller=True),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=train_params['roi_size'],
                num_samples=train_params['num_samples'],
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    return train_transforms, val_transforms



def load_im_dataset_kfolds(train_params):
    """
    Splits a dataset of medical images into training, validation, and test sets using K-Fold cross-validation.
    Args:
        train_params (dict): A dictionary containing the following keys:
            - "data_path" (str): Path to the directory containing the images.
            - "n_folds" (int): Number of folds for K-Fold cross-validation.
            - "fold_id" (int): The index of the fold to use for the current split.
    Returns:
        dict: A dictionary with the following keys:
            - 'train' (list): List of file paths for the training images.
            - 'val' (list): List of file paths for the validation images.
            - 'test' (list): List of file paths for the test images (same as validation set).
    Notes:
        - The function assumes the images are stored in the "images" subdirectory of the provided "data_path".
        - Supported image file extensions are "*.nii" and "*.nii.gz".
        - The K-Fold split is randomized with a fixed random state (42) for reproducibility.
        - The test set is identical to the validation set in this implementation.
    Example:
        train_params = {
            "data_path": "/path/to/data",
            "n_folds": 5,
            "fold_id": 0
        }
        images = load_im_dataset_kfolds(train_params)
    """
    
    image_path = f'{train_params["data_path"]}/images'

    extensions = ["*.nii", "*.nii.gz"]
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(image_path, ext)))
    
    all_images = sorted(all_images)  

    kfold =  KFold(n_splits=train_params["n_folds"], shuffle=True, random_state=42).split(all_images)
    kfold_list = list(kfold)
    train_idx = kfold_list[train_params["fold_id"]][0]
    val_idx = kfold_list[train_params["fold_id"]][1]

    images = {}
    images['train'] = [all_images[i] for i in train_idx]
    images['val'] = [all_images[i] for i in val_idx]
    images['test'] = images['val']
    print(f"Train: {(images['train'])}, Val: {(images['val'])}, Test: {(images['test'])}")

    return images