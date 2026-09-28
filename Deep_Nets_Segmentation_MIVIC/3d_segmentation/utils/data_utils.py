"""
utils.py
Utility functions for data loading, augmentation, and dataset preparation
for 3D medical image segmentation using MONAI and PyTorch.

Key design decision for val_transforms
---------------------------------------
CropForegroundd is intentionally ABSENT from val_transforms.

Reason: CropForegroundd crops to the foreground bounding box, which introduces
a spatial origin shift (e.g. [246, 160, 33] voxels) relative to the full RAS
volume. This shift is not recorded anywhere and cannot be cleanly inverted in
evaluate.py. When the prediction is zoomed back to the original shape, the
origin offset causes the segmentation mask to land in the wrong location,
producing the visible dispersion/elongation seen in ITK-Snap.

sliding_window_inference handles full volumes of arbitrary size via patch-based
inference with configurable overlap — it does not require CropForegroundd.

CropForegroundd is kept in train_transforms only, where it is followed
immediately by RandCropByPosNegLabeld (random patch sampling), so the absolute
origin of the cropped volume is irrelevant to training.
"""

import os
import glob
import torch
from monai.data import ArrayDataset, DataLoader, CacheDataset
from argparse import ArgumentParser
from sklearn.model_selection import KFold, train_test_split

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    RandFlip,
    ScaleIntensity,
    RandGaussianNoise,
    RandRotate,
    Rand3DElastic,
)

from monai.transforms import (
    EnsureTyped,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityd,
    Spacingd,
    RandRotate90d,
    SpatialPadd,
)

from monai.data import DataLoader


def get_argparser():
    parser = ArgumentParser(description='Fit a model defined in a project folder.')
    parser.add_argument("--project_dir", type=str, default='./',
                        help="Path to project directory.")
    parser.add_argument('--mdname', '-m', type=str,
                        choices=['UNET', 'UNETR', 'SWIN_UNETR', 'SWIN_UNETRV2'],
                        default='UNET', help='Model name')
    parser.add_argument('--deterministic', action='store_true',
                        help='Disable cudnn.benchmark for reproducibility')
    parser.add_argument('--parallel', action='store_true',
                        help='Use multiple GPUs for training')
    parser.add_argument('--optimize_lr', action='store_true',
                        help='Optimize learning rate')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='Use pretrained weights')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained weights')
    return parser


def augment_array_dataset(images, segs, train_params, train_mode=True):
    imtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Rand3DElastic(sigma_range=(0, 1), magnitude_range=(10, 15)),
        RandFlip(),
        RandGaussianNoise(),
        RandRotate(),
    ])
    segtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Rand3DElastic(sigma_range=(0, 1), magnitude_range=(10, 15)),
        RandFlip(),
        RandRotate(),
    ])
    val_imtrans = Compose([
        LoadImage(image_only=True),
        ScaleIntensity(),
        EnsureChannelFirst(),
    ])
    val_segtrans = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
    ])

    if train_mode:
        train_ds = ArrayDataset(images['train'], imtrans, segs['train'], segtrans)
        train_loader = DataLoader(train_ds, batch_size=train_params['batch_size'],
                                  pin_memory=torch.cuda.is_available())
        val_ds = ArrayDataset(images['val'], val_imtrans, segs['val'], val_segtrans)
        val_loader = DataLoader(val_ds, batch_size=1,
                                pin_memory=torch.cuda.is_available())
        test_ds = None
        test_loader = None
    else:
        train_ds = train_loader = val_ds = val_loader = None
        test_ds = ArrayDataset(images['test'], val_imtrans, segs['test'], val_segtrans)
        test_loader = DataLoader(test_ds, batch_size=1,
                                 pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def augment_list_dataset(images, segs, device, train_params, train_mode=True):
    """
    Prepares data loaders for 3D medical image segmentation.

    train_transforms  — includes CropForegroundd + RandCropByPosNegLabeld
                        (origin offset from CropForegroundd does not matter
                         because random patches are sampled immediately after)

    val_transforms    — NO CropForegroundd at all
                        The full resampled volume is passed directly to
                        sliding_window_inference, which handles any shape.
                        This guarantees the prediction array has the same
                        spatial origin as the resampled input, so the inverse
                        transform in evaluate.py (zoom + reorient) is exact.
    """
    num_samples = train_params['rand_crop_num_samples']

    # ── Training transforms ───────────────────────────────────────────────────
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=train_params['pix_dim'],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
        SpatialPadd(keys=["image", "label"],
                    spatial_size=train_params['roi_size']),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=train_params['roi_size'],   # fine for training
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=train_params['roi_size'],
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])

    # ── Validation / test transforms ──────────────────────────────────────────
    # CropForegroundd is intentionally absent.
    # The full volume (after Spacingd resampling) is fed to sliding_window_inference.
    # No spatial origin shift → evaluate.py inverse transform is exact.
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=train_params['pix_dim'],
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
        # SpatialPadd is also not needed: sliding_window_inference pads internally
        # if the volume is smaller than roi_size in any dimension.
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    if train_mode:
        train_data_dict = [
            {"image": img, "label": lbl}
            for img, lbl in zip(images['train'], segs['train'])
        ]
        train_ds = CacheDataset(
            data=train_data_dict, transform=train_transforms,
            cache_rate=1.0, num_workers=4,
        )
        train_loader = DataLoader(train_ds,
                                  batch_size=train_params['batch_size'])

        val_data_dict = [
            {"image": img, "label": lbl}
            for img, lbl in zip(images['val'], segs['val'])
        ]
        val_ds = CacheDataset(
            data=val_data_dict, transform=val_transforms,
            cache_rate=1.0, num_workers=4,
        )
        val_loader   = DataLoader(val_ds, batch_size=1)
        test_ds      = None
        test_loader  = None

    else:
        train_ds = train_loader = val_ds = val_loader = None

        test_data_dict = [
            {"image": img, "label": lbl}
            for img, lbl in zip(images['test'], segs['test'])
        ]
        test_ds = CacheDataset(
            data=test_data_dict, transform=val_transforms,
            cache_rate=1.0, num_workers=4,
        )
        test_loader = DataLoader(test_ds, batch_size=1)

    return train_loader, val_loader, test_loader


def fixed_split_dataset(train_params):
    image_paths = {
        'train': f'{train_params["data_path"]}/train/images',
        'val':   f'{train_params["data_path"]}/val/images',
        'test':  f'{train_params["data_path"]}/test/images',
    }
    label_paths = {
        'train': f'{train_params["data_path"]}/train/labels',
        'val':   f'{train_params["data_path"]}/val/labels',
        'test':  f'{train_params["data_path"]}/test/labels',
    }

    TPS = ['train', 'val', 'test']
    extensions = ["*.nii", "*.nii.gz"]

    images = {tp: [] for tp in TPS}
    segs   = {tp: [] for tp in TPS}

    for tp in TPS:
        for ext in extensions:
            images[tp] += sorted(glob.glob(os.path.join(image_paths[tp], ext)))
            segs[tp]   += sorted(glob.glob(os.path.join(label_paths[tp], ext)))
        images[tp] = sorted(set(images[tp]))
        segs[tp]   = sorted(set(segs[tp]))

    return images, segs


def kfold_split_dataset(train_params):
    """
    Splits dataset into train/val/test using k-fold cross-validation.
    """
    image_path = f'{train_params["data_path"]}/images'
    label_path = f'{train_params["data_path"]}/labels'

    extensions = ["*.nii", "*.nii.gz"]
    all_images, all_segs = [], []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(image_path, ext)))
        all_segs.extend(glob.glob(os.path.join(label_path, ext)))

    all_images = sorted(all_images)
    all_segs   = sorted(all_segs)

    kfold = KFold(n_splits=train_params["n_folds"], shuffle=True,
                  random_state=42).split(all_images, all_segs)
    kfold_list = list(kfold)

    fold_id = train_params["fold_id"] - 1
    if fold_id < 0 or fold_id >= train_params["n_folds"]:
        raise ValueError(
            f"fold_id must be between 1 and {train_params['n_folds']}, "
            f"got {train_params['fold_id']}"
        )

    train_idx, val_idx = train_test_split(
        kfold_list[fold_id][0],
        test_size=1 / train_params["n_folds"],
        random_state=42,
    )
    test_idx = kfold_list[fold_id][1]

    images = {
        'train': [all_images[i] for i in train_idx],
        'val':   [all_images[i] for i in val_idx],
        'test':  [all_images[i] for i in test_idx],
    }
    segs = {
        'train': [all_segs[i] for i in train_idx],
        'val':   [all_segs[i] for i in val_idx],
        'test':  [all_segs[i] for i in test_idx],
    }

    print(f"Train: {images['train']}")
    print(f"Val  : {images['val']}")
    print(f"Test : {images['test']}")

    return images, segs