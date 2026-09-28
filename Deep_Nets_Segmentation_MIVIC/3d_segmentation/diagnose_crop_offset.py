"""
diagnose_crop_offset.py
=======================
Checks whether CropForegroundd introduces a spatial offset between
the image and label arrays, and whether label interpolation is correct.

Run this on one image+label pair to diagnose overlay dispersion.

Usage
-----
python diagnose_crop_offset.py \
    --image  /path/to/image.nii.gz \
    --label  /path/to/label.nii.gz \
    --pix-dim 0.2469 0.2469 0.5 \
    --roi-size 96 96 96
"""

import argparse
import numpy as np
import nibabel as nib
import torch

from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd,
    ScaleIntensityd, SpatialPadd, CropForegroundd, EnsureTyped,
)


def check(image_path, label_path, pix_dim, roi_size):

    print("\n" + "="*70)
    print("CROP OFFSET + INTERPOLATION DIAGNOSTIC")
    print("="*70)

    data = {"image": image_path, "label": label_path}

    # ── Base transforms up to Spacingd ────────────────────────────────────────
    t_base = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pix_dim,
            mode=("bilinear", "nearest"),   # ← nearest for label
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
    ])

    # ── With SpatialPad ───────────────────────────────────────────────────────
    t_pad = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pix_dim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
    ])

    # ── With CropForegroundd ──────────────────────────────────────────────────
    t_crop = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pix_dim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            k_divisible=roi_size,
        ),
    ])

    print("\n[1] After Spacingd (before any crop/pad):")
    out = t_base(data)
    img_shape = tuple(out["image"].shape[1:])
    lbl_shape = tuple(out["label"].shape[1:])
    lbl_arr   = out["label"].numpy().squeeze()
    lbl_vals  = np.unique(lbl_arr)
    print(f"    image shape : {img_shape}")
    print(f"    label shape : {lbl_shape}")
    print(f"    label values: {lbl_vals}  ← should be integers only (0,1,2...)")
    bad_vals = lbl_vals[(lbl_vals != np.round(lbl_vals))]
    if len(bad_vals) > 0:
        print(f"    ⚠ NON-INTEGER label values found: {bad_vals[:10]}")
        print(f"      Spacingd label mode is NOT nearest-neighbour!")
    else:
        print(f"    ✓ All label values are integers after Spacingd")

    print("\n[2] After SpatialPadd:")
    out = t_pad(data)
    img_shape_pad = tuple(out["image"].shape[1:])
    lbl_shape_pad = tuple(out["label"].shape[1:])
    print(f"    image shape : {img_shape_pad}")
    print(f"    label shape : {lbl_shape_pad}")
    pad_added = tuple(img_shape_pad[i] - img_shape[i] for i in range(3))
    print(f"    padding added (per axis): {pad_added}")
    if any(p > 0 for p in pad_added):
        print(f"    ⚠ Padding was added — this shifts the foreground bbox origin")
        print(f"      CropForegroundd will anchor to a corner of this padded volume")
    else:
        print(f"    ✓ No padding added (volume already >= roi_size)")

    print("\n[3] After CropForegroundd:")
    out = t_crop(data)
    img_shape_crop = tuple(out["image"].shape[1:])
    lbl_shape_crop = tuple(out["label"].shape[1:])
    lbl_arr_crop   = out["label"].numpy().squeeze()
    lbl_vals_crop  = np.unique(lbl_arr_crop)
    print(f"    image shape : {img_shape_crop}")
    print(f"    label shape : {lbl_shape_crop}")
    print(f"    label values: {lbl_vals_crop}")

    # Check crop offset: compare foreground bbox before and after
    lbl_before = t_pad(data)["label"].numpy().squeeze()
    lbl_after  = lbl_arr_crop

    fg_before = np.argwhere(lbl_before > 0)
    fg_after  = np.argwhere(lbl_after  > 0)

    if fg_before.size > 0 and fg_after.size > 0:
        bbox_before_min = fg_before.min(axis=0)
        bbox_after_min  = fg_after.min(axis=0)
        offset = bbox_after_min - bbox_before_min

        print(f"\n    Foreground bbox origin (before crop): {bbox_before_min}")
        print(f"    Foreground bbox origin (after crop) : {bbox_after_min}")
        print(f"    Offset introduced by CropForegroundd: {offset}")

        if np.any(offset != 0):
            print(f"    ⚠ CropForegroundd shifted the label origin by {offset} voxels")
            print(f"      This will cause label-to-image misalignment after inverse transform")
        else:
            print(f"    ✓ No spatial offset introduced by CropForegroundd")
    else:
        print(f"    ⚠ No foreground voxels found in label — check label file")

    # ── Check shapes are consistent ───────────────────────────────────────────
    print("\n[4] Shape consistency check:")
    print(f"    image == label at every stage: "
          f"{img_shape==lbl_shape}, {img_shape_pad==lbl_shape_pad}, {img_shape_crop==lbl_shape_crop}")

    # ── Interpolation check: verify no fractional label values ────────────────
    print("\n[5] Label interpolation check (full pipeline):")
    all_vals = np.unique(lbl_arr_crop)
    non_int  = all_vals[np.abs(all_vals - np.round(all_vals)) > 1e-6]
    if len(non_int) > 0:
        print(f"    ⚠ FRACTIONAL label values detected: {non_int[:10]}")
        print(f"      Somewhere in the pipeline, bilinear interpolation is being")
        print(f"      applied to the label. Check ALL transforms for mode='nearest'.")
    else:
        print(f"    ✓ All label values are integers: {all_vals}")
        print(f"      Interpolation is correct — overlay mismatch is NOT due to")
        print(f"      interpolation but to a spatial offset or affine mismatch.")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print(f"  If offset != 0 → fix: remove k_divisible from CropForegroundd")
    print(f"                        OR track and invert the crop offset in evaluate.py")
    print(f"  If non-integer labels → fix: ensure mode='nearest' for ALL label transforms")
    print(f"  If both OK → overlay issue is in evaluate.py inverse transform")
    print("="*70 + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True)
    p.add_argument("--label",    required=True)
    p.add_argument("--pix-dim",  nargs=3, type=float, required=True)
    p.add_argument("--roi-size", nargs=3, type=int,   required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check(
        image_path=args.image,
        label_path=args.label,
        pix_dim=tuple(args.pix_dim),
        roi_size=tuple(args.roi_size),
    )