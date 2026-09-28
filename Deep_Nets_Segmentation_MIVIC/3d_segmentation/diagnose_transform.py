"""
diagnose_transforms.py
======================
Traces exactly what each val_transforms step does to your image shape and affine.
Run this BEFORE fixing anything — it will tell you the correct pix_dim and roi_size
to use so the model input shape maps cleanly back to the original.

Usage
-----
python diagnose_transforms.py \
    --image  /path/to/image.nii.gz \
    --label  /path/to/label.nii.gz \
    --pix-dim 0.98214287 0.98214287 5.0 \
    --roi-size 96 96 32

Dependencies: monai, nibabel, numpy
"""

import argparse
import numpy as np
import nibabel as nib
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    SpatialPadd,
    CropForegroundd,
    EnsureTyped,
)


def trace_transforms(image_path, label_path, pix_dim, roi_size):

    print("\n" + "="*70)
    print("TRANSFORM TRACE")
    print("="*70)

    # ── Load raw ──────────────────────────────────────────────────────────────
    orig = nib.load(image_path)
    print(f"\n[0] RAW NIfTI")
    print(f"    shape       : {orig.shape}")
    print(f"    orientation : {nib.aff2axcodes(orig.affine)}")
    print(f"    spacing(mm) : {orig.header.get_zooms()[:3]}")
    print(f"    affine diag : {np.diag(orig.affine)[:3].round(4)}")

    # ── Step by step ──────────────────────────────────────────────────────────
    data = {"image": image_path, "label": label_path}

    steps = [
        ("LoadImaged",
         Compose([LoadImaged(keys=["image", "label"], ensure_channel_first=True)])),

        ("Orientationd(RAS)",
         Compose([LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                  Orientationd(keys=["image", "label"], axcodes="RAS")])),

        ("Spacingd",
         Compose([LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                  Orientationd(keys=["image", "label"], axcodes="RAS"),
                  Spacingd(keys=["image", "label"],
                           pixdim=pix_dim,
                           mode=("bilinear", "nearest"))])),

        ("ScaleIntensity + SpatialPad",
         Compose([LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                  Orientationd(keys=["image", "label"], axcodes="RAS"),
                  Spacingd(keys=["image", "label"],
                           pixdim=pix_dim,
                           mode=("bilinear", "nearest")),
                  ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                  SpatialPadd(keys=["image", "label"], spatial_size=roi_size)])),

        ("CropForegroundd",
         Compose([LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                  Orientationd(keys=["image", "label"], axcodes="RAS"),
                  Spacingd(keys=["image", "label"],
                           pixdim=pix_dim,
                           mode=("bilinear", "nearest")),
                  ScaleIntensityd(keys=["image"], minv=0, maxv=1),
                  SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
                  CropForegroundd(keys=["image", "label"], source_key="image",
                                  k_divisible=roi_size)])),
    ]

    prev_shape = None
    for name, transform in steps:
        out = transform(data)
        shape = tuple(out["image"].shape[1:])   # remove channel dim
        changed = shape != prev_shape if prev_shape else True
        marker  = "  ←── CHANGES SHAPE" if changed and prev_shape else ""
        print(f"\n[→] After {name}")
        print(f"    image shape : {shape}{marker}")
        print(f"    label shape : {tuple(out['label'].shape[1:])}")
        if hasattr(out["image"], "affine"):
            print(f"    affine diag : {np.diag(out['image'].affine)[:3].round(4)}")
        prev_shape = shape

    final_shape = shape

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    orig_shape  = orig.shape[:3]
    orig_zooms  = np.array(orig.header.get_zooms()[:3])
    pred_zooms  = np.array(pix_dim)

    # Zoom factors from final inference shape → original shape
    zoom_factors = np.array(orig_shape, dtype=float) / np.array(final_shape, dtype=float)
    print(f"\n  Inference shape    : {final_shape}")
    print(f"  Original shape     : {orig_shape}")
    print(f"  Zoom factors (back): {tuple(round(z,4) for z in zoom_factors)}")

    all_close = np.allclose(zoom_factors, zoom_factors[0], rtol=0.05)
    print(f"  Isotropic zoom?    : {'✓ YES' if all_close else '✗ NO — asymmetric, overlay will fail'}")

    # What pix_dim would make Spacingd a no-op?
    print(f"\n  To make Spacingd a no-op, set:")
    print(f"    pix_dim: [{orig_zooms[0]:.8f}, {orig_zooms[1]:.8f}, {orig_zooms[2]:.8f}]")

    # What roi_size would give a clean crop of the original?
    ras_nib   = nib.as_closest_canonical(orig)
    ras_shape = np.array(ras_nib.shape[:3])

    print(f"\n  Original shape in RAS: {tuple(ras_shape)}")
    print(f"  Suggested roi_size options (must be <= RAS shape, divisible):")
    for divisor in [16, 32, 48, 64, 96]:
        candidate = (ras_shape // divisor) * divisor
        candidate = np.maximum(candidate, divisor)   # at least one block
        print(f"    roi_size divisible by {divisor:2d}: {tuple(candidate)}")

    # Ideal: no Spacingd + roi = full image (sliding window handles it)
    print(f"\n  RECOMMENDED for this dataset:")
    print(f"    pix_dim  : [{orig_zooms[0]:.8f}, {orig_zooms[1]:.8f}, {orig_zooms[2]:.8f}]")
    best = (ras_shape // 32) * 32
    best = np.maximum(best, 32)
    print(f"    roi_size : {tuple(best)}   (divisible by 32, fits in image)")
    print(f"\n  With these settings, zoom factors become:")
    spacingd_shape = np.round(ras_shape * orig_zooms / orig_zooms).astype(int)  # = ras_shape (no-op)
    zf = np.array(orig_shape, dtype=float) / spacingd_shape
    print(f"    {tuple(round(z,4) for z in zf)}  (should be 1.0 or very close)")
    print("="*70 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Trace val_transforms shape changes step by step.")
    p.add_argument("--image",    required=True, help="Path to image NIfTI")
    p.add_argument("--label",    required=True, help="Path to label NIfTI")
    p.add_argument("--pix-dim",  nargs=3, type=float, required=True,
                   metavar=("X", "Y", "Z"), help="pix_dim from train_params.yaml")
    p.add_argument("--roi-size", nargs=3, type=int, required=True,
                   metavar=("X", "Y", "Z"), help="roi_size from train_params.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trace_transforms(
        image_path=args.image,
        label_path=args.label,
        pix_dim=tuple(args.pix_dim),
        roi_size=tuple(args.roi_size),
    )
