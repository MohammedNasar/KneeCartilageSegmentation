"""
evaluate.py
===========
Evaluates a segmentation model and saves predictions in the ORIGINAL image space.

Correct inverse transform strategy
------------------------------------
val_transforms applies:
    Orientationd(RAS)          → may flip/permute axes (e.g. LAS→RAS flips X)
    Spacingd(pix_dim)          → resamples voxel spacing
    SpatialPadd(roi_size)      → pads to minimum roi_size
    CropForegroundd(k_divisible=roi_size) → crops to foreground bbox

When pix_dim == native spacing (recommended), Spacingd is a no-op and
CropForegroundd only changes shape by the foreground crop, not by resampling.

Inverse pipeline:
    1. Resample pred (inference shape) → RAS original shape  [order=0 zoom]
    2. Wrap in NIfTI with canonical RAS affine
    3. Reorient RAS NIfTI → original orientation via nibabel  [handles flips]
    4. Save with original affine + header

NOTE: When pix_dim matches native spacing, zoom factors should be very close
to (1.0, 1.0, 1.0) and the resample step is nearly a no-op. The reorient
step (e.g. RAS→LAS X-flip) is then the only meaningful transform being inverted.
"""

import numpy as np
import nibabel as nib
import nibabel.orientations as nio
import torch
import os
from scipy.ndimage import zoom

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch


# ──────────────────────────────────────────────────────────────────────────────
# Inverse transform
# ──────────────────────────────────────────────────────────────────────────────

def pred_to_original_space(label_array, pred_spacing, original_nifti):
    """
    Invert Orientationd(RAS) + Spacingd for a segmentation prediction.

    Parameters
    ----------
    label_array    : (H,W,D) int ndarray — argmax output, in RAS resampled space
    pred_spacing   : voxel spacing at inference (from train_params['pix_dim'])
    original_nifti : original NIfTI from disk

    Returns
    -------
    uint8 ndarray, shape == original_nifti.shape[:3], in original orientation
    """
    orig_shape = original_nifti.shape[:3]
    orig_zooms = np.array(original_nifti.header.get_zooms()[:3])
    pred_sp    = np.array(pred_spacing)

    # ── RAS shape: what Orientationd produces before Spacingd ─────────────────
    ras_nib        = nib.as_closest_canonical(original_nifti)
    orig_ras_shape = ras_nib.shape[:3]
    ras_affine     = ras_nib.affine

    print(f"\n    Original orientation : {nib.aff2axcodes(original_nifti.affine)}")
    print(f"    Original shape       : {orig_shape}  @ {tuple(orig_zooms.round(4))} mm")
    print(f"    RAS shape            : {orig_ras_shape}")

    # ── Zoom factors: inference grid → RAS original grid ─────────────────────
    source       = np.array(label_array.shape[:3], dtype=float)
    target       = np.array(orig_ras_shape, dtype=float)
    zoom_factors = target / source

    print(f"    [resample] pred {tuple(label_array.shape)} @ {tuple(pred_sp)} mm"
          f"  →  RAS {orig_ras_shape} @ {tuple(orig_zooms.round(4))} mm")
    print(f"    [resample] zoom factors: {tuple(round(z,4) for z in zoom_factors)}")

    near_unity = np.allclose(zoom_factors, 1.0, atol=0.05)
    if near_unity:
        print(f"    [resample] ✓ zoom ≈ 1.0 — pix_dim matches native spacing, "
              f"resample is near no-op")
    else:
        print(f"    [resample] ⚠ zoom factors differ from 1.0 — "
              f"pix_dim does not match native spacing")

    # order=0: nearest-neighbour — preserves integer labels exactly
    resampled_ras = zoom(label_array.astype(np.float32), zoom_factors,
                         order=0, prefilter=False).astype(np.uint8)

    # Hard-clamp (float rounding in zoom can produce shape ±1)
    slices = tuple(slice(0, int(s)) for s in orig_ras_shape)
    resampled_ras = resampled_ras[slices]

    # Pad if undersized (rare)
    pad_widths = [(0, max(0, int(orig_ras_shape[i]) - resampled_ras.shape[i]))
                  for i in range(3)]
    if any(p[1] > 0 for p in pad_widths):
        resampled_ras = np.pad(resampled_ras, pad_widths,
                               mode='constant', constant_values=0)

    # ── Wrap in NIfTI with canonical RAS affine ───────────────────────────────
    # This tells nibabel the exact world-space meaning of each voxel in RAS.
    pred_ras_nib = nib.Nifti1Image(resampled_ras, ras_affine)

    # ── Reorient: RAS → original orientation ──────────────────────────────────
    # nib.as_reoriented() correctly handles BOTH axis permutations AND flips
    # (e.g. the X-axis flip when going RAS→LAS), which a raw numpy flip cannot.
    orig_ornt = nio.io_orientation(original_nifti.affine)
    ras_ornt  = nio.io_orientation(ras_affine)
    transform = nio.ornt_transform(ras_ornt, orig_ornt)

    print(f"    [reorient] RAS → {nib.aff2axcodes(original_nifti.affine)}")
    print(f"    [reorient] ornt_transform:\n{transform}")

    pred_orig_nib = pred_ras_nib.as_reoriented(transform)
    result        = np.asarray(pred_orig_nib.dataobj).astype(np.uint8)

    # ── Sanity check ──────────────────────────────────────────────────────────
    if result.shape != tuple(orig_shape):
        raise RuntimeError(
            f"Shape after inverse transform {result.shape} != "
            f"expected original {tuple(orig_shape)}.\n"
            f"  RAS shape   : {orig_ras_shape}\n"
            f"  transform   :\n{transform}\n"
            f"  Check that pix_dim in train_params.yaml matches the dataset's "
            f"native voxel spacing."
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ──────────────────────────────────────────────────────────────────────────────

def eval_pred_images(
    root_dir,
    model,
    post_label,
    post_pred,
    my_data_loader,
    segs_dset,
    dice_metric,
    train_params,
):
    pred_spacing = tuple(train_params['pix_dim'])

    print("\n" + "="*60)
    print("EVALUATION — predictions saved in original image space")
    print(f"  pix_dim (inference spacing): {pred_spacing}")
    print("="*60)

    with torch.no_grad():
        for case_num, batch in enumerate(my_data_loader):

            val_inputs = batch["image"].cuda()
            val_labels = batch["label"].cuda()

            seg_path       = segs_dset[case_num]
            original_nifti = nib.load(seg_path)

            filename = os.path.splitext(os.path.basename(seg_path))[0]
            if filename.endswith('.nii'):
                filename = os.path.splitext(filename)[0]

            print(f"\n{'─'*60}")
            print(f"Case {case_num + 1}/{len(my_data_loader)}: {filename}")
            print(f"  Inference input shape  : {val_inputs.shape}")
            print(f"  Original NIfTI shape   : {original_nifti.shape}")
            print(f"  Original orientation   : {nib.aff2axcodes(original_nifti.affine)}")
            print(f"  Original spacing (mm)  : {original_nifti.header.get_zooms()[:3]}")

            # ── Sliding window inference ──────────────────────────────────────
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=train_params['roi_size'],
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode='gaussian',
                device=torch.device('cuda'),
            )

            # ── Argmax → integer label map (H, W, D) ──────────────────────────
            label_image = (
                torch.argmax(val_outputs, dim=1)
                .detach().cpu().numpy()
                .astype(np.int16)
            )
            if label_image.ndim == 4 and label_image.shape[0] == 1:
                label_image = label_image[0]

            print(f"  Raw prediction shape   : {label_image.shape}")
            print(f"  Unique labels          : {np.unique(label_image)}")

            # ── Invert preprocessing: Spacingd + Orientationd ─────────────────
            label_original = pred_to_original_space(
                label_image,
                pred_spacing,
                original_nifti,
            )

            nonzero_pct = 100.0 * np.count_nonzero(label_original) / label_original.size
            match       = label_original.shape == original_nifti.shape[:3]
            print(f"  Restored shape         : {label_original.shape}  "
                  f"({'✓ matches' if match else '⚠ MISMATCH — check pix_dim'})")
            print(f"  Unique labels (final)  : {np.unique(label_original)}")
            print(f"  Non-zero voxels        : {np.count_nonzero(label_original):,}  "
                  f"({nonzero_pct:.2f}%)")

            if nonzero_pct < 0.01:
                print(f"  ⚠  Very sparse mask — verify pix_dim matches training config")

            # ── Save with original affine + header ────────────────────────────
            out_header = original_nifti.header.copy()
            out_header.set_data_dtype(np.uint8)
            out_header.set_data_shape(label_original.shape)

            nifti_out   = nib.Nifti1Image(label_original, original_nifti.affine, out_header)
            output_path = os.path.join(root_dir, f"{filename}_pred.nii.gz")
            nib.save(nifti_out, output_path)

            print(f"  ✓ Saved : {os.path.basename(output_path)}")
            print(f"  Spacing : {out_header.get_zooms()[:3]}")

            # ── Dice (in inference space — consistent with training metric) ───
            val_labels_list  = decollate_batch(val_labels)
            val_outputs_list = decollate_batch(val_outputs)
            val_labels_conv  = [post_label(t) for t in val_labels_list]
            val_outputs_conv = [post_pred(t)  for t in val_outputs_list]
            dice_metric(y_pred=val_outputs_conv, y=val_labels_conv)

    val_dice      = dice_metric.get_buffer().cpu().numpy()
    mean_val_dice = dice_metric.aggregate().item()

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Average Dice Score : {mean_val_dice:.4f}")
    print(f"Predictions saved  : {root_dir}")
    print(f"Cases processed    : {case_num + 1}")
    print("="*60 + "\n")

    return val_dice

# """
# evaluate.py
# ===========
# Evaluates a segmentation model and saves predictions in the ORIGINAL image space.

# Root cause of orientation mismatch
# ------------------------------------
# val_transforms applies Orientationd(axcodes="RAS").
# Your originals are LAS. Even though LAS and RAS have the same shape (640,640,224),
# the X-axis is FLIPPED (L↔R). A naive axis-permutation without tracking the flip
# sign produces a mirror image of the correct segmentation.

# Correct inverse strategy
# -------------------------
# Instead of manually tracking flips, we:
# 1. Build a temporary NIfTI in RAS space using the canonical (RAS) affine.
# 2. Resample prediction → RAS grid shape (scipy zoom, order=0).
# 3. Place the resampled data into that RAS NIfTI.
# 4. Use nibabel to reorient the NIfTI back to the original orientation — this
#    correctly handles BOTH axis permutations AND axis flips.
# 5. Extract the data array and save with the original affine + header.

# This is equivalent to what MONAI's Orientationd inverse does internally.
# """

# import numpy as np
# import nibabel as nib
# import nibabel.orientations as nio
# import torch
# import os
# from scipy.ndimage import zoom

# from monai.inferers import sliding_window_inference
# from monai.data import decollate_batch


# # ──────────────────────────────────────────────────────────────────────────────
# # Core inverse transform
# # ──────────────────────────────────────────────────────────────────────────────

# def pred_to_original_space(label_array, pred_spacing, original_nifti):
#     """
#     Invert the full val_transforms preprocessing (Orientationd RAS + Spacingd)
#     for a segmentation prediction array.

#     Strategy
#     --------
#     We build a proper NIfTI in RAS space for the prediction, then use nibabel's
#     reorientation machinery to convert it back to the original orientation.
#     This correctly handles axis flips (e.g. L↔R) as well as axis permutations.

#     Parameters
#     ----------
#     label_array    : (H, W, D) int ndarray — argmax output from model (in RAS space)
#     pred_spacing   : spacing used at inference, e.g. (1.0, 1.0, 1.0)
#     original_nifti : original NIfTI from disk (provides target space)

#     Returns
#     -------
#     uint8 ndarray, shape == original_nifti.shape[:3], in original orientation
#     """
#     orig_shape = original_nifti.shape[:3]
#     orig_zooms = original_nifti.header.get_zooms()[:3]

#     # ── 1. Get the canonical (RAS) version of the original image ─────────────
#     #    This is what Orientationd(RAS) would produce.
#     #    Its affine is the RAS affine; its shape is the RAS shape.
#     ras_nib        = nib.as_closest_canonical(original_nifti)
#     orig_ras_shape = ras_nib.shape[:3]
#     ras_affine     = ras_nib.affine

#     print(f"\n    Original orientation : {nib.aff2axcodes(original_nifti.affine)}")
#     print(f"    Original shape       : {orig_shape}")
#     print(f"    RAS shape            : {orig_ras_shape}")
#     print(f"    RAS affine diagonal  : {np.diag(ras_affine)[:3].round(4)}")

#     # ── 2. Resample prediction → RAS grid (nearest-neighbour) ─────────────────
#     source       = np.array(label_array.shape[:3], dtype=float)
#     target       = np.array(orig_ras_shape, dtype=float)
#     zoom_factors = target / source

#     print(f"    [resample] {tuple(label_array.shape)} @ {pred_spacing}mm "
#           f"→ {orig_ras_shape} @ {tuple(round(z,4) for z in orig_zooms)}mm")
#     print(f"    [resample] zoom factors: {tuple(round(z,4) for z in zoom_factors)}")

#     resampled_ras = zoom(label_array.astype(np.float32), zoom_factors,
#                          order=0, prefilter=False).astype(np.uint8)

#     # Hard-clamp to target shape (float rounding in zoom can give shape ±1)
#     slices = tuple(slice(0, int(s)) for s in orig_ras_shape)
#     resampled_ras = resampled_ras[slices]

#     # Pad if undersized (rare)
#     pad_widths = [(0, max(0, int(orig_ras_shape[i]) - resampled_ras.shape[i]))
#                   for i in range(3)]
#     if any(p[1] > 0 for p in pad_widths):
#         resampled_ras = np.pad(resampled_ras, pad_widths,
#                                mode='constant', constant_values=0)

#     # ── 3. Wrap resampled data in a NIfTI using the RAS affine ────────────────
#     #    This is the key step: we tell nibabel "this array IS in RAS space"
#     #    by giving it the canonical affine.
#     pred_ras_nib = nib.Nifti1Image(resampled_ras, ras_affine)

#     # ── 4. Reorient the NIfTI from RAS → original orientation ─────────────────
#     #    nib.as_reoriented() applies the correct axis permutation AND flip signs.
#     orig_ornt   = nio.io_orientation(original_nifti.affine)
#     ras_ornt    = nio.io_orientation(ras_affine)
#     transform   = nio.ornt_transform(ras_ornt, orig_ornt)

#     print(f"    [reorient] RAS → {nib.aff2axcodes(original_nifti.affine)}")
#     print(f"    [reorient] ornt_transform:\n{transform}")

#     pred_orig_nib = pred_ras_nib.as_reoriented(transform)
#     result        = np.asarray(pred_orig_nib.dataobj).astype(np.uint8)

#     # ── 5. Sanity check ────────────────────────────────────────────────────────
#     if result.shape != tuple(orig_shape):
#         raise RuntimeError(
#             f"Shape after inverse transform {result.shape} "
#             f"!= expected {tuple(orig_shape)}.\n"
#             f"  RAS shape was: {orig_ras_shape}\n"
#             f"  transform:\n{transform}"
#         )

#     return result


# # ──────────────────────────────────────────────────────────────────────────────
# # Main evaluation function
# # ──────────────────────────────────────────────────────────────────────────────

# def eval_pred_images(
#     root_dir,
#     model,
#     post_label,
#     post_pred,
#     my_data_loader,
#     segs_dset,
#     dice_metric,
#     train_params,
# ):
#     pred_spacing = tuple(train_params['pix_dim'])   # e.g. (1.0, 1.0, 1.0)

#     print("\n" + "="*60)
#     print("EVALUATION — predictions saved in original image space")
#     print("  (inverting Orientationd RAS + Spacingd, including axis flips)")
#     print("="*60)

#     with torch.no_grad():
#         for case_num, batch in enumerate(my_data_loader):

#             val_inputs = batch["image"].cuda()
#             val_labels = batch["label"].cuda()

#             seg_path       = segs_dset[case_num]
#             original_nifti = nib.load(seg_path)

#             filename = os.path.splitext(os.path.basename(seg_path))[0]
#             if filename.endswith('.nii'):
#                 filename = os.path.splitext(filename)[0]

#             print(f"\n{'─'*60}")
#             print(f"Case {case_num + 1}/{len(my_data_loader)}: {filename}")
#             print(f"  Inference input shape  : {val_inputs.shape}")
#             print(f"  Original NIfTI shape   : {original_nifti.shape}")
#             print(f"  Original orientation   : {nib.aff2axcodes(original_nifti.affine)}")
#             print(f"  Original spacing (mm)  : {original_nifti.header.get_zooms()[:3]}")

#             # Sliding window inference
#             val_outputs = sliding_window_inference(
#                 inputs=val_inputs,
#                 roi_size=train_params['roi_size'],
#                 sw_batch_size=4,
#                 predictor=model,
#                 overlap=0.5,
#                 mode='gaussian',
#                 device=torch.device('cuda'),
#             )

#             # Argmax → integer label map (H, W, D)
#             label_image = (
#                 torch.argmax(val_outputs, dim=1)
#                 .detach().cpu().numpy()
#                 .astype(np.int16)
#             )
#             if label_image.ndim == 4 and label_image.shape[0] == 1:
#                 label_image = label_image[0]

#             print(f"  Raw prediction shape   : {label_image.shape}")
#             print(f"  Unique labels          : {np.unique(label_image)}")

#             # Invert full preprocessing transform
#             label_original = pred_to_original_space(
#                 label_image,
#                 pred_spacing,
#                 original_nifti,
#             )

#             nonzero_pct = 100.0 * np.count_nonzero(label_original) / label_original.size
#             match       = label_original.shape == original_nifti.shape[:3]
#             print(f"  Restored shape         : {label_original.shape}  "
#                   f"({'✓ matches' if match else '⚠ MISMATCH'})")
#             print(f"  Unique labels (final)  : {np.unique(label_original)}")
#             print(f"  Non-zero voxels        : {np.count_nonzero(label_original):,}  "
#                   f"({nonzero_pct:.2f}%)")

#             # Save with original affine + header
#             out_header = original_nifti.header.copy()
#             out_header.set_data_dtype(np.uint8)
#             out_header.set_data_shape(label_original.shape)

#             nifti_out   = nib.Nifti1Image(label_original, original_nifti.affine, out_header)
#             output_path = os.path.join(root_dir, f"{filename}_pred.nii.gz")
#             nib.save(nifti_out, output_path)

#             print(f"  ✓ Saved : {os.path.basename(output_path)}")
#             print(f"  Spacing : {out_header.get_zooms()[:3]}")

#             # Dice computed in inference space
#             val_labels_list  = decollate_batch(val_labels)
#             val_outputs_list = decollate_batch(val_outputs)
#             val_labels_conv  = [post_label(t) for t in val_labels_list]
#             val_outputs_conv = [post_pred(t)  for t in val_outputs_list]
#             dice_metric(y_pred=val_outputs_conv, y=val_labels_conv)

#     val_dice      = dice_metric.get_buffer().cpu().numpy()
#     mean_val_dice = dice_metric.aggregate().item()

#     print("\n" + "="*60)
#     print("EVALUATION COMPLETE")
#     print("="*60)
#     print(f"Average Dice Score : {mean_val_dice:.4f}")
#     print(f"Predictions saved  : {root_dir}")
#     print(f"Cases processed    : {case_num + 1}")
#     print("="*60 + "\n")

#     return val_dice


# import numpy as np
# import nibabel as nib
# import torch
# import os
# from monai.inferers import sliding_window_inference
# from monai.data import decollate_batch


# def eval_pred_images(root_dir, model, post_label, post_pred, my_data_loader, 
#                      segs_dset, dice_metric, train_params):
#     """
#     Evaluate and save predictions at FULL RESOLUTION.
#     Uses sliding window inference on complete images.
#     """
    
#     print("\n" + "="*60)
#     print("EVALUATION WITH FULL-SIZE PREDICTIONS")
#     print("="*60)
    
#     with torch.no_grad():
#         for case_num, batch in enumerate(my_data_loader):
#             # ==========================================
#             # Get full-resolution inputs
#             # ==========================================
#             val_inputs = batch["image"].cuda()
#             val_labels = batch["label"].cuda()
            
#             # Load original file for metadata
#             seg_path = segs_dset[case_num]
#             original_nifti = nib.load(seg_path)
#             affine = original_nifti.affine
#             header = original_nifti.header.copy()
            
#             filename = os.path.splitext(os.path.basename(seg_path))[0]
#             if filename.endswith('.nii'):
#                 filename = os.path.splitext(filename)[0]
            
#             print(f"\n{'─'*60}")
#             print(f"Case {case_num + 1}/{len(my_data_loader)}: {filename}")
#             print(f"  Input shape: {val_inputs.shape}")
#             print(f"  Original shape: {original_nifti.shape}")
            
#             # ==========================================
#             # SLIDING WINDOW INFERENCE on FULL image
#             # ==========================================
#             val_outputs = sliding_window_inference(
#                 inputs=val_inputs,
#                 roi_size=train_params['roi_size'],
#                 sw_batch_size=4,  # Process 4 patches simultaneously
#                 predictor=model,
#                 overlap=0.5,  # 50% overlap for smooth predictions
#                 mode='gaussian',  # Gaussian weighting in overlap regions
#                 device=torch.device('cuda')
#             )
            
#             # Convert to segmentation map
#             label_image = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().astype(np.int16)
#             if label_image.shape[0] == 1:
#                 label_image = np.squeeze(label_image, axis=0)
            
#             print(f"  Prediction shape: {label_image.shape}")
            
#             # ==========================================
#             # Verify shape matches
#             # ==========================================
#             if label_image.shape != original_nifti.shape:
#                 print(f"  ⚠ WARNING: Shape mismatch!")
#                 print(f"    Expected: {original_nifti.shape}")
#                 print(f"    Got: {label_image.shape}")
#             else:
#                 print(f"  ✓ Shape matches original")
            
#             # ==========================================
#             # Save with correct metadata
#             # ==========================================
#             nifti_image = nib.Nifti1Image(label_image, affine, header)
#             nifti_image.set_data_dtype(np.int16)
            
#             output_path = os.path.join(root_dir, f"{filename}_pred.nii.gz")
#             nib.save(nifti_image, output_path)
            
#             print(f"  ✓ Saved: {os.path.basename(output_path)}")
#             print(f"  Spacing: {header.get_zooms()[:3]}")
            
#             # ==========================================
#             # Compute Dice metrics
#             # ==========================================
#             val_labels_list = decollate_batch(val_labels)
#             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             val_outputs_list = decollate_batch(val_outputs)
#             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    
#     # ==========================================
#     # Final results
#     # ==========================================
#     val_dice = dice_metric.get_buffer().cpu().numpy()
#     mean_val_dice = dice_metric.aggregate().item()
    
#     print("\n" + "="*60)
#     print("EVALUATION COMPLETE")
#     print("="*60)
#     print(f"Average Dice Score: {mean_val_dice:.4f}")
#     print(f"Predictions saved to: {root_dir}")
#     print(f"Total cases processed: {case_num + 1}")
#     print("="*60 + "\n")

#     return val_dice

# import numpy as np
# import nibabel as nib
# import torch
# import os
# from monai.inferers import sliding_window_inference
# from monai.data import (
#     decollate_batch,
# )


# def eval_pred_images(root_dir, model, post_label, post_pred, my_data_loader, 
#                      segs_dset, dice_metric, train_params):
#     """
#     Evaluate on FULL resolution images using sliding window.
#     """
    
#     with torch.no_grad():
#         for case_num, batch in enumerate(my_data_loader):
#             # ==========================================
#             # Load FULL resolution image
#             # ==========================================
#             seg_path = segs_dset[case_num]
#             original_nifti = nib.load(seg_path)
            
#             # Get full resolution input
#             # (Assuming your dataloader already has it in batch["image"])
#             val_inputs = batch["image"].cuda()  # Full size: [1, 1, 640, 640, 224]
#             val_labels = batch["label"].cuda()
            
#             affine = original_nifti.affine
#             header = original_nifti.header.copy()
#             filename = os.path.splitext(os.path.basename(seg_path))[0]
            
#             print(f"\nProcessing: {filename}")
#             print(f"  Input shape: {val_inputs.shape}")
            
#             # ==========================================
#             # Sliding window on FULL image
#             # ==========================================
#             val_outputs = sliding_window_inference(
#                 inputs=val_inputs,
#                 roi_size=train_params['roi_size'],  # e.g., [96, 96, 96]
#                 sw_batch_size=4,  # Process 4 windows at once
#                 predictor=model,
#                 overlap=0.5,  # 50% overlap for smooth predictions
#                 mode='gaussian',  # Gaussian weighting for overlap regions
#                 device=torch.device('cuda')
#             )
            
#             # Convert to segmentation
#             label_image = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().astype(np.int16)
#             if label_image.shape[0] == 1:
#                 label_image = np.squeeze(label_image, axis=0)
            
#             print(f"  Output shape: {label_image.shape}")
            
#             # ==========================================
#             # Save with metadata
#             # ==========================================
#             nifti_image = nib.Nifti1Image(label_image, affine, header)
#             output_path = os.path.join(root_dir, f"{filename}_pred.nii.gz")
#             nib.save(nifti_image, output_path)
            
#             print(f"  ✓ Saved: {output_path}")
            
#             # Compute metrics
#             val_labels_list = decollate_batch(val_labels)
#             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             val_outputs_list = decollate_batch(val_outputs)
#             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    
#     val_dice = dice_metric.get_buffer().cpu().numpy()
#     mean_val_dice = dice_metric.aggregate().item()
#     print(f"\nAverage Dice: {mean_val_dice:.4f}")

#     return val_dice
# # def eval_pred_images(root_dir, model, post_label, post_pred, my_data_loader, 
# #                      images_dset, segs_dset, dice_metric, train_params):
# #     """
# #     Evaluate and save predictions matching original input images.
    
# #     Args:
# #         images_dset: List of paths to original input images
# #         segs_dset: List of paths to ground truth segmentations
# #     """
    
# #     with torch.no_grad():
# #         for case_num, batch in enumerate(my_data_loader):
# #             val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            
# #             # Sliding window inference
# #             val_outputs = sliding_window_inference(
# #                 val_inputs, 
# #                 train_params['roi_size'], 
# #                 1, 
# #                 model
# #             )
            
# #             # ==========================================
# #             # Load metadata from ORIGINAL INPUT IMAGE
# #             # ==========================================
# #             input_image_path = images_dset[case_num]
# #             original_nifti = nib.load(input_image_path)
            
# #             affine = original_nifti.affine
# #             header = original_nifti.header.copy()
            
# #             # Get base filename
# #             filename = os.path.splitext(os.path.basename(input_image_path))[0]
            
# #             # Convert prediction
# #             label_image = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().astype(np.int16)
# #             if label_image.shape[0] == 1:
# #                 label_image = np.squeeze(label_image, axis=0)
            
# #             # ==========================================
# #             # Create NIfTI with matching metadata
# #             # ==========================================
# #             nifti_image = nib.Nifti1Image(label_image, affine, header)
            
# #             # Set data type in header
# #             nifti_image.set_data_dtype(np.int16)
            
# #             output_path = os.path.join(root_dir, f"{filename}_pred.nii.gz")
# #             nib.save(nifti_image, output_path)
            
# #             # Print verification
# #             print(f"\n{'='*60}")
# #             print(f"Case {case_num + 1}:")
# #             print(f"  Input:  {os.path.basename(input_image_path)}")
# #             print(f"  Output: {os.path.basename(output_path)}")
# #             print(f"  Shape:  {label_image.shape}")
# #             print(f"  Spacing: {header.get_zooms()[:3]}")
# #             print(f"{'='*60}")
            
# #             # Compute Dice
# #             val_labels_list = decollate_batch(val_labels)
# #             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
# #             val_outputs_list = decollate_batch(val_outputs)
# #             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
# #             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            
# #     val_dice = dice_metric.get_buffer().cpu().numpy()
# #     mean_val_dice = dice_metric.aggregate().item()
# #     print(f"\nAverage Dice: {mean_val_dice:.4f}")

# #     return val_dice

# # def eval_pred_images(root_dir, model, post_label, post_pred, my_data_loader, 
# #                      images_dset, segs_dset, dice_metric, train_params):
# #     """
# #     Evaluate predicted images using a given model and validation data loader, and compute the Dice metric.

# #     Args:
# #         root_dir (str): The root directory where the predicted segmentation results will be saved.
# #         model (torch.nn.Module): The trained model used for inference.
# #         my_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
# #         segs (dict): Dictionary containing paths to the segmentation files.
# #         dice_metric (monai.metrics.DiceMetric): Dice metric object to compute the Dice score.

# #     Returns:
# #         numpy.ndarray: Array containing the Dice scores for the validation dataset.
# #     """

# #     # Produce and display segmentation results.
# #     # affine = np.eye(4)

# #     """
# #     Evaluate and save predictions with metadata from MONAI batch.
# #     """

    

# #     with torch.no_grad():
# #         for case_num, batch in enumerate(my_data_loader):
# #             val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
# #             # val_inputs = torch.unsqueeze(img, 1).cuda()
# #             # val_labels = torch.unsqueeze(label, 1).cuda()
# #             original_nifti = nib.load(segs_dset[case_num])
# #             affine = original_nifti.affine
# #             header = original_nifti.header
# #             val_outputs = sliding_window_inference(val_inputs, train_params['roi_size'], 1, model)
# #             filename = os.path.splitext(os.path.basename(segs_dset[case_num]))[0]
# #             label_image = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().astype(np.int16)
# #             if label_image.shape[0] == 1:
# #                 label_image = np.squeeze(label_image, axis=0)
# #             nifti_image = nib.Nifti1Image(label_image, affine, header)
# #             nib.save(nifti_image, os.path.join(root_dir, f"{filename[0]}_pred.nii"))
# #             val_labels_list = decollate_batch(val_labels)
# #             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
# #             val_outputs_list = decollate_batch(val_outputs)
# #             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
# #             dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    
# #     val_dice = dice_metric.get_buffer().cpu().numpy()
# #     mean_val_dice = dice_metric.aggregate().item()
# #     print(f"Average Dice: {mean_val_dice}")

# #     return val_dice


# # def eval_pred_images(root_dir, model, post_label, post_pred, my_data_loader,
# #                      segs_dset, dice_metric, train_params):

# #     with torch.no_grad():
# #         for case_num, batch in enumerate(my_data_loader):
# #             val_inputs = batch["image"].cuda()
# #             val_labels = batch["label"].cuda()

# #             val_outputs = sliding_window_inference(
# #                 val_inputs, train_params['roi_size'], 1, model
# #             )

# #             filename = os.path.splitext(os.path.basename(segs_dset[case_num]))[0]

# #             label_image = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().astype(np.uint8)
# #             if label_image.shape[0] == 1:
# #                 label_image = np.squeeze(label_image, axis=0)

# #             meta = batch["image_meta_dict"]
# #             if "original_affine" in meta:
# #                 affine = meta["original_affine"][0].cpu().numpy()
# #             else:
# #                 affine = meta["affine"][0].cpu().numpy()

# #             nifti_image = nib.Nifti1Image(label_image, affine)
# #             out_path = os.path.join(root_dir, f"{filename}_pred.nii.gz")
# #             nib.save(nifti_image, out_path)

# #             # label_image = torch.argmax(val_outputs, dim=1).detach().cpu().numpy().astype(np.uint8)
# #             # if label_image.shape[0] == 1:
# #             #     label_image = np.squeeze(label_image, axis=0)

# #             # if hasattr(batch["image"], "meta") and batch["image"].meta is not None:
# #             #     meta = batch["image"].meta
# #             #     if "original_affine" in meta:
# #             #         affine = meta["original_affine"][0].cpu().numpy()
# #             #     elif "affine" in meta:
# #             #         affine = meta["affine"][0].cpu().numpy()
# #             #     else:
# #             #         affine = np.eye(4)
# #             # else:
# #             #     affine = np.eye(4)

# #             # nifti_image = nib.Nifti1Image(label_image, affine)
# #             # nib.save(nifti_image, os.path.join(root_dir, f"{filename}_pred.nii.gz"))

# #             val_labels_list = decollate_batch(val_labels)
# #             val_labels_convert = [post_label(v) for v in val_labels_list]
# #             val_outputs_list = decollate_batch(val_outputs)
# #             val_output_convert = [post_pred(v) for v in val_outputs_list]
# #             dice_metric(y_pred=val_output_convert, y=val_labels_convert)

# #     mean_val_dice = dice_metric.aggregate().item()
# #     print(f"Average Dice: {mean_val_dice}")
# #     return dice_metric.get_buffer().cpu().numpy()