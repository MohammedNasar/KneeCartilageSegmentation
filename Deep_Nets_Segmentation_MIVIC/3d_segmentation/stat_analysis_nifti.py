import os
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes   # or just use `nib.aff2axcodes`
import csv

# Configure your paths and patterns
IMAGE_DIR = r"C:\Users\mnibrahim\Documents\Projects\KneeCartilage\Data\GE_scanner\Phase1_10\images_nifti"           # .nii images
LABEL_DIR = r"C:\Users\mnibrahim\Documents\Projects\KneeCartilage\Data\GE_scanner\Phase1_10\labels"           # corresponding .nii labels
OUT_CSV = "nifti_stats.csv"

# Match images and labels by filename (change pattern as needed)
imagelist = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".nii", ".nii.gz"))]
label_files = {f: os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith((".nii", ".nii.gz"))}

# CSV header
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "is_label",
        "shape",
        "voxel_size",
        "orientation",
        "label_min",
        "label_max",
        "is_label_012",
    ])

    for img_name in sorted(imagelist):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = nib.load(img_path)
        data = np.asarray(img.dataobj)
        shape = list(data.shape)
        zooms = list(img.header.get_zooms()[:3])
        axcodes = aff2axcodes(img.affine)   # or nib.aff2axcodes(img.affine)
        orientation = "".join(axcodes)

        writer.writerow([
            img_name,
            0,
            str(shape),
            str(zooms),
            orientation,
            "", "", ""   # dummy label fields
        ])

        # --- LABEL stats (if present) ---
        if img_name in label_files:
            lbl_path = label_files[img_name]

            lbl = nib.load(lbl_path)
            lbl_data = np.asarray(lbl.dataobj)
            lbl_shape = list(lbl_data.shape)
            lbl_zooms = list(lbl.header.get_zooms()[:3])
            lbl_axcodes = aff2axcodes(lbl.affine)
            lbl_orientation = "".join(lbl_axcodes)

            lbl_min, lbl_max = lbl_data.min(), lbl_data.max()
            is_label_012 = (lbl_min == 0 and lbl_max == 2)

            writer.writerow([
                img_name,
                1,
                str(lbl_shape),
                str(lbl_zooms),
                lbl_orientation,
                lbl_min,
                lbl_max,
                str(is_label_012),
            ])
