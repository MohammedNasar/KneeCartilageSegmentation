# -*- coding: utf-8 -*-
"""
@author: mivic lab
"""

import argparse
import os, glob
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import pandas as pd
import torch
# from monai.config import print_config
# from monai.handlers import write_metrics_reports
from monai.metrics import (DiceMetric,HausdorffDistanceMetric,SurfaceDiceMetric)
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    KeepLargestConnectedComponentd,
    LoadImaged,
    ScaleIntensityd,
    ToDeviced,
)

"""
Usage example:
python ~/Codes/src/gitrepos/ShapeModelingMIVIC/Scripts/PerformanceEvaluationMetrics.py --ref_dir ../MPUnet/data/test/labels/ --pred_file_patt _From_AtlasList5_VXM_fold1v2_JLF.nii --ref_file_patt _crop.nii
"""


# print_config()

# parse commandline args
parser = argparse.ArgumentParser()

parser.add_argument('--pred_file_patt', required=True, type=str, help='filename string pattern for prediction files')
parser.add_argument('--ref_file_patt', required=True, type=str, help='filename string pattern for reference files')
parser.add_argument('--pred_dir', type=str, default='./', help='predictions folder')
parser.add_argument('--ref_dir', type=str, default='./', help='references folder')
parser.add_argument('--n_labels', default=9, type=int, help='number of tissue labels')

args = parser.parse_args()

# Retrieve file lists of predictions and references.
pred_files = sorted(glob.glob(os.path.join(args.pred_dir, '*'+args.pred_file_patt)))
ref_files = sorted(glob.glob(os.path.join(args.ref_dir, '*'+args.ref_file_patt)))
n_files = len(pred_files)

datalist = [{"pred": pred, "label": label} for pred, label in zip(pred_files, ref_files)]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device) 

make_n_class = Compose([
    LoadImaged(keys=["pred", "label"]),
    ToDeviced(keys=["pred", "label"], device=device),
    EnsureChannelFirstd(keys=["pred", "label"]),
    AsDiscreted(keys=["pred", "label"],
    to_onehot=args.n_labels)])

data_processed =  [make_n_class(item) for item in datalist]

# Compute metrics
dsc_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=False)
hd_metric = HausdorffDistanceMetric(include_background=False, reduction="none", get_not_nans=False, percentile=95)
sd_metric = SurfaceDiceMetric([1.414,1.414,1.414,1.414,1.414,1.414,1.414,1.414], include_background=False, reduction="none", get_not_nans=False)

dsc_metric(y_pred=[i["pred"] for i in data_processed], y=[i["label"] for i in data_processed])
hd_metric(y_pred=[i["pred"] for i in data_processed], y=[i["label"] for i in data_processed])
sd_metric(y_pred=[i["pred"] for i in data_processed], y=[i["label"] for i in data_processed])


# all-gather results from all the processes and reduce for final result
dsc_result = dsc_metric.aggregate()
hd_result = hd_metric.aggregate()
sd_result = sd_metric.aggregate()

print("mean dice: ", dsc_result.cpu())
print("mean hd: ", hd_result.cpu())
print("mean sd: ", sd_result.cpu())

# breakpoint()

# Write results out to a file.
# generate metrics reports at: output/mean_dice_raw.csv, output/mean_dice_summary.csv, output/metrics.csv
# write_metrics_reports(
#     save_dir="./output",
#     images=pred_files,
#     metrics={"mean_dice": dsc_result},
#     metric_details={"mean_dice": dsc_result},
#     summary_ops="*",
# )

# write_metrics_reports(
#     save_dir="./output",
#     images=pred_files,
#     metrics={"mean_hd": hd_result},
#     metric_details={"mean_hd": hd_result},
#     summary_ops="*",
# )


dice_results_df = pd.DataFrame(
    data = dsc_result.cpu(),
    index = pred_files)
dice_results_df.to_csv('DSC_Results.csv')

hd_results_df = pd.DataFrame(
    data = hd_result.cpu(),
    index = pred_files)
hd_results_df.to_csv('HD_Results.csv')

sd_results_df = pd.DataFrame(
    data = sd_result.cpu(),
    index = pred_files)
sd_results_df.to_csv('SD_Results.csv')


