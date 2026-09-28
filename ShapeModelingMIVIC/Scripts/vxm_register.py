#!/usr/bin/env python
"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.h5 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

Copyright 2020 Adrian V. Dalca
Edits: 2022 Sokratis Makrogiannis


Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
# import pdb

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--moving_labels', help='moving label image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--moved_labels', help='warped label image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
user_resize_factor = 1
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis, resize_factor=user_resize_factor)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True, resize_factor=user_resize_factor)

# Convert to float and normalize voxel intensities.
if not moving.dtype.name == 'float32':
    moving = np.single(moving)
    moving_max_intensity = np.max(moving)
    moving = moving / moving_max_intensity
else:
    moving_max_intensity = 1.

if not fixed.dtype.name == 'float32':
    fixed = np.single(fixed)
    fixed_max_intensity = np.max(fixed)
    fixed = fixed / fixed_max_intensity
else:
    fixed_max_intensity = 1.

# Perform registration.
inshape = moving.shape[1:-1]
nb_feats = moving.shape[-1]

with tf.device(device):
    # load model and predict
    config = dict(inshape=inshape, input_model=None)
    warp = vxm.networks.VxmDense.load(args.model, **config).register(moving, fixed)
    moved = vxm.networks.Transform(inshape, nb_feats=nb_feats).predict([moving, warp])
    if args.moving_labels:
        moving_labels = vxm.py.utils.load_volfile(args.moving_labels, add_feat_axis=add_feat_axis)
        moving_labels = vxm.py.utils.resize(moving_labels, user_resize_factor)
        moving_labels = moving_labels[np.newaxis, ...]
        moved_labels = vxm.networks.Transform(inshape, nb_feats=nb_feats, interp_method='nearest').predict([moving_labels, warp])

# save warp
if args.warp:
    vxm.py.utils.save_volfile(warp.squeeze(), args.warp, fixed_affine)

# Re-scale moved image intensities.
if moved.dtype.name == 'float32':
    moved = fixed_max_intensity * moved  
    moved = np.short(moved)

# Save original moved image.
# pdb.set_trace()
vxm.py.utils.save_volfile(moved.squeeze(), 'moved.nii', fixed_affine)

# Resize back to original and write to file.
final_moved = vxm.py.utils.resize(moved[0,:], 1/user_resize_factor, label_map=False)
vxm.py.utils.save_volfile(final_moved.squeeze(), args.moved, fixed_affine)
if args.moving_labels:
    final_moved_labels = vxm.py.utils.resize(moved_labels[0,:], 1/user_resize_factor)
    final_moved_labels = np.short(final_moved_labels)
    vxm.py.utils.save_volfile(final_moved_labels.squeeze(), args.moved_labels, fixed_affine)
