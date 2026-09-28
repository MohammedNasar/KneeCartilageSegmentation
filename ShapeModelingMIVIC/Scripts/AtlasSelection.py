import nibabel as nib, numpy as np, glob

for f in sorted(glob.glob('/home/users/mibrahim/segmentation/Knee/ShapeModelingMIVIC/Scripts/test/participant_*_label.nii')):
    vol = np.sum(nib.load(f).get_fdata() > 0)
    print(f, vol)