#!/usr/bin/env python

#==========================================================================
#   Segment a subject list using the atlas-based framework.
#   S. Makrogiannis, MIVIC/PEMACS/DSU
#==========================================================================

import pdb
import sys, os
import pdb

# Check for command syntax.
if len(sys.argv) < 3:
    print('Usage (multiple atlases):\n' + str(sys.argv[0]) + ' <Subject List> <Atlas List>\n')
    print('OR:\n')
    print('Usage (single atlas):\n' + str(sys.argv[0]) + ' <Subject List> <Atlas> <Atlas labels>\n')
    sys.exit(0)

pdb.set_trace()
# Pass lists of subjects and atlases.

if len(sys.argv) == 3: # For multiple atlases.
    subject_list_filename = sys.argv[1]
    subject_list = os.path.splitext(subject_list_filename)[0]
    atlas_list_filename = sys.argv[2]
    atlas_list = os.path.splitext(atlas_list_filename)[0]
elif len(sys.argv) == 4: # For a single atlas.
    subject_list_filename = sys.argv[1]
    subject_list = os.path.splitext(subject_list_filename)[0]
    atlas_filename = sys.argv[2]
    atlas = os.path.splitext(atlas_filename)[0]
    atlas_labels_filename = sys.argv[3]
    atlas_labels  = os.path.splitext(atlas_labels_filename)[0]
else:
    print('Unrecognized argument list.\n')
    sys.exit(0)


# Program paths.
project_path = "/home/users/mibrahim/segmentation/knee/ge_scanner/ShapeModelingMIVIC/"
segmentation_program_multi = project_path + "Scripts/SegmentByMultipleAtlases3D.py"
segmentation_program_single = project_path + "Scripts/SegmentBySingleAtlas.py"


# Read subject list and run segmentation.
f = open(subject_list_filename, 'r')
print(f)

for line in f:
    subject_filename = line.rstrip('\n')
    subject = os.path.splitext(subject_filename)[0]
    
    if len(sys.argv) == 3:
        command = segmentation_program_multi + " " + subject_filename +  " " + atlas_list_filename
        print( command + "\n")
        os.system( command )
    elif len(sys.argv) == 4:
        command = segmentation_program_single + " " + subject_filename +  " " + atlas_filename +  " " + atlas_labels_filename
        print( command + "\n")
        os.system( command )
        


# Close subject list.
f.close()
