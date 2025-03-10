#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
#python p1_histology_preprocess.py ${prefix}
#python p2_superpixel_quality_control.py ${prefix} --m 0.7

# extract histology features
#python p3_feature_extraction.py  ${prefix} --device=${device} --model_path './checkpoints/' --down_samp_step 1

# get histology segmentations
python p4_get_histology_segmentation.py ${prefix} --down_samp_step 1 --num_histology_clusters 25
python p5_merge_over_clusters.py ${prefix}

# ROI selection
python p6_roi_selection.py ${prefix} --down_samp_step 1 --roi_size 2 2 

# cell-level label broadcasting
python p7_cell_label_broadcasting.py ${prefix} --roi_size 2 2 
