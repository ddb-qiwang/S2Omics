#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
python p1_histology_preprocess.py ${prefix}
# higher m will filter more cells, we recommand m=0.7 if the H&E image is of poor quality
python p2_superpixel_quality_control.py ${prefix} --m 1.0 

# extract histology features
# larger down_samp_step will lower the resolution ot image segmentation results, but can save time
# we recommand user set down_samp_step as 5, 10 or 15, this parameter should be consistent in this file
python p3_feature_extraction.py  ${prefix} --device=${device} --down_samp_step 10

# get histology segmentations
python p4_get_histology_segmentation.py ${prefix} --down_samp_step 10
python p5_merge_over_clusters.py ${prefix}

# ROI selection
# roi_size refers the physical size of ROI to be selected, 
# here we are selecting 1 ROI for Visium HD experiment, so roi_size should be 6.5 6.5
# if user want S2Omics to automatically determine the optimal number of ROIs, please set num_roi as 0
python p6_roi_selection.py ${prefix} --down_samp_step 10 --num_roi 1 --roi_size 6.5 6.5 