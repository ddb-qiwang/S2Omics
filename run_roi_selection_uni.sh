#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
save_folder=$2 # e.g. test_0710

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
python p1_histology_preprocess.py ${prefix}
# use histosweep for quality control
python p2_superpixel_quality_control.py ${prefix} --save_folder ${save_folder} --patch_size 16

# extract histology features
# larger down_samp_step will lower the resolution ot image segmentation results, but can save time
# we recommand user set down_samp_step as 5, 10 or 15, this parameter should be consistent in this file
python p3_feature_extraction.py  ${prefix} --save_folder ${save_folder} --foundation_model 'uni' --ckpt_path '/home/msyuan/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/' --device=${device} --down_samp_step 10

# get histology segmentations
# default setting use PCA + K-means++ + merging over-clusters
python p4_get_histology_segmentation.py ${prefix} --save_folder ${save_folder} --foundation_model 'uni' --down_samp_step 1
python p5_merge_over_clusters.py ${prefix} --save_folder ${save_folder}

# ROI selection
# roi_size refers the physical size of ROI to be selected, 
# here we are selecting 1 ROI for Visium HD experiment, so roi_size should be 6.5 6.5
# if user want S2Omics to automatically determine the optimal number of ROIs, please set num_roi as 0
python p6_roi_selection_rectangle.py ${prefix} --save_folder ${save_folder} --down_samp_step 10 --num_roi 1 --roi_size 6.5 6.5
# if user what to select TMA cores or other circle-shaped ROIs, a 3mm-radius circle for an instance
# python p6_roi_selection_circle.py ${prefix} --down_samp_step 10 --num_roi 1 --roi_size 3 3 