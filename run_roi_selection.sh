#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
save_folder=$2 # e.g. test_0710
ds_step=$3
target_n_clusters=$4
roi_size=$5
num_roi=$6

device='cuda:0'  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
#python p1_histology_preprocess.py ${prefix}
# higher m will filter more cells, we recommand m=0.7 if the H&E image is of poor quality
#python p2_superpixel_quality_control.py ${prefix} --save_folder ${save_folder} --patch_size 16 --m 1.0

# extract histology features
# larger down_samp_step will lower the resolution ot image segmentation results, but can save time
# we recommand user set down_samp_step as 1, 2, 3, 5, 10, this parameter should be consistent in this file
#python p3_feature_extraction.py  ${prefix} --save_folder ${save_folder} --foundation_model 'uni' --ckpt_path './checkpoints/uni/' --device=${device} --down_samp_step ${ds_step}

# get histology segmentations
# default setting use PCA + K-means++ + merging over-clusters
python p4_get_histology_segmentation.py ${prefix} --save_folder ${save_folder} --foundation_model 'uni' --down_samp_step ${ds_step}
python p5_merge_over_clusters.py ${prefix} --save_folder ${save_folder} --target_n_clusters ${target_n_clusters}

# ROI selection
# roi_size refers the physical size of ROI to be selected, 
# here we are selecting 1 ROI for Visium HD experiment, so roi_size should be 6.5 6.5
# if user want S2Omics to automatically determine the optimal number of ROIs, please set num_roi as 0
python p6_roi_selection_rectangle.py ${prefix} --save_folder ${save_folder} --down_samp_step ${ds_step} --num_roi ${num_roi} --roi_size ${roi_size} ${roi_size}
# if user what to select TMA cores or other circle-shaped ROIs, a 3mm-radius circle for an instance
# python p6_roi_selection_circle.py ${prefix} --down_samp_step 10 --num_roi 1 --roi_size 3 3 