#!/bin/bash
set -e

multiple_images_prefix=$1
multiple_images_save_folder=$2
ds_step=$3
n_clusters=$4
roi_size=$5
num_roi=$6

# get histology segmentations
# default setting use PCA + K-means++ 
#python p4_get_histology_segmentation.py ${multiple_images_prefix} ${multiple_images_save_folder} --foundation_model 'uni' --down_samp_step ${ds_step} --n_clusters ${n_clusters}

# ROI selection
# roi_size refers the physical size of ROI to be selected, 
# here we are selecting 1.5*1.5 mm^2 ROI
# if user want S2Omics to automatically determine the optimal number of ROIs, please set num_roi as 0
python p5_roi_selection_rectangle.py ${multiple_images_prefix} ${multiple_images_save_folder} --down_samp_step ${ds_step} --num_roi ${num_roi} --roi_size ${roi_size} ${roi_size}
# if user what to select TMA cores or other circle-shaped ROIs, a 3mm-radius circle for an instance
# python p5_roi_selection_circle.py ${prefix} --down_samp_step 10 --num_roi 1 --roi_size 3 3 