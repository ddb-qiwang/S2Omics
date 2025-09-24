#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
save_folder=$2 # e.g. test_0710
ds_step=$3
device=$4

pixel_size=0.5  # desired pixel size for the whole analysis

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
python p1_histology_preprocess.py ${prefix}
# use histosweep for quality control
python p2_superpixel_quality_control.py ${prefix} --save_folder ${save_folder} --patch_size 16

# extract histology features
# larger down_samp_step will lower the resolution ot image segmentation results, but can save time
# we recommand user set down_samp_step as 5, 10 or 15, this parameter should be consistent in this file
python p3_feature_extraction.py  ${prefix} --save_folder ${save_folder} --foundation_model 'uni' --ckpt_path './checkpoints/uni/' --device=${device} --down_samp_step ${ds_step}
