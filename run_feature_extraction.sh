#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
save_folder=$2 # e.g. test_0710
m=$3
ds_step=$4
device=$5

pixel_size=0.5  # desired pixel size for the whole analysis

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt
python p1_histology_preprocess.py ${prefix}
# higher m will filter more cells, we recommand m=0.7 if the H&E image is of poor quality
python p2_superpixel_quality_control.py ${prefix} --save_folder ${save_folder} --patch_size 16 --m ${m} --manual True --manual_x_vertex 180

# extract histology features
# larger down_samp_step will lower the resolution ot image segmentation results, but can save time
# we recommand user set down_samp_step as 5, 10 or 15, this parameter should be consistent in this file
python p3_feature_extraction.py  ${prefix} --save_folder ${save_folder} --foundation_model 'uni' --ckpt_path './checkpoints/uni/' --device=${device} --down_samp_step ${ds_step}