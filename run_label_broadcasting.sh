#!/bin/bash
set -e

WSI_datapath=$1  # the data path of WSI, containing he-raw.jpg, pixel-size-raw.jpg
SO_datapath=$2  # the data path of spatial omics data, containing he-raw.jpg, pixel-size-raw.jpg, annotation_file.csv

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis

# extract the histology feature of whole-slide H&E image
echo $pixel_size > ${WSI_datapath}pixel-size.txt
python p1_histology_preprocess.py ${WSI_datapath}
python p2_superpixel_quality_control.py ${WSI_datapath} --patch_size 16
python p3_feature_extraction.py  ${WSI_datapath} --foundation_model 'uni' --ckpt_path './checkpoints/uni/' --device=${device} --down_samp_step 1

# if the ST data itselt has whole-slide H&E image, we can do in-sample broadcasting and thus only need to extract the features once
if [ "${WSI_datapath}" != "${SO_datapath}" ]; then
	# extract the histology feature of SO H&E image
	echo $pixel_size > ${SO_datapath}pixel-size.txt
	python p1_histology_preprocess.py ${SO_datapath}
	python p2_superpixel_quality_control.py ${SO_datapath}
	python p3_feature_extraction.py  ${SO_datapath} --device=${device} --down_samp_step 1
fi

# cell-level label broadcasting
python p7_cell_label_broadcasting.py ${WSI_datapath} ${SO_datapath} --foundation_model 'uni' --device=${device}
