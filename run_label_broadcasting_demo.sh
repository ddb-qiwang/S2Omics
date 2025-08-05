#!/bin/bash
set -e

WSI_datapath="./demo/colorectal_cancer_p1/"
SO_datapath="./demo/colorectal_cancer_p1/"

# run pipeline
./run_label_broadcasting.sh $WSI_datapath $SO_datapath