#!/bin/bash
set -e

WSI_datapath="./demo_visiumhd_crc/"
SO_datapath="./demo_visiumhd_crc/"

# run pipeline
./run_label_broadcasting.sh $WSI_datapath $SO_datapath