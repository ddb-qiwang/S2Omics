#!/bin/bash
set -e

bash run_feature_extraction.sh './demo/breast_cancer_g1/' 'example_3d' 1 10 'cuda:1'
bash run_feature_extraction.sh './demo/breast_cancer_g2/' 'example_3d' 1 10 'cuda:1'
bash run_feature_extraction.sh './demo/breast_cancer_g3/' 'example_3d' 1 10 'cuda:1'