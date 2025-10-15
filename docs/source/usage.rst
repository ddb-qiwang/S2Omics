Usage
=====

We recommand users to start from running our tutorials.

Please first download the demo data and pretrained model checkpoints file from:

google drive: https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_?usp=sharing

Please place both 'checkpoints' and 'demo' folder under the 'S2Omics' main folder.

User can either refer to the tutorial notebooks or run the python codes in the main folder.

For example, to select ROI on the demo colorectal cancer section:

.. code-block:: bash

   python run_roi_selection_single.py --prefix './demo/Tutorial_1_VisiumHD_ROI_selection_colon/' --save_folder './demo/Tutorial_1_VisiumHD_ROI_selection_colon/S2Omics_output' --device 'cuda:0' --roi_size 6.5 6.5 --num_roi 

Typical output:

.. image:: images/best_roi_on_histology_segmentations_scaled.jpg
   :alt: Best ROI example
   :width: 60%
   :align: center

To select ROI on the demo consecutive breast cancer sections

.. code-block:: bash

   python run_roi_selection_multiple.py --prefix_list './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g1/' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g2/' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g3/' --save_folder_list './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g1/S2Omics_output' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g2/S2Omics_output' './demo/Tutorial_3_Consecutive_ROI_selection_breast/breast_cancer_g3/S2Omics_output' --device 'cuda:0' --roi_size 1.5 1.5 --num_roi 1

To broadcast the cell type label within th selected ROI to the entire slide on the demo colorectal cancer section:

.. code-block:: bash

   python run_label_broadcasting.py --WSI_datapath './demo/Tutorial_1_VisiumHD_ROI_selection_colon/' --SO_datapath './demo/Tutorial_1_VisiumHD_ROI_selection_colon/' --WSI_save_folder './demo/Tutorial_1_VisiumHD_ROI_selection_colon/S2Omics_output' --SO_save_folder './demo/Tutorial_1_VisiumHD_ROI_selection_colon/S2Omics_output' --need_preprocess True --need_feature_extraction True

Output example:

.. image:: images/S2Omics_whole_slide_prediction_scaled.jpg
   :alt: Whole slide cell type prediction
   :width: 60%
   :align: center
