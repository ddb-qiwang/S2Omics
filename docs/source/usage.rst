Usage
=====

Installation
------------
We recommend using Python 3.11 or above for S2Omics.

.. code-block:: bash

   git clone https://github.com/ddb-qiwang/S2Omics.git
   cd S2Omics
   conda create -n S2Omics python=3.11
   conda activate S2Omics
   pip install -r requirements.txt  # or requirements_old_gcc.txt if GCC is outdated

Downloading Demo Data and Models
--------------------------------
You can download demo datasets and pretrained foundation model checkpoints from:

Google Drive: https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_?usp=sharing

Place both `checkpoints` and `demo` folders inside the `S2Omics` main directory.

Running ROI Selection Demo
--------------------------
Example: selecting a single 6.5 mm × 6.5 mm ROI for a Visium HD experiment.

.. code-block:: bash

   chmod +x run_*
   ./run_roi_selection_demo.sh

Typical output:
.. image:: /readme_images/best_roi_on_histology_segmentations_scaled.jpg
   :alt: Best ROI example
   :width: 60%
   :align: center

Running Cell Type Label Broadcasting Demo
-----------------------------------------
If spatial omics data is available with cell type annotations inside the ROI (`annotation_file.csv`), you can broadcast these labels to the entire slide:

.. code-block:: bash

   ./run_label_broadcasting_demo.sh

Output example:
.. image:: /readme_images/S2Omics_whole_slide_prediction_scaled.jpg
   :alt: Whole slide cell type prediction
   :width: 60%
   :align: center

Data Format
-----------
Required file inputs for S2Omics modules:

- ``he-raw.jpg``: Raw histology image.
- ``pixel-size-raw.txt``: Side length in micrometers per pixel of the raw image (e.g., `0.2`).
- ``pixel-size.txt``: Desired target microns per pixel after rescaling (e.g., `0.5`).
- ``annotation_file.csv`` *(optional for label broadcasting)*:
  - **super_pixel_x**, **super_pixel_y**, **annotation**
  - Example row: `267, 1254, Myofibroblasts`.

Example annotation table:

+---------------------+----------------+----------------+-----------------------------------+
| barcode             | super_pixel_x  | super_pixel_y  | annotation                        |
+=====================+================+================+===================================+
| s_008um_00000_00276 | 267             | 1254           | Myofibroblasts                     |
| s_008um_00000_00279 | 270             | 1254           | Epithelial cells (Malignant)       |
+---------------------+----------------+----------------+-----------------------------------+

Pipeline Overview
-----------------
.. image:: /readme_images/S2Omics_pipeline.png
   :alt: Pipeline overview
   :width: 85%
   :align: center
