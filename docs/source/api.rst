API Reference
=============

Below are the main modules in the S2Omics pipeline, with purpose, CLI parameters,
and I/O descriptions. Autodoc will generate detailed function and class listings from docstrings.

----------------------------------------
s2omics.p1_histology_preprocess
----------------------------------------
**Purpose:**  
Scale and pad raw H&E stained images to a target resolution so that image dimensions are divisible by the patch size.

**Parameters:**

+--------------------+----------+---------------------------------------------------+
| Argument           | Default  | Description                                       |
+====================+==========+===================================================+
| prefix             |          | path to H&E image folder, str                     |
+--------------------+----------+---------------------------------------------------+
| --show_image       | False    | if output the H&E image or not                        |
+--------------------+----------+---------------------------------------------------+


**Return:**
- `he-scaled.jpg`: rescaled image. Saved under prefix folder
- `he.jpg`: scaled + padded image. Saved under prefix folder

----------------------------------------
s2omics.p2_superpixel_quality_control
----------------------------------------
**Purpose:**  
Split histology image (`he.jpg`) into superpixels and filter out tiles without nuclei
or with low structural quality, using density and texture analysis. The new version of S2-omics use QC package HistoSweep for this step.

**Parameters:**
+--------------------+----------+---------------------------------------------------+
| Argument           | Default  | Description                                       |
+====================+==========+===================================================+
| prefix             | (pos.)   | Input folder.                                     |
+--------------------+----------+---------------------------------------------------+
| --save_folder      | S2Omics_output | Output directory.                           |
+--------------------+----------+---------------------------------------------------+
| --clustering_method| kmeans   | kmeans, fcm, agglo, bisect, birch, louvain, leiden |
+--------------------+----------+---------------------------------------------------+
| --n_clusters       | 20       | Initial clusters (kmeans/fcm).                     |
+--------------------+----------+---------------------------------------------------+
| --resolution       | 1.0      | Graph-based method resolution (louvain/leiden).    |
+--------------------+----------+---------------------------------------------------+
- prefix: path to H&E image folder, str
- save_folder: path to results folder, str
- density_thresh: HistoSweep parameter, threshold for identifying low density superpixels, int, default=100
- clean_background_flag: HistoSweep parameter, whether to preserve fibrous regions that are otherwise being incorrectly filtered out, bool, default=False
- patch_size: the shape of superpixels, int, default=16 means that all superpixels are 16*16 pathces
- show_image: if output the H&E image or not, bool, default = False

**Return:**
- `shapes.pickle`: image and superpixel shape information in pickle format, saved under save_folder/pickle_files
- `qc_preserve_indicator.pickle`: binary mask in pickle format, saved under save_folder/pickle_files
- `mask-small.png`: binary mask, saved under HistoSweep_output folder

----------------------------------------
s2omics.p3_feature_extraction
----------------------------------------
**Purpose:**  
Apply a foundation model (UNI / Virchow / GigaPath) to extract hierarchical features from superpixels.

**Parameters:**
- prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
- save_folder: the name of save folder
- foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath
- ckpt_path: the path to foundation model parameter files (should be named as 'pytorch_model.bin'), './checkpoints/uni/' for an example
- device: default = 'cuda'
- batch_size: default = 32
- down_samp_step: the down-sampling step, default = 10 refers to only extract features for superpixels whose row_index and col_index can both be divided by 10 (roughly 1:100 down-sampling rate). down_samp_step = 1 means extract features for every superpixel
- num_workers: default = 4

**Return:**
- Hierarchical embeddings saved in multiple `.pickle` parts, saved under save_folder/pickle_files

----------------------------------------
p4_get_histology_segmentation
----------------------------------------
**Purpose:**  
Cluster PCA-reduced embeddings into morphological clusters using chosen algorithm.

**Outputs:**
- `cluster_image.pickle` – Cluster map.
- Cluster RGB image.

**CLI Arguments:**

+--------------------+----------+---------------------------------------------------+
| Argument           | Default  | Description                                       |
+====================+==========+===================================================+
| prefix             | (pos.)   | Input folder.                                     |
+--------------------+----------+---------------------------------------------------+
| --save_folder      | S2Omics_output | Output directory.                           |
+--------------------+----------+---------------------------------------------------+
| --clustering_method| kmeans   | kmeans, fcm, agglo, bisect, birch, louvain, leiden |
+--------------------+----------+---------------------------------------------------+
| --n_clusters       | 20       | Initial clusters (kmeans/fcm).                     |
+--------------------+----------+---------------------------------------------------+
| --resolution       | 1.0      | Graph-based method resolution (louvain/leiden).    |
+--------------------+----------+---------------------------------------------------+

.. automodule:: p4_get_histology_segmentation
   :members:

----------------------------------------
p5_merge_over_clusters
----------------------------------------
**Purpose:**  
Merge morphological clusters with high similarity to target number using hierarchical linkage.

**Outputs:**
- `adjusted_cluster_image.pickle` – Merged cluster map.
- Adjusted segmentation image.

**CLI Arguments:**

+----------------------+----------+---------------------------------------------+
| Argument             | Default  | Description                                 |
+======================+==========+=============================================+
| prefix               | (pos.)   | Input folder.                               |
+----------------------+----------+---------------------------------------------+
| --save_folder        | S2Omics_output | Output directory.                     |
+----------------------+----------+---------------------------------------------+
| --target_n_clusters  | 15       | Desired final cluster number.               |
+----------------------+----------+---------------------------------------------+

.. automodule:: p5_merge_over_clusters
   :members:

----------------------------------------
p6_roi_selection_rectangle
----------------------------------------
**Purpose:**  
Automatically select rectangular ROIs based on scoring criteria:
- **Scale score** (size coverage)
- **Coverage score** (valid cell proportion)
- **Balance score** (match desired cluster composition)

**Outputs:**
- ROI visualizations on segmentation and raw histology image.
- `best_roi.pickle` – ROI details and score breakdown.

**CLI Arguments:**

+--------------------+----------+------------------------------------------------------+
| Argument           | Default  | Description                                          |
+====================+==========+======================================================+
| prefix             | (pos.)   | Input folder.                                        |
+--------------------+----------+------------------------------------------------------+
| --save_folder      | S2Omics_output | Output folder.                                 |
+--------------------+----------+------------------------------------------------------+
| --roi_size         | [6.5,6.5]| Physical size in mm (width height).                  |
+--------------------+----------+------------------------------------------------------+
| --num_roi          | 0        | Number of ROIs (0 = auto-determine optimal).         |
+--------------------+----------+------------------------------------------------------+
| --positive_prior   | []       | Clusters to emphasize.                               |
+--------------------+----------+------------------------------------------------------+
| --negative_prior   | []       | Clusters to de-prioritize.                           |
+--------------------+----------+------------------------------------------------------+
| --prior_preference | 2        | Weight for emphasis clusters.                        |

.. automodule:: p6_roi_selection_rectangle
   :members:

----------------------------------------
p6_roi_selection_circle
----------------------------------------
**Purpose:**  
Same as rectangular ROI selection, but using circular geometry. Suitable for TMA core or circular ROI scans.

**CLI Arguments:** Similar to rectangle, with `--roi_size` interpreted as radius.

.. automodule:: p6_roi_selection_circle
   :members:

----------------------------------------
p7_cell_label_broadcasting
----------------------------------------
**Purpose:**  
Train an Autoencoder-based classifier using ROI-scale spatial omics cell annotations, then broadcast labels to the entire slide.

**Outputs:**
- `S2Omics_whole_slide_prediction.jpg` – Predicted whole-slide cell type map.

**CLI Arguments:**

+-------------------+----------+--------------------------------------------------+
| Argument          | Default  | Description                                      |
+===================+==========+==================================================+
| WSI_datapath      | (pos.)   | Whole-slide input folder.                        |
+-------------------+----------+--------------------------------------------------+
| SO_datapath       | (pos.)   | Spatial omics ROI input folder.                  |
+-------------------+----------+--------------------------------------------------+
| --foundation_model| uni      | Model for embeddings.                            |
+-------------------+----------+--------------------------------------------------+
| --device          | cuda     | Compute device.                                  |

.. automodule:: p7_cell_label_broadcasting
   :members:

----------------------------------------
Utility Modules
----------------------------------------
Low-level utilities used in multiple steps (I/O helpers, seeding, image operations).

.. automodule:: s1_utils
   :members:

.. automodule:: s2_label_broadcasting
   :members:

.. automodule:: utils
   :members:
