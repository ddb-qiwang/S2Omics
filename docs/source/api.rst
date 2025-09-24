API Reference
=============

Below are the main modules in the S2Omics pipeline, with purpose, CLI parameters,
and I/O descriptions. Autodoc will generate detailed function and class listings from docstrings.

----------------------------------------
p1_histology_preprocess
----------------------------------------
**Purpose:**  
Scale and pad raw H&E stained images to a target resolution so that image dimensions are divisible by the patch size.

**Inputs:**
- Raw image file (`he-raw.jpg` or `.png`/`.tiff`/`.svs`).
- `pixel-size-raw.txt` – Physical resolution in µm/pixel.
- `pixel-size.txt` – Target resolution after rescaling (default 0.5 µm).

**Outputs:**
- `he-scaled.jpg` – rescaled image.
- `he.jpg` – scaled + padded image.

**CLI Arguments:**

+----------------+----------+---------------------------------------------+
| Argument       | Default  | Description                                 |
+================+==========+=============================================+
| prefix         | (pos.)   | Path to H&E image folder.                   |
+----------------+----------+---------------------------------------------+

.. automodule:: p1_histology_preprocess
   :members:
   :undoc-members:
   :show-inheritance:

----------------------------------------
p2_superpixel_quality_control
----------------------------------------
**Purpose:**  
Split histology image (`he.jpg`) into superpixels and filter out tiles without nuclei
or with low structural quality, using density and texture analysis.

**Outputs:**
- `qc_preserve_indicator.pickle` – Boolean matrix of valid superpixels.
- QC mask image (`qc_mask.jpg`).

**CLI Arguments:**

+------------------------+----------+---------------------------------------------------+
| Argument               | Default  | Description                                       |
+========================+==========+===================================================+
| prefix                 | (pos.)   | Input histology folder.                           |
+------------------------+----------+---------------------------------------------------+
| --save_folder          | S2Omics_output | Output directory name.                      |
+------------------------+----------+---------------------------------------------------+
| --patch_size           | 16       | Superpixel dimension (px).                        |
+------------------------+----------+---------------------------------------------------+
| --density_thresh       | 100      | RGB density threshold.                            |
+------------------------+----------+---------------------------------------------------+
| --clean_background_flag| off      | Preserve fibrous regions if set.                  |
+------------------------+----------+---------------------------------------------------+

.. automodule:: p2_superpixel_quality_control
   :members:

----------------------------------------
p3_feature_extraction
----------------------------------------
**Purpose:**  
Apply a foundation model (UNI / Virchow / GigaPath) to extract hierarchical features from superpixels.

**Outputs:**
- Hierarchical embeddings saved in multiple `.pickle` parts.

**CLI Arguments:**

+-------------------+----------+------------------------------------------------------+
| Argument          | Default  | Description                                          |
+===================+==========+======================================================+
| prefix            | (pos.)   | Input histology folder.                              |
+-------------------+----------+------------------------------------------------------+
| --save_folder     | S2Omics_output | Output folder name.                            |
+-------------------+----------+------------------------------------------------------+
| --foundation_model| uni      | Base model: uni / virchow / gigapath.                 |
+-------------------+----------+------------------------------------------------------+
| --ckpt_path       | ./checkpoints/uni/ | Model checkpoint path.                      |
+-------------------+----------+------------------------------------------------------+
| --down_samp_step  | 10       | Downsampling factor.                                  |
+-------------------+----------+------------------------------------------------------+

.. automodule:: p3_feature_extraction
   :members:

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
