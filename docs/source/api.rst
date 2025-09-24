API Reference
=============

Below are the main modules in S2Omics pipeline with their purpose, CLI arguments, and outputs.

p1_histology_preprocess
-----------------------
**Purpose**:  
Scale and pad the raw H&E stained image to target resolution and size alignment.

**Key Functions / CLI Arguments**:
- `prefix` (positional) – folder containing `he-raw.jpg` and pixel size files.
- Output: `he-scaled.jpg` and padded `he.jpg`.

.. automodule:: p1_histology_preprocess
   :members:

p2_superpixel_quality_control
-----------------------------
Splits `he.jpg` into superpixels and applies quality control:
- Density filtering using RGB variance.
- Texture analysis to remove tiles with poor cellular structure.
- Background cleaning (optional).

**CLI Args**:
- `--save_folder`: name/path for outputs.
- `--patch_size`: size per superpixel (default 16 px).
- `--density_thresh`: RGB density cutoff.
- `--clean_background_flag`: preserve fibrous regions.

Outputs:
- `qc_preserve_indicator.pickle`
- Mask images showing valid superpixels.

.. automodule:: p2_superpixel_quality_control
   :members:

p3_feature_extraction
---------------------
Apply a chosen **foundation model** (UNI/Virchow/GigaPath) to extract hierarchical embeddings:
- Global 224×224-level feature
- Local patch-level feature

CLI Args:
- `--foundation_model`: one of {uni, virchow, gigapath}
- `--ckpt_path`: path to model checkpoint
- `--down_samp_step`: controls patch sample density.

Output files:
- `{model}_embeddings_downsamp_{step}_part_{n}.pickle`

.. automodule:: p3_feature_extraction
   :members:

p4_get_histology_segmentation
-----------------------------
Cluster PCA-reduced embeddings into morphological classes using chosen clustering method.

CLI Args:
- `--clustering_method`: kmeans, fcm, agglo, bisect, birch, louvain, leiden.
- `--n_clusters`: initial cluster number (for kmeans/fcm).
- `--resolution`: resolution for graph-based clustering.

Outputs:
- `cluster_image.pickle` – matrix of cluster IDs per superpixel.
- RGB cluster-preview image.

.. automodule:: p4_get_histology_segmentation
   :members:

p5_merge_over_clusters
----------------------
Merge clusters with high similarity to avoid over-segmentation; uses `linkage_matrix.pickle` from p4.

CLI Args:
- `--target_n_clusters`: desired final count.

Outputs:
- `adjusted_cluster_image.pickle`
- Dendrogram showing cluster merging.

.. automodule:: p5_merge_over_clusters
   :members:

p6_roi_selection_rectangle
--------------------------
Search and scoring for **rectangular ROIs**.

Scoring metrics:
- Scale (size coverage)
- Validity (cell coverage)
- Balance (match target cluster composition)

CLI Args:
- `--roi_size`: physical size in mm×mm.
- `--num_roi`: number of ROIs to select (0 = auto-determine).
- `--positive_prior`, `--negative_prior`: emphasize or de-prioritize clusters.

Outputs:
- ROI visualization over histology segmentation and raw H&E.
- Best ROI pickle with score breakdown.

.. automodule:: p6_roi_selection_rectangle
   :members:

p6_roi_selection_circle
-----------------------
Same as rectangle ROI selection but uses circular masks; suited for TMA core selection.

.. automodule:: p6_roi_selection_circle
   :members:

p7_cell_label_broadcasting
--------------------------
Broadcast cell type annotations from ROI-scale spatial omics to whole-slide.

**Workflow**:
1. Load ROI annotation (`annotation_file.csv`).
2. Train small AE-classifier using ROI histology features and labels.
3. Apply to valid whole-slide superpixels to predict cell types.

Output:
- `S2Omics_whole_slide_prediction.jpg`

.. automodule:: p7_cell_label_broadcasting
   :members:

Utility Modules
---------------
.. automodule:: s1_utils
   :members:
.. automodule:: s2_label_broadcasting
   :members:
.. automodule:: utils
   :members:
