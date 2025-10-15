API Reference
=============

Below are the main modules in the S2Omics pipeline, with purpose, CLI parameters,
and I/O descriptions. Autodoc will generate detailed function and class listings from docstrings.

----------------------------------------
s2omics.p1_histology_preprocess.histology_preprocess
----------------------------------------

**Purpose:**  

Scale and pad raw H&E stained images to a target resolution so that image dimensions are divisible by the patch size.

**Parameters:**

- prefix: path to H&E image folder, str

- show_image: if output the H&E image or not, bool, default = False


**Return:**

- `he-scaled.jpg`: rescaled image. Saved under prefix folder

- `he.jpg`: scaled + padded image. Saved under prefix folder

----------------------------------------
s2omics.p2_superpixel_quality_control.superpixel_quality_control
----------------------------------------

**Purpose:**  

Split histology image (`he.jpg`) into superpixels and filter out tiles without nuclei
or with low structural quality, using density and texture analysis. The new version of S2-omics use QC package HistoSweep for this step.

**Parameters:**

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
s2omics.p3_feature_extraction.feature_extraction
----------------------------------------

**Purpose:**  

Apply a foundation model (UNI / Virchow / GigaPath) to extract hierarchical features from superpixels.

**Parameters:**

- prefix: folder path of H&E stained image, '/home/H&E_image/' for an example

- save_folder: the name of save folder

- foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath

- ckpt_path: the path to foundation model parameter files (should be named as 'pytorch_model.bin'), './checkpoints/uni/' for an example

- device: default = 'cuda:0'

- batch_size: default = 32

- down_samp_step: the down-sampling step, default = 10 refers to only extract features for superpixels whose row_index and col_index can both be divided by 10 (roughly 1:100 down-sampling rate). down_samp_step = 1 means extract features for every superpixel

- num_workers: default = 4

**Return:**

- Hierarchical embeddings saved in multiple `.pickle` parts, saved under save_folder/pickle_files

----------------------------------------
s2omics.single_section.p4_get_histology_segmentation.get_histology_segmentation
----------------------------------------

**Purpose:**  
Cluster PCA-reduced embeddings into morphological clusters using chosen algorithm.

**Parameters:**

- prefix: folder path of H&E stained image, '/home/H&E_image/' for an example

- save_folder: the name of save folder

- foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath

- cache_path: the path to exatracted feature embedding files

- down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate

- clustering_method: the clustering method used for H&E image segmentation, user can select among {'kmeans': k-means++, 'fcm': fuzzy c-means, 'louvain': Louvain algorithm, 'leiden': Leiden algorithm}, default = 'kmeans'

- n_clusters: initial number of clusters for histology segmentation when using kmeans or fcm for clustering. default=20. Please notice that this is not the final number of clusters when clustering method is fcm.

- resolution: resolution for leiden algorithm, default=1.0

- if_evaluate: if evaluate the clustering results by quantitative metrics, default=False

**Return:**

- `cluster_image.pickle`: Cluster map.

- Cluster RGB image.

----------------------------------------
s2omics.single_section.p5_merge_over_clusters.merge_over_clusters
----------------------------------------

**Purpose:**  

Merge morphological clusters with high similarity to target number using hierarchical linkage.

**Parameters:**

- prefix: folder path of H&E stained image, '/home/H&E_image/' for an example

- save_folder: the name of save folder

- target_n_clusters: the final number of clusters user want to preserve, default=15

**Return:**

- `adjusted_cluster_image.pickle`: Merged cluster map.

- Adjusted segmentation image.

----------------------------------------
s2omics.single_section.p6_roi_selection_rectangle.roi_selection_for_single_section
----------------------------------------

**Purpose:**  

Automatically select rectangular ROIs based on scoring criteria:

- **Scale score** (size coverage)

- **Coverage score** (valid cell proportion)

- **Balance score** (match desired cluster composition)

**Parameters:**

- prefix: folder path of H&E stained image, '/home/H&E_image/' for an example

- save_folder: the name of save folder

- has_annotation: if True, use the cell type annotation file instead of histology segmentation results for ROI selection

- cache_path: if user want to specify another segmentation result for ROi selection, please insert the path here

- down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate

- roi_size: the physical size (mm x mm) of ROIs, default = [6.5, 6.5] which is the physical size for Visium HD ROI

- rotation_seg: the number of difference angles ROI can rotate, default=6 means the a ROI can rotate to 30/60/90/120/150/180 degrees

- num_roi: number of ROIs to be selected, default = 0 refers to automatic determination

- optimal_roi_thres: hyper-parameter for automatic ROI determination, default = 0.03 is suitable for most cases, recommend to be set as 0 when selecting FOVs. If you want to select more ROIs, please lower this parameter

- fusion_weights: the weight of three scores, default=[0.33,0.33,0.33], the sum of three weights should be equal to 1 (if not they will be normalized)

- emphasize_clusters, discard_clusters: prior information about interested and not-interested histology clusters, default = [],[]

- prior_preference: the larger this parameter is, S2Omics will focus more on those interested histology clusters, default=  1

**Return:**

- ROI visualizations on segmentation and raw histology image.

- `best_roi.pickle`: ROI details and score breakdown.

----------------------------------------
s2omics.single_section.p6_roi_selection_circle.roi_selection_for_single_section
----------------------------------------

**Purpose:**  
Same as rectangular ROI selection, but using circular geometry. Suitable for TMA core or circular ROI scans.

**Parameters:**

- prefix: folder path of H&E stained image, '/home/H&E_image/' for an example

- save_folder: the name of save folder

- has_annotation: if True, use the cell type annotation file instead of histology segmentation results for ROI selection

- cache_path: if user want to specify another segmentation result for ROi selection, please insert the path here

- down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate

- roi_size: the physical size (mm x mm) of circle-shaped ROIs, default = [0.5, 0.5] means the r=0.5

- rotation_seg: the number of difference angles ROI can rotate, default=6 means the a ROI can rotate to 30/60/90/120/150/180 degrees

- num_roi: number of ROIs to be selected, default = 0 refers to automatic determination

- optimal_roi_thres: hyper-parameter for automatic ROI determination, default = 0.03 is suitable for most cases, recommend to be set as 0 when selecting FOVs. If you want to select more ROIs, please lower this parameter

- fusion_weights: the weight of three scores, default=[0.33,0.33,0.33], the sum of three weights should be equal to 1 (if not they will be normalized)

- emphasize_clusters, discard_clusters: prior information about interested and not-interested histology clusters, default = [],[]

- prior_preference: the larger this parameter is, S2Omics will focus more on those interested histology clusters, default=  1

**Return:**

- ROI visualizations on segmentation and raw histology image.

- `best_roi.pickle`: ROI details and score breakdown.

----------------------------------------
s2omics.single_section.p7_cell_label_broadcasting.label_broadcasting
----------------------------------------

**Purpose:**  

After user obtained the spatial omics data of the selected small ROI, we can annotate the superpixels in the paired H&E image with cell type labels.

Afterwards, we can transfer the label information to the previously stained whole-slide H&E image to obtain whole-slide level cell type spatial distribution.

This function trains an Autoencoder-based classifier using ROI-scale spatial omics cell annotations, then broadcast labels to the entire slide.

**Parameters:**

- WSI_datapath: path to the whole slide H&E image

- WSI_save_folder: save path to the whole slide H&E image results
                      
- SO_datapath: path to the spatial omics data and accroding H&E image

- SO_save_folder: save path to the spatial omics data and accroding H&E image results
                      
- WSI_cache_path: path to the extracted histology feature of the WSI, if it is already obtained, default=''

- SO_cache_path: path to the extracted histology feature of the SO, if it is already obtained, default=''

- device: default='cuda:0'

- foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath

**Return:**

- `S2Omics_whole_slide_prediction.jpg`:  Predicted whole-slide cell type map.

----------------------------------------
s2omics.multiple_sections.p4_get_histology_segmentation.get_joint_histology_segmentation
----------------------------------------

**Purpose:**  
Jointly cluster PCA-reduced embeddings of multiple slides into morphological clusters using chosen algorithm.

**Parameters:**

- prefix_list: list of folder path of H&E stained image, ['/home/H&E_image/'] for an example

- save_folder_list: list of the name of save folder

- foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath

- cache_path: the path to exatracted feature embedding files

- down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate

- clustering_method: the clustering method used for H&E image segmentation, user can select among {'kmeans': k-means++, 'fcm': fuzzy c-means, 'louvain': Louvain algorithm, 'leiden': Leiden algorithm}, default = 'kmeans'

- n_clusters: initial number of clusters for histology segmentation when using kmeans or fcm for clustering. default=20. Please notice that this is not the final number of clusters when clustering method is fcm.

- resolution: resolution for leiden algorithm, default=1.0

- if_evaluate: if evaluate the clustering results by quantitative metrics, default=False

**Return:**

- `cluster_image.pickle`: Cluster map.

- Cluster RGB image.

----------------------------------------
s2omics.multiple_sections.p6_roi_selection_rectangle.roi_selection_for_multiple_sections
----------------------------------------

**Purpose:**  

Automatically select rectangular ROIs based on scoring criteria:

- **Scale score** (size coverage)

- **Coverage score** (valid cell proportion)

- **Balance score** (match desired cluster composition)

**Parameters:**

- prefix_list: list of folder path of H&E stained image, ['/home/H&E_image/'] for an example

- save_folder_list: list of the name of save folder

- has_annotation: if True, use the cell type annotation file instead of histology segmentation results for ROI selection

- cache_path: if user want to specify another segmentation result for ROi selection, please insert the path here

- down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate

- roi_size: the physical size (mm x mm) of ROIs, default = [6.5, 6.5] which is the physical size for Visium HD ROI

- rotation_seg: the number of difference angles ROI can rotate, default=6 means the a ROI can rotate to 30/60/90/120/150/180 degrees

- num_roi: number of ROIs to be selected, default = 0 refers to automatic determination

- optimal_roi_thres: hyper-parameter for automatic ROI determination, default = 0.03 is suitable for most cases, recommend to be set as 0 when selecting FOVs. If you want to select more ROIs, please lower this parameter

- fusion_weights: the weight of three scores, default=[0.33,0.33,0.33], the sum of three weights should be equal to 1 (if not they will be normalized)

- emphasize_clusters, discard_clusters: prior information about interested and not-interested histology clusters, default = [],[]

- prior_preference: the larger this parameter is, S2Omics will focus more on those interested histology clusters, default=  1

**Return:**

- ROI visualizations on segmentation and raw histology image.

- `best_roi.pickle`: ROI details and score breakdown.

----------------------------------------
s2omics.single_section.p6_roi_selection_circle.roi_selection_for_multiple_sections
----------------------------------------

**Purpose:**  
Same as rectangular ROI selection, but using circular geometry. Suitable for TMA core or circular ROI scans.

**Parameters:**

- prefix_list: list of folder path of H&E stained image, ['/home/H&E_image/'] for an example

- save_folder_list: list of the name of save folder

- has_annotation: if True, use the cell type annotation file instead of histology segmentation results for ROI selection

- cache_path: if user want to specify another segmentation result for ROi selection, please insert the path here

- down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate

- roi_size: the physical size (mm x mm) of circle-shaped ROIs, default = [0.5, 0.5] means the r=0.5

- rotation_seg: the number of difference angles ROI can rotate, default=6 means the a ROI can rotate to 30/60/90/120/150/180 degrees

- num_roi: number of ROIs to be selected, default = 0 refers to automatic determination

- optimal_roi_thres: hyper-parameter for automatic ROI determination, default = 0.03 is suitable for most cases, recommend to be set as 0 when selecting FOVs. If you want to select more ROIs, please lower this parameter

- fusion_weights: the weight of three scores, default=[0.33,0.33,0.33], the sum of three weights should be equal to 1 (if not they will be normalized)

- emphasize_clusters, discard_clusters: prior information about interested and not-interested histology clusters, default = [],[]

- prior_preference: the larger this parameter is, S2Omics will focus more on those interested histology clusters, default=  1

**Return:**

- ROI visualizations on segmentation and raw histology image.

- `best_roi.pickle`: ROI details and score breakdown.

----------------------------------------
s2omics.multiple_sections.p6_cell_label_broadcasting.label_broadcasting
----------------------------------------

**Purpose:**  

After user obtained the spatial omics data of the selected small ROI, we can annotate the superpixels in the paired H&E image with cell type labels.

Afterwards, we can transfer the label information to the previously stained whole-slide H&E image to obtain whole-slide level cell type spatial distribution.

This function trains an Autoencoder-based classifier using ROI-scale spatial omics cell annotations, then broadcast labels to the entire slide.

**Parameters:**

- WSI_datapath: path to the whole slide H&E image

- WSI_save_folder: save path to the whole slide H&E image results
                      
- SO_datapath: path to the spatial omics data and accroding H&E image

- SO_save_folder: save path to the spatial omics data and accroding H&E image results
                      
- WSI_cache_path: path to the extracted histology feature of the WSI, if it is already obtained, default=''

- SO_cache_path: path to the extracted histology feature of the SO, if it is already obtained, default=''

- device: default='cuda:0'

- foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath

**Return:**

- `S2Omics_whole_slide_prediction.jpg`:  Predicted whole-slide cell type map.
