# instructions

before run S2Omics, you need to:
1. create an environment referring to requirements.txt
2. download the model file for foundation model UNI
2. rename the raw H&E image file into 'he-raw.jpg/tiff/svs/png'
3. create pixel-size.txt & pixel-size-raw.txt under the same folder as he-raw, the files refer to raw/target microns per pixel, please set pixel-size.txt as 0.5
4*. if you want to run p6_cell_label_broadcasting.py, 3 additional files are needed, please refer to the python file descriptions

Use xenium breast cancer s1r1 as an example,
suppose that all files require are under the folder '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1'
and the model parameters file is under '/home/msyuan/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k'

# if you only want to select ROI and don't need to predict cell type labels for unsampled cells:
1. python p1_histology_preprocess.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/'
2. python p2_superpixel_quality_control.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --m 0.7 
    (if you think this step filter too many superpixels, select a smaller m)
3. python p3_feature_extraction.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --model_path '/home/msyuan/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/' --down_samp_step 5
    (if you want to extract histology features for more superpixels, select a smaller positive integar down_samp_step)
4. python p4_get_histology_segmentation.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --down_samp_step 5 --num_histology_clusters 10
    (we recommand using a bigger num_histology_clusters as we can mitigate overclustering through p5)
5. python p5_merge_over_clusters.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --dist_thres 0
    (please refer to the cluster_image_plt.jpg and cluster_distance.jpg to select a proper dist_thres, deault = 0 means do not merge any cluster)
6. python p6_roi_selection.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --down_samp_step 5 --roi_size 2 2 --num_roi 2
    (here the command means select 2 2mm x 2mm ROIs, if you want to automatically determine the optimal number of ROIs, set --num_roi 0)

# if you also want to predict cell type labels for unsampled cells:
1. python p1_histology_preprocess.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/'
2. python p2_superpixel_quality_control.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --m 0.7 
    (if you think this step filter too many superpixels, select a smaller m)
3. python p3_feature_extraction.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --model_path '/home/msyuan/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/' --down_samp_step 1
    (if you want to extract histology features for more superpixels, select a smaller positive integar down_samp_step)
4. python p4_get_histology_segmentation.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --down_samp_step 1 --num_histology_clusters 10
    (we recommand using a bigger num_histology_clusters as we can mitigate overclustering through p5)
5. python p5_merge_over_clusters.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --down_samp_step 1 --dist_thres 0
    (please refer to the cluster_image_plt.jpg and cluster_distance.jpg to select a proper dist_thres, deault = 0 means do not merge any cluster)
6. python p6_roi_selection.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --down_samp_step 1 --roi_size 2 2 --num_roi 2
    (here the command means select 2 2mm x 2mm ROIs, if you want to automatically determine the optimal number of ROIs, set --num_roi 0)
7. python p7_cell_label_broadcasting.py '/home/msyuan/Datasets/S2Omics_test_xenium_bc_s1r1/' --roi_size 2 2

