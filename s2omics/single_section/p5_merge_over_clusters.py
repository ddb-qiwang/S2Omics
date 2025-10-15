import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from ..s1_utils import (
        load_pickle, save_pickle, setup_seed)


def merge_over_clusters(prefix, save_folder,
                       target_n_clusters=15):
    '''
    merging histology clusters with high similarity
    Parameters:
        prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
        save_folder: the name of save folder, user can input the complete path or just the folder name, 
            if so, the folder will be placed under the prefix folder
        target_n_clusters: the final number of clusters user want to preserve
    '''
     # load in previously obtained params
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = save_folder+'/'
    if not os.path.exists(save_folder+'image_files'):
        os.makedirs(save_folder+'image_files')
    image_folder = save_folder+'image_files/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    dpi = 1200
    length = np.max(image_shape)//100
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    if dpi*length > np.power(2,16):
        reduce_ratio = np.power(2,16)/(dpi*length)
        plt_figsize = ((image_shape[1]*reduce_ratio)//100,(image_shape[0]*reduce_ratio)//100)
    qc_preserve_indicator = load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)

    # load in previous cluster image and generate its rgb image
    cluster_image = load_pickle(pickle_folder+'cluster_image.pickle')
    num_histology_clusters = len(np.unique(cluster_image[cluster_image>-1]))
    
    if num_histology_clusters > target_n_clusters:
        print('Merging over-clusters...')
        setup_seed(42)
        
        color_list = np.loadtxt(os.path.join(os.path.dirname(__file__), '../color_list.txt'), dtype='int').tolist()
        with open(os.path.join(os.path.dirname(__file__), '../color_list_16bit.txt'), "r", encoding="utf-8") as file:
            lines = file.readlines()
        color_list_16bit = []
        for line in lines:
            color_list_16bit.append(line.strip())
        cluster_color_mapping = np.arange(len(color_list))
        colors = np.array(color_list_16bit)[cluster_color_mapping]

        cluster_color_mapping = np.arange(num_histology_clusters)
        colors = np.array(color_list_16bit)[cluster_color_mapping]
        cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
        cluster_color_mapping = np.arange(num_histology_clusters)
        for cluster in range(num_histology_clusters):
            cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
        cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
        cluster_image_mask = np.full(np.shape(cluster_image), False)
        cluster_image_mask[np.where(cluster_image>-1)] = True

        # show distances between clusters
        Z = load_pickle(pickle_folder+'linkage_matrix.pickle')
        f = fcluster(Z, 4, 'distance')
        dn = dendrogram(Z)

        ncluster_to_reduce = num_histology_clusters - target_n_clusters -1
        dist_thres = Z[ncluster_to_reduce, 2]
        # merge cluster pairs whose distances are smaller than the threshold
        adjusted_he_cluster = cluster_image[cluster_image>-1].copy()
        merge_index = np.where(Z[:,2]<=dist_thres)[0]
        for index in merge_index:
            cluster_1 = Z[index,0]
            cluster_2 = Z[index,1]
            adjusted_he_cluster[adjusted_he_cluster==cluster_1] = num_histology_clusters+index
            adjusted_he_cluster[adjusted_he_cluster==cluster_2] = num_histology_clusters+index
        curr_cluster = 0
        adjusted_unique_clusters = np.sort(np.unique(adjusted_he_cluster))
        for cluster in adjusted_unique_clusters:
            adjusted_he_cluster[adjusted_he_cluster==cluster] = curr_cluster
            curr_cluster += 1
        adjusted_cluster_image = cluster_image.copy()
        adjusted_cluster_image[cluster_image>-1] = adjusted_he_cluster
        num_adjusted_clusters = len(np.unique(adjusted_cluster_image[adjusted_cluster_image>-1]))
        curr_num_clusters = num_adjusted_clusters
        save_pickle(adjusted_cluster_image, pickle_folder+'adjusted_cluster_image.pickle')
        print(f'Combined the original {num_histology_clusters} clusters into {num_adjusted_clusters} clusters.')
        # visualize the adjusted cluster image
        adjusted_cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],
                                                  np.shape(cluster_image)[1],3])
        for cluster in range(num_adjusted_clusters):
            adjusted_cluster_image_rgb[adjusted_cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
        adjusted_cluster_image_rgb = np.array(adjusted_cluster_image_rgb, dtype='int')
        
        fig = plt.figure(figsize=(2*plt_figsize[0],plt_figsize[1]))
        plt.subplot(1,2,1)
        dn = dendrogram(Z)
        plt.hlines(y=dist_thres, xmin=0, xmax=np.max(dn['icoord'])+10, color='r')
        plt.title('Distances between default clusters', fontsize=20)

        plt.subplot(1,2,2)
        plt.imshow(adjusted_cluster_image_rgb)
        plt.title('histology segmentation', fontsize=20)
        legend_x = legend_y = np.zeros(num_adjusted_clusters)
        for i in range(num_adjusted_clusters):
            plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
        cluster_names = [f'cluster {i}' for i in range(1, num_adjusted_clusters+1)]
        plt.legend((cluster_names), fontsize=12)
        plt.savefig(image_folder+f'adjusted_cluster_image_num_clusters_{num_adjusted_clusters}.jpg', 
                    format='jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)
        print('Adjusted segmentation image is stored at: '+image_folder+f'adjusted_cluster_image_num_clusters_{num_adjusted_clusters}.jpg')