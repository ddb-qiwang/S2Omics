import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from s1_utils import (
        load_pickle, save_pickle, setup_seed)


'''merging histology clusters with high similarity
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    nclusters: final number of clusters for histology segmentation after merging similar groups, default = 15
Return:
    --prefix (the main folder)
    ----pickle_files (subfolder)
        default_cluster_image.pickle: histology cluster matrix, number of clusters is set as default setting 25, 
                                      superpixels with no cell are assigned with -1
        adjusted_cluster_image_num_clusters_{nclusters}.pickle: adjusted histology cluster matrix, 
                                                                superpixels with no cell are assigned with -1
    --prefix (the main folder)
    ----p4_histo_seg_image_output (subfolder)
        adjusted_cluster_image_num_clusters_{nclusters}.jpg: including a dendrogram depicting how we merge the default clusters 
                                                             and the adjusted histology segmentation
'''

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--nclusters', type=int, default=15)
    return parser.parse_args()

def main():
    args = get_args()
    setup_seed(42)

    color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],
                  [188,189,34],[23,190,207],[174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],
                  [196,156,148],[247,182,210],[199,199,199],[219,219,141],[158,218,229],[16,60,90],[128,64,7],
                  [22,80,22],[107,20,20],[74,52,94],[70,43,38],[114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
    color_list_16bit = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  '#E377C2', '#7F7F7F', '#BCBD22', 
                        '#17BECF', '#AEC7E8','#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94',  '#F7B6D2', 
                        '#C7C7C7', '#DBDB8D', '#9EDAE5', '#103C5A', '#804007', '#165016', '#6B1414', '#4A345E', 
                        '#462B26', '#723C61', '#404040', '#5E5E11', '#0C5F68', '#000000']
    
    # load in previously obtained params
    if not os.path.exists(args.prefix+'pickle_files'):
        os.makedirs(args.prefix+'pickle_files')
    pickle_folder = args.prefix+'pickle_files/'
    if not os.path.exists(args.prefix+'p4_histo_seg_image_output'):
        os.makedirs(args.prefix+'p4_histo_seg_image_output')
    save_folder = args.prefix+'p4_histo_seg_image_output/'
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
    cluster_image = load_pickle(pickle_folder+'default_cluster_image.pickle')
    num_histology_clusters = len(np.unique(cluster_image[cluster_image>-1]))
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
    plt.figure()
    Z = load_pickle(pickle_folder+'linkage_matrix.pickle')
    f = fcluster(Z, 4, 'distance')
    fig = plt.figure(figsize=(5, 3))
    dn = dendrogram(Z)

    ncluster_to_reduce = num_histology_clusters - args.nclusters -1
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
    save_pickle(adjusted_cluster_image, pickle_folder+f'adjusted_cluster_image_num_clusters_{num_adjusted_clusters}.pickle')
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
    cluster_names = [f'cluster {i}' for i in range(num_adjusted_clusters)]
    plt.legend((cluster_names), fontsize=12)
    plt.savefig(save_folder+f'adjusted_cluster_image_num_clusters_{num_adjusted_clusters}.jpg', 
                format='jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)

if __name__ == '__main__':
    main()