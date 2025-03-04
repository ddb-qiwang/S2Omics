import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from s1_utils import (
        load_pickle, save_pickle, setup_seed)


'''extracting hierarchical features of superpixels using a modified version of UNI
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate
    num_histology_clusters: number of clusters for histology segmentation, default = 25
Return:
    --prefix (the main folder)
    ----pickle_files (subfolder)
        default_cluster_image.pickle: histology cluster matrix, superpixels with no cell are assigned with -1
        linkage_matrix.pickle: the linkage matrix of clustering results
    --prefix (the main folder)
    ----histo_seg_image_output (subfolder)
        default_cluster_image.jpg: including a dendrogram depicting how we merge the default clusters 
                                   and the adjusted histology segmentation, , number of clusters is set as default setting 25
'''

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--num_histology_clusters', type=int, default=25)
    return parser.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.prefix+'pickle_files'):
        os.makedirs(args.prefix+'pickle_files')
    pickle_folder = args.prefix+'pickle_files/'
    if not os.path.exists(args.prefix+'histo_seg_image_output'):
        os.makedirs(args.prefix+'histo_seg_image_output')
    save_folder = args.prefix+'histo_seg_image_output/'
    
    # load in previously obtained params
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    dpi = 1200
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    sns_figsize = (image_shape[1]//100+5,image_shape[0]//100)
    qc_preserve_indicator =load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)
    
    # load in histology features
    he_embed_total = []
    i = 0
    while 1 > 0:
        if os.path.exists(pickle_folder+f'uni_embeddings_downsamp_{args.down_samp_step}_part_{i}.pickle'):
            he_embed_part = load_pickle(pickle_folder+f'uni_embeddings_downsamp_{args.down_samp_step}_part_{i}.pickle')
            he_embed_total.append(he_embed_part)
            i += 1
        else:
            break
    he_embed_total = np.concatenate(he_embed_total)
    del he_embed_part

    # define color palette
    color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],
                  [188,189,34],[23,190,207],[174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],
                  [196,156,148],[247,182,210],[199,199,199],[219,219,141],[158,218,229],[16,60,90],[128,64,7],
                  [22,80,22],[107,20,20],[74,52,94],[70,43,38],[114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
    color_list_16bit = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  '#E377C2', '#7F7F7F', '#BCBD22', 
                        '#17BECF', '#AEC7E8','#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94',  '#F7B6D2', 
                        '#C7C7C7', '#DBDB8D', '#9EDAE5', '#103C5A', '#804007', '#165016', '#6B1414', '#4A345E', 
                        '#462B26', '#723C61', '#404040', '#5E5E11', '#0C5F68', '#000000']
    cluster_color_mapping = np.arange(args.num_histology_clusters)
    colors = np.array(color_list_16bit)[cluster_color_mapping]

    setup_seed(42)
    # create a mask for down-sampled superpixels in all superpixels
    down_samp_mask = np.full(image_shape, False)
    down_samp_shape = [(image_shape[0]-1)//args.down_samp_step+1, (image_shape[1]-1)//args.down_samp_step+1]
    for i in range(down_samp_shape[0]):
        for j in range(down_samp_shape[1]):
            down_samp_mask[i*args.down_samp_step,j*args.down_samp_step] = True
    
    # PCA+kmeans to cluster the superpixels into morphology clusters
    he_embed_qc = he_embed_total[qc_mask[down_samp_mask]]
    del he_embed_total
    pca_encoder = PCA(n_components=80)
    pca_encoder.fit(he_embed_qc)
    he_embed_qc_pca = pca_encoder.fit_transform(he_embed_qc)
    uni_cluster = KMeans(n_clusters=args.num_histology_clusters).fit_predict(he_embed_qc_pca).astype('int')
    cluster_image = -1*np.ones(image_shape)
    cluster_image[qc_mask & down_samp_mask] = uni_cluster
    cluster_image = cluster_image[down_samp_mask]
    cluster_image = np.reshape(cluster_image, [down_samp_shape[0],down_samp_shape[1]])
    histology_clusters_new = np.unique(cluster_image[cluster_image>-1])
    num_histology_clusters_new = len(histology_clusters_new)
    cluster_image_copy = cluster_image.copy()
    for i in range(num_histology_clusters_new):
        cluster_image[cluster_image_copy==histology_clusters_new[i]] = i
    save_pickle(cluster_image, pickle_folder+'default_cluster_image.pickle')

    # calculate the distances between clusters and visualize them using dendrogram
    cluster_centroids = []
    for cluster in range(num_histology_clusters_new):
        cluster_centroid = np.mean(he_embed_qc[uni_cluster==cluster], axis=0)
        cluster_centroids.append(cluster_centroid)
    Z = linkage(cluster_centroids, 'average')
    save_pickle(Z, pickle_folder+'linkage_matrix.pickle')
    f = fcluster(Z, 4, 'distance')

    fig = plt.figure(figsize=(2*plt_figsize[0],plt_figsize[1]))
    plt.subplot(1,2,1)
    dn = dendrogram(Z)
    plt.show()
    plt.title('distances between clusters', fontsize=20)

    plt.subplot(1,2,2)
    # output rgb cluster image
    cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
    for cluster in range(num_histology_clusters_new):
        cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
    cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
    plt.imshow(cluster_image_rgb)
    plt.title('histology segmentation', fontsize=20)
    legend_x = legend_y = np.zeros(num_histology_clusters_new)
    for i in range(num_histology_clusters_new):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    cluster_names = [f'cluster {i}' for i in range(num_histology_clusters_new)]
    plt.legend((cluster_names), fontsize=12)
    plt.savefig(save_folder+f'default_cluster_image_num_clusters_{num_histology_clusters_new}.jpg', 
                format='jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)


if __name__ == '__main__':
    main()