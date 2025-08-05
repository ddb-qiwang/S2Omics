import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering, BisectingKMeans
from skfuzzy.cluster import cmeans
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from s1_utils import (
        load_pickle, save_pickle, setup_seed)


'''extracting hierarchical features of superpixels using a modified version of UNI
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    save_folder: the name of save folder, user can input the complete path or just the folder name, 
        if so, the folder will be placed under the prefix folder
    foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath
    cache_path: the path to exatracted feature embedding files
    down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate
    n_pc_s: the number of principle components in PCA, default = 80
    clustering_method: the clustering method used for H&E image segmentation, user can select among 
        'kmeans': k-means++, 'fcm': fuzzy c-means, 'louvain': Louvain algorithm, 'leiden': Leiden algorithm 
        default = 'kmeans'
    n_clusters: initial number of clusters for histology segmentation when using kmeans or fcm for clustering. 
        Please notice that this is not the final number of clusters when clustering method is fcm.
Return:
    --prefix (the main folder)
    ---save_folder (subfolder)
    ----pickle_files (subsubfolder)
        cluster_image.pickle: histology cluster matrix, superpixels with no cell are assigned with -1
        linkage_matrix.pickle (if clustering_method='kmeans'): the linkage matrix of clustering results
    ----image_files (subsubfolder)
        cluster_image.jpg: including a dendrogram depicting how we merge the default clusters 
            and the adjusted histology segmentation, , number of clusters is set as default setting 25
'''

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--save_folder', type=str, default='S2Omics_output')
    parser.add_argument('--foundation_model', type=str, default='uni', help='uni, virchow, gigapath')
    parser.add_argument('--cache_path', type=str, default='')
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--n_pcs', type=int, default=80)
    parser.add_argument('--clustering_method', type=str, default='kmeans',help='kmeans, fcm, agglo, bisect, birch, louvain, leiden')
    parser.add_argument('--n_clusters', type=int, default=20, help='if kmeans or fcm')
    parser.add_argument('--resolution', type=float, default=1.0, help='if louvain or leiden')
    parser.add_argument('--if_evaluate', type=bool, default=False)
    return parser.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.prefix+args.save_folder):
        os.makedirs(args.prefix+args.save_folder)
    save_folder = args.prefix+args.save_folder+'/'
    if not os.path.exists(save_folder+'image_files'):
        os.makedirs(save_folder+'image_files')
    image_folder = save_folder+'image_files/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    
    # load in previously obtained params
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    dpi = 1200
    length = np.max(image_shape)//100
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    if dpi*length > np.power(2,16):
        reduce_ratio = np.power(2,16)/(dpi*length)
        plt_figsize = ((image_shape[1]*reduce_ratio)//100,(image_shape[0]*reduce_ratio)//100)
    qc_preserve_indicator =load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)
    
    # load in histology features
    print('Loading histology feature embeddings...')
    he_embed_total = []
    if len(args.cache_path) > 0:
        cache_path = args.cache_path
    else:
        cache_path = pickle_folder
    i = 0
    while 1 > 0:
        if os.path.exists(cache_path+args.foundation_model+f'_embeddings_downsamp_{args.down_samp_step}_part_{i}.pickle'):
            he_embed_part = load_pickle(cache_path+args.foundation_model+f'_embeddings_downsamp_{args.down_samp_step}_part_{i}.pickle')
            he_embed_total.append(he_embed_part)
            i += 1
        else:
            break
    he_embed_total = np.concatenate(he_embed_total)
    del he_embed_part
    print('Sucessfully loaded and normalized all histology feature embeddings!')

    # define color palette
    color_list = np.loadtxt('color_list.txt', dtype='int').tolist()
    with open("color_list_16bit.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
    color_list_16bit = []
    for line in lines:
        color_list_16bit.append(line.strip())
    cluster_color_mapping = np.arange(len(color_list))
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
    pca_encoder = PCA(n_components=args.n_pcs)
    pca_encoder.fit(he_embed_qc)
    he_embed_qc_pca = pca_encoder.fit_transform(he_embed_qc)

    print(f'Start segmenting the histology image, clustering method: {args.clustering_method}')
    if args.clustering_method == 'kmeans':
        uni_cluster = KMeans(n_clusters=args.n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if args.clustering_method == 'fcm':
        train = he_embed_qc_pca.T
        center, u,_,_,_,_,_ = cmeans(train, m=2, c=args.n_clusters, error=0.005, maxiter=1000)
        for i in u:
            uni_cluster = np.argmax(u, axis=0)
    if args.clustering_method == 'agglo':
        uni_cluster = AgglomerativeClustering(n_clusters=args.n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if args.clustering_method == 'bisect':
        uni_cluster = BisectingKMeans(n_clusters=args.n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if args.clustering_method == 'birch':
        uni_cluster = Birch(n_clusters=args.n_clusters).fit_predict(he_embed_qc_pca).astype('int')
    if args.clustering_method == 'louvain':
        adata = sc.AnnData(he_embed_qc_pca)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
        sc.tl.louvain(adata, resolution=args.resolution)
        uni_cluster = adata.obs['louvain']
    if args.clustering_method == 'leiden':
        adata = sc.AnnData(he_embed_qc_pca)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
        sc.tl.leiden(adata, resolution=args.resolution)
        uni_cluster = adata.obs['leiden']
    
    cluster_image = -1*np.ones(image_shape)
    cluster_image[qc_mask & down_samp_mask] = uni_cluster
    cluster_image = cluster_image[down_samp_mask]
    cluster_image = np.reshape(cluster_image, [down_samp_shape[0],down_samp_shape[1]])
    histology_clusters_new = np.unique(cluster_image[cluster_image>-1])
    n_clusters = len(histology_clusters_new)
    cluster_image_copy = cluster_image.copy()
    for i in range(n_clusters):
        cluster_image[cluster_image_copy==histology_clusters_new[i]] = i
    save_pickle(cluster_image, pickle_folder+'cluster_image.pickle')
    
    cluster_vector = cluster_image[cluster_image>-1]
    cluster_centroids = []
    for cluster in range(n_clusters):
        cluster_centroid = np.mean(he_embed_qc[cluster_vector==cluster], axis=0)
        #print(cluster, cluster_centroid)
        cluster_centroids.append(cluster_centroid)
    Z = linkage(cluster_centroids, 'average')
    save_pickle(Z, pickle_folder+'linkage_matrix.pickle')

    if args.if_evaluate:
        s1 = silhouette_score(he_embed_qc_pca, uni_cluster, metric='euclidean') 
        s2 = calinski_harabasz_score(he_embed_qc_pca, uni_cluster) 
        s3 = davies_bouldin_score(he_embed_qc_pca, uni_cluster)
        print(f'''Finish segmentation. Segmented the histology image into {n_clusters} clusters. 
        Sihouette score: {s1}
        C-H score: {s2}
        DBI: {s3}''')
        save_pickle([s1,s2,s3], pickle_folder+'clustering_metrics.pickle')

    fig = plt.figure(figsize=(plt_figsize[0],plt_figsize[1]))
    # output rgb cluster image
    cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
    for cluster in range(n_clusters):
        cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
    cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
    plt.imshow(cluster_image_rgb)
    plt.title('histology segmentation', fontsize=20)
    legend_x = legend_y = np.zeros(n_clusters)
    for i in range(n_clusters):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    cluster_names = [f'cluster {i}' for i in range(1, n_clusters+1)]
    plt.legend((cluster_names), fontsize=12)
    plt.savefig(image_folder+f'cluster_image_num_clusters_{n_clusters}.jpg', 
                format='jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)
    print('Segmentation image is stored at: '+image_folder+f'cluster_image_num_clusters_{n_clusters}.jpg')


if __name__ == '__main__':
    main()