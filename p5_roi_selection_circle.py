import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import math
from s1_utils import (
        load_image, load_pickle, save_pickle, setup_seed)


''' select best ROI(s)
Args:
    multiple_images_prefix: .txt file that contains the folder paths of H&E stained images, user can refer to 'multiple_images_prefix.txt'
    multiple_images_save_folder: .txt file that contains the names of save folders, user can refer to 'multiple_images_save_folder.txt'
    has_annotation: if use annotation instead of histology segmentation for ROI selection, default=False
    multiple_images_cache_path: .txt file that contains the paths to exatracted feature embedding files, user can refer to 'multiple_images_cache_path.txt'
    down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate
    roi_size: the physical size (mm x mm) of ROIs, default = [6.5, 6.5] which is the physical size for Visium HD ROI
    num_roi: number of ROIs to be selected, default = 0 refers to automatic determination
    fusion_weights: the weight of three scores, default=[0.33,0.33,0.33], the sum of three weights should be equal to 1 (if not they will be normalized)
    positive_prior, negative_prior: prior information about interested and not-interested histology clusters, default = [],[]
    prior_preference: to what extend should the positive prior clusters be emphasized, default = 2
    optimal_roi_thres: hyper-parameter for automatic ROI determination, default = 0.03 is suitable for most cases, recommend to be set as 0 when selecting FOVs. If you want to select more ROIs, please lower this parameter
Return:
    --prefix (the main folder)
    ---save_folder (subfolder)
    ----main_output (subsubfolder)
            best_roi_on_histology_segmentations.jpg: the selected ROIs on histology segmentation result
            best_roi_on_he.jpg: the selected ROIs on H&E image
    ----roi_selection_detailed_output (subsubfolder)
    -----roi_size_{roi_size[0]}_{roi_size[1]} (subsubsubfolder)
    ------prior_preference_{prior_preference} (subsubsubsubfolder)
              best_roi.pickle: [best_num_roi,best_roi_list,best_roi_mask_list,best_comp_list,best_roi_score_list] 
                               contains the ROIs information for best 1/2/.../best_num_roi ROIs
              best_{n}_roi_on_histo_clusters.jpg: additional information about ROI selection based on current histology segmentation.
                                                  the number of ROIs R is automatically determined by S2Omics, we save and show our 
                                                  selection for best R ROIs, R+1 ROIs and R+2 ROIs for reference
'''

def euclid_distance(point1, point2):
    tmp = np.array(point1)-np.array(point2)
    
    return np.sqrt(np.sum(tmp*tmp))
    
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0
    return num / denom

def logistic_func(x, L=1, k=2):  
    return L/(1+np.exp(-k*x))

def region_selection_few(cluster_image, valid_mask, num_clusters, window_size, target_proportion,
                         curr_roi, curr_roi_mask, curr_comp, num_roi=2, 
                         fusion_weights=[1,1,1], num_samp=10000, samp_step=5):
    '''
    region selection function for less than 2 ROIs, based on previous selections (curr_roi,...)
    '''
    total_cell_num = np.sum(cluster_image>-1)
    valid_points = np.where(valid_mask > 0)
    valid_region = [np.min(valid_points[0]),np.max(valid_points[0]),
                    np.min(valid_points[1]),np.max(valid_points[1])]

    # introduce previous ROI selections
    # best_roi = [] means no previous selection information 
    best_roi_score = [0]
    best_roi = curr_roi.copy()
    best_roi_mask = curr_roi_mask.copy()
    best_comp = curr_comp.copy()
    mask_shape = np.shape(cluster_image)
    min_window_size = np.min(window_size)
    if len(target_proportion) == 0:
        target_proportion = np.ones(num_clusters)/num_clusters
    
    # randomly sample num_samp times and keep the seletion with highest total roi score (together with previous selections)
    for samp in tqdm(range(num_samp)):       
        tmp_roi = curr_roi.copy()
        tmp_roi_mask = curr_roi_mask.copy()
        tmp_comp = curr_comp.copy()
        centroids_index = np.random.choice(np.arange(0,len(valid_points[0]),samp_step), num_roi)
        x_centroids = valid_points[0][centroids_index]
        y_centroids = valid_points[1][centroids_index]

        # sample num_roi ROIs
        for roi in range(num_roi):      
            centroid = [x_centroids[roi], y_centroids[roi]]
            tmp_roi.append(centroid)
            tmp_mask = np.zeros(np.shape(cluster_image), dtype=np.uint8)  # 返回与图像 img1 尺寸相同的全零数组
            cv2.circle(tmp_mask, centroid, window_size[0], (255, 255, 255), -1)
            tmp_roi_mask.append(tmp_mask.astype('bool'))

        tmp_roi_mask_total = tmp_roi_mask[0]
        for roi in range(1, len(tmp_roi)):
            tmp_roi_mask_total = tmp_roi_mask_total | tmp_roi_mask[roi]
        tmp_clusters = cluster_image[tmp_roi_mask_total]
        for cluster in range(num_clusters):
            tmp_comp[cluster] = np.sum(tmp_clusters==cluster)
        tmp_prop = tmp_comp/(len(tmp_roi)*int(np.pi*window_size[0]**2))

        # compute the current roi score for len(best_roi) ROIs
        tmp_scale_score = logistic_func(np.sum(tmp_comp)/total_cell_num)
        tmp_valid_score = np.sqrt(np.sum(tmp_prop))
        tmp_balance_score = cosine_similarity(tmp_prop, target_proportion)
        roi_score = np.power(np.power(tmp_scale_score,fusion_weights[0])*np.power(tmp_valid_score,fusion_weights[1])*np.power(tmp_balance_score,fusion_weights[2]), 1/np.sum(fusion_weights))
        
        if roi_score > best_roi_score[0]:
            best_roi_score = [roi_score, tmp_scale_score, tmp_valid_score, tmp_balance_score]
            best_roi = tmp_roi
            best_roi_mask = tmp_roi_mask
            best_comp = tmp_comp
    print(f'''Current best ROI: {best_roi}
    roi score: {best_roi_score[0]}
    scale score: {best_roi_score[1]}
    valid score: {best_roi_score[2]}
    balance score: {best_roi_score[3]}''')

    return best_roi, best_roi_mask, best_comp, best_roi_score
        
    

def region_selection_random(save_folder, cluster_image, cluster_image_rgb, valid_mask, num_clusters, 
                            window_size, num_roi, fusion_weights=[1,1,1], target_proportion=[], 
                            optimal_roi_thres=0.03, num_samp_per_iter=10000, samp_step=5, save_plot=True):
    '''
    valid_points: [[valid_xs], [valid_ys]]
    window_size = [width, length]
    
    '''
    best_roi_list = []
    best_roi_mask_list = []
    best_comp_list = []
    best_roi_score_list = []
    if len(target_proportion) == 0:
        target_proportion = np.ones(num_clusters)/num_clusters
    
    if_stop = False
    curr_num_roi = 0
    pre_best_roi, pre_best_roi_mask = [], []
    pre_best_comp = np.zeros(num_clusters)
    pre1_best_roi_score, pre2_best_roi_score = [0], [0]
    
    while if_stop == False:      
        # find the best (curr_num_roi+1)^th ROI
        best_roi,best_roi_mask,best_comp,best_roi_score = region_selection_few(cluster_image, valid_mask,
                                                                               num_clusters, window_size,
                                                                               target_proportion, pre_best_roi,
                                                                               pre_best_roi_mask, pre_best_comp, 1, fusion_weights,
                                                                               num_samp_per_iter, samp_step)
        # show the best (curr_num_roi+1)^th ROI
        if save_plot==True:
            plt.figure()
            plt.imshow(cluster_image_rgb)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            for i in range(len(best_roi)):
                ax.add_patch(plt.Circle([best_roi[i][0],best_roi[i][1]], window_size[0],
                                        color='red',fill=False, linewidth=2))
            plt.savefig(save_folder+f'best_{curr_num_roi+1}_roi_on_histo_clusters.jpg', 
                        format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
        # if roi score increased less than optimal_roi_thres, there's no need to select this one more ROI  
        if best_roi_score[0] - pre2_best_roi_score[0] < 2*optimal_roi_thres and pre1_best_roi_score[0] - pre2_best_roi_score[0] < optimal_roi_thres and curr_num_roi >= num_roi:  
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            break    
        else:
            best_roi_list.append(best_roi)
            best_roi_mask_list.append(best_roi_mask)
            best_comp_list.append(best_comp)
            best_roi_score_list.append(best_roi_score)   
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            pre2_best_roi_score = pre1_best_roi_score
            pre1_best_roi_score = best_roi_score
            if curr_num_roi == num_roi:
                if_stop = True
                return curr_num_roi, best_roi_list, best_roi_mask_list, best_comp_list, best_roi_score_list

        # find the best (curr_num_roi+1)^th ROI
        best_roi,best_roi_mask,best_comp,best_roi_score = region_selection_few(cluster_image, valid_mask,
                                                                               num_clusters, window_size, 
                                                                               target_proportion, pre_best_roi,
                                                                               pre_best_roi_mask, pre_best_comp, 2, fusion_weights,
                                                                               num_samp_per_iter, samp_step)
        # save the best (curr_num_roi+1)^th ROI
        if save_plot==True:
            plt.figure()
            plt.imshow(cluster_image_rgb)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            for i in range(len(best_roi)):
                ax.add_patch(plt.Circle([best_roi[i][0], best_roi[i][1]], window_size[0],
                                        color='red', fill=False, linewidth=2))
            plt.savefig(save_folder+f'best_{curr_num_roi+1}_roi_on_histo_clusters.jpg', 
                        format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
        # if roi score increased less than optimal_roi_thres, there's no need to select this one more ROI
        if best_roi_score[0] - pre2_best_roi_score[0] < 2*optimal_roi_thres and pre1_best_roi_score[0] - pre2_best_roi_score[0] < optimal_roi_thres and curr_num_roi >= num_roi:  
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            break
        else:
            pre_best_roi,pre_best_roi_mask,pre_best_comp = best_roi,best_roi_mask,best_comp
            best_roi_list.append(best_roi)
            best_roi_mask_list.append(best_roi_mask)
            best_comp_list.append(best_comp)
            best_roi_score_list.append(best_roi_score)   
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            pre2_best_roi_score = pre1_best_roi_score
            pre1_best_roi_score = best_roi_score
            if curr_num_roi == num_roi:
                if_stop = True
                return curr_num_roi, best_roi_list, best_roi_mask_list, best_comp_list, best_roi_score_list
                
    print(f'Found the optimal number of ROI: {curr_num_roi-1}. Program finished.')
    return curr_num_roi-1, best_roi_list[:-1], best_roi_mask_list[:-1], best_comp_list[:-1], best_roi_score_list[:-1]



def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('multiple_images_prefix', type=str)
    parser.add_argument('multiple_images_save_folder', type=str)
    parser.add_argument('--has_annotation', type=bool, default=False)
    parser.add_argument('--multiple_images_cache_path', type=str, default='')
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--roi_size', type=float, nargs='+', default=[6.5,6.5])
    parser.add_argument('--num_roi', type=int, default=0) # 0 means automatically determine the optimal number of ROIs
    parser.add_argument('--fusion_weights', type=float, nargs='+', default=[0.33,0.33,0.33]) # [size_score_wight, coverage_score_weight, balance_score_weight], sum of them should be equal to 1
    parser.add_argument('--positive_prior', type=int, nargs='+', default=[]) # [emphasize_clusters]
    parser.add_argument('--negative_prior', type=int, nargs='+', default=[]) # [discard_clusters]
    parser.add_argument('--prior_preference', type=int, default=1) 
    parser.add_argument('--optimal_roi_thres', type=float, default=0.03) # threshold for optimal ROI number determination 
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_samp_init', type=int, default=500)
    return parser.parse_args()

def main():
    args = get_args()
    setup_seed(args.random_seed)

    # define color palette
    color_list = np.loadtxt('color_list.txt', dtype='int').tolist()
    with open("color_list_16bit.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
    color_list_16bit = []
    for line in lines:
        color_list_16bit.append(line.strip())

    with open(args.multiple_images_prefix, "r", encoding="utf-8") as file:
        lines = file.readlines()
    prefix_list = [line.split()[0] for line in lines]

    with open(args.multiple_images_save_folder, "r", encoding="utf-8") as file:
        lines = file.readlines()
    save_folder_list_raw = [line.split()[0] for line in lines]
    assert len(prefix_list) == len(save_folder_list_raw)
    
    if len(args.multiple_images_cache_path) > 0:
        with open(args.multiple_images_cache_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
        cache_path_list = [line.split()[0] for line in lines]
        assert len(prefix_list) == len(cache_path_list)

    n_images = len(prefix_list)

    dpi = 600
    save_folder_list = []
    image_folder_list = []
    pickle_folder_list = []
    image_shape_list = []
    cluster_image_list = []
    cluster_image_rgb_list = []
    cluster_image_mask_list = []
    
    for i in range(n_images):
        prefix = prefix_list[i]
        save_folder = save_folder_list_raw[i]
        if len(args.multiple_images_cache_path) > 0:
            cache_path = cache_path_list[i]
        if not os.path.exists(prefix+save_folder):
            os.makedirs(prefix+save_folder)
        save_folder = prefix+save_folder+'/'
        save_folder_list.append(save_folder)
        if not os.path.exists(save_folder+'image_files'):
            os.makedirs(save_folder+'image_files')
        image_folder = save_folder+'image_files/'
        image_folder_list.append(image_folder)
        if not os.path.exists(save_folder+'pickle_files'):
            os.makedirs(save_folder+'pickle_files')
        pickle_folder = save_folder+'pickle_files/'
        pickle_folder_list.append(pickle_folder)
        if i == 0:
            if not os.path.exists(save_folder+'main_output'):
                os.makedirs(save_folder+'main_output')
            main_output_folder = save_folder+'main_output/'
            if not os.path.exists(save_folder+f'roi_selection_detailed_output/circle_roi_size_{args.roi_size[0]}_{args.roi_size[1]}'):
                os.makedirs(save_folder+f'roi_selection_detailed_output/circle_roi_size_{args.roi_size[0]}_{args.roi_size[1]}')
            roi_save_folder = save_folder+f'roi_selection_detailed_output/circle_roi_size_{args.roi_size[0]}_{args.roi_size[1]}/'
            if not os.path.exists(roi_save_folder+f'prior_preference_{args.prior_preference}'):
                os.makedirs(roi_save_folder+f'prior_preference_{args.prior_preference}')
            save_subfolder = roi_save_folder+f'prior_preference_{args.prior_preference}/'
    
        # load in previously obtained params
        shapes = load_pickle(pickle_folder+'shapes.pickle')
        image_shape = shapes['tiles']
        image_shape_list.append(image_shape)

        if args.has_annotation:
            cluster_image = load_pickle(pickle_folder+'annotation.pickle')
            category_names = load_pickle(pickle_folder+'category_names.pickle')
        elif len(args.multiple_images_cache_path) > 0:
            cluster_image = load_pickle(cache_path_list[i])
        else:
            if os.path.exists(pickle_folder+'adjusted_cluster_image.pickle'):
                cluster_image = load_pickle(pickle_folder+'adjusted_cluster_image.pickle')
            else:
                cluster_image = load_pickle(pickle_folder+'cluster_image.pickle')
        cluster_image_list.append(cluster_image)

    print([np.shape(image) for image in cluster_image_list])
    
    num_roi = args.num_roi
    physical_size = args.roi_size
    emphasize_clusters = args.positive_prior
    fusion_weights = args.fusion_weights
    discard_clusters = args.negative_prior
    optimal_roi_thres = args.optimal_roi_thres
    window_size_raw = [int(125*physical_size[0]),int(125*physical_size[1])]
    window_size = [int(125*physical_size[0]/args.down_samp_step),int(125*physical_size[1]/args.down_samp_step)]
    num_samp = args.num_samp_init*math.ceil((image_shape[0]*image_shape[1])/(window_size_raw[0]*window_size_raw[1]))
    samp_step = math.ceil(5/args.down_samp_step)

        
    # concatenate the cluster images and masks into a row
    padding_size = int(np.sqrt(window_size[0]**2+window_size[1]**2)/2)
    concat_image_shape = [np.sum(image_shape_list,axis=0)[0]+(n_images-1)*padding_size,np.max(image_shape_list,axis=0)[1]]
    concat_cluster_image = -1*np.ones(concat_image_shape)
    concat_cluster_image_mask = np.full(concat_image_shape, False)
    concat_cluster_image_rgb = 255*np.ones([concat_image_shape[0],concat_image_shape[1],3])
    starting_x = 0
    for i in range(n_images):
        image_shape = image_shape_list[i]
        cluster_image = cluster_image_list[i]
        concat_cluster_image[starting_x:starting_x+image_shape[0],0:image_shape[1]] = cluster_image
        starting_x += image_shape[0] + padding_size

    num_histology_clusters = len(np.unique(concat_cluster_image[concat_cluster_image>-1]))
    cluster_color_mapping = np.arange(num_histology_clusters)
    cluster_color_mapping = [10,9,3,1,5,4,7,0,11,8]
    for cluster in range(num_histology_clusters):
        concat_cluster_image_rgb[concat_cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
    concat_cluster_image_rgb = np.array(concat_cluster_image_rgb, dtype='int')
    concat_cluster_image_mask = np.full(np.shape(concat_cluster_image), False)
    concat_cluster_image_mask[np.where(concat_cluster_image>-1)] = True
    
    target_proportion = np.ones(num_histology_clusters)
    if len(discard_clusters) > 0:
        target_proportion[discard_clusters] -= 1
    if len(emphasize_clusters) > 0:
        emphasize_clusters = [emphasize_clusters[i]-1 for i in range(len(emphasize_clusters))]
        target_proportion[emphasize_clusters] += args.prior_preference
    target_proportion = target_proportion/np.sum(target_proportion)

    length = np.max(concat_image_shape)//100
    plt_figsize = (concat_image_shape[1]//100,concat_image_shape[0]//100)
    if dpi*length > np.power(2,16):
        reduce_ratio = np.power(2,16)/(dpi*length)
        plt_figsize = ((concat_image_shape[1]*reduce_ratio)//100,(concat_image_shape[0]*reduce_ratio)//100)
    
    # select ROIs
    print('Sampling ROI candidates...')
    best_num_roi, best_roi_list, best_roi_mask_list, best_comp_list, best_roi_score_list = \
    region_selection_random(save_subfolder, concat_cluster_image, concat_cluster_image_rgb, concat_cluster_image_mask,
                            num_histology_clusters, window_size, num_roi, fusion_weights=fusion_weights, target_proportion=target_proportion, 
                            optimal_roi_thres=optimal_roi_thres, 
                            num_samp_per_iter=num_samp, samp_step=samp_step, save_plot=True)
    print(f'''Find the best {best_num_roi} ROI(s) with: 
    ROI score: {best_roi_score_list[-1][0]}
    Scale score: {best_roi_score_list[-1][1]}
    Coverage score: {best_roi_score_list[-1][2]}
    Balance score: {best_roi_score_list[-1][3]}
    ''')
    save_pickle([best_roi_list, best_roi_mask_list, best_comp_list, best_roi_score_list], 
                save_subfolder+'best_roi.pickle')
    
    plt.figure(figsize=plt_figsize)
    best_roi = best_roi_list[-1]
    best_roi_score = best_roi_score_list[-1][0]
    plt.imshow(concat_cluster_image_rgb)
    fontdict = {'fontsize':12}
    plt.text(1,np.shape(concat_cluster_image)[0]-1,f'ROI score: {round(best_roi_score,3)}',fontdict=fontdict)
    if args.has_annotation:
        plt.title('annotation', fontsize=20)
    else:
        plt.title(f'num_clusters = {num_histology_clusters}', fontsize=20)
    ax = plt.gca()

    legend_x = legend_y = np.zeros(num_histology_clusters)
    for i in range(num_histology_clusters):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    if args.has_annotation:
        plt.legend((category_names), fontsize=12)
    else:
        plt.legend(([f'Cluster {i}' for i in range(1, num_histology_clusters+1)]), fontsize=12)

    for i in range(len(best_roi)):
        ax.add_patch(plt.Circle([best_roi[i][0], best_roi[i][1]], window_size[0],
                                color='red', fill=False, linewidth=2))
    if args.has_annotation:
        plt.savefig(main_output_folder+'best_roi_on_annotation.jpg', 
                format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
    else:
        plt.savefig(main_output_folder+'best_roi_on_histology_segmentations.jpg', 
                    format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
    plt.close()
    if args.has_annotation:
        print('Best ROI on annotation image is stored at '+main_output_folder+'best_roi_on_annotation.jpg')
    else:
        print('Best ROI on histology segmentation image is stored at '+main_output_folder+'best_roi_on_histology_segmentations.jpg')

    plt.figure(figsize=plt_figsize)
    plt.imshow(he)
    if args.has_annotation:
        plt.title('annotation', fontsize=20)
    else:
        plt.title('H&E image', fontsize=20)
    plt.text(1,np.shape(he)[0]-1,f'ROI score: {round(best_roi_score,3)}',fontdict=fontdict)
    ax = plt.gca()
    for i in range(len(best_roi)):
        ax.add_patch(plt.Circle([best_roi[i][0]*args.down_samp_step*16, best_roi[i][1]*args.down_samp_step*16],
                                window_size[0]*args.down_samp_step*16, color='red',fill=False, linewidth=3))
    plt.savefig(main_output_folder+'best_roi_on_he.jpg', 
                format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
    print('Best ROI on H&E image is stored at '+main_output_folder+'best_roi_on_he.jpg')

if __name__ == '__main__':
    main()