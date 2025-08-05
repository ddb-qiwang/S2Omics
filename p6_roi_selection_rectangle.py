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
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    save_folder: the name of save folder, user can input the complete path or just the folder name, 
        if so, the folder will be placed under the prefix folder
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
              best_roi.pickle: [best_num_roi,best_roi_list, best_rotate_list,best_roi_mask_list,best_comp_list,best_roi_score_list] 
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

def get_vertex_set_from_centroid(centroid, window_size, rotate_deg):
    '''
    Input:
        cetroid: [x,y] coordinates of ROI rectangle centroid
        window_size: [length, width]
        rotate_deg: degre of rotation n*Pi
    Output:
        vertex_set: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] anti-clockwise, start from bottom-left vertex
    '''
    [x, y] = centroid
    hypo = np.sqrt(window_size[0]**2+window_size[1]**2)/2
    [x0, y0] = int(x-hypo*np.cos(rotate_deg+np.pi/4)), int(y-hypo*np.sin(rotate_deg+np.pi/4))
    [x1, y1] = int(x+hypo*np.sin(rotate_deg+np.pi/4)), int(y-hypo*np.cos(rotate_deg+np.pi/4))
    [x2, y2] = int(x+hypo*np.cos(rotate_deg+np.pi/4)), int(y+hypo*np.sin(rotate_deg+np.pi/4))
    [x3, y3] = int(x-hypo*np.sin(rotate_deg+np.pi/4)), int(y+hypo*np.cos(rotate_deg+np.pi/4))
    vertex_set = [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]  
    return vertex_set

def generate_roi_mask(point_bl, window_size, rotate_deg, mask_shape):
    corners = np.array([
        [0, 0],
        [window_size[0], 0],
        [window_size[0], window_size[1]],
        [0, window_size[1]]
    ])
    rotate_deg = rotate_deg / np.pi * 180
    point_bl = point_bl[::-1]
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), rotate_deg, 1.0)
    rotated_corners = np.dot(corners, rotation_matrix[:, :2].T) + rotation_matrix[:, 2]
    final_corners = rotated_corners + point_bl
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [final_corners.astype(np.int32)], 1)   
    return mask.astype(bool)

def cal_roi_score(curr_comp, target_proportion, window_size, num_roi, total_cell_num):
    
    scale_score = logistic_func(np.sum(curr_comp)/total_cell_num)
    curr_prop = curr_comp/(num_roi*window_size[0]*window_size[1])
    valid_score = np.sqrt(np.sum(curr_prop))
    balance_score = cosine_similarity(curr_prop, target_proportion)
    roi_score = np.power(scale_score*valid_score*balance_score, 1/3)
    return roi_score

def region_selection_few(cluster_image, valid_mask, num_clusters, window_size, target_proportion,
                         curr_roi, curr_rotate, curr_roi_mask, curr_comp,
                         rotation_seg=6, num_roi=2, fusion_weights=[1,1,1], num_samp=10000, samp_step=5):
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
    best_rotate = curr_rotate.copy()
    best_roi_mask = curr_roi_mask.copy()
    best_comp = curr_comp.copy()
    mask_shape = np.shape(cluster_image)
    min_window_size = np.min(window_size)
    if len(target_proportion) == 0:
        target_proportion = np.ones(num_clusters)/num_clusters
    
    # randomly sample num_samp times and keep the seletion with highest total roi score (together with previous selections)
    for samp in tqdm(range(num_samp)):       
        tmp_roi = curr_roi.copy()
        tmp_rotate = curr_rotate.copy()
        tmp_roi_mask = curr_roi_mask.copy()
        tmp_comp = curr_comp.copy()
        centroids_index = np.random.choice(np.arange(0,len(valid_points[0]),samp_step), num_roi)
        x_centroids = valid_points[0][centroids_index]
        y_centroids = valid_points[1][centroids_index]

        # sample num_roi ROIs
        for roi in range(num_roi):      
            centroid = [x_centroids[roi], y_centroids[roi]]
            rotate = np.random.choice(rotation_seg)
            rotate_deg = 90*np.pi/180/rotation_seg*rotate
            tmp_rotate.append(90*rotate/rotation_seg)
            point_set = get_vertex_set_from_centroid(centroid, window_size, rotate_deg)
            tmp_roi.append(point_set)
            tmp_roi_mask.append(generate_roi_mask(point_set[0], window_size, rotate_deg, mask_shape))

        tmp_roi_mask_total = tmp_roi_mask[0]
        for roi in range(1, len(tmp_roi)):
            tmp_roi_mask_total = tmp_roi_mask_total | tmp_roi_mask[roi]
        tmp_clusters = cluster_image[tmp_roi_mask_total]
        for cluster in range(num_clusters):
            tmp_comp[cluster] = np.sum(tmp_clusters==cluster)
        tmp_prop = tmp_comp/(len(tmp_roi)*window_size[0]*window_size[1])

        # compute the current roi score for len(best_roi) ROIs
        tmp_scale_score = logistic_func(np.sum(tmp_comp)/total_cell_num)
        tmp_valid_score = np.sqrt(np.sum(tmp_prop))
        tmp_balance_score = cosine_similarity(tmp_prop, target_proportion)
        roi_score = np.power(np.power(tmp_scale_score,fusion_weights[0])*np.power(tmp_valid_score,fusion_weights[1])*np.power(tmp_balance_score,fusion_weights[2]), 1/np.sum(fusion_weights))

        
        if roi_score > best_roi_score[0]:
            best_roi_score = [roi_score, tmp_scale_score, tmp_valid_score, tmp_balance_score]
            best_rotate = tmp_rotate
            best_roi = tmp_roi
            best_roi_mask = tmp_roi_mask
            best_comp = tmp_comp
    print(f'''Current best ROI: {best_roi}
    roi score: {best_roi_score[0]}
    scale score: {best_roi_score[1]}
    valid score: {best_roi_score[2]}
    balance score: {best_roi_score[3]}''')

    return best_roi, best_rotate, best_roi_mask, best_comp, best_roi_score
        
    

def region_selection_random(save_folder, cluster_image, cluster_image_rgb, valid_mask, num_clusters, 
                            window_size, num_roi, fusion_weights=[1,1,1], target_proportion=[], rotation_seg=6, 
                            optimal_roi_thres=0.03, num_samp_per_iter=10000, samp_step=5, save_plot=True):
    '''
    valid_points: [[valid_xs], [valid_ys]]
    window_size = [width, length]
    rotation_seg = 6 means we seg the 90 degrees in to 6 rotations, each time we rotate the window for 90/6 degree
    
    '''
    best_roi_list = []
    best_rotate_list = []
    best_roi_mask_list = []
    best_comp_list = []
    best_roi_score_list = []
    if len(target_proportion) == 0:
        target_proportion = np.ones(num_clusters)/num_clusters
    
    if_stop = False
    curr_num_roi = 0
    pre_best_roi, pre_best_rotate, pre_best_roi_mask = [], [], []
    pre_best_comp = np.zeros(num_clusters)
    pre1_best_roi_score, pre2_best_roi_score = [0], [0]
    
    while if_stop == False:      
        # find the best (curr_num_roi+1)^th ROI
        best_roi,best_rotate,best_roi_mask,best_comp,best_roi_score = region_selection_few(cluster_image, valid_mask, 
                                                                                           num_clusters, window_size,
                                                                                           target_proportion, pre_best_roi, 
                                                                                           pre_best_rotate, pre_best_roi_mask,
                                                                                           pre_best_comp, rotation_seg, 1, fusion_weights,
                                                                                           num_samp_per_iter, samp_step)
        # show the best (curr_num_roi+1)^th ROI
        if save_plot==True:
            plt.figure()
            plt.imshow(cluster_image_rgb)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            for i in range(len(best_roi)):
                ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                           window_size[0],window_size[1],color='red',fill=False,
                                           linewidth=2,angle=-best_rotate[i]))
            plt.savefig(save_folder+f'best_{curr_num_roi+1}_roi_on_histo_clusters.jpg', 
                        format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
        # if roi score increased less than optimal_roi_thres, there's no need to select this one more ROI  
        if best_roi_score[0] - pre2_best_roi_score[0] < 2*optimal_roi_thres and pre1_best_roi_score[0] - pre2_best_roi_score[0] < optimal_roi_thres and curr_num_roi >= num_roi:  
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            break   
        else:
            best_roi_list.append(best_roi)
            best_rotate_list.append(best_rotate)
            best_roi_mask_list.append(best_roi_mask)
            best_comp_list.append(best_comp)
            best_roi_score_list.append(best_roi_score)   
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            pre2_best_roi_score = pre1_best_roi_score
            pre1_best_roi_score = best_roi_score
            if curr_num_roi == num_roi:
                if_stop = True
                return curr_num_roi, best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list

        # find the best (curr_num_roi+1)^th ROI
        best_roi,best_rotate,best_roi_mask,best_comp,best_roi_score = region_selection_few(cluster_image, valid_mask,
                                                                                           num_clusters, window_size, 
                                                                                           target_proportion, pre_best_roi,
                                                                                           pre_best_rotate, pre_best_roi_mask,
                                                                                           pre_best_comp, rotation_seg, 2, fusion_weights,
                                                                                           num_samp_per_iter, samp_step)
        # save the best (curr_num_roi+1)^th ROI
        if save_plot==True:
            plt.figure()
            plt.imshow(cluster_image_rgb)
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            for i in range(len(best_roi)):
                ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                           window_size[0],window_size[1],color='red',fill=False,
                                           linewidth=2,angle=-best_rotate[i]))
            plt.savefig(save_folder+f'best_{curr_num_roi+1}_roi_on_histo_clusters.jpg', 
                        format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
        # if roi score increased less than optimal_roi_thres, there's no need to select this one more ROI
        if best_roi_score[0] - pre2_best_roi_score[0] < 2*optimal_roi_thres and pre1_best_roi_score[0] - pre2_best_roi_score[0] < optimal_roi_thres and curr_num_roi >= num_roi:  
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            break
        else:
            pre_best_roi,pre_best_rotate,pre_best_roi_mask,pre_best_comp = best_roi,best_rotate,best_roi_mask,best_comp
            best_roi_list.append(best_roi)
            best_rotate_list.append(best_rotate)
            best_roi_mask_list.append(best_roi_mask)
            best_comp_list.append(best_comp)
            best_roi_score_list.append(best_roi_score)   
            curr_num_roi = len(best_roi)
            print(f'Current number of ROIs is {curr_num_roi}.')
            pre2_best_roi_score = pre1_best_roi_score
            pre1_best_roi_score = best_roi_score
            if curr_num_roi == num_roi:
                if_stop = True
                return curr_num_roi, best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list
                
    print(f'Found the optimal number of ROI: {curr_num_roi-2}. Program finished.')
    return curr_num_roi-1, best_roi_list[:-1], best_rotate_list[:-1], best_roi_mask_list[:-1], best_comp_list[:-1], best_roi_score_list[:-1]



def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--has_annotation', type=bool, default=False)
    parser.add_argument('--save_folder', type=str, default='S2Omics_output')
    parser.add_argument('--cache_path', type=str, default='')
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--roi_size', type=float, nargs='+', default=[6.5,6.5])
    parser.add_argument('--num_roi', type=int, default=0) # 0 means automatically determine the optimal number of ROIs
    parser.add_argument('--fusion_weights', type=float, nargs='+', default=[0.33,0.33,0.33]) # [size_score_wight, coverage_score_weight, balance_score_weight], sum of them should be equal to 1
    parser.add_argument('--positive_prior', type=int, nargs='+', default=[]) # [emphasize_clusters]
    parser.add_argument('--negative_prior', type=int, nargs='+', default=[]) # [discard_clusters]
    parser.add_argument('--prior_preference', type=int, default=2) 
    parser.add_argument('--rotation_seg', type=int, default=6) # whether ROIs can be rotated
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
    
    # load in previously obtained params
    if not os.path.exists(args.prefix+args.save_folder):
        os.makedirs(args.prefix+args.save_folder)
    save_folder = args.prefix+args.save_folder+'/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    if not os.path.exists(save_folder+'main_output'):
        os.makedirs(save_folder+'main_output')
    main_output_folder = save_folder+'main_output/'
    if not os.path.exists(save_folder+f'roi_selection_detailed_output/rectangle_roi_size_{args.roi_size[0]}_{args.roi_size[1]}'):
        os.makedirs(save_folder+f'roi_selection_detailed_output/rectangle_roi_size_{args.roi_size[0]}_{args.roi_size[1]}')
    roi_save_folder = save_folder+f'roi_selection_detailed_output/rectangle_roi_size_{args.roi_size[0]}_{args.roi_size[1]}/'
    
    he = load_image(f'{args.prefix}he.jpg')
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    dpi = 600
    length = np.max(image_shape)//100
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    if dpi*length > np.power(2,16):
        reduce_ratio = np.power(2,16)/(dpi*length)
        plt_figsize = ((image_shape[1]*reduce_ratio)//100,(image_shape[0]*reduce_ratio)//100)
    qc_preserve_indicator = load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)

    num_roi = args.num_roi
    physical_size = args.roi_size
    fusion_weights = args.fusion_weights
    emphasize_clusters = args.positive_prior
    discard_clusters = args.negative_prior
    rotation_seg = args.rotation_seg
    optimal_roi_thres = args.optimal_roi_thres
    window_size_raw = [int(125*physical_size[0]),int(125*physical_size[1])]
    window_size = [int(125*physical_size[0]/args.down_samp_step),int(125*physical_size[1]/args.down_samp_step)]
    num_samp = args.num_samp_init*math.ceil((image_shape[0]*image_shape[1])/(window_size_raw[0]*window_size_raw[1]))
    samp_step = math.ceil(5/args.down_samp_step)

    if args.has_annotation:
        cluster_image = load_pickle(pickle_folder+'annotation.pickle')
        category_names = load_pickle(pickle_folder+'category_names.pickle')
    elif len(args.cache_path) > 0:
        cluster_image = load_pickle(args.cache_path)
    else:
        if os.path.exists(pickle_folder+'adjusted_cluster_image.pickle'):
            cluster_image = load_pickle(pickle_folder+'adjusted_cluster_image.pickle')
        else:
            cluster_image = load_pickle(pickle_folder+'cluster_image.pickle')
    num_histology_clusters = len(np.unique(cluster_image[cluster_image>-1]))
    
    target_proportion = np.ones(num_histology_clusters)
    if len(discard_clusters) > 0:
        target_proportion[discard_clusters] -= 1
    if len(emphasize_clusters) > 0:
        emphasize_clusters = [emphasize_clusters[i]-1 for i in range(len(emphasize_clusters))]
        target_proportion[emphasize_clusters] += args.prior_preference
    target_proportion = target_proportion/np.sum(target_proportion)
    
    cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
    cluster_color_mapping = np.arange(num_histology_clusters)
    for cluster in range(num_histology_clusters):
        cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
    cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
    cluster_image_mask = np.full(np.shape(cluster_image), False)
    cluster_image_mask[np.where(cluster_image>-1)] = True

    if not os.path.exists(roi_save_folder+f'prior_preference_{args.prior_preference}'):
        os.makedirs(roi_save_folder+f'prior_preference_{args.prior_preference}')
    save_subfolder = roi_save_folder+f'prior_preference_{args.prior_preference}/'
        
    # select ROIs
    print('Sampling ROI candidates...')
    best_num_roi, best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list = \
    region_selection_random(save_subfolder, cluster_image, cluster_image_rgb, cluster_image_mask,
                            num_histology_clusters, window_size, num_roi, fusion_weights=fusion_weights, target_proportion=target_proportion, 
                            rotation_seg=rotation_seg, optimal_roi_thres=optimal_roi_thres, 
                            num_samp_per_iter=num_samp, samp_step=samp_step, save_plot=True)
    print(f'''Find the best {best_num_roi} ROI(s) with: 
    ROI score: {best_roi_score_list[-1][0]}
    Scale score: {best_roi_score_list[-1][1]}
    Coverage score: {best_roi_score_list[-1][2]}
    Balance score: {best_roi_score_list[-1][3]}
    ''')
    save_pickle([best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list], 
                save_subfolder+'best_roi.pickle')
    
    plt.figure(figsize=plt_figsize)
    best_roi = best_roi_list[-1]
    best_rotate = best_rotate_list[-1]
    best_roi_score = best_roi_score_list[-1][0]
    plt.imshow(cluster_image_rgb)
    fontdict = {'fontsize':12}
    plt.text(1,np.shape(cluster_image)[0]-1,f'ROI score: {round(best_roi_score,3)}',fontdict=fontdict)
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
        ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                    window_size[0],window_size[1],color='red',fill=False,
                                    linewidth=2,angle=-best_rotate[i]))
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
        ax.add_patch(plt.Rectangle([best_roi[i][0][1]*args.down_samp_step*16,
                                    best_roi[i][0][0]*args.down_samp_step*16],
                                    window_size[0]*args.down_samp_step*16,
                                    window_size[1]*args.down_samp_step*16,
                                    color='red',fill=False, linewidth=3,
                                    angle=-best_rotate[i]))
    plt.savefig(main_output_folder+'best_roi_on_he.jpg', 
                format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
    print('Best ROI on H&E image is stored at '+main_output_folder+'best_roi_on_he.jpg')

if __name__ == '__main__':
    main()