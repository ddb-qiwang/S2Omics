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
    down_samp_step: the down-sampling step for feature extraction, default = 10, which refers to 1:10^2 down-sampling rate
    roi_size: the physical size (mm x mm) of ROIs, default = [6.5, 6.5] which is the physical size for Visium HD ROI
    num_roi: number of ROIs to be selected, default = 0 refers to automatic determination
    prior: prior information about interested and not-interested histology clusters, default=[[],[]]
    optimal_roi_thres: hyper-parameter for automatic ROI determination, default = 0.03 is suitable for most cases, recommend to be set as 0 when selecting FOVs. If you want to select more ROIs, please lower this parameter
Return:
    --prefix (the main folder)
        best_roi_on_histology_segmentations.jpg: the selected ROIs based on different histology segmentation results
        best_roi_on_he.jpg: the selected ROIs on H&E image
    --prefix
    ----roi_selection_output (subfolder)
    ------num_cluster_{n} (subsubfolder, here n is in {7,10,13,16,19,22,25})
              best_roi.pickle: [best_num_roi,best_roi_list, best_rotate_list,best_roi_mask_list,best_comp_list,best_roi_score_list] 
                               contains the ROIs information for best 1/2/.../best_num_roi ROIs
              best_{n}_roi_on_histo_clusters.jpg: additional information about ROI selection based on current histology segmentation.
                                                  the number of ROIs R is automatically determined by S2Omics, we save and show our selection
                                                  for best R ROIs, R+1 ROIs and R+2 ROIs for reference
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
                         rotation_seg=6, num_roi=2, num_samp=10000, samp_step=5):
    '''
    region selection function for less than 2 ROIs, based on previous selections (curr_roi,...)
    '''
    total_cell_num = np.sum(cluster_image>-1)
    valid_points = np.where(valid_mask > 0)
    valid_region = [np.min(valid_points[0]),np.max(valid_points[0]),
                    np.min(valid_points[1]),np.max(valid_points[1])]

    # introduce previous ROI selections
    # best_roi = [] means no previous selection information 
    best_roi_score = 0
    best_roi_score_comp = []
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
        roi_score = np.power(tmp_scale_score*tmp_valid_score*tmp_balance_score, 1/3)
        
        if roi_score > best_roi_score:
            best_roi_score = roi_score
            best_roi_score_comp = [tmp_scale_score,tmp_valid_score,tmp_balance_score]
            best_rotate = tmp_rotate
            best_roi = tmp_roi
            best_roi_mask = tmp_roi_mask
            best_comp = tmp_comp
    print(f'Current best ROI:{best_roi}, roi score: {best_roi_score}, scale score: {best_roi_score_comp[0]},valid score: {best_roi_score_comp[1]}, balance score: {best_roi_score_comp[2]}')

    return best_roi, best_rotate, best_roi_mask, best_comp, best_roi_score
        
    

def region_selection_random(save_folder, cluster_image, cluster_image_rgb, valid_mask, num_clusters, 
                            window_size, num_roi, target_proportion=[], rotation_seg=6, 
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
    pre1_best_roi_score, pre2_best_roi_score = 0, 0
    
    while if_stop == False:      
        # find the best (curr_num_roi+1)^th ROI
        best_roi,best_rotate,best_roi_mask,best_comp,best_roi_score = region_selection_few(cluster_image, valid_mask, 
                                                                                           num_clusters, window_size,
                                                                                           target_proportion, pre_best_roi, 
                                                                                           pre_best_rotate, pre_best_roi_mask,
                                                                                           pre_best_comp, rotation_seg, 1, 
                                                                                           num_samp_per_iter, samp_step)
        # show the best (curr_num_roi+1)^th ROI
        if save_plot==True:
            plt.figure()
            plt.imshow(cluster_image_rgb)
            ax = plt.gca()
            for i in range(len(best_roi)):
                ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                           window_size[1],window_size[0],color='red',fill=False,
                                           linewidth=2,angle=-best_rotate[i]))
            plt.savefig(save_folder+f'best_{curr_num_roi+1}_roi_on_histo_clusters.jpg', 
                        format='jpg', dpi=1200, bbox_inches='tight',pad_inches=0)
        # if roi score increased less than optimal_roi_thres, there's no need to select this one more ROI  
        if best_roi_score - pre2_best_roi_score < 2*optimal_roi_thres and pre1_best_roi_score - pre2_best_roi_score < optimal_roi_thres and curr_num_roi >= num_roi:  
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
                                                                                           pre_best_comp, rotation_seg, 2,
                                                                                           num_samp_per_iter, samp_step)
        # save the best (curr_num_roi+1)^th ROI
        if save_plot==True:
            plt.figure()
            plt.imshow(cluster_image_rgb)
            ax = plt.gca()
            for i in range(len(best_roi)):
                ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                           window_size[1],window_size[0],color='red',fill=False,
                                           linewidth=2,angle=-best_rotate[i]))
            plt.savefig(save_folder+f'best_{curr_num_roi+1}_roi_on_histo_clusters.jpg', 
                        format='jpg', dpi=1200, bbox_inches='tight',pad_inches=0)
        # if roi score increased less than optimal_roi_thres, there's no need to select this one more ROI
        if best_roi_score - pre2_best_roi_score < 2*optimal_roi_thres and pre1_best_roi_score - pre2_best_roi_score < optimal_roi_thres and curr_num_roi >= num_roi:  
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
                
    print(f'Found the optimal number of ROI: {curr_num_roi-1}. Program finished.')
    return curr_num_roi-1, best_roi_list[:-1], best_rotate_list[:-1], best_roi_mask_list[:-1], best_comp_list[:-1], best_roi_score_list[:-1]



def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--roi_size', type=float, nargs='+', default=[6.5,6.5])
    parser.add_argument('--num_roi', type=int, default=0) # 0 means automatically determine the optimal number of ROIs
    parser.add_argument('--prior', type=list, default=[[],[]]) # [[emphasize_clusters], [discard_clusters]]
    parser.add_argument('--rotation_seg', type=int, default=6) # whether ROIs can be rotated
    parser.add_argument('--optimal_roi_thres', type=float, default=0.03) # threshold for optimal ROI number determination 
    return parser.parse_args()

def main():
    args = get_args()
    setup_seed(42)

    color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],
                  [188,189,34],[23,190,207],[174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],
                  [196,156,148],[247,182,210],[199,199,199],[219,219,141],[158,218,229],[16,60,90],[128,64,7],
                  [22,80,22],[107,20,20],[74,52,94],[70,43,38],[114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
    
    # load in previously obtained params
    if not os.path.exists(args.prefix+'pickle_files'):
        os.makedirs(args.prefix+'pickle_files')
    pickle_folder = args.prefix+'pickle_files/'
    if not os.path.exists(args.prefix+'roi_selection_output'):
        os.makedirs(args.prefix+'roi_selection_output')
    save_folder = args.prefix+'roi_selection_output/'
    he = load_image(f'{args.prefix}he.jpg')
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    qc_preserve_indicator = load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)

    num_roi = args.num_roi
    physical_size = args.roi_size
    [emphasize_clusters, disgard_clusters] = args.prior
    rotation_seg = args.rotation_seg
    optimal_roi_thres = args.optimal_roi_thres
    window_size_raw = [int(125*physical_size[0]),int(125*physical_size[1])]
    window_size = [int(125*physical_size[0]/args.down_samp_step),int(125*physical_size[1]/args.down_samp_step)]
    num_samp = 500*math.ceil((image_shape[0]*image_shape[1])/(window_size_raw[0]*window_size_raw[1]))
    samp_step = math.ceil(5/args.down_samp_step)
    
    cluster_image = load_pickle(pickle_folder+'default_cluster_image.pickle')
    default_num_histology_clusters = len(np.unique(cluster_image[cluster_image>-1]))

    gap = 3
    minimum = 5
    num_iters = (default_num_histology_clusters-minimum-1)//gap+1
    num_histology_clusters = default_num_histology_clusters
    plt_cluster_image_rgb_list = []
    plt_best_roi_list = []
    plt_best_rotate_list = []
    plt_best_roi_score_list = []
    for iter in range(num_iters):
        if iter > 0:
            num_histology_clusters = default_num_histology_clusters-iter*gap
            cluster_image = load_pickle(pickle_folder+f'adjusted_cluster_image_num_clusters_{num_histology_clusters}.pickle')
        target_proportion = np.ones(num_histology_clusters)
        if len(disgard_clusters) > 0:
            target_proportion[disgard_clusters] -= 1
        if len(emphasize_clusters) > 0:
            target_proportion[emphasize_clusters] += 1
        target_proportion = target_proportion/np.sum(target_proportion)
    
        cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
        cluster_color_mapping = np.arange(num_histology_clusters)
        for cluster in range(num_histology_clusters):
            cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
        cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
        cluster_image_mask = np.full(np.shape(cluster_image), False)
        cluster_image_mask[np.where(cluster_image>-1)] = True

        if not os.path.exists(save_folder+f'num_clusters_{num_histology_clusters}'):
            os.makedirs(save_folder+f'num_clusters_{num_histology_clusters}')
        save_subfolder = save_folder+f'num_clusters_{num_histology_clusters}/'
        
        # select ROIs
        best_num_roi, best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list = \
        region_selection_random(save_subfolder, cluster_image, cluster_image_rgb, cluster_image_mask,
                                num_histology_clusters, window_size, num_roi, target_proportion=target_proportion, 
                                rotation_seg=rotation_seg, optimal_roi_thres=optimal_roi_thres, 
                                num_samp_per_iter=num_samp, samp_step=samp_step, save_plot=True)
        save_pickle([best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list], 
                    save_subfolder+'best_roi.pickle')

        plt_cluster_image_rgb_list.append(cluster_image_rgb)
        plt_best_roi_list.append(best_roi_list[-1])
        plt_best_rotate_list.append(best_rotate_list[-1])
        plt_best_roi_score_list.append(best_roi_score_list[-1])

    length = np.ceil(np.sqrt(num_iters)).astype('int')
    output_figsize = (length*plt_figsize[0],length*plt_figsize[1])
    plt.figure(figsize=output_figsize)
    cluster_image = load_pickle(pickle_folder+'default_cluster_image.pickle')
    num_histology_clusters = default_num_histology_clusters
    for iter in range(num_iters):
        if iter > 0:
            num_histology_clusters = default_num_histology_clusters-iter*gap
            cluster_image = load_pickle(pickle_folder+f'adjusted_cluster_image_num_clusters_{num_histology_clusters}.pickle')
        cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
        cluster_color_mapping = np.arange(num_histology_clusters)
        for cluster in range(num_histology_clusters):
            cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
        cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
        cluster_image_mask = np.full(np.shape(cluster_image), False)
        cluster_image_mask[np.where(cluster_image>-1)] = True
        
        best_roi = plt_best_roi_list[iter]
        best_rotate = plt_best_rotate_list[iter]
        best_roi_score = plt_best_roi_score_list[iter]
        plt.subplot(length,length,iter+1)
        plt.imshow(cluster_image_rgb)
        fontdict = {'fontsize':12}
        plt.text(1,np.shape(cluster_image)[0]-1,f'ROI score: {round(best_roi_score,3)}',fontdict=fontdict)
        plt.title(f'num_clusters = {num_histology_clusters}', fontsize=20)
        ax = plt.gca()
        for i in range(len(best_roi)):
            ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                        window_size[1],window_size[0],color='red',fill=False,
                                        linewidth=2,angle=-best_rotate[i]))
    plt.savefig(args.prefix+'best_roi_on_histology_segmentations.jpg', 
                format='jpg', dpi=1200, bbox_inches='tight',pad_inches=0)
    plt.close()

    plt.figure(figsize=output_figsize)
    cluster_image = load_pickle(pickle_folder+'default_cluster_image.pickle')
    num_histology_clusters = default_num_histology_clusters
    for iter in range(num_iters):
        if iter > 0:
            num_histology_clusters = default_num_histology_clusters-iter*gap
            cluster_image = load_pickle(pickle_folder+f'adjusted_cluster_image_num_clusters_{num_histology_clusters}.pickle')
        cluster_image_rgb = 255*np.ones([np.shape(cluster_image)[0],np.shape(cluster_image)[1],3])
        cluster_color_mapping = np.arange(num_histology_clusters)
        for cluster in range(num_histology_clusters):
            cluster_image_rgb[cluster_image==cluster] = color_list[cluster_color_mapping[cluster]]
        cluster_image_rgb = np.array(cluster_image_rgb, dtype='int')
        cluster_image_mask = np.full(np.shape(cluster_image), False)
        cluster_image_mask[np.where(cluster_image>-1)] = True

        best_roi = plt_best_roi_list[iter]
        best_rotate = plt_best_rotate_list[iter]
        best_roi_score = plt_best_roi_score_list[iter]
        plt.subplot(length,length,iter+1)
        plt.imshow(he)
        plt.title(f'num_clusters = {num_histology_clusters}', fontsize=20)
        fontdict = {'fontsize':12}
        plt.text(1,np.shape(he)[0]-1,f'ROI score: {round(best_roi_score,3)}',fontdict=fontdict)
        ax = plt.gca()
        for i in range(len(best_roi_list[-1])):
            ax.add_patch(plt.Rectangle([best_roi_list[-1][i][0][1]*args.down_samp_step*16,
                                        best_roi_list[-1][i][0][0]*args.down_samp_step*16],
                                        window_size[1]*args.down_samp_step*16,
                                        window_size[0]*args.down_samp_step*16,
                                        color='red',fill=False, linewidth=3,
                                        angle=-best_rotate_list[-1][i]))
    plt.savefig(args.prefix+'best_roi_on_he.jpg', 
                format='jpg', dpi=1200, bbox_inches='tight',pad_inches=0)

if __name__ == '__main__':
    main()