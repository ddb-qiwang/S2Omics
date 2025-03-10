########################################################################################################
########################################### warning! ###################################################
### cell label broadcasting can only be run with feature extraction conducted with down_samp_step=1 ####
########################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
from s1_utils import *
from s2_label_broadcasting import *
import random


''' predict cell-level labels
    warning! running this files need cell_type_image.pickle & unique_cell_type.pickle
    cell_type_image.pickle is a image-shaped matrix indicating the cell type label (0~n-1) of superpixels, -1 means no cells in current superpixel
    unique_cell_type.pickle is cell type names [cell_type_1,..., cell_type_n] in the order of cell_type_label (0~n-1)
    if you want to predict the spatial domain or other cell-level label, 
    just name the according file as cell_type_image.pickle & unique_cell_type.pickle
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    roi_size: the physical size (mm x mm) of ROIs, default = [6.5, 6.5] which is the physical size for Visium HD ROI
Return:
    cell_type_prediction: predicted cell type distribution from anntations inside ROI(s) produced by pyplot
'''

def get_args():
    parser = argparse.ArgumentParser(description = ' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--best_num_clusters', type=int, default=25)
    parser.add_argument('--roi_size', type=float, nargs='+', default=[6.5,6.5])
    return parser.parse_args()

def main():
    args = get_args()
    if not os.path.exists(args.prefix+'pickle_files'):
        os.makedirs(args.prefix+'pickle_files')
    pickle_folder = args.prefix+'pickle_files/'
    shapes = load_pickle(pickle_folder+'shapes.pickle')
    image_shape = shapes['tiles']
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    sns_figsize = (image_shape[1]//100+5,image_shape[0]//100)
    qc_preserve_indicator = load_pickle(pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)
    unique_cell_type = load_pickle(args.prefix+'unique_cell_type.pickle')
    cell_type_image = load_pickle(args.prefix+'cell_type_image.pickle')
    cell_type_image_mask = np.full(shapes['tiles'], False)
    cell_type_image_mask[np.where(cell_type_image>-1)] = True
    physical_size = args.roi_size
    window_size = [int(125*physical_size[0]),int(125*physical_size[1])]

    he_embed_total = []
    i = 0
    while 1 > 0:
        if os.path.exists(pickle_folder+f'uni_embeddings_downsamp_1_part_{i}.pickle'):
            he_embed_part = load_pickle(pickle_folder+f'uni_embeddings_downsamp_1_part_{i}.pickle')
            he_embed_total.append(he_embed_part)
            i += 1
        else:
            break
    he_embed_total = np.concatenate(he_embed_total)
    del he_embed_part

    [best_roi_list, best_rotate_list, best_roi_mask_list, best_comp_list, best_roi_score_list] = \
    load_pickle(args.prefix+f'roi_selection_output/num_clusters_{args.best_num_clusters}/best_roi.pickle')
    best_roi = best_roi_list[-1]
    num_roi = len(best_roi)
    best_rotate = best_rotate_list[-1]
    best_comp = best_comp_list[-1]
    best_roi_mask = best_roi_mask_list[-1]
    best_roi_mask_total = best_roi_mask[0]
    for i in range(num_roi-1):
        best_roi_mask_total = best_roi_mask_total | best_roi_mask[i+1]

    setup_seed(42)
    num_cell_types = len(unique_cell_type) 
    onehot_label = np.eye(len(unique_cell_type))
    train_mask = cell_type_image_mask & best_roi_mask_total
    train_x = he_embed_total[train_mask.flatten(),:]
    train_y = np.array(cell_type_image[train_mask], dtype='int64')
    total_x = he_embed_total[qc_mask.flatten(),:]
    total_y = -1*np.ones(np.sum(qc_mask), dtype='int64')
    TrainSet = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
    TotalSet = TensorDataset(torch.from_numpy(total_x).float(), torch.from_numpy(total_y))
    del he_embed_total
    train_loader = DataLoader(TrainSet,shuffle=True,batch_size=512,num_workers=0,drop_last=False)
    total_loader = DataLoader(TotalSet,shuffle=False,batch_size=512,num_workers=0,drop_last=False)

    # Train AE
    device = torch.device('cuda:0')
    model = S2Omics_Predictor(
        n_input = 2048,
        n_enc_1=1024,
        n_enc_2=1024,
        n_enc_3=1024,
        n_z=256,
        n_cls_1=64,
        n_cls_out=num_cell_types+1).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = GCE_loss(0.6, num_cell_types+1)

    epochs = 100
    test_interval = 20
    for epoch in range(epochs):  # loop over the dataset multiple times
        for (i, data) in enumerate(train_loader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = len(labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            x_bar, z, cls_h1, cls_out = model(inputs)
            
            loss_recon = 0.5*(F.mse_loss(x_bar, inputs)+nn.L1Loss()(x_bar, inputs))
            loss_cls =  criterion(cls_out, labels)
            loss = loss_recon + loss_cls
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % test_interval == 0:  
            train_cor,train_tot = 0,0
            with torch.no_grad():
                for i, data in enumerate(train_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    _,_,_,outputs = model(inputs)
                    pred = torch.argmax(outputs, axis=1)
                    train_cor += torch.sum(pred==labels)
                    train_tot += len(labels)
            print('Epoch [%d] loss: %.3f, train accuracy %.3f' %(epoch + 1, loss.item(),train_cor/train_tot))
            
    print('Finished Training')

    model.eval()
    total_pred = []
    with torch.no_grad():
        for i, data in enumerate(total_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            _,_,_,outputs = model(inputs)
            pred = torch.argmax(outputs, axis=1)
            total_pred.append(pred)
    total_pred = np.array(torch.concat(total_pred).cpu().numpy(), dtype='int')
    total_pred_ct = unique_cell_type[total_pred]
    
    color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],[227,119,194],[127,127,127],[188,189,34],
                  [23,190,207],[174,199,232],[255,187,120],[152,223,138],[255,152,150],[197,176,213],[196,156,148],[247,182,210],
                  [199,199,199],[219,219,141],[158,218,229],[16,60,90],[128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                  [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
    color_list_16bit = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B',  '#E377C2', '#7F7F7F', '#BCBD22', 
                        '#17BECF', '#AEC7E8','#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94',  '#F7B6D2', 
                        '#C7C7C7', '#DBDB8D', '#9EDAE5', '#103C5A', '#804007', '#165016', '#6B1414', '#4A345E', 
                        '#462B26', '#723C61', '#404040', '#5E5E11', '#0C5F68', '#000000']
    ct_color_mapping = np.arange(num_cell_types)
    colors = np.array(color_list_16bit)[ct_color_mapping]
    
    pred_image = -1*np.ones(image_shape)
    pred_image[qc_mask] = total_pred
    pred_image_rgb = 255*np.ones([image_shape[0],image_shape[1],3])
    for cluster in range(num_cell_types):
        pred_image_rgb[pred_image==cluster] = color_list[ct_color_mapping[cluster]]
    pred_image_rgb = np.array(pred_image_rgb, dtype='int')

    plt.figure(figsize=plt_figsize)
    plt.imshow(pred_image_rgb)
    ax = plt.gca()
    legend_x = legend_y = np.zeros(num_cell_types)
    for i in range(num_cell_types):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    plt.legend((unique_cell_type), fontsize=12)
    for i in range(num_roi):
        ax.add_patch(plt.Rectangle([best_roi[i][0][1],best_roi[i][0][0]],
                                    window_size[1],window_size[0],color='red',fill=False,
                                    linewidth=2,angle=-best_rotate[i]))
    plt.savefig(args.prefix+'S2Omics_prediction.jpg',
                format='jpg', dpi=1200, bbox_inches='tight',pad_inches=0)

if __name__ == '__main__':
    main()