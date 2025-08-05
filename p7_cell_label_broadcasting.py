########################################################################################################
########################################### warning! ###################################################
### cell label broadcasting can only be run with feature extraction conducted with down_samp_step=1 ####
########################################################################################################

import matplotlib.pyplot as plt
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
    After user obtained the spatial omics data of the selected small ROI,
    we can annotate the superpixels in the paired H&E image with cell type labels,
    Afterwards, we can transfer the label information to the previously stained whole-slide H&E image 
    to obtain whole-slide level cell type spatial distribution.
    Args:
        foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath
        WSI_datapath: the folder of whole-slide H&E image, should contain he-raw.jpg, pixel-size-raw.jpg
        SO_datapath: the folder of Spatial Omics data, should contain he-raw.jpg, pixel-size-raw.jpg, annotation_file.csv
            The annotation_file.csv should at least contain three columns {'super_pixel_x','super_pixel_y','annotation'} which 
            separately refer to the x, y coordinates and cell-level annotation of superpixels in the SO-paired H&E image.
            Notice: For some technologies like 10x Xenium, alignment between H&E and cell type spatial distribution need to be
            conducted.
        device: default = 'cuda'
    return:
        S2Omics_whole_slide_prediction.jpg: the predicted cell label of the whole-slide H&E image.
'''

def get_args():
    parser = argparse.ArgumentParser(description = ' ')
    parser.add_argument('WSI_datapath', type=str)
    parser.add_argument('SO_datapath', type=str)
    parser.add_argument('--WSI_save_folder', type=str, default='S2Omics_output')
    parser.add_argument('--SO_save_folder', type=str, default='S2Omics_output')
    parser.add_argument('--foundation_model', type=str, default='uni', help='uni, virchow, gigapath')
    parser.add_argument('--WSI_cache_path', type=str, default='')
    parser.add_argument('--SO_cache_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = get_args()
    # load in spatial omics data
    if not os.path.exists(args.SO_datapath+args.SO_save_folder):
        os.makedirs(args.SO_datapath+args.SO_save_folder)
    SO_save_folder = args.SO_datapath+args.SO_save_folder+'/'
    if not os.path.exists(SO_save_folder+'pickle_files'):
        os.makedirs(SO_save_folder+'pickle_files')
    SO_pickle_folder = SO_save_folder+'pickle_files/'
    
    shapes = load_pickle(SO_pickle_folder+'shapes.pickle')
    SO_image_shape = shapes['tiles']
    qc_preserve_indicator = load_pickle(SO_pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, SO_image_shape)
    annotation_file = pd.read_csv(args.SO_datapath+'annotation_file.csv')
    unique_cell_type = np.unique(annotation_file['annotation'])
    label_vector = np.ones(len(annotation_file), dtype='int64')
    for ct in range(len(unique_cell_type)):
        ct_index = np.arange(len(annotation_file))[annotation_file['annotation']==unique_cell_type[ct]]
        label_vector[ct_index] = ct
    annotation_file['label'] = label_vector
    cell_type_image = -1*np.ones(SO_image_shape)
    for i in range(len(annotation_file)):
        x = annotation_file['super_pixel_x'][annotation_file.index[i]]
        y = annotation_file['super_pixel_y'][annotation_file.index[i]]
        label = annotation_file['label'][annotation_file.index[i]]
        cell_type_image[x, y] = label
    cell_type_image = np.array(cell_type_image, dtype='int64')
    cell_type_image_mask = np.full(shapes['tiles'], False)
    cell_type_image_mask[np.where(cell_type_image>-1)] = True
    # load in the histology features of spatial omics data 
    print('Loading histology feature embeddings of the Spatial Omics data...')
    if len(args.SO_cache_path) > 0:
        SO_cache_path = args.SO_cache_path
    else:
        SO_cache_path = SO_pickle_folder
    SO_he_embed_total = []
    i = 0
    while 1 > 0:
        if os.path.exists(SO_cache_path+args.foundation_model+f'_embeddings_downsamp_1_part_{i}.pickle'):
            SO_he_embed_part = load_pickle(SO_cache_path+args.foundation_model+f'_embeddings_downsamp_1_part_{i}.pickle')
            SO_he_embed_total.append(SO_he_embed_part)
            i += 1
        else:
            break
    SO_he_embed_total = np.concatenate(SO_he_embed_total)
    del SO_he_embed_part
    print('Sucessfully loaded all histology feature embeddings of the Spatial Omics data!')

    # load in whole-slide H&E data
    if not os.path.exists(args.WSI_datapath+args.WSI_save_folder):
        os.makedirs(args.WSI_datapath+args.WSI_save_folder)
    WSI_save_folder = args.WSI_datapath+args.WSI_save_folder+'/'
    if not os.path.exists(WSI_save_folder+'pickle_files'):
        os.makedirs(WSI_save_folder+'pickle_files')
    WSI_pickle_folder = WSI_save_folder+'pickle_files/'
    
    shapes = load_pickle(WSI_pickle_folder+'shapes.pickle')
    WSI_image_shape = shapes['tiles']
    plt_figsize = (WSI_image_shape[1]//100,WSI_image_shape[0]//100)
    qc_preserve_indicator = load_pickle(WSI_pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, WSI_image_shape)
    # load in the histology features of whole-lide H&E data 
    print('Loading histology feature embeddings of the whole-lide H&E data...')
    if len(args.WSI_cache_path) > 0:
        WSI_cache_path = args.WSI_cache_path
    else:
        WSI_cache_path = WSI_pickle_folder
    WSI_he_embed_total = []
    i = 0
    while 1 > 0:
        if os.path.exists(WSI_cache_path+args.foundation_model+f'_embeddings_downsamp_1_part_{i}.pickle'):
            WSI_he_embed_part = load_pickle(WSI_cache_path+args.foundation_model+f'_embeddings_downsamp_1_part_{i}.pickle')
            WSI_he_embed_total.append(WSI_he_embed_part)
            i += 1
        else:
            break
    WSI_he_embed_total = np.concatenate(WSI_he_embed_total)
    del WSI_he_embed_part
    print('Sucessfully loaded all histology feature embeddings of the whole-lide H&E data!')

    # construct pytorch dataset, spatial omics (histo-feature, annotation) as training data
    setup_seed(42)
    num_cell_types = len(unique_cell_type) 
    onehot_label = np.eye(len(unique_cell_type))
    train_mask = cell_type_image_mask
    train_x = SO_he_embed_total[train_mask.flatten(),:]
    train_y = np.array(cell_type_image[train_mask], dtype='int64')
    del SO_he_embed_total
    total_x = WSI_he_embed_total[qc_mask.flatten(),:]
    total_y = -1*np.ones(np.sum(qc_mask), dtype='int64')
    del WSI_he_embed_total
    TrainSet = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train_y))
    TotalSet = TensorDataset(torch.from_numpy(total_x).float(), torch.from_numpy(total_y))
    train_loader = DataLoader(TrainSet,shuffle=True,batch_size=512,num_workers=0,drop_last=False)
    total_loader = DataLoader(TotalSet,shuffle=False,batch_size=512,num_workers=0,drop_last=False)

    # Train AE
    print('Start training the label transferring model...')
    device = args.device
    model = S2Omics_Predictor(
        n_input = np.shape(train_x)[1],
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

    # visualize the prediction results
    color_list = np.loadtxt('color_list.txt', dtype='int').tolist()
    with open("color_list_16bit.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
    color_list_16bit = []
    for line in lines:
        color_list_16bit.append(line.strip())
    cluster_color_mapping = np.arange(len(color_list))
    colors = np.array(color_list_16bit)[cluster_color_mapping]
    
    pred_image = -1*np.ones(WSI_image_shape)
    pred_image[qc_mask] = total_pred
    pred_image_rgb = 255*np.ones([WSI_image_shape[0],WSI_image_shape[1],3])
    for cluster in range(num_cell_types):
        pred_image_rgb[pred_image==cluster] = color_list[cluster_color_mapping[cluster]]
    pred_image_rgb = np.array(pred_image_rgb, dtype='int')
    plt.figure(figsize=plt_figsize)
    plt.imshow(pred_image_rgb)
    ax = plt.gca()
    legend_x = legend_y = np.zeros(num_cell_types)
    for i in range(num_cell_types):
        plt.scatter(legend_x, legend_y, c=color_list_16bit[i])
    plt.legend((unique_cell_type), fontsize=12)
    plt.savefig(args.WSI_datapath+'S2Omics_whole_slide_prediction.jpg',
                format='jpg', dpi=600, bbox_inches='tight',pad_inches=0)
    print('Predicted cell type distribution for the whole-slide H&E data is stored at: '+args.WSI_datapath+'S2Omics_whole_slide_prediction.jpg')

if __name__ == '__main__':
    main()