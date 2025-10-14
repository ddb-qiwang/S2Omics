import sys 
sys.path.append('./HistoSweep')
import shutil
import argparse
import pandas as pd
import numpy as np
import os 
from time import time
import os
import matplotlib.pyplot as plt
from HistoSweep.saveParameters import saveParams
from HistoSweep.computeMetrics import compute_metrics_memory_optimized
from HistoSweep.densityFiltering import compute_low_density_mask
from HistoSweep.textureAnalysis import run_texture_analysis
from HistoSweep.ratioFiltering import run_ratio_filtering
from HistoSweep.generateMask import generate_final_mask
from HistoSweep.additionalPlots import generate_additionalPlots
from PIL import Image
from s1_utils import (
        load_image, save_pickle)
from UTILS import get_image_filename,load_image



def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    pad_w = shape_ext[0] - x.shape[0]
    pad_h = shape_ext[1] - x.shape[1]
    print(pad_w,pad_h)
    x = np.pad(x, ((0, pad_w),(0, pad_h),(0, 0)), mode='edge')
    patch_index_mask = np.zeros(np.shape(x)[:2])
    tiles_shape = np.array(x.shape[:2]) // patch_size
    tiles = []
    counter = 0
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size
        b0 = a0 + patch_size
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size
            b1 = a1 + patch_size
            tiles.append(x[a0:b0, a1:b1])
            patch_index_mask[a0:b0, a1:b1] = counter
            counter += 1

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    patch_index_mask = patch_index_mask[:np.shape(x)[0]-pad_w,:np.shape(x)[1]-pad_h]
    return tiles, shapes, patch_index_mask

def superpixel_quality_control(prefix, save_folder, 
                              density_thresh = 100,
                              clean_background_flag=False,
                              min_size = 10, patch_size = 16,
                              show_image = False):
    """
    clean_background_flag: Whether to preserve fibrous regions that are otherwise being incorrectly filtered out
    """
    histosweep_folder = 'HistoSweep_output'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = save_folder+'/'
    if not os.path.exists(save_folder+'image_files'):
        os.makedirs(save_folder+'image_files')
    image_folder = save_folder+'image_files/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    
    image = load_image(prefix+'he.jpg')
    _,shapes,_ = patchify(image, patch_size)
    # save shapes parameter to pickle folder
    save_pickle(shapes, pickle_folder+'shapes.pickle')

    # Flag for whether to rescale the image 
    need_scaling_flag = False  # True if image resolution ≠ 0.5µm (or desired size) per pixel
    # Flag for whether to preprocess the image 
    need_preprocessing_flag = False  # True if image dimensions are not divisible by patch_size

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics_memory_optimized(image, patch_size=patch_size)
    
    # identify low density superpixels
    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
    
    print('Total selected for density filtering: ', mask1_lowdensity.sum())
    
    # perform texture analysis 
    mask1_lowdensity_update = run_texture_analysis(prefix=prefix[:-1], image=image, tissue_mask=mask1_lowdensity, output_dir=histosweep_folder, patch_size=patch_size, glcm_levels=64)

    
    # identify low ratio superpixels
    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
    print(mask2_lowratio.shape)
    
    
    generate_final_mask(prefix=prefix[:-1], he=image, output_dir=histosweep_folder, 
                    mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio, 
                    clean_background = clean_background_flag, 
                    super_pixel_size=patch_size, minSize = min_size)

    ###########################################################
    
    print("Running successfully!")

    # transform the mask image to matrix and save to a pickle file
    # Load the image
    img = Image.open(prefix+histosweep_folder+'/mask-small.png')
    if show_image:
        plt.imshow(img)

    arr = np.array(img)

    # Define threshold (0=black, 255=white)
    threshold = 128
    mask = arr > threshold  # True for white, False for black

    # Save pickle for later use
    save_pickle(mask, pickle_folder+'qc_preserve_indicator.pickle')