import sys 
sys.path.append('./HistoSweep')
import shutil
import argparse
import pandas as pd
import numpy as np
import os 
from time import time
import os
#from utils import load_image
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


'''combine he.jpg into superpixels and filter out superpixels that do not contain nuclei
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    save_folder: the name of save folder, user can input the complete path or just the folder name, 
        if so, the folder will be placed under the prefix folder
    m: default = 1, can be any float>0, smaller m filters less superpixels, another common setting is 0.7 
        which is useful when the H&E image quality is not satisfactory (post-Xenium or post-CosMx H&E for example)
Return:
    --prefix (the main folder)
    ---save_folder (subfolder)
    ----pickle_files (subsubfolder)
        shapes.pickle: the shape of he and superpixels
        qc_preserve_indicator.pickle: True/False indicator of all superpixels, True means it passed quality control, False means it didn't
    ----image_files (subsubfolder)
        linear_boundary.jpg: the fitted RGB average-variance quadratic of all superpixels and the linear boundary to 
                             filter out superpixels with high RGB average (bright) and low RGB variance (no strcture). 
        qc_mask.jpg: the mask of all valid superpixels that have passed quality control
    
'''

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
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help="folder path of H&E stained image, '/home/H&E_image/' for an example") 
    parser.add_argument('--save_folder', type=str, default='S2Omics_output')
    parser.add_argument('--pixel_size_raw',type=float,default = 0.5)
    parser.add_argument('--density_thresh',type=int,default = 100)
    parser.add_argument('--clean_background_flag', action='store_true', help='Whether to preserve fibrous regions that are otherwise being incorrectly filtered out')
    parser.add_argument('--min_size',type=int,default = 10)
    parser.add_argument('--patch_size',type=int,default = 16)
    parser.add_argument('--pixel_size',type=float,default = 0.5)
    
    ##########################
    return parser.parse_args()

def main():

    args = get_args()
    pixel_size_raw = args.pixel_size_raw
    density_thresh = args.density_thresh
    clean_background_flag = args.clean_background_flag
    min_size = args.min_size
    patch_size = args.patch_size
    pixel_size = args.pixel_size
    
    if '/' not in args.save_folder:
        histosweep_folder = args.save_folder
        save_folder = args.prefix+args.save_folder
    else:
        histosweep_folder = 'HistoSweep_output'
        os.makedirs(args.prefix+args.save_folder)
    if not os.path.exists(args.prefix+histosweep_folder):
        os.makedirs(args.prefix+args.save_folder)
    save_folder = args.prefix+args.save_folder+'/'
    if not os.path.exists(save_folder+'image_files'):
        os.makedirs(save_folder+'image_files')
    image_folder = save_folder+'image_files/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    
    image = load_image(args.prefix+'he.jpg')
    _,shapes,_ = patchify(image, patch_size=args.patch_size)

    # Flag for whether to rescale the image 
    need_scaling_flag = False  # True if image resolution ≠ 0.5µm (or desired size) per pixel
    # Flag for whether to preprocess the image 
    need_preprocessing_flag = False  # True if image dimensions are not divisible by patch_size

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics_memory_optimized(image, patch_size=patch_size)
    
    # identify low density superpixels
    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
    
    print('Total selected for density filtering: ', mask1_lowdensity.sum())
    
    # perform texture analysis 
    mask1_lowdensity_update = run_texture_analysis(prefix=args.prefix[:-1], image=image, tissue_mask=mask1_lowdensity, output_dir=histosweep_folder, patch_size=patch_size, glcm_levels=64)

    
    # identify low ratio superpixels
    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
    print(mask2_lowratio.shape)
    
    
    generate_final_mask(prefix=args.prefix[:-1], he=image,output_dir=histosweep_folder+'/image_files', 
                    mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio, 
                    clean_background = clean_background_flag, 
                    super_pixel_size=patch_size, minSize = min_size)

    ###########################################################
    
    print("Running successfully!")
    
    # transform the mask image to matrix and save to a pickle file
    # Load the image
    img = Image.open(args.prefix+histosweep_folder+'/image_files/mask-small.png')

    arr = np.array(img)

    # Define threshold (0=black, 255=white)
    threshold = 128
    mask = arr > threshold  # True for white, False for black

    # Save pickle for later use
    save_pickle(mask, pickle_folder+'qc_preserve_indicator.pickle')
    

if __name__ == '__main__':
    main()
