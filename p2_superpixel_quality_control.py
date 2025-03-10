import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from s1_utils import (
        load_image, save_pickle)


'''combine he.jpg into superpixels and filter out superpixels that do not contain nuclei
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    m: default = 1, can be any float>0, smaller m filters less superpixels, another common setting is 0.7 which is useful when the H&E image quality is not satisfactory (post-Xenium or post-CosMx H&E for example)
Return:
    --prefix (the main folder)
    ----pickle_files (subfolder)
        shapes.pickle: the shape of he and superpixels
        qc_preserve_indicator.pickle: True/False indicator of all superpixels, True means it passed quality control, False means it didn't
    --prefix (the main folder)
    ----qc_image_output (subfolder)
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

def linear_line(x,x_vertex,m):
    return m*x - x_vertex 

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--m', type=float, default=1.0)
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    if not os.path.exists(args.prefix+'qc_image_output'):
        os.makedirs(args.prefix+'qc_image_output')
    save_folder = args.prefix+'qc_image_output/'
    if not os.path.exists(args.prefix+'pickle_files'):
        os.makedirs(args.prefix+'pickle_files')
    pickle_folder = args.prefix+'pickle_files/'
    he = load_image(args.prefix+'he.jpg')
    he_tiles,shapes,_ = patchify(he, patch_size=16)
    he_mean = [np.mean(he_tile) for he_tile in he_tiles]
    he_mean_image = np.reshape(he_mean, shapes['tiles'])
    he_std = [np.std(he_tile) for he_tile in he_tiles]
    he_std_image = np.reshape(he_std, shapes['tiles'])
    save_pickle(shapes, pickle_folder+'shapes.pickle')
    
    image_shape = shapes['tiles']
    dpi = 600
    length = np.max(image_shape)//100
    plt_figsize = (image_shape[1]//100,image_shape[0]//100)
    if dpi*length > np.power(2,16):
        reduce_ratio = np.power(2,16)/(dpi*length)
        plt_figsize = ((image_shape[1]*reduce_ratio)//100,(image_shape[0]*reduce_ratio)//100)

    # fit the superpixel RGB mean-std distribution with a parabola
    mean_intensity = he_mean_image.copy().flatten()
    std_dev = he_std_image.copy().flatten()
    coeffs = np.polyfit(mean_intensity, std_dev, 2) 
    a, b, c = coeffs  
    x_vertex = -b / (2 * a)
    y_vertex = a * x_vertex**2 + b * x_vertex + c
    print(f"Peak of the parabola occurs at Mean Intensity: {x_vertex:.2f}, Standard Deviation: {y_vertex:.2f}")

    # calculate the linear bound to filter out superpixels without nuclei
    linear_boundary = args.m*mean_intensity - x_vertex 
    below_boundary_mask = std_dev < linear_boundary
    below_boundary_indices = np.where(below_boundary_mask)[0]

    # plot the parabola
    plt.figure()
    plt.scatter(mean_intensity[~below_boundary_mask], std_dev[~below_boundary_mask], color='blue', s=.1)
    plt.scatter(mean_intensity[below_boundary_mask], std_dev[below_boundary_mask], color='red', s=.1)
    x_vals = np.linspace(np.min(mean_intensity), np.max(mean_intensity), 500)
    y_vals = a * x_vals**2 + b * x_vals + c
    plt.plot(x_vals, y_vals, 'g--', label='Fitted Quadratic')
    plt.scatter([x_vertex], [y_vertex], color='green', s=50, zorder=5)
    # Plot the linear boundary line
    x_line = np.linspace(x_vertex, np.max(mean_intensity), 500)
    y_line = linear_line(x_line, x_vertex, args.m)
    plt.plot(x_line, y_line, 'r--', label='Linear Boundary')
    plt.xlim(0, he_mean_image.max()+10)  
    plt.ylim(-5, he_std_image.max()+25)  
    # Plot settings
    plt.xlabel('Mean Intensity')
    plt.ylabel('Standard Deviation')
    plt.title(f'Linear Boundary with M = {args.m}')
    plt.legend()
    plt.savefig(save_folder+'linear_boundary.jpg', format='jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)

    # get the index of superpixels that have passed quality control and output the qc_mask 
    qc_preserve_indicator = [True if he_std[i] >= linear_boundary[i] else False for i in range(len(he_tiles))]
    save_pickle(qc_preserve_indicator, pickle_folder+'qc_preserve_indicator.pickle')
    qc_mask = np.reshape(qc_preserve_indicator, image_shape)
    plt.figure(figsize=plt_figsize)
    plt.imshow(qc_mask)
    plt.savefig(save_folder+'qc_mask.jpg', format='jpg', dpi=dpi, bbox_inches='tight',pad_inches=0)

if __name__ == '__main__':
    main()
