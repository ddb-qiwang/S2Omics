import argparse
import os
from time import time

from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt

from .s1_utils import (
        crop_image, load_image, save_image, 
        read_string, write_string)

def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.ome.tif', '.tiff', '.tif', '.svs']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename

def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img

def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img

def histology_preprocess(prefix, show_image=False):
    
    pixel_size_raw = float(read_string(prefix+'pixel-size-raw.txt'))
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size

    img = load_image(get_image_filename(prefix+'he-raw'))
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    t0 = time()
    img = rescale_image(img, scale)
    print(int(time() - t0), 'sec')
    img = img.astype(np.uint8)
    save_image(img, prefix+'he-scaled.jpg')

    pad = 256
    img = adjust_margins(img, pad=pad, pad_value=255)
    save_image(img, f'{prefix}he.jpg')
    print('Preprocessed H&E image saved!')

    if show_image:
        plt.imshow(img)