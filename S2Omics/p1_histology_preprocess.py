import argparse
import os
from time import time

from skimage.transform import rescale
import numpy as np

from s1_utils import (
        crop_image, load_image, save_image, 
        read_string, write_string)


'''preprocess H&E stained image
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
        The folder should contain three files: 
        1. he-raw.jpg/png/tiff/svs
        2. pixel-size-raw.txt describe microns/pixel 0.3 for an example
        3. pixel-size.txt the target microns/pixel after rescaling, please set as 0.5
return:
    he-scale.jpg: rescaled H&E stained image
    he.jpg: rescale and padded H&E stained image, all following procedures will be based on this image file
'''

def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff', '.svs']:
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

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    args = parser.parse_args()
    return args

def main():

    args = get_args()

    pixel_size_raw = float(read_string(args.prefix+'pixel-size-raw.txt'))
    pixel_size = float(read_string(args.prefix+'pixel-size.txt'))
    scale = pixel_size_raw / pixel_size

    img = load_image(get_image_filename(args.prefix+'he-raw'))
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    t0 = time()
    img = rescale_image(img, scale)
    print(int(time() - t0), 'sec')
    img = img.astype(np.uint8)
    save_image(img, args.prefix+'he-scaled.jpg')

    pad = 256
    img = adjust_margins(img, pad=pad, pad_value=255)
    save_image(img, f'{args.prefix}he.jpg')

if __name__ == '__main__':
    main()
