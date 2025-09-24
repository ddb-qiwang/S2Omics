#### Load package ####
import argparse
import os
from time import time
from PIL import Image
import PIL
from skimage.transform import rescale
import numpy as np
import tifffile
import cv2


from utils import (
        load_image, save_image, read_string, write_string,
        load_tsv, save_tsv, smart_save_image)

Image.MAX_IMAGE_PIXELS = None

PIL.Image.MAX_IMAGE_PIXELS = 10e100


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff', '.tif']:
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

def rescale_image_cv2(img, scale):
    h,w = img.shape[:2]
    new_size = (int(w*scale), int(h*scale))
    img_rescaled = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return img_rescaled





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--pixelSizeRaw', type=float, default=None)
    parser.add_argument('--pixelSize', type=float, default=0.5)
    args = parser.parse_args()
    return args


def main():
    

    args = get_args()

    pixel_size_raw = args.pixelSizeRaw
    pixel_size = args.pixelSize
    scale = pixel_size_raw / pixel_size

    if args.image:
        img = load_image(get_image_filename(args.prefix+'he-raw'))
        img = img.astype(np.float32)
        print(f'Rescaling image (scale: {scale:.3f})...')
        t0 = time()
        img = rescale_image_cv2(img, scale)
        print(int(time() - t0), 'sec')
        img = img.astype(np.uint8)
        print(img.shape)
        #save_image(img, args.prefix+'he-scaled.jpg')
        #tifffile.imwrite(args.prefix+"he-scaled.tiff", img, bigtiff=True)
        smart_save_image(img, args.prefix, base_name="he-scaled", size_threshold=10000)




if __name__ == '__main__':
    main()
