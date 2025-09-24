#### Load package ####
import argparse
import numpy as np
from einops import reduce
from utils import load_image, save_image, load_mask, get_image_filename, smart_save_image
from image import crop_image
from PIL import Image
import os


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


def reduce_mask(mask, factor):
    mask = reduce(
            mask.astype(np.float32),
            '(h0 h1) (w0 w1) -> h0 w0', 'mean',
            h1=factor, w1=factor) > 0.5
    return mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--patchSize', type=int, default=16)
    args = parser.parse_args()
    return args



def main():

    args = get_args()
    pad = args.patchSize**2


    # === Usage ===
    if args.image:
        # Load histology image from .jpg or .tif

        img = load_image(get_image_filename(args.prefix+'he-scaled'))

        # Pad image to match model input constraints
        img = adjust_margins(img, pad=pad, pad_value=255)

        # Save padded version as .jpg or .tif depending on your needs
        smart_save_image(img, args.prefix, base_name="he", size_threshold=10000)



if __name__ == '__main__':
    main()




