#### Load package ####
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from PIL import Image
from .UTILS import measure_peak_memory

@measure_peak_memory
def generate_final_mask(prefix, he, mask1_updated, mask2, output_dir, clean_background = True, super_pixel_size=16, minSize = 10):


    # Combine masks
    masked = (mask1_updated.flatten() | mask2.flatten())
    image_height, image_width = he.shape[:2]

    # Reshape to super-pixel grid
    num_super_pixels_y = image_height // super_pixel_size
    num_super_pixels_x = image_width // super_pixel_size
    mask = masked.reshape((num_super_pixels_y, num_super_pixels_x))
    cleaned = 1 - mask
    cleaned = (cleaned * 255).astype(np.uint8)

    # Clean artifacts (specs) in super-pixel space
    if clean_background:
        binary = 1 - mask  # invert to remove specs (foreground = 1)
        cleaned = morphology.remove_small_objects(binary.astype(bool), min_size = minSize, connectivity=2)
        cleaned = (cleaned * 255).astype(np.uint8)

    # Save the cleaned mask at super-pixel level
    #save_pickle(cleaned, os.path.join(f"{prefix}/{output_dir}", 'conserve_index_mask-small.pickle'))
    Image.fromarray(cleaned).save(os.path.join(f"{prefix}/{output_dir}", 'mask-small.png'))

    # Build full-size binary mask
    super_pixel_values = cleaned == 0  
    mask_final = np.zeros((image_height, image_width), dtype=np.uint8)


    for i in range(num_super_pixels_y):
        for j in range(num_super_pixels_x):
            value = 0 if super_pixel_values[i, j] else 255
            mask_final[i * super_pixel_size:(i + 1) * super_pixel_size,
                       j * super_pixel_size:(j + 1) * super_pixel_size] = value

    # Save full-resolution final mask
    #save_pickle(mask_final, prefix+output_dir+'conserve_index_mask.pickle')
    Image.fromarray(mask_final).save(os.path.join(f"{prefix}/{output_dir}", 'mask.png'))

    print("✅ Final masks saved in:", output_dir)


