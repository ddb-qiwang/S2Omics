#### Load package ####
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from UTILS import load_image
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.filters import threshold_otsu
from brokenaxes import brokenaxes


def generate_additionalPlots(prefix, he, he_std_image, he_std_norm_image, z_v_image, z_v_norm_image, ratio_norm, 
ratio_norm_image, mask1, mask1_updated, mask2, output_dir, super_pixel_size=16, generate_masked_plots = False):
    """
    Generate additional plots for quality control and visualization.

    Parameters:
    - prefix: str, path prefix where plots will be saved
    - he: np.array, original H&E image (H x W x 3)
    - he_std_image: np.array, standard deviation map
    - he_std_norm_image: np.array, normalized standard deviation map
    - z_v_image: np.array, weighted mean map
    - z_v_norm_image: np.array, normalized weighted mean map
    - ratio_norm: np.array, flattened ratio values
    - ratio_norm_image: np.array, ratio of std/mean map
    - mask1: np.array, initial low-density mask
    - mask1_updated: np.array, updated mask after texture filtering
    - mask2: np.array, ratio-based mask
    - super_pixel_size: int, patch size used in superpixel processing
    - generate_masked_plots: bool, whether to generate masked H&E images

    """

    # Step 1: Create directories
    output_dir_plots = os.path.join(prefix, output_dir, "AdditionalPlots")
    masked_he_dir = os.path.join(output_dir_plots, "maskedHE_plots")
    filtering_dir = os.path.join(output_dir_plots, "filtering_plots")
    texture_dir = os.path.join(output_dir_plots, "textureAnalysis_plots/masks")

    os.makedirs(masked_he_dir, exist_ok=True)
    os.makedirs(filtering_dir, exist_ok=True)
    os.makedirs(texture_dir, exist_ok=True)

    print(f"✅ Created folders:\n- {masked_he_dir}\n- {filtering_dir}\n- {texture_dir}")


    # Step 2: Load masks
    mask_path = os.path.join(f'{prefix}/{output_dir}', "mask.png")
    mask_small_path = os.path.join(f'{prefix}/{output_dir}', "mask-small.png")

    mask_fullres = np.array(Image.open(mask_path))
    mask_small = np.array(Image.open(mask_small_path))

    # Ensure masks are binary (0, 1)
    mask_fullres = (mask_fullres > 0).astype(np.uint8)
    mask_small = (mask_small > 0) #.astype(np.uint8)




   ###############  Scatter plot - he_std vs z_v colored by final mask  ###############
    fig, ax = plt.subplots(figsize=(8, 6))

    # Flatten mask and match to points
    mask_flat3 = mask_small.flatten()

    mean_intensity = z_v_image.flatten()
    std_dev = he_std_image.flatten()


    # Plot points for False (keep) and True (filter) separately
    ax.scatter(mean_intensity[mask_flat3], std_dev[mask_flat3], color='mediumseagreen', s=.2, label='Keep', edgecolor='none')
    ax.scatter(mean_intensity[~mask_flat3], std_dev[~mask_flat3], color='black', s=.2, label='Filter', edgecolor='none')


    ax.set_xlabel('Weighted Mean RGB"')
    ax.set_ylabel('Standard Deviation RGB')
    ax.set_title('Final Filtering')
    ax.legend(markerscale=20, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "scatter_mask_final.png"), dpi=500)
    plt.close()
    print(f"✅ Saved scatter plot: scatter_mask_final.png")



    ###############  Plot he_std_image and z_v_image side-by-side  ##############
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(he_std_norm_image - 1, cmap='jet')  # Subtract 1
    axes[0].set_title('Standard Deviation Image')
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(z_v_norm_image - 1, cmap='jet')  # Subtract 1
    axes[1].set_title('Weighted Mean Image')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "he_std_and_zv_images.png"), dpi=500)
    plt.close()
    print(f"✅ Saved image: he_std_and_zv_images.png")

    # Plot ratio_norm_image separately
    plt.figure(figsize=(8, 6))
    im = plt.imshow(ratio_norm_image, cmap='jet')
    plt.title('Ratio Image (Standard Deviation / Weighted Mean)')
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "ratio_norm_image.png"), dpi=500)
    plt.close()
    print(f"✅ Saved image: ratio_norm_image.png")


   ##############  Plot side-by-side histograms of he_std_norm_image and z_v_norm_image  ##############
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(he_std_norm_image.flatten()-1, bins=500, color='darkcyan', alpha=0.8)
    axes[0].set_title('Histogram: Standard Deviation')
    axes[0].set_xlabel('Normalized Standard Deviation')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(z_v_norm_image.flatten()-1, bins=500, color='lightsteelblue', alpha=0.8)
    axes[1].set_title('Histogram: Weighted Mean')
    axes[1].set_xlabel('Normalized Weighted Mean')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "he_std_and_zv_histograms.png"), dpi=500)
    plt.close()
    print(f"✅ Saved histograms: he_std_and_zv_histograms.png")


   ##############  Plot histogram of ratio distribution (with Otsu's threshold)  ##############

    # Assume ratio_norm is your 1D numpy array of normalized ratios.
    # Compute Otsu's threshold on the raw data (no smoothing)
    otsu_thresh = threshold_otsu(ratio_norm)

    # Define number of bins (for visualization)
    bin_n = int(len(ratio_norm) * 0.0005)

    # Compute histogram for reference (not needed for plotting via brokenaxes)
    counts, bins = np.histogram(ratio_norm, bins=bin_n)
    max_count = counts.max()

    # Define where to break the y-axis.Here, break_low covers the lower y-range and break_high the upper portion.
    # Adjust these values based on your histogram's count distribution. (for visualization)
    break_low = max_count * 0.1 
    break_high = max_count * 0.99

    # Directly create brokenaxes (this internally creates fig properly)
    bax = brokenaxes(
        ylims=((0, break_low), (break_high, max_count + 5)),
        hspace=0.05, despine=False
    )

    # Get fig handle from bax
    fig = bax.fig

    # Plot histogram
    bax.hist(ratio_norm, bins=bin_n, alpha=0.7, label="Histogram")

    # Plot threshold
    bax.axvline(otsu_thresh, color='red', linestyle='--', label=f'Otsu Threshold = {otsu_thresh:.3f}')

    # Set labels
    bax.set_xlabel("Ratio Norm")
    fig.suptitle("Histogram of Ratio Norm with Otsu Threshold")
    bax.legend()

    # Save figure correctly
    fig.savefig(os.path.join(filtering_dir, "ratio_histogram_with_threshold.png"), dpi=500)

    plt.close(fig)
    print(f"✅ Saved scatter plot: ratio_histogram_with_threshold.png")



    ###############  Create scatter plot colored by density  ##############

    mean_intensity = z_v_image.flatten()
    std_dev = he_std_image.flatten()

    plt.figure(figsize=(11, 6))
    plt.hexbin(mean_intensity, std_dev, gridsize=700, cmap='viridis', bins='log')
    #plt.xlim(0, z_v_image.max())  
    plt.ylim(-10, he_std_image.max()+10)  
    # Add colorbar
    plt.colorbar(label="Density (log scale)")
    # Labels
    plt.xlabel("Weighted Mean RGB")
    plt.ylabel("Standard Deviation RGB")
    plt.title("Std vs Mean RGB: Colored by Density", fontsize = 15)
    #plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "scatter_density.png"), dpi=500)
    plt.close()
    print(f"✅ Saved scatter plot: scatter_density.png")

   


   ###############  Create scatter plot colored by ratio  ###############  
    plt.figure(figsize=(11, 6))
    plt.scatter(mean_intensity, std_dev, cmap='inferno', s=.025, c=ratio_norm)
    #plt.xlim(0, z_v_image.max())  
    plt.ylim(-10, he_std_image.max()+10)  
    # Add colorbar
    plt.colorbar(label="Ratio")
    # Labels
    plt.xlabel("Weighted Mean RGB")
    plt.ylabel("Standard Deviation RGB")
    plt.title("Std vs Mean RGB: Colored by Ratio", fontsize = 15)
    #plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "scatter_ratio.png"), dpi=500)
    plt.close()
    print(f"✅ Saved scatter plot: scatter_ratio.png")


    ###############  Scatter plot - he_std vs z_v colored by mask2 (ratio)  ###############
    fig, ax = plt.subplots(figsize=(8, 6))

    # Flatten masks
    mask_flat1_updated = mask1_updated.flatten()

    # Plot keep and filter points
    ax.scatter(mean_intensity[~mask_flat1_updated], std_dev[~mask_flat1_updated], color='mediumseagreen', s=0.2, label='Keep', edgecolor='none')
    ax.scatter(mean_intensity[mask_flat1_updated], std_dev[mask_flat1_updated], color='black', s=0.2, label='Filter', edgecolor='none')

    ax.set_xlabel('Weighted Mean RGB')
    ax.set_ylabel('Standard Deviation RGB')
    ax.set_title('Low Density Filtering')
    ax.legend(markerscale=20, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "scatter_mask1_lowdensity.png"), dpi=500)
    plt.close()
    print(f"✅ Saved scatter plot: scatter_mask1_lowdensity.png")


    ###############  Scatter plot - he_std vs z_v colored by mask2 (ratio) with texture analysis ###############
    fig, ax = plt.subplots(figsize=(8, 6))

    # Flatten masks
    mask_flat1 = mask1.flatten()
    mask_flat1_updated = mask1_updated.flatten()

    # Plot keep and filter points
    ax.scatter(mean_intensity[~mask_flat1_updated], std_dev[~mask_flat1_updated], color='mediumseagreen', s=0.15, label='Keep', edgecolor='none')
    ax.scatter(mean_intensity[mask_flat1_updated], std_dev[mask_flat1_updated], color='black', s=0.15, label='Filter', edgecolor='none')

    # ➡️ Highlight points that were True originally but became False after update
    rescued_mask1 = (mask_flat1 == True) & (mask_flat1_updated == False)
    ax.scatter(mean_intensity[rescued_mask1], std_dev[rescued_mask1], color='blue', s=0.15, label='Rescued', edgecolor='none')

    ax.set_xlabel('Weighted Mean RGB')
    ax.set_ylabel('Standard Deviation RGB')
    ax.set_title('Low Density Filtering with Rescued Points')
    ax.legend(markerscale=20, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "scatter_mask1_lowdensity_with_rescue.png"), dpi=500)
    plt.close()
    print(f"✅ Saved scatter plot: scatter_mask1_lowdensity_with_rescue.png")


    ###############  Scatter plot - he_std vs z_v colored by mask2 (ratio)  ###############
    fig, ax = plt.subplots(figsize=(8, 6))

    # Flatten mask and match to points
    mask_flat2 = mask2.flatten()

    # Plot points for False (keep) and True (filter) separately
    ax.scatter(mean_intensity[~mask_flat2], std_dev[~mask_flat2], color='mediumseagreen', s=.2, label='Keep', edgecolor='none')
    ax.scatter(mean_intensity[mask_flat2], std_dev[mask_flat2], color='black', s=.2, label='Filter', edgecolor='none')


    ax.set_xlabel('Weighted Mean RGB"')
    ax.set_ylabel('Standard Deviation RGB')
    ax.set_title('Ratio Filtering')
    ax.legend(markerscale=20, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(filtering_dir, "scatter_mask2_ratio.png"), dpi=500)
    plt.close()
    print(f"✅ Saved scatter plot: scatter_mask2_ratio.png")






    ###############  Generate additional masked H&E Plots  ###############

    if generate_masked_plots:


        ###############  Masked H&E Plots (final) ###############

        # Load and resize mask to match H&E dimensions if needed

        # 1. Black where mask = 1 (filtered out)
        he_black_mask = he.copy()
        he_black_mask[mask_fullres == 0] = [0, 0, 0]  # Set masked pixels to black

        # Save directly as image
        Image.fromarray(he_black_mask).save(os.path.join(masked_he_dir, "he_remove_black_mask_overlay.png"))
        print(f"✅ Saved: he_remove_black_mask_overlay.png")

        # 2. Green where mask = 0 (keep regions highlighted)
        he_msea_mask = he.copy()

        # Define mediumseagreen RGB color
        mediumseagreen = np.array([60, 179, 113], dtype=np.uint8)

        # Create overlay
        green_overlay = np.zeros_like(he)
        green_overlay[:, :, 0] = mediumseagreen[0]
        green_overlay[:, :, 1] = mediumseagreen[1]
        green_overlay[:, :, 2] = mediumseagreen[2]

        alpha = 0.8  # Transparency factor

        mask_keep = (mask_fullres == 1)
        he_msea_mask[mask_keep] = (alpha * green_overlay[mask_keep] + (1 - alpha) * he_msea_mask[mask_keep]).astype(np.uint8)

        # Save directly as image
        Image.fromarray(he_msea_mask).save(os.path.join(masked_he_dir, "he_keep_green_mask_overlay.png"))
        print(f"✅ Saved: he_keep_green_mask_overlay.png")




        ###############  Masked H&E Plots (ratio)  ###############

        # Reshape masks to match the superpixel grid
        image_height, image_width = he.shape[:2]
        super_pixel_size = 16  # (or whatever your setting is)
        num_super_pixels_y = image_height // super_pixel_size
        num_super_pixels_x = image_width // super_pixel_size

        # Build full-size mask showing differences
        # Want pixels that were False in mask1_updated but True in mask1 (i.e., newly filtered after texture analysis)

        mask2_flat = mask2.flatten()

        # Expand this difference mask to full image size
        mask2_large = np.zeros((image_height, image_width), dtype=np.uint8)

        for i in range(num_super_pixels_y):
            for j in range(num_super_pixels_x):
                value = 255 if mask2_flat[i * num_super_pixels_x + j] else 0
                mask2_large[i * super_pixel_size:(i + 1) * super_pixel_size,
                    j * super_pixel_size:(j + 1) * super_pixel_size] = value


        # Load and resize mask to match H&E dimensions if needed

        # 1. Black where mask = 1 (filtered out)
        he_black_mask = he.copy()
        he_black_mask[mask2_large == 0] = [0, 0, 0]  # Set masked pixels to black

        # Save directly as image
        Image.fromarray(he_black_mask).save(os.path.join(masked_he_dir, "he_remove_black_mask_ratio.png"))
        print(f"✅ Saved: he_remove_black_mask_ratio.png")

        # 2. Green where mask = 0 (keep regions highlighted)
        he_msea_mask = he.copy()

        # Define mediumseagreen RGB color
        mediumseagreen = np.array([60, 179, 113], dtype=np.uint8)

        # Create overlay
        green_overlay = np.zeros_like(he)
        green_overlay[:, :, 0] = mediumseagreen[0]
        green_overlay[:, :, 1] = mediumseagreen[1]
        green_overlay[:, :, 2] = mediumseagreen[2]

        alpha = 0.8  # Transparency factor

        mask_keep = (mask2_large == 1)
        he_msea_mask[mask_keep] = (alpha * green_overlay[mask_keep] + (1 - alpha) * he_msea_mask[mask_keep]).astype(np.uint8)

        # Save directly as image
        Image.fromarray(he_msea_mask).save(os.path.join(masked_he_dir, "he_keep_green_mask_ratio.png"))
        print(f"✅ Saved: he_keep_green_mask_ratio.png")





        ############### Masked H&E Plots (texture analysis - final density mask ) ###############

        # Reshape to super-pixel grid
        image_height, image_width = he.shape[:2]
        num_super_pixels_y = image_height // super_pixel_size
        num_super_pixels_x = image_width // super_pixel_size


        # Build full-size binary mask
        super_pixel_values = mask1_updated == 0  
        mask1_large = np.zeros((image_height, image_width), dtype=np.uint8)

        for i in range(num_super_pixels_y):
            for j in range(num_super_pixels_x):
                value = 0 if super_pixel_values[i, j] else 255
                mask1_large[i * super_pixel_size:(i + 1) * super_pixel_size,
                    j * super_pixel_size:(j + 1) * super_pixel_size] = value


        # 1. Black where mask = 1 (filtered out)
        he_black_mask = he.copy()
        he_black_mask[mask1_large == 255] = [0, 0, 0]  # Set masked pixels to black

        # Save directly as image
        Image.fromarray(he_black_mask).save(os.path.join(texture_dir, "he_remove_black_mask_density.png"))
        print(f"✅ Saved: he_remove_black_mask_density.png")

        # 2. Green where mask = 0 (keep regions highlighted)
        he_msea_mask = he.copy()

        # Define mediumseagreen RGB color
        mediumseagreen = np.array([60, 179, 113], dtype=np.uint8)

        # Create overlay
        green_overlay = np.zeros_like(he)
        green_overlay[:, :, 0] = mediumseagreen[0]
        green_overlay[:, :, 1] = mediumseagreen[1]
        green_overlay[:, :, 2] = mediumseagreen[2]

        alpha = 1  # Transparency factor

        mask_keep = (mask1_large == 0)
        he_msea_mask[mask_keep] = (alpha * green_overlay[mask_keep] + (1 - alpha) * he_msea_mask[mask_keep]).astype(np.uint8)

        # Save directly as image
        Image.fromarray(he_msea_mask).save(os.path.join(texture_dir, "he_keep_green_mask_density.png"))
        print(f"✅ Saved: he_keep_green_mask_density.png")






        ############### Masked H&E Plots (texture analysis - showing changed regions only) ###############

        # Reshape masks to match the superpixel grid
        image_height, image_width = he.shape[:2]
        super_pixel_size = 16  # (or whatever your setting is)
        num_super_pixels_y = image_height // super_pixel_size
        num_super_pixels_x = image_width // super_pixel_size

        # Build full-size mask showing differences
        # Want pixels that were False in mask1_updated but True in mask1 (i.e., newly filtered after texture analysis)

        mask1_flat = mask1.flatten()
        mask1_updated_flat = mask1_updated.flatten()

        # Pixels that were kept in mask1 but now filtered in mask1_updated
        difference_mask = (mask1_flat) & (~mask1_updated_flat)

        # Expand this difference mask to full image size
        difference_large = np.zeros((image_height, image_width), dtype=np.uint8)

        for i in range(num_super_pixels_y):
            for j in range(num_super_pixels_x):
                value = 255 if difference_mask[i * num_super_pixels_x + j] else 0
                difference_large[i * super_pixel_size:(i + 1) * super_pixel_size,
                j * super_pixel_size:(j + 1) * super_pixel_size] = value

        # 1. Black out the changed regions
        he_black_diff = he.copy()
        he_black_diff[difference_large == 0] = [0, 0, 0]

        Image.fromarray(he_black_diff).save(os.path.join(texture_dir, "he_rescued_shown_mask_textureFiltered.png"))
        print(f"✅ Saved: he_rescued_shown_mask_textureFiltered.png")

        he_blue_diff = he.copy()

        # Blue color
        blue = np.array([0, 0, 255], dtype=np.uint8)

        blue_overlay = np.zeros_like(he)
        blue_overlay[:, :, 0] = blue[0]
        blue_overlay[:, :, 1] = blue[1]
        blue_overlay[:, :, 2] = blue[2]

        alpha = 0.8  # Transparency level

        mask_changed = (difference_large == 255)
        he_blue_diff[mask_changed] = (alpha * blue_overlay[mask_changed] + (1 - alpha) * he_blue_diff[mask_changed]).astype(np.uint8)

        Image.fromarray(he_blue_diff).save(os.path.join(texture_dir, "he_rescued_blue_mask_textureFiltered.png"))
        print(f"✅ Saved: he_rescued_blue_mask_textureFiltered.png")





