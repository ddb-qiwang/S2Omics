import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from PIL import Image
from matplotlib.colors import ListedColormap
from UTILS import load_image,measure_peak_memory
import pandas as pd
from sklearn.mixture import GaussianMixture
import cv2

def efficient_skimage_rgb2gray(image):
    '''memory efficitnt and keep same result with skimag rgb2gray'''
    h, w = image.shape[:2]
    result = np.zeros((h, w), dtype=np.float32) 
    result += image[:,:,0].astype(np.float32) * 0.2125
    result += image[:,:,1].astype(np.float32) * 0.7154
    result += image[:,:,2].astype(np.float32) * 0.0721
    return result.astype(np.uint8)

@measure_peak_memory
def run_texture_analysis(prefix, image, tissue_mask, output_dir, patch_size=16, glcm_levels=64):

    output = f"{prefix}/{output_dir}/AdditionalPlots/textureAnalysis_plots/"
    
    os.makedirs(output, exist_ok=True)

    # Convert image to grayscale
#     gray_image = rgb2gray(image)
#     gray_image = (gray_image * 255).astype(np.uint8)
    gray_image = efficient_skimage_rgb2gray(image)

    # Process mask
    mask = tissue_mask.astype(bool)

    # Get dimensions
    h, w = gray_image.shape
    h_mask, w_mask = mask.shape
    assert h // patch_size == h_mask and w // patch_size == w_mask, "Mask dimensions do not match superpixel grid"

    # Normalize grayscale values
    gray_image = (gray_image / 255 * (glcm_levels - 1)).astype(np.uint8)


    # === EARLY RETURN IF NO SUPERPIXELS WERE SELECTED
    if mask.sum() == 0 :
        print("✅ Skipping texture analysis — no low density superpixels.")
        updated_mask = mask.copy()
        return updated_mask


    # Initialize maps
    energy_map = np.full(mask.shape, np.nan)
    homogeneity_map = np.full(mask.shape, np.nan)
    entropy_map = np.full(mask.shape, np.nan)
    sharpness_map = np.full(mask.shape, np.nan)
    color_filter_mask = np.zeros(mask.shape, dtype=bool)  # Track bad color removals
    tracker = 0

    for i in range(h_mask):
        for j in range(w_mask):
            if mask[i, j]:
                patch_gray = gray_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                patch_rgb = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]

                mean_r = np.mean(patch_rgb[:, :, 0])
                mean_g = np.mean(patch_rgb[:, :, 1])
                mean_b = np.mean(patch_rgb[:, :, 2])

                is_green = (mean_g > mean_r + 20) and (mean_g > mean_b + 20)
                is_gray = (np.std([mean_r, mean_g, mean_b]) < 10)
                is_too_bright = (mean_r > 230) and (mean_g > 230) and (mean_b > 230)

                if is_green or is_gray or is_too_bright:
                    color_filter_mask[i, j] = True
                    tracker = tracker+1
                    continue

                # Compute GLCM features
                glcm = graycomatrix(patch_gray, distances=[1], angles=[0], levels=glcm_levels, symmetric=True, normed=True)
                energy_map[i, j] = graycoprops(glcm, 'energy')[0, 0]
                homogeneity_map[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]

                glcm_vals = glcm[:, :, 0, 0]
                glcm_nonzero = glcm_vals[glcm_vals > 0]
                entropy_map[i, j] = -np.sum(glcm_nonzero * np.log2(glcm_nonzero))

                sharpness_map[i, j] = cv2.Laplacian(patch_gray, cv2.CV_64F).var()

    # Normalize maps
    energy_map_norm = (energy_map - np.nanmin(energy_map)) / (np.nanmax(energy_map) - np.nanmin(energy_map))
    homogeneity_map_norm = (homogeneity_map - np.nanmin(homogeneity_map)) / (np.nanmax(homogeneity_map) - np.nanmin(homogeneity_map))
    entropy_map_norm = (entropy_map - np.nanmin(entropy_map)) / (np.nanmax(entropy_map) - np.nanmin(entropy_map))
    sharpness_map_norm = (sharpness_map - np.nanmin(sharpness_map)) / (np.nanmax(sharpness_map) - np.nanmin(sharpness_map))


    # === EARLY RETURN IF EVERYTHING IS ALREADY KEPT
    if mask.sum() - tracker < 2:
        print("✅ Skipping GMM clustering — all low density superpixels already marked as remove.")
        updated_mask = mask.copy()
        return updated_mask


    def save_colormapped_map(map_norm, title, filename):
        colormap = plt.get_cmap("jet")
        colored = colormap(map_norm)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        Image.fromarray(colored).save(os.path.join(output, filename))
        print(f"✅ {title} map saved as '{filename}'")

    save_colormapped_map(entropy_map_norm, "Entropy", "glcm_entropy_map_colored.png")
    save_colormapped_map(energy_map_norm, "Energy", "glcm_energy_map_colored.png")
    save_colormapped_map(homogeneity_map_norm, "Homogeneity", "glcm_homogeneity_map_colored.png")


    # Clustering based on features
    features = pd.DataFrame({
        'homogeneity': homogeneity_map_norm.flatten(),
        'energy': energy_map_norm.flatten(),
        'entropy': entropy_map_norm.flatten(),
    })

    valid_features = features.dropna().reset_index(drop=True)
    gmm = GaussianMixture(n_components=4, random_state=45)
    labels = gmm.fit_predict(valid_features)

    cluster_map = np.full(mask.shape, np.nan)
    valid_mask = ~np.isnan(homogeneity_map_norm)
    cluster_map[valid_mask] = labels

    means = valid_features.groupby(labels).mean()

    # === Print GLCM feature means
    print("\n=== GLCM Metric Means ===")
    print(means)

    # Calculate and print scores
    scores = means['energy'] + means['homogeneity'] - means['entropy']
    print("\n=== Cluster Scores ===")
    for cluster_label, score in scores.items():
        print(f"Cluster {cluster_label}: Score = {score:.4f}")

    # Print number of observations
    print("\n=== Number of Observations per Cluster ===")
    counts = valid_features.groupby(labels).size()
    total_count = counts.sum()
    for cluster_label, count in counts.items():
        print(f"Cluster {cluster_label}: {count}")
    print(f"Total: {total_count}")

    output_path = os.path.join(output, "texture_analysis_summary.txt")

    with open(output_path, "w") as f:
        f.write("=== GLCM Metric Means ===\n")
        f.write(means.to_string())
        f.write("\n\n=== Cluster Scores ===\n")
        for cluster_label, score in scores.items():
            f.write(f"Cluster {cluster_label}: Score = {score:.4f}\n")

        f.write("\n=== Number of Observations per Cluster ===\n")
        for cluster_label, count in counts.items():
            f.write(f"Cluster {cluster_label}: {count}\n")
        f.write(f"Total: {total_count}\n")


    # Now select the clusters with the two lowest scores
    keep_labels = scores.nsmallest(2).index.tolist()  # list of 2 cluster labels
    keep_coords = np.where(np.isin(cluster_map, keep_labels))

    updated_mask = mask.copy()
    updated_mask[keep_coords] = False

    # Color clusters (blue = bad color, red = bad texture, green = keep)
    cluster_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(h_mask):
        for j in range(w_mask):
            if color_filter_mask[i, j]:
                cluster_rgb[i, j] = [30, 144, 255]  # blue (dodgerblue)
            elif np.isnan(cluster_map[i, j]):
                cluster_rgb[i, j] = [0, 0, 0]  # black
            elif cluster_map[i, j] in keep_labels:
                cluster_rgb[i, j] = [0, 255, 0]  # green
            else:
                cluster_rgb[i, j] = [255, 0, 0]  # red

    Image.fromarray(cluster_rgb).save(os.path.join(output, "cluster_labels_colored.png"))
    print("\n✅ Clustered texture map saved as 'cluster_labels_colored.png'")


    return updated_mask

