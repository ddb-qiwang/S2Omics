#### Load package ####
import numpy as np
from skimage.filters import threshold_otsu
from UTILS import measure_peak_memory

@measure_peak_memory

def run_ratio_filtering(ratio_norm, mask1_lowdensity_updated):
    """
    Applies ratio-based filtering using Otsu's threshold after excluding regions filtered by low density.

    Parameters:
        ratio_norm (np.ndarray): 1D array of normalized std/mean ratio values.
        mask1_lowdensity_updated (np.ndarray): Boolean mask (True for regions to keep).

    Returns:
        mask2_lowratio (np.ndarray): Boolean mask indicating which regions pass the ratio threshold.
        otsu_thresh (float): The computed Otsu threshold.
    """
    # Invert the mask to get the regions to apply filtering to
    r = ~mask1_lowdensity_updated.flatten()
    data_filtered1 = ratio_norm[r]  # Apply only to regions that passed low density filtering

    # Remove top 10% values to prevent outlier influence
    upper_bound = np.percentile(data_filtered1, 90)
    h = data_filtered1 < upper_bound
    data_filtered2 = data_filtered1[h]

    # Compute Otsu's threshold on the filtered values
    otsu_thresh = threshold_otsu(data_filtered2)
    #print(f"Otsu Threshold: {otsu_thresh:.3f}")

    # Generate mask based on threshold
    mask2_lowratio = ratio_norm < otsu_thresh

    return mask2_lowratio, otsu_thresh

