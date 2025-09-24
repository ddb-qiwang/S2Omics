#### Load package ####
import numpy as np
from scipy.ndimage import gaussian_filter
from UTILS import measure_peak_memory

@measure_peak_memory
def compute_low_density_mask(z_v_image, he_std_image, ratio_norm,  density_thresh=100):
    # Step 1: Histogram binning
    bin_n = int(len(ratio_norm) * 0.00005)
    #print(f"Number of bins: {bin_n}")

    sigma = max(1, int(bin_n * 0.02))
    #print(f"Sigma: {sigma}")

    # Flatten inputs
    mean = z_v_image.flatten()
    std = he_std_image.flatten()

    # Step 2: Compute 2D histogram (density map)
    hist, xedges, yedges = np.histogram2d(mean, std, bins=[bin_n, bin_n])

    # Step 3: Smooth the histogram
    smoothed_density = gaussian_filter(hist, sigma=sigma)

    # Step 4: Assign density values to each patch
    xidx = np.searchsorted(xedges, mean) - 1
    yidx = np.searchsorted(yedges, std) - 1

    valid = (
        (xidx >= 0) & (xidx < smoothed_density.shape[0]) &
        (yidx >= 0) & (yidx < smoothed_density.shape[1])
    )

    density = np.zeros_like(std)
    density[valid] = smoothed_density[xidx[valid], yidx[valid]]

    # Step 5: Threshold to create mask
    mask1_lowdensity = density < density_thresh
    return mask1_lowdensity.reshape(z_v_image.shape)


# Optional CLI support
if __name__ == '__main__':
    import argparse, pickle, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to a pickle file with z_v_image, he_std_image, ratio_norm')
    parser.add_argument('--output', type=str, required=True, help='Path to save the low density mask')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    mask = compute_low_density_mask(
        z_v_image=data['z_v_image'],
        he_std_image=data['he_std_image'],
        ratio_norm=data['ratio_norm']
    )

    with open(args.output, 'wb') as f:
        pickle.dump(mask, f)

    print(f"Low density mask saved to {args.output}")
