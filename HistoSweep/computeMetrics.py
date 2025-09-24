#### Load package ####
import numpy as np
from UTILS import load_image, measure_peak_memory
import psutil
import os
import gc

def get_memory_usage():
    """Get current process memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def memory_efficient_patchify(x, patch_size):
    """Memory-efficient patchify with identical results"""
    shape_ori = np.array(x.shape[:2])
    shape_ext = (shape_ori + patch_size - 1) // patch_size * patch_size
    
    # Check if padding is actually needed
    needs_padding = not np.array_equal(shape_ori, shape_ext)
    
    if needs_padding:
        x_padded = np.pad(
            x,
            ((0, shape_ext[0] - x.shape[0]),
             (0, shape_ext[1] - x.shape[1]),
             (0, 0)),
            mode='edge')
    else:
        # If no padding needed, use original array directly
        x_padded = x
    
    tiles_shape = shape_ext // patch_size
    patch_strides = (
        x_padded.strides[0] * patch_size,
        x_padded.strides[1] * patch_size,
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2])
    
    patches = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=(tiles_shape[0], tiles_shape[1], patch_size, patch_size, x.shape[2]),
        strides=patch_strides)
    
    patches_reshaped = patches.reshape(-1, patch_size, patch_size, x.shape[2])
    shapes = dict(original=shape_ori, padded=shape_ext, tiles=tiles_shape)
    
    return patches_reshaped, shapes

def chunked_computation(he_tiles, compute_func, chunk_size=30000):
    """Generic chunked computation function to reduce peak memory"""
    n_patches = he_tiles.shape[0]
    
    # Compute first chunk to determine output shape
    first_chunk_size = min(chunk_size, n_patches)
    first_result = compute_func(he_tiles[:first_chunk_size])
    
    # Determine full output array shape based on first result
    if first_result.ndim == 1:
        full_result = np.zeros(n_patches, dtype=first_result.dtype)
    else:
        full_result = np.zeros((n_patches,) + first_result.shape[1:], dtype=first_result.dtype)
    
    full_result[:first_chunk_size] = first_result
    
    # Process remaining chunks
    for i in range(first_chunk_size, n_patches, chunk_size):
        end_i = min(i + chunk_size, n_patches)
        chunk_result = compute_func(he_tiles[i:end_i])
        full_result[i:end_i] = chunk_result
        del chunk_result
    
    return full_result

@measure_peak_memory
def compute_metrics_memory_optimized(he, patch_size=16):
    """Memory-optimized compute_metrics with identical results"""
    
    # Step 1: Memory-efficient patchify
    he_tiles, shapes = memory_efficient_patchify(he, patch_size=patch_size)
    
    # Step 2: Chunked standard deviation computation
    def std_func(chunk):
        return np.std(chunk, axis=(1, 2, 3))
    
    he_std_flat = chunked_computation(he_tiles, std_func)
    he_std_image = he_std_flat.reshape(shapes['tiles'])
    
    # Step 3: Chunked RGB mean computation
    def mean_func(chunk):
        return np.mean(chunk, axis=(1, 2))
    
    mean_rgb_per_patch = chunked_computation(he_tiles, mean_func)
    
    # Release he_tiles to free memory
    del he_tiles
    gc.collect()
    
    # Step 4: Compute variance and z_v
    V_r = np.var(mean_rgb_per_patch[:, 0])
    V_g = np.var(mean_rgb_per_patch[:, 1])
    V_b = np.var(mean_rgb_per_patch[:, 2])
    numerators = (
        mean_rgb_per_patch[:, 0] * V_r +
        mean_rgb_per_patch[:, 1] * V_g +
        mean_rgb_per_patch[:, 2] * V_b)
    denominator = V_r + V_g + V_b
    z_v = numerators / denominator
    z_v_image = z_v.reshape(shapes['tiles'])
    
    # Step 5: Normalize and compute ratio
    flattened_std = he_std_image.flatten()
    flattened_std = (flattened_std - flattened_std.min()) / (flattened_std.max() - flattened_std.min()) + 1
    he_std_norm_image = flattened_std.reshape(z_v_image.shape)
    
    flattened_mean = z_v_image.flatten()
    flattened_mean = (flattened_mean - flattened_mean.min()) / (flattened_mean.max() - flattened_mean.min()) + 1
    z_v_norm_image = flattened_mean.reshape(z_v_image.shape)
    
    ratio_norm = flattened_std / flattened_mean
    ratio_norm = (ratio_norm - ratio_norm.min()) / (ratio_norm.max() - ratio_norm.min())
    ratio_norm_image = ratio_norm.reshape(z_v_image.shape)
    
    return he_std_norm_image, he_std_image, z_v_norm_image, z_v_image, ratio_norm, ratio_norm_image

def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (shape_ori + patch_size - 1) // patch_size * patch_size
    x_padded = np.pad(
        x,
        ((0, shape_ext[0] - x.shape[0]),
         (0, shape_ext[1] - x.shape[1]),
         (0, 0)),
        mode='edge')
    tiles_shape = shape_ext // patch_size
    patch_strides = (
        x_padded.strides[0] * patch_size,
        x_padded.strides[1] * patch_size,
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2])
    patches = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=(tiles_shape[0], tiles_shape[1], patch_size, patch_size, x.shape[2]),
        strides=patch_strides)
    patches = patches.reshape(-1, patch_size, patch_size, x.shape[2])
    shapes = dict(original=shape_ori, padded=shape_ext, tiles=tiles_shape)
    return patches, shapes

@measure_peak_memory
def compute_metrics(he, patch_size=16):
    he_tiles, shapes = patchify(he, patch_size=patch_size)

    he_std_image = np.std(he_tiles, axis=(1, 2, 3))
    he_std_image = he_std_image.reshape(shapes['tiles'])

    mean_rgb_per_patch = np.mean(he_tiles, axis=(1, 2))
    V_r = np.var(mean_rgb_per_patch[:, 0])
    V_g = np.var(mean_rgb_per_patch[:, 1])
    V_b = np.var(mean_rgb_per_patch[:, 2])
    numerators = (
        mean_rgb_per_patch[:, 0] * V_r +
        mean_rgb_per_patch[:, 1] * V_g +
        mean_rgb_per_patch[:, 2] * V_b)
    denominator = V_r + V_g + V_b
    z_v = numerators / denominator
    z_v_image = z_v.reshape(shapes['tiles'])

    # Normalize and compute ratio
    flattened_std = he_std_image.flatten()
    flattened_std = (flattened_std - flattened_std.min()) / (flattened_std.max() - flattened_std.min()) + 1
    he_std_norm_image = flattened_std.reshape(z_v_image.shape)

    flattened_mean = z_v_image.flatten()
    flattened_mean = (flattened_mean - flattened_mean.min()) / (flattened_mean.max() - flattened_mean.min()) + 1
    z_v_norm_image = flattened_mean.reshape(z_v_image.shape)

    ratio_norm = flattened_std / flattened_mean
    ratio_norm = (ratio_norm - ratio_norm.min()) / (ratio_norm.max() - ratio_norm.min())
    ratio_norm_image = ratio_norm.reshape(z_v_image.shape)

    return he_std_norm_image, he_std_image, z_v_norm_image, z_v_image, ratio_norm, ratio_norm_image






































# #### Load package ####
# import numpy as np
# from utils import load_image

# def patchify(x, patch_size):
#     shape_ori = np.array(x.shape[:2])
#     shape_ext = (shape_ori + patch_size - 1) // patch_size * patch_size
#     x_padded = np.pad(
#         x,
#         ((0, shape_ext[0] - x.shape[0]),
#          (0, shape_ext[1] - x.shape[1]),
#          (0, 0)),
#         mode='edge')
#     tiles_shape = shape_ext // patch_size
#     patch_strides = (
#         x_padded.strides[0] * patch_size,
#         x_padded.strides[1] * patch_size,
#         x_padded.strides[0],
#         x_padded.strides[1],
#         x_padded.strides[2])
#     patches = np.lib.stride_tricks.as_strided(
#         x_padded,
#         shape=(tiles_shape[0], tiles_shape[1], patch_size, patch_size, x.shape[2]),
#         strides=patch_strides)
#     patches = patches.reshape(-1, patch_size, patch_size, x.shape[2])
#     shapes = dict(original=shape_ori, padded=shape_ext, tiles=tiles_shape)
#     return patches, shapes


# def compute_metrics(he, patch_size=16):
#     he_tiles, shapes = patchify(he, patch_size=patch_size)

#     he_std_image = np.std(he_tiles, axis=(1, 2, 3))
#     he_std_image = he_std_image.reshape(shapes['tiles'])

#     mean_rgb_per_patch = np.mean(he_tiles, axis=(1, 2))
#     V_r = np.var(mean_rgb_per_patch[:, 0])
#     V_g = np.var(mean_rgb_per_patch[:, 1])
#     V_b = np.var(mean_rgb_per_patch[:, 2])
#     numerators = (
#         mean_rgb_per_patch[:, 0] * V_r +
#         mean_rgb_per_patch[:, 1] * V_g +
#         mean_rgb_per_patch[:, 2] * V_b)
#     denominator = V_r + V_g + V_b
#     z_v = numerators / denominator
#     z_v_image = z_v.reshape(shapes['tiles'])

#     # Normalize and compute ratio
#     flattened_std = he_std_image.flatten()
#     flattened_std = (flattened_std - flattened_std.min()) / (flattened_std.max() - flattened_std.min()) + 1
#     he_std_norm_image = flattened_std.reshape(z_v_image.shape)

#     flattened_mean = z_v_image.flatten()
#     flattened_mean = (flattened_mean - flattened_mean.min()) / (flattened_mean.max() - flattened_mean.min()) + 1
#     z_v_norm_image = flattened_mean.reshape(z_v_image.shape)

#     ratio_norm = flattened_std / flattened_mean
#     ratio_norm = (ratio_norm - ratio_norm.min()) / (ratio_norm.max() - ratio_norm.min())
#     ratio_norm_image = ratio_norm.reshape(z_v_image.shape)

#     return he_std_norm_image, he_std_image, z_v_norm_image, z_v_image, ratio_norm, ratio_norm_image







