import psutil
import os
import platform
import torch
from datetime import datetime
import threading
import time
import atexit
import argparse
import os 
import numpy as np
from PIL import Image
import PIL
import tifffile
import cv2
import pdb 

Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100
import tracemalloc
from functools import wraps

def measure_peak_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        current_gb = current / (1024 ** 3)
        peak_gb = peak / (1024 ** 3)
        print(f"[{func.__name__}] Current memory: {current_gb:.4f} GB; Peak memory: {peak_gb:.4f} GB")
        tracemalloc.stop()
        return result
    return wrapper


# modified raw load_image
def load_image(filename, verbose=True):
    ext = os.path.splitext(filename)[-1].lower()
    # use tifffile to open .tif / .tiff
    if ext in [".svs"]:
        import pyvips
        # Load the SVS file
        slide = pyvips.Image.new_from_file(filename, access='sequential')
        # Get image properties
        print(f"Width: {slide.width}, Height: {slide.height}, Bands: {slide.bands}")
        # Convert to NumPy array
        img = np.ndarray(
            buffer=slide.write_to_memory(),
            dtype=np.uint8,  # assuming 8-bit image
            shape=(slide.height, slide.width, slide.bands)
        )
    elif ext in ['.tif', '.tiff']:
        img = tifffile.imread(filename)
        img = np.array(img)
    else:
        img = Image.open(filename)
        img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
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


def smart_save_image(img, prefix, base_name="base", size_threshold=60000):
    """
    Save image as JPG if both dimensions are under `size_threshold`, otherwise as TIFF.
    """
    h, w = img.shape[:2]
    print(f"Image size: {h}x{w}")

    if h < size_threshold and w < size_threshold:
        # Save as JPG
        path = f"{prefix}{base_name}.jpg"
        Image.fromarray(img.astype(np.uint8)).save(path, quality=90)
        print(f"✅ Saved as JPG: {path}")
    else:
        # Save as TIFF
        path = f"{prefix}{base_name}.tiff"
        tifffile.imwrite(path, img, bigtiff=True)
        print(f"✅ Saved as TIFF: {path}")


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
        
    
def get_image_filename(prefix):
    print(f"prefix = {prefix}")
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff', ".tif",".svs"]:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename    

# padding image tiles considering different directions
def reflect_pad(img: Image.Image, pad, left=False, top=False, right=False, bottom=False):
    arr = np.array(img)
    padded = np.pad(
        arr,
        (
            (pad if top else 0, pad if bottom else 0),
            (pad if left else 0, pad if right else 0),
            (0, 0)
        ),
        mode='constant',
        constant_values=0
    )
    return Image.fromarray(padded)

def get_center_weights(size=14, sigma=0.5):
    """Create a 2D Gaussian-like center weight matrix of shape [size, size]."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    weights /= weights.max()  # Normalize to [0,1]
    return weights

# Cleanup function to handle termination signals
def cleanup(signum, frame):
    print(f"\n⚠️ Received signal {signum}, cleaning up CUDA...")
    # Explicitly clear CUDA cache
    torch.cuda.empty_cache()
    # Optional: Also clear custom models/tensors if you keep references
    # del model, data
    # torch.cuda.empty_cache()
    print("✅ CUDA cache cleared, exiting.")
    sys.exit(1)

# log function
def log_system_info(tag="START"):
    print(f"\n[INFO-{tag}] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO-{tag}] CPU usage: {psutil.cpu_percent()}%")
    print(f"[INFO-{tag}] Memory usage: {psutil.virtual_memory().percent}%")
    print(f"[INFO-{tag}] Platform: {platform.platform()}")
    print(f"[INFO-{tag}] Process ID: {os.getpid()}")

# monitor cuda memory usage
def monitor_cuda_memory():
    def decorator(func):
        def wrapper(*args, **kwargs):
            # get args.device automatically
            if args:
                args_obj = args[0]
                device = getattr(args_obj, "device", "cuda:0")
            else:
                device = kwargs.get("device", "cuda:0")
            
            print(f"[Debug] Using device: {device}")
            if torch.cuda.is_available():
                torch.cuda.set_device(device)
                _ = torch.tensor(0., device=device)  # activate CUDA allocator
                torch.cuda.reset_peak_memory_stats()
            else:
                print("⚠️ CUDA is not available，all steps will be done on CPU")

            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(device=device) / 1024 ** 3
                print(f"\n[CUDA:{device}] Maximum CUDA Memory Usage: {peak:.4f} GB")
            print(f"[TIME] main() cost: {end - start:.2f} s")

            return result
        return wrapper
    return decorator


class CPUMemoryMonitor:
    def __init__(self, interval=0.1):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.max_mem = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor)
        atexit.register(self.stop)  #

    def _monitor(self):
        while not self._stop_event.is_set():
            mem = self.process.memory_info().rss
            if mem > self.max_mem:
                self.max_mem = mem
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        print(f"Maximum CPU Memory Usage：{self.max_mem / (1024 ** 3):.2f} GB")
