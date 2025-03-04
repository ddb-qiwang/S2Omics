import itertools
from PIL import Image
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import torch
import random


Image.MAX_IMAGE_PIXELS = None

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

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename, quality=95)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')
    print(filename)


def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)


def load_yaml(filename, verbose=False):
    with open(filename, 'r') as file:
        content = yaml.safe_load(file)
    if verbose:
        print(f'YAML loaded from {filename}')
    return content


def save_yaml(filename, content):
    with open(filename, 'w') as file:
        yaml.dump(content, file)
    print(file)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]