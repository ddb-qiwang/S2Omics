import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from timm.layers import SwiGLUPacked
import numpy as np
from s1_utils import save_pickle, load_image
from PIL import Image
import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List

class PatchDataset(Dataset):
    def __init__(self, image, patch_size=16, stride=16, model='uni'):
        self.image = image
        self.patch_size = patch_size
        self.stride = stride
        if model == 'gigapath':
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        
        self.shape_ori = np.array(image.shape[:2])
        self.num_patches = ((self.shape_ori - patch_size) // stride + 1)
        self.total_patches = self.num_patches[0] * self.num_patches[1]

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        i = (idx // self.num_patches[1]) * self.stride
        j = (idx % self.num_patches[1]) * self.stride
        
        # Extract 224x224 patch centered on the 16x16 patch
        center_i, center_j = i + self.patch_size//2, j + self.patch_size//2
        start_i, start_j = max(0, center_i - 112), max(0, center_j - 112)
        end_i, end_j = min(self.shape_ori[0], center_i + 112), min(self.shape_ori[1], center_j + 112)
        
        patch = self.image[start_i:end_i, start_j:end_j]
        
        # Pad if necessary to ensure 224x224 size
        if patch.shape[0] < 224 or patch.shape[1] < 224:
            padded_patch = np.zeros((224, 224, 3), dtype=patch.dtype)
            padded_patch[(224-patch.shape[0])//2:(224-patch.shape[0])//2+patch.shape[0], 
                         (224-patch.shape[1])//2:(224-patch.shape[1])//2+patch.shape[1]] = patch
            patch = padded_patch
        
        patch = Image.fromarray(patch.astype('uint8')).convert('RGB')
        return self.transform(patch), (i, j)

def create_model_uni(local_dir):
    model = timm.create_model(
        "vit_large_patch16_224", 
        img_size=224, 
        patch_size=16, 
        init_values=1e-5, 
        num_classes=0,  # This ensures no classification head
        global_pool='',  # This removes global pooling
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
    return model

def create_model_virchow(local_dir):
    model = timm.create_model(
        "vit_huge_patch14_224", 
        img_size=224,
        init_values=1e-5,
        num_classes=0,
        reg_tokens=4,
        mlp_ratio=5.3375,
        global_pool='',
        dynamic_img_size=True,
        mlp_layer=SwiGLUPacked, 
        act_layer=torch.nn.SiLU
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
    return model

def create_model_gigapath(local_dir):
    model = timm.create_model(
        "vit_giant_patch14_dinov2",
        img_size=224,
        in_chans=3,
        patch_size=16,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        init_values=1e-05,
        mlp_ratio=5.33334,
        num_classes=0,
        global_pool="token"
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=False)
    return model

@torch.inference_mode()
def extract_features(model, batch):
    # Get 224-level embedding
    feature_emb = model(batch)
    
    # Get 16-level embedding
    _, intermediates = model.forward_intermediates(batch, return_prefix_tokens=False)
    patch_emb = intermediates[-1]  # Use the last intermediate output
    
    return feature_emb, patch_emb

@torch.inference_mode()
def histology_feature_extraction(prefix, save_folder,
                                 foundation_model='uni',
                                 ckpt_path='../checkpoints/uni/',
                                 device='cuda:0',
                                 batch_size=32,
                                 down_samp_step=10,
                                 num_workers=4):
    '''
    extracting hierarchical features of superpixels using a modified version of current pathology foundation models
    Parameters:
        prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
        save_folder: the name of save folder, user can input the complete path or just the folder name, 
            if so, the folder will be placed under the prefix folder
        foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath
        ckpt_path: the path to foundation model parameter files (should be named as 'pytorch_model.bin'), './checkpoints/uni/' for an example
        device: default = 'cuda'
        batch_size: default = 128
        down_samp_step: the down-sampling step, default = 10 refers to only extract features for superpixels whose row_index and col_index can both be divided by 10 (roughly 1:100 down-sampling rate). down_samp_step = 1 means extract features for every superpixel
        num_workers: default = 4
    '''
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_folder = save_folder+'/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    
    local_dir = ckpt_path
    if foundation_model == 'uni':
        model = create_model_uni(local_dir)
    elif foundation_model == 'virchow':
        model = create_model_virchow(local_dir)
    elif foundation_model == 'gigapath':
        model = create_model_gigapath(local_dir)
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    print(f'''Histology foundation model loaded! 
    Foundation model name: {foundation_model}
    Start extracting histology feature embeddings...''')
    
    he = load_image(f'{prefix}he.jpg')
    if foundation_model == 'uni' or foundation_model == 'gigapath':
        stride_init = 16
        patch_size = 16
    if foundation_model == 'virchow':
        stride_init = 14
        patch_size = 14
        
    dataset = PatchDataset(he, patch_size=patch_size, stride=stride_init*down_samp_step, model=foundation_model)
    save_pickle(dataset.num_patches, pickle_folder+'num_patches.pickle')
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    patch_embeddings = []
    part_cnts = 0
    for batch_idx, (patches, positions) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):

        patches = patches.to(device, non_blocking=True)
        
        if batch_idx == 0:
            print(f"Batch {batch_idx}:")
            print(f"Shape of patches: {patches.shape}")
            print(f"Shape of positions[0]: {positions[0].shape}")
            print(f"Content of positions[0][:10]: {positions[0][:10]}")
            print(f"Content of positions[1][:10]: {positions[1][:10]}")
        
        feature_emb, patch_emb = extract_features(model, patches)
        
        if batch_idx == 0:
            print(f"Shape of feature_emb: {feature_emb.shape}")
            print(f"Shape of patch_emb: {patch_emb.shape}")
        
        # Process each patch
        for idx in range(len(positions[0])):
            
            # Extract features
            if foundation_model != 'gigapath':
                center_feature = feature_emb[idx, 0]
            else:
                center_feature = feature_emb[idx, :]
            patch_feature = patch_emb[idx, :, patch_size//2-1, patch_size//2-1]
            
            # layernorm the local features and global features
            feat_shape = center_feature.shape[-1]
            layernorm = nn.LayerNorm(feat_shape, eps=1e-6).to(device)
            center_feature = layernorm(center_feature)
            patch_feature = layernorm(patch_feature)
            # Concatenate 224-level and 16-level features
            combined_feature = torch.cat([center_feature, patch_feature])
            patch_embeddings.append(combined_feature.cpu().numpy())
            
        if (batch_idx*batch_size)//100000 < ((batch_idx+1)*batch_size)//100000 or batch_idx == len(dataloader) - 1:
            print(f"Part {part_cnts} patch number: {len(patch_embeddings)}")
            save_pickle(patch_embeddings, pickle_folder+foundation_model+
                        f'_embeddings_downsamp_{down_samp_step}_part_{part_cnts}.pickle')
            patch_embeddings = []
            part_cnts += 1