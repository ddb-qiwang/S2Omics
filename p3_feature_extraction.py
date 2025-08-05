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


'''extracting hierarchical features of superpixels using a modified version of UNI
Args:
    prefix: folder path of H&E stained image, '/home/H&E_image/' for an example
    save_folder: the name of save folder, user can input the complete path or just the folder name, 
        if so, the folder will be placed under the prefix folder
    foundation_model: the name of foundation model used for feature extraction, user can select from uni, virchow and gigapath
    ckpt_path: the path to foundation model parameter files (should be named as 'pytorch_model.bin'), './checkpoints/uni/' for an example
    device: default = 'cuda'
    batch_size: default = 128
    down_samp_step: the down-sampling step, default = 10 refers to only extract features for superpixels whose row_index and col_index can both be divided by 10 (roughly 1:100 down-sampling rate). down_samp_step = 1 means extract features for every superpixel
    num_workers: default = 4
Return:
    {args.prefix}_{args.foundation_model}_embeddings_downsamp_{args.down_samp_step}_part_{part_cnts}.pickle: part_cnts: 0, 1, 2,... Save the files into several subfiles ensures every pickle file contains no more than features for 100,000 superpixels and can avoid memory overflow
'''

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

def get_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('prefix', type=str)
    parser.add_argument('--save_folder', type=str, default='S2Omics_output')
    parser.add_argument('--foundation_model', type=str, default='uni', help='uni, virchow, gigapath')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/uni/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--down_samp_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()

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
def main():
    args = get_args()
    
    if not os.path.exists(args.prefix+args.save_folder):
        os.makedirs(args.prefix+args.save_folder)
    save_folder = args.prefix+args.save_folder+'/'
    if not os.path.exists(save_folder+'pickle_files'):
        os.makedirs(save_folder+'pickle_files')
    pickle_folder = save_folder+'pickle_files/'
    
    local_dir = args.ckpt_path
    if args.foundation_model == 'uni':
        model = create_model_uni(local_dir)
    elif args.foundation_model == 'virchow':
        model = create_model_virchow(local_dir)
    elif args.foundation_model == 'gigapath':
        model = create_model_gigapath(local_dir)
    
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    print(f'''Histology foundation model loaded! 
    Foundation model name: {args.foundation_model}
    Start extracting histology feature embeddings...''')
    
    he = load_image(f'{args.prefix}he.jpg')
    if args.foundation_model == 'uni' or args.foundation_model == 'gigapath':
        stride_init = 16
        patch_size = 16
    if args.foundation_model == 'virchow':
        stride_init = 14
        patch_size = 14
    dataset = PatchDataset(he, patch_size=patch_size, stride=stride_init*args.down_samp_step, model=args.foundation_model)
    save_pickle(dataset.num_patches, pickle_folder+'num_patches.pickle')
    dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

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
            if args.foundation_model != 'gigapath':
                center_feature = feature_emb[idx, 0]
            else:
                center_feature = feature_emb[idx, :]
            patch_feature = patch_emb[idx, :, patch_size//2-1, patch_size//2-1]
            
            # layernorm the local features and global features
            feat_shape = center_feature.shape[-1]
            layernorm = nn.LayerNorm(feat_shape, eps=1e-6).to(args.device)
            center_feature = layernorm(center_feature)
            patch_feature = layernorm(patch_feature)
            # Concatenate 224-level and 16-level features
            combined_feature = torch.cat([center_feature, patch_feature])
            patch_embeddings.append(combined_feature.cpu().numpy())
            
        if (batch_idx*args.batch_size)//100000 < ((batch_idx+1)*args.batch_size)//100000 or batch_idx == len(dataloader) - 1:
            print(f"Part {part_cnts} patch number: {len(patch_embeddings)}")
            save_pickle(patch_embeddings, pickle_folder+args.foundation_model+
                        f'_embeddings_downsamp_{args.down_samp_step}_part_{part_cnts}.pickle')
            patch_embeddings = []
            part_cnts += 1

if __name__ == '__main__':
    main()