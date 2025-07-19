#!/usr/bin/env python
# coding=utf-8
import json
import os
import json

import numpy as np
from PIL import Image
import torch

import cv2

from .base import BaseDataset
from .all_data import load_triplet_paths

from .bucket_utils import (
    find_nearest_bucket,
)

class TripletBucketDataset(BaseDataset):
    """
    为三元组数据(源图像、蒙版、编辑后图像)设计的桶数据集
    """
    def __init__(
        self,
        json_path,
        buckets,
        metadata_file=None,
        custom=False,
    ):
        super().__init__()
        self.json_path = json_path
        self.root_path = os.path.dirname(json_path)
        self.buckets = buckets
        self.custom = custom
        
        self.triplet_paths = []
        self.bucket_indices = []
        
        # 尺度加载得根据传入的尺度序列不同，也需要重新计算json TODO 
        # 保存传入的桶，以及原始数据对应桶的索引
        if metadata_file and os.path.exists(metadata_file):
            print(f"从{metadata_file}加载数据集元数据")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.bucket_indices = metadata['bucket_indices']
        else:
            print(f"从{json_path}加载三元组数据并分配桶...")
            
            self.triplet_paths = load_triplet_paths(json_path)
            
            for paths in self.triplet_paths:
                input_image_path = paths["input_image"]
                
                with Image.open(input_image_path) as img:
                    w, h = img.size
                
                bucket_idx = find_nearest_bucket(h, w, self.buckets)
                self.bucket_indices.append(bucket_idx)
            
            if metadata_file:
                print(f"保存元数据到{metadata_file}")
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump({
                        'bucket_indices': self.bucket_indices
                    }, f)
    
    def __len__(self):
        return len(self.triplet_paths)
    
    def get_sample(self, idx):
        paths = self.triplet_paths[idx]
        target_size = self.buckets[self.bucket_indices[idx]]
        
        input_image = cv2.imread(paths["input_image"])
        if input_image is None:
            raise ValueError(f"无法读取图像: {paths['input_image']}")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
        mask_img = cv2.imread(paths["mask"])
        if mask_img is None:
            raise ValueError(f"无法读取mask: {paths['mask']}")
        removed_mask = (mask_img > 128).astype(np.uint8)[:,:,0]
        
        # 推理数据集可能没有edited_image
        if paths.get("edited_image") and os.path.exists(paths["edited_image"]):
            removed_image = cv2.imread(paths["edited_image"])
            if removed_image is None:
                raise ValueError(f"无法读取擦除后图像: {paths['edited_image']}")
            removed_image = cv2.cvtColor(removed_image, cv2.COLOR_BGR2RGB)
        else:
            removed_image = input_image.copy()
        
        # TODO add simple mode
        if self.custom:
            return self.process_custom(input_image, removed_mask, removed_image, target_size)
        return self.process_triplets(input_image, removed_mask, removed_image, target_size)
    
    def __getitem__(self, idx):
        return self.get_sample(idx)


def triplet_collate_fn(examples):
    ref_values = [example["ref"] for example in examples]
    src_values = [example["src"] for example in examples]
    mask_values = [example["mask"] for example in examples]
    result_values = [example["result"] for example in examples]
    
    return {
        "ref": torch.stack(ref_values),
        "src": torch.stack(src_values),
        "mask": torch.stack(mask_values),
        "result": torch.stack(result_values),
    }
    
def triplet_collate_fn_simple(examples):
    input_image = [example["input_image"] for example in examples]
    mask = [example["mask"] for example in examples]
    edited_image = [example["edited_image"] for example in examples]
    # remove the masked region and keep the background harmonized,may generate the zhedang object
    captions = [example["captions"] for example in examples] 
    
    return {
        "input_image": torch.stack(input_image),
        "mask": torch.stack(mask),
        "edited_image": torch.stack(edited_image),
        "captions": captions
    }   
