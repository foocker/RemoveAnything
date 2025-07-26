#!/usr/bin/env python
# coding=utf-8

"""
GRPO标签数据集，支持包含人工标注3个分数的数据格式
"""

import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple

from .triplet_bucket_dataset import TripletBucketDataset


class GRPOLabeledDataset(TripletBucketDataset):
    """
    扩展TripletBucketDataset，支持人工标注的3个分数标签
    
    数据格式示例:
    {
        "ref": "path/to/ref.jpg",
        "src": "path/to/src.jpg", 
        "mask": "path/to/mask.jpg",
        "result": "path/to/result.jpg",
        "scores": {
            "removal_score": 0.8,      # 移除质量分数 [0, 1]
            "background_score": 0.7,   # 背景一致性分数 [0, 1]  
            "smoothness_score": 0.9    # 边界平滑度分数 [0, 1]
        }
    }
    
    或者tensor格式:
    {
        ...,
        "scores": [0.8, 0.7, 0.9]  # [removal, background, smoothness]
    }
    """
    
    def __init__(self, 
                 json_path: str,
                 buckets: str = "512:1,768:0.5,1024:0.25",
                 bucket_side_min: int = 256,
                 bucket_side_max: int = 1536,
                 use_human_scores: bool = True,
                 score_format: str = "dict",  # "dict" or "tensor"
                 **kwargs):
        """
        Args:
            json_path: JSON数据文件路径
            buckets: 桶配置字符串
            bucket_side_min: 最小桶尺寸
            bucket_side_max: 最大桶尺寸
            use_human_scores: 是否使用人工标注分数
            score_format: 分数格式，"dict"或"tensor"
        """
        super().__init__(json_path, buckets, bucket_side_min, bucket_side_max, **kwargs)
        
        self.use_human_scores = use_human_scores
        self.score_format = score_format
        
        # 验证数据格式
        if self.use_human_scores:
            self._validate_score_format()
    
    def _validate_score_format(self):
        """验证数据中是否包含正确格式的分数标签"""
        if not self.data_list:
            return
            
        sample = self.data_list[0]
        if 'scores' not in sample:
            raise ValueError("use_human_scores=True but no 'scores' field found in data")
        
        scores = sample['scores']
        if self.score_format == "dict":
            if not isinstance(scores, dict):
                raise ValueError(f"Expected dict format for scores, got {type(scores)}")
            required_keys = ['removal_score', 'background_score', 'smoothness_score']
            missing_keys = [k for k in required_keys if k not in scores]
            if missing_keys:
                raise ValueError(f"Missing score keys: {missing_keys}")
        elif self.score_format == "tensor":
            if not isinstance(scores, (list, tuple)) or len(scores) != 3:
                raise ValueError(f"Expected list/tuple of length 3 for scores, got {type(scores)} with length {len(scores) if hasattr(scores, '__len__') else 'unknown'}")
        else:
            raise ValueError(f"Unsupported score_format: {self.score_format}")
    
    def __getitem__(self, idx):
        """获取数据项，包含人工标注分数"""
        # 获取基础数据
        item = super().__getitem__(idx)
        
        # 添加人工标注分数
        if self.use_human_scores:
            data_item = self.data_list[idx]
            scores = data_item['scores']
            
            if self.score_format == "dict":
                # dict格式：直接使用
                item['scores'] = scores
            elif self.score_format == "tensor":
                # tensor格式：转换为tensor
                if isinstance(scores, (list, tuple)):
                    item['scores'] = torch.tensor(scores, dtype=torch.float32)
                else:
                    raise ValueError(f"Invalid scores format: {type(scores)}")
        
        return item


def grpo_labeled_collate_fn(batch):
    """
    GRPO标签数据的collate函数，处理包含分数标签的批次数据
    """
    from .triplet_bucket_dataset import triplet_collate_fn
    
    # 使用原有的collate函数处理基础数据
    collated = triplet_collate_fn(batch)
    
    # 处理分数标签
    if 'scores' in batch[0]:
        scores_list = [item['scores'] for item in batch]
        
        if isinstance(scores_list[0], dict):
            # dict格式：保持dict结构
            collated['scores'] = {
                'removal_score': [s['removal_score'] for s in scores_list],
                'background_score': [s['background_score'] for s in scores_list], 
                'smoothness_score': [s['smoothness_score'] for s in scores_list]
            }
        elif isinstance(scores_list[0], torch.Tensor):
            # tensor格式：堆叠为批次tensor
            collated['scores'] = torch.stack(scores_list, dim=0)
        else:
            raise ValueError(f"Unsupported scores type: {type(scores_list[0])}")
    
    return collated


def create_grpo_dataset(json_path: str, 
                       use_human_scores: bool = True,
                       score_format: str = "dict",
                       **kwargs) -> GRPOLabeledDataset:
    """
    创建GRPO标签数据集的便捷函数
    
    Args:
        json_path: JSON数据文件路径
        use_human_scores: 是否使用人工标注分数
        score_format: 分数格式，"dict"或"tensor"
        **kwargs: 其他参数传递给数据集构造函数
        
    Returns:
        GRPOLabeledDataset实例
    """
    return GRPOLabeledDataset(
        json_path=json_path,
        use_human_scores=use_human_scores,
        score_format=score_format,
        **kwargs
    )


# 示例数据格式
EXAMPLE_DATA_FORMAT = {
    "dict_format": {
        "ref": "data/images/ref_001.jpg",
        "src": "data/images/src_001.jpg", 
        "mask": "data/images/mask_001.jpg",
        "result": "data/images/result_001.jpg",
        "scores": {
            "removal_score": 0.85,      # 移除质量：0-1，越高越好
            "background_score": 0.72,   # 背景一致性：0-1，越高越好
            "smoothness_score": 0.91    # 边界平滑度：0-1，越高越好
        }
    },
    "tensor_format": {
        "ref": "data/images/ref_002.jpg",
        "src": "data/images/src_002.jpg",
        "mask": "data/images/mask_002.jpg", 
        "result": "data/images/result_002.jpg",
        "scores": [0.78, 0.83, 0.69]  # [removal, background, smoothness]
    }
}


if __name__ == "__main__":
    # 测试代码
    print("GRPO Labeled Dataset Example")
    print("=" * 40)
    
    # 打印示例数据格式
    print("Example data formats:")
    print(json.dumps(EXAMPLE_DATA_FORMAT, indent=2))
