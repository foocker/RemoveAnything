#!/usr/bin/env python
# coding=utf-8

"""
Flux-based Object Removal with LoRA MoE using GRPO (Generalized Reward-based Policy Optimization)
Based on train_bucket_moe_simple.py with modular GRPO integration
"""

import os
import argparse
import logging
import math
import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from tqdm.auto import tqdm
from peft import LoraConfig
from PIL import Image
import cv2
import wandb
from typing import Dict, List, Tuple
from collections import defaultdict
import traceback

from optimizer.muon import MuonWithAuxAdam

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory
)

from data.all_data import load_triplet_paths, load_triplet_paths_from_dir
from models.pipeline_tools import encode_images, prepare_text_input, Flux_fill_encode_masks_images
from models.image_project import image_output
from models.transformer import tranformer_forward
from data.triplet_bucket_dataset import TripletBucketDataset, triplet_collate_fn
from data.grpo_labeled_dataset import GRPOLabeledDataset, grpo_labeled_collate_fn, create_grpo_dataset
from data.bucket_utils import BucketBatchSampler
from data.bucket_utils import (
    parse_buckets_string,
    find_nearest_bucket
)

from data.data_utils import (
    get_bbox_from_mask,
    expand_bbox,
    expand_image_mask,
    pad_to_square,
    box2squre,
    crop_back
)

logger = get_logger(__name__, log_level="INFO")


# ============================================================================
# MoE Components (optimized from train_bucket_moe_simple.py)
# ============================================================================

def log_infer_moe(accelerator, args, save_path, epoch, global_step, 
              pipefill: "FluxFillPipeline", pipeprior: "FluxPriorReduxPipeline"):
    # 设置所有模型组件都使用BFloat16
    model_dtype = torch.bfloat16
    logger.info(f"Setting all model components to use {model_dtype}")
    
    # 重编VAE的_decode方法以确保类型一致性
    if hasattr(pipefill, "vae") and hasattr(pipefill.vae, "_decode"):
        # 保存原始方法
        original_decode = pipefill.vae._decode
        
        # 创建一个引用模型的辅助类，而不是使用闭包函数
        class DecoderPatch:
            def __init__(self, pipe, orig_decode, dtype):
                self.pipe = pipe
                self.original_decode = orig_decode
                self.dtype = dtype
                
            def __call__(self, z):
                # 确保所有输入和模型参数都使用相同类型
                z = z.to(dtype=self.dtype)
                # 临时将所有decoder组件转换为一致类型
                with torch.no_grad():
                    for name, module in self.pipe.vae.decoder.named_modules():
                        if hasattr(module, "weight") and module.weight is not None:
                            if module.weight.dtype != self.dtype:
                                module.weight.data = module.weight.data.to(self.dtype)
                        if hasattr(module, "bias") and module.bias is not None:
                            if module.bias.dtype != self.dtype:
                                module.bias.data = module.bias.data.to(self.dtype)
                    
                    # 调用原始方法
                    try:
                        return self.original_decode(z)
                    except Exception as e:
                        logger.warning(f"Error in patched decode: {e}")
                        # 如果失败，尝试使用float32
                        logger.info("Falling back to float32 for this operation")
                        z_float = z.to(dtype=torch.float32)
                        # 将decoder暂时转换为float32
                        self.pipe.vae.decoder = self.pipe.vae.decoder.to(dtype=torch.float32)
                        result = self.original_decode(z_float)
                        # 还原为原始类型
                        self.pipe.vae.decoder = self.pipe.vae.decoder.to(dtype=self.dtype)
                        return result
        
        # 替换方法
        pipefill.vae._decode = DecoderPatch(pipefill, original_decode, model_dtype)
        logger.info("Successfully patched VAE decoder method")
    
    # 将所有模型转换为BFloat16
    pipefill.to(dtype=model_dtype)
    pipeprior.to(dtype=model_dtype)

    set_seed(args.seed)
    logger.info(f"Running MoE inference... \nEpoch: {epoch}, Step: {global_step}")
    save_dir = os.path.join(save_path, f"infer_moe_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isdir(args.val_json_path):
        triplet_paths = load_triplet_paths_from_dir(args.val_json_path)
    else:
        triplet_paths = load_triplet_paths(args.val_json_path)
    buckets = parse_buckets_string(args.aspect_ratio_buckets)

    for paths in triplet_paths:
        source_image_path = paths["input_image"]  # 被消除
        mask_image_path = paths["mask"]  # 待消除区域
        file_name = os.path.basename(source_image_path)
        removed_image_path = paths["edited_image"] if os.path.exists(paths["edited_image"]) else source_image_path # 消除后的结果 

        ref_image_path = removed_image_path
        ref_mask_path = mask_image_path

        ref_image = cv2.imread(ref_image_path)
        h, w = ref_image.shape[:2]
        target_size = buckets[find_nearest_bucket(h, w, buckets)]
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tar_image = cv2.imread(source_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]
        tar_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]
        tar_mask_original = tar_mask.copy()

        if tar_mask.shape != tar_image.shape:
            tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:] 
        ref_mask = ref_mask[y1:y2,x1:x2] 
        ratio = 1.3
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 

        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 

        kernel = np.ones((7, 7), np.uint8)
        iterations = 2
        tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
        # tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2.0)
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx) # crop box
        y1,y2,x1,x2 = tar_box_yyxx_crop

        old_tar_image = tar_image.copy()

        tar_image = tar_image[y1:y2,x1:x2,:]
        tar_mask = tar_mask[y1:y2,x1:x2]

        H1, W1 = tar_image.shape[0], tar_image.shape[1]
        # zome in
        tar_mask = pad_to_square(tar_mask, pad_value=0)
        tar_mask = cv2.resize(tar_mask, target_size)

        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), target_size).astype(np.uint8)
        
        # 确保 pipeprior 也使用正确的数据类型
        with torch.no_grad():
            pipe_prior_output = pipeprior(Image.fromarray(masked_ref_image))
        
        # 转换pipe_prior_output中的所有张量到与模型相同的数据类型
        for key, value in pipe_prior_output.items():
            if isinstance(value, torch.Tensor):
                pipe_prior_output[key] = value.to(dtype=model_dtype)

        tar_image = pad_to_square(tar_image, pad_value=255)
        H2, W2 = tar_image.shape[0], tar_image.shape[1]

        tar_image = cv2.resize(tar_image, target_size)
        diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)

        tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
        mask_black = np.ones_like(tar_image) * 0  # TODO may not reasonable, tar mask may be better? if in-eontext is better
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych = Image.fromarray(mask_diptych)

        generator = torch.Generator(accelerator.device,).manual_seed(args.seed)
        edited_image = pipefill(
            image=diptych_ref_tar,
            mask_image=mask_diptych,
            height=mask_diptych.size[1],
            width=mask_diptych.size[0],
            max_sequence_length=512,
            num_inference_steps=30,
            generator=generator,
            **pipe_prior_output,  # Use the output from the prior redux model
        ).images[0]

        t_width, t_height = edited_image.size
        start_x = t_width // 2
        edited_image = edited_image.crop((start_x, 0, t_width, t_height))

        edited_image = np.array(edited_image)
        edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
        edited_image = Image.fromarray(edited_image)
        
        # Create a composite image: original input (left) + mask (middle) + output (right)
        original_image = Image.fromarray(old_tar_image)
        # Convert mask to a visible format (ensure it's RGB)
        visible_mask = Image.fromarray(
            np.where(tar_mask_original > 0, np.ones_like(old_tar_image) * 255, old_tar_image).astype(np.uint8)
            if len(tar_mask_original.shape) == 3 else
            np.stack([tar_mask_original * 255, tar_mask_original * 255, tar_mask_original * 255], axis=-1).astype(np.uint8)
        )
        
        # Create composite image by concatenating horizontally
        total_width = original_image.width + visible_mask.width + edited_image.width
        composite_image = Image.new('RGB', (total_width, original_image.height))
        
        composite_image.paste(original_image, (0, 0))
        composite_image.paste(visible_mask, (original_image.width, 0))
        composite_image.paste(edited_image, (original_image.width + visible_mask.width, 0))
        
        edited_image_save_path = os.path.join(save_dir, f"moe_seed{args.seed}_epoch_{epoch}_step_{global_step}_{file_name}")
        composite_image.save(edited_image_save_path)
        
    del pipefill, pipeprior
    torch.cuda.empty_cache()
    gc.collect()


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def compute_simple_mse_loss(pred, target, weighting=None):
    """
    简化的MSE损失函数，用于固定权重MoE训练。
    
    Args:
        pred: 预测的latent，形状 [batch_size, seq_len, hidden_dim]
        target: 目标latent，形状 [batch_size, seq_len, hidden_dim] 
        weighting: 基础权重，用于时间步加权
        
    Returns:
        dict: 包含损失的字典
    """
    # 计算基础MSE损失
    mse_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    mse_loss = mse_loss.mean(dim=[1, 2])  # [batch_size]
    
    # 应用权重（如果提供）
    if weighting is not None:
        mse_loss = mse_loss * weighting
    
    # 返回损失字典
    return {
        "base_loss": mse_loss.mean(),
        "total_loss": mse_loss.mean()
    }


class FeatureExtractor(nn.Module):
    """
    简化的特征提取器，用于固定权重MoE训练。
    """
    def __init__(self):
        super().__init__()
        self.projection = None  # 将在第一次前向传播时初始化
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim] 或 [batch_size, hidden_dim]
        Returns:
            features: 提取的特征，形状为 [batch_size, output_dim]
        """
        if len(x.shape) == 3:
            # 如果是3D张量，进行平均池化
            x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 延迟初始化投影层
        if self.projection is None:
            input_dim = x.shape[-1]
            output_dim = min(256, input_dim // 4)  # 简化的输出维度
            self.projection = nn.Linear(input_dim, output_dim).to(x.device, dtype=x.dtype)
        
        return self.projection(x)


class LearnableRouter(nn.Module):
    """
    可学习的路由器，支持固定权重和动态权重两种模式。
    """
    def __init__(self, num_experts=3, initial_weights=None, dtype=None):
        super().__init__()
        self.num_experts = num_experts
        
        if initial_weights is None:
            initial_weights = [1.0 / num_experts] * num_experts
        
        if dtype is None:
            dtype = torch.float32
        
        # 转换为logits进行学习
        logits = torch.log(torch.tensor(initial_weights, dtype=dtype))
        self.weight_logits = nn.Parameter(logits)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            weights: 三个专家的可学习权重，形状为 [batch_size, 3]
        """
        batch_size = x.shape[0]
        weights = F.softmax(self.weight_logits, dim=-1)
        weights = weights.unsqueeze(0).expand(batch_size, -1)
        return weights
    
    def get_current_weights(self):
        """获取当前的权重值（用于监控和调试）"""
        with torch.no_grad():
            weights = F.softmax(self.weight_logits, dim=-1)
            weights_cpu = weights.cpu()
            if weights_cpu.dtype == torch.bfloat16:
                weights_cpu = weights_cpu.float()
            return weights_cpu.numpy().tolist()


class MoELoRAManager:
    """
    简化的MoE LoRA管理器，支持固定权重和可学习权重两种模式。
    """
    def __init__(self, model, lora_configs, adapter_names, initial_weights=None):
        """
        Args:
            model: 基础模型，LoRA将应用于此模型
            lora_configs: 列表，每个元素是一个LoRA配置字典
            adapter_names: 列表，LoRA适配器名称
            initial_weights: 初始权重列表，默认为均匀分配
        """
        self.model = model
        self.adapter_names = adapter_names
        self.num_experts = len(adapter_names)
        
        # 添加LoRA适配器
        for i, (config, name) in enumerate(zip(lora_configs, adapter_names)):
            self.model.add_adapter(config, adapter_name=name)
        
        # 初始化路由器
        model_dtype = next(self.model.parameters()).dtype
        self.router = LearnableRouter(
            num_experts=self.num_experts, 
            initial_weights=initial_weights, 
            dtype=model_dtype
        )
        
        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 路由统计
        self._routing_stats = defaultdict(int)
        self.current_weights = None
    
    def compute_adapter_weights(self, features):
        """
        计算当前输入应该使用的适配器权重（使用可学习路由器）
        
        Args:
            features: 输入特征，用于计算路由权重
            
        Returns:
            torch.Tensor: 可学习适配器权重，形状 [batch_size, num_experts]
        """
        return self.router(features)
    
    def set_adapter_weights(self, weights):
        """
        设置适配器权重
        
        PEFT库期望weights参数是一个一维列表，例如[0.3, 0.3, 0.4]。
        输入的weights可能是二维张量[batch_size, num_experts]，
        所以需要进行处理以匹配库的期望。
        """
        if torch.is_tensor(weights):
            if len(weights.shape) == 2:
                # 如果是二维张量，取第一个样本的权重
                weights = weights.mean(dim=0).detach()
            weights = weights.cpu().numpy().tolist()
        
        # 设置所有适配器的权重
        self.model.set_adapter(self.adapter_names, weights)
        self.current_weights = weights
    
    def get_routing_stats(self):
        """获取路由统计信息（可学习版本）"""
        stats = dict(self._routing_stats)
        stats["current_weights"] = self.router.get_current_weights()
        return stats
    
    def forward_with_routing(self, features, **forward_kwargs):
        """
        使用路由权重进行前向传播
        
        Args:
            features: 输入特征，用于计算路由权重
            **forward_kwargs: 传递给模型的参数
        Returns:
            result: 模型输出
            adapter_weights: 使用的适配器权重
        """
        # 提取特征
        extracted_features = self.feature_extractor(features)
        
        # 计算适配器权重
        adapter_weights = self.compute_adapter_weights(extracted_features)
        
        # 设置权重
        self.set_adapter_weights(adapter_weights)
        
        # 前向传播
        result = self.model(**forward_kwargs)
        
        # 更新统计
        self.current_weights = adapter_weights.mean(dim=0).detach()
        
        # 统计专家使用情况
        expert_usage = torch.argmax(adapter_weights, dim=1)
        for expert_idx in range(self.num_experts):
            count = (expert_usage == expert_idx).sum().item()
            self._routing_stats[f"expert_{expert_idx}_usage"] += count
        
        return result, adapter_weights


# ============================================================================
# GRPO-specific Components (modular implementation)
# ============================================================================

class GRPORewardComputer:
    """
    GRPO奖励计算器，支持多种奖励类型。
    """
    def __init__(self, reward_weights=None):
        """
        Args:
            reward_weights: 各个专家的奖励权重 [removal, background, completion]
        """
        self.reward_weights = reward_weights or [0.4, 0.3, 0.3]
    
    def compute_removal_strength_reward(self, generated_images, masks, targets=None):
        """计算消除强度奖励"""
        # 简化实现：基于遮罩区域的差异
        batch_size = generated_images.shape[0]
        rewards = []
        
        for i in range(batch_size):
            # 这里可以实现更复杂的奖励计算逻辑
            # 目前使用简化版本
            reward = np.random.uniform(0.5, 1.0)  # 占位实现
            rewards.append(reward)
        
        return np.array(rewards)
    
    def compute_background_blend_reward(self, generated_images, masks, targets=None):
        """计算背景融合奖励"""
        batch_size = generated_images.shape[0]
        rewards = []
        
        for i in range(batch_size):
            # 简化实现
            reward = np.random.uniform(0.3, 0.9)
            rewards.append(reward)
        
        return np.array(rewards)
    
    def compute_occlusion_repair_reward(self, generated_images, masks, targets=None):
        """计算遭挡修复奖励"""
        batch_size = generated_images.shape[0]
        rewards = []
        
        for i in range(batch_size):
            # 简化实现
            reward = np.random.uniform(0.2, 0.8)
            rewards.append(reward)
        
        return np.array(rewards)
    
    def compute_rewards(self, generated_images, masks, targets=None, use_human_scores=False, human_scores=None):
        """
        计算所有奖励。
        
        Args:
            generated_images: 生成的图像
            masks: 遮罩
            targets: 目标图像（可选）
            use_human_scores: 是否使用人工标注分数
            human_scores: 人工标注分数
            
        Returns:
            dict: 包含各种奖励的字典
        """
        if use_human_scores and human_scores is not None:
            # 使用人工标注分数
            rewards = {
                'removal_strength': np.array(human_scores.get('removal_strength', [0.5] * generated_images.shape[0])),
                'background_blend': np.array(human_scores.get('background_blend', [0.5] * generated_images.shape[0])),
                'occlusion_repair': np.array(human_scores.get('occlusion_repair', [0.5] * generated_images.shape[0]))
            }
        else:
            # 自动计算奖励
            rewards = {
                'removal_strength': self.compute_removal_strength_reward(generated_images, masks, targets),
                'background_blend': self.compute_background_blend_reward(generated_images, masks, targets),
                'occlusion_repair': self.compute_occlusion_repair_reward(generated_images, masks, targets)
            }
        
        # 计算整体奖励
        overall_reward = (
            rewards['removal_strength'] * self.reward_weights[0] +
            rewards['background_blend'] * self.reward_weights[1] +
            rewards['occlusion_repair'] * self.reward_weights[2]
        )
        rewards['overall'] = overall_reward
        
        return rewards, {'source': 'human' if use_human_scores else 'auto'}


class GRPOLossComputer:
    """
    GRPO损失计算器，实现PPO风格的策略梯度优化。
    """
    def __init__(self, clip_range=0.2, diversity_weight=0.1):
        """
        Args:
            clip_range: PPO裁剪范围
            diversity_weight: 专家多样性损失权重
        """
        self.clip_range = clip_range
        self.diversity_weight = diversity_weight
    
    def compute_grpo_loss(self, log_probs, old_log_probs, rewards, adapter_weights):
        """
        计算GRPO损失。
        
        Args:
            log_probs: 当前策略的对数概率
            old_log_probs: 旧策略的对数概率
            rewards: 奖励信号
            adapter_weights: 适配器权重
            
        Returns:
            dict: 包含各种损失的字典
        """
        # 计算比率
        ratio = torch.exp(log_probs - old_log_probs)
        
        # 计算裁剪损失
        surr1 = ratio * rewards
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * rewards
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算专家多样性损失
        diversity_loss = self.compute_diversity_loss(adapter_weights)
        
        # 总损失
        total_loss = policy_loss + self.diversity_weight * diversity_loss
        
        return {
            'policy_loss': policy_loss,
            'diversity_loss': diversity_loss,
            'total_loss': total_loss
        }
    
    def compute_diversity_loss(self, adapter_weights):
        """计算专家多样性损失"""
        # 鼓励专家权重的多样性
        mean_weights = adapter_weights.mean(dim=0)
        uniform_weights = torch.ones_like(mean_weights) / mean_weights.size(0)
        diversity_loss = F.kl_div(F.log_softmax(mean_weights, dim=0), uniform_weights, reduction='sum')
        return diversity_loss


class RemovalPromptDataset(Dataset):
    """Dataset for removal prompts with optional human annotations
    
    Supports two modes:
    1. Human-annotated mode: Uses pre-computed human scores
    2. Auto-compute mode: Only provides prompts, scores computed during training
    """
    
    def __init__(self, dataset_path, split='train', use_human_scores=False):
        self.dataset_path = dataset_path
        self.use_human_scores = use_human_scores
        
        if use_human_scores:
            # Load dataset with human annotations
            self.annotation_file = os.path.join(dataset_path, f'{split}_annotations.jsonl')
        else:
            # Load dataset with prompts only
            self.annotation_file = os.path.join(dataset_path, f'{split}_prompts.jsonl')
        
        # Load data
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        print(f"Loaded {len(self.data)} {split} samples")
        print(f"Mode: {'Human-annotated scores' if use_human_scores else 'Auto-compute scores'}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        
        # Load original image and mask
        original_path = os.path.join(self.dataset_path, data_item['original_image'])
        mask_path = os.path.join(self.dataset_path, data_item['mask'])
        
        original_image = Image.open(original_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        result = {
            "original_image": original_image,
            "mask": mask,
            "metadata": data_item.get('metadata', {})
        }
        
        # Add human scores if available
        if self.use_human_scores and 'scores' in data_item:
            result["human_scores"] = data_item['scores']
        
        # Add ground truth if available (for comparison)
        if 'ground_truth' in data_item:
            gt_path = os.path.join(self.dataset_path, data_item['ground_truth'])
            result["ground_truth"] = Image.open(gt_path).convert('RGB')
        
        return result
    
    @staticmethod
    def collate_fn(examples):
        batch = {}
        
        # Stack images and masks
        batch["original_images"] = [ex["original_image"] for ex in examples]
        batch["masks"] = [ex["mask"] for ex in examples]
        batch["metadata"] = [ex["metadata"] for ex in examples]
        
        # Stack human scores if available
        if "human_scores" in examples[0]:
            batch["human_scores"] = [ex["human_scores"] for ex in examples]
        
        # Stack ground truth if available
        if "ground_truth" in examples[0]:
            batch["ground_truth"] = [ex["ground_truth"] for ex in examples]
        
        return batch


def compute_removal_strength(gen_img, orig_img, mask_img):
    """Compute removal strength score"""
    # Convert to numpy arrays if needed
    gen_np = np.array(gen_img)
    orig_np = np.array(orig_img)
    mask_np = np.array(mask_img)
    
    # Ensure mask is binary and single channel
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    mask_np = (mask_np > 127).astype(float)
    
    # Compute difference in masked region
    diff = np.abs(gen_np.astype(float) - orig_np.astype(float))
    masked_diff = diff * mask_np[:, :, np.newaxis]
    
    # Higher difference = better removal
    removal_score = np.mean(masked_diff) / 255.0
    
    # Normalize to [0, 1] range
    return min(1.0, removal_score * 2.0)


def compute_background_blend(gen_img, orig_img, mask_img):
    """Compute background blending score"""
    # Convert to numpy arrays
    gen_np = np.array(gen_img)
    orig_np = np.array(orig_img)
    mask_np = np.array(mask_img)
    
    # Ensure mask is binary and single channel
    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]
    mask_np = (mask_np > 127).astype(float)
    
    # Compute difference in non-masked region (background)
    inv_mask = 1.0 - mask_np
    diff = np.abs(gen_np.astype(float) - orig_np.astype(float))
    background_diff = diff * inv_mask[:, :, np.newaxis]
    
    # Lower difference = better blending
    blend_score = 1.0 - (np.mean(background_diff) / 255.0)
    
    # Normalize to [0, 1] range
    return max(0.0, min(1.0, blend_score))


def compute_occlusion_repair(gen_img, orig_img, mask_img):
    """Compute occlusion repair score"""
    # Convert to numpy arrays
    gen_np = np.array(gen_img)
    mask_np = np.array(mask_img) / 255.0
    
    # Compute texture variance in masked region
    gray_gen = cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY)
    masked_region = gray_gen * mask_np
    
    # Higher variance = better texture repair
    if np.sum(mask_np) > 0:
        variance = np.var(masked_region[mask_np > 0.5])
        repair_score = min(1.0, variance / 1000.0)  # Normalize
    else:
        repair_score = 0.0
    
    return repair_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flux MoE with GRPO")
    
    # Basic training arguments (copied from simple version)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--pretrained_redux_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Redux-dev")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing an image.")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="flux-moe-grpo")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--use_8bit_optimizer", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", action="store_true")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--prodigy_use_bias_correction", action="store_true")
    parser.add_argument("--prodigy_safeguard_warmup", action="store_true")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--prior_generation_precision", type=str, default=None, choices=["no", "fp32", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=30)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--validation_epochs", type=int, default=None)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--text_encoder_lora_rank", type=int, default=64)
    parser.add_argument("--text_encoder_lora_alpha", type=int, default=64)
    parser.add_argument("--text_encoder_lora_dropout", type=float, default=0.0)
    parser.add_argument("--text_encoder_lora_bias", type=str, default="none")
    
    # MoE specific arguments
    parser.add_argument("--num_experts", type=int, default=3, help="Number of experts in MoE")
    parser.add_argument("--expert_names", type=str, nargs="+", default=["removal_expert", "background_expert", "completion_expert"])
    parser.add_argument("--fixed_expert_weights", type=float, nargs="+", default=[0.4, 0.3, 0.3])
    
    # GRPO specific arguments
    parser.add_argument("--use_grpo", action="store_true", help="Enable GRPO training")
    parser.add_argument("--enable_grpo_rewards", action="store_true", help="Enable GRPO reward computation")
    parser.add_argument("--enable_grpo_loss", action="store_true", help="Enable GRPO loss computation")
    parser.add_argument("--use_human_scores", action="store_true", help="Use human-annotated scores")
    parser.add_argument("--score_format", type=str, default="dict", choices=["dict", "tensor"], help="Format of human scores in data")
    
    # GRPO reward weights
    parser.add_argument("--removal_reward_weight", type=float, default=0.4, help="Weight for removal quality reward")
    parser.add_argument("--background_reward_weight", type=float, default=0.3, help="Weight for background consistency reward")
    parser.add_argument("--smoothness_reward_weight", type=float, default=0.3, help="Weight for boundary smoothness reward")
    
    # GRPO loss weights
    parser.add_argument("--base_loss_weight", type=float, default=1.0, help="Weight for base reconstruction loss")
    parser.add_argument("--policy_loss_weight", type=float, default=0.1, help="Weight for policy loss")
    parser.add_argument("--diversity_loss_weight", type=float, default=0.05, help="Weight for expert diversity loss")
    
    # GRPO training parameters
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clipping range")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_freq", type=int, default=500, help="Model saving frequency")
    
    # Dataset arguments
    parser.add_argument("--train_json_path", type=str, required=True)
    parser.add_argument("--val_json_path", type=str, default=None)
    parser.add_argument("--buckets", type=str, default="512:1,768:0.5,1024:0.25")
    parser.add_argument("--bucket_side_min", type=int, default=256)
    parser.add_argument("--bucket_side_max", type=int, default=1536)
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset_name is None and args.train_data_dir is None and args.train_json_path is None:
        raise ValueError("Need either a dataset name or a training folder or train_json_path.")
    
    return args


def main():
    args = parse_args()
    
    # Initialize accelerator
    if args.report_to == "wandb" and args.use_wandb:
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    logger.info("Loading models...")
    
    # Load Fill pipeline
    flux_fill_pipe = FluxFillPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16
    )
    
    # Load Redux pipeline
    flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(
        args.pretrained_redux_model_name_or_path,
        revision=args.revision,
        torch_dtype=torch.bfloat16
    )
    
    # Get transformer and VAE
    transformer = flux_fill_pipe.transformer
    vae = flux_fill_pipe.vae
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    
    # Setup LoRA
    if args.use_lora:
        logger.info("Setting up LoRA...")
        
        # Create LoRA configs for each expert
        lora_configs = []
        for expert_name in args.expert_names:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=[
                    "to_k", "to_q", "to_v", "to_out.0",
                    "proj_in", "proj_out",
                    "ff.net.0.proj", "ff.net.2",
                    "norm.linear"
                ],
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )
            lora_configs.append(config)
        
        # Add adapters to transformer
        for i, (config, adapter_name) in enumerate(zip(lora_configs, args.expert_names)):
            transformer.add_adapter(config, adapter_name)
        
        # Initialize MoE manager
        moe_manager = MoELoRAManager(
            model=transformer,
            lora_configs=lora_configs,
            adapter_names=args.expert_names,
            initial_weights=args.fixed_expert_weights
        )
        
        logger.info(f"MoE LoRA setup complete with {args.num_experts} experts")
    
    # Initialize GRPO components if enabled
    if args.use_grpo:
        logger.info("Initializing GRPO components...")
        reward_computer = GRPORewardComputer(reward_weights=args.reward_weights)
        loss_computer = GRPOLossComputer(clip_range=args.clip_range, diversity_weight=args.diversity_weight)
    
    # Setup datasets
    logger.info("Loading datasets...")
    
    if args.use_grpo:
        # Use GRPO dataset
        train_dataset = RemovalPromptDataset(
            dataset_path=args.train_json_path,
            split='train',
            use_human_scores=args.use_human_scores
        )
        collate_fn = RemovalPromptDataset.collate_fn
    else:
        # 根据是否使用人工标签选择数据集类型
        if args.use_human_scores:
            # 使用GRPO标签数据集
            logger.info("Using GRPO labeled dataset with human scores")
            train_dataset = create_grpo_dataset(
                json_path=args.train_json_path,
                use_human_scores=True,
                score_format=args.score_format,  # 使用用户指定的格式
                buckets=args.buckets,
                bucket_side_min=args.bucket_side_min,
                bucket_side_max=args.bucket_side_max
            )
            collate_fn = grpo_labeled_collate_fn
        else:
            # 使用标准triplet数据集
            logger.info("Using standard triplet dataset")
            train_paths = load_triplet_paths(args.train_json_path)
            buckets = parse_buckets_string(args.buckets)
            
            train_dataset = TripletBucketDataset(
                triplet_paths=train_paths,
                buckets=buckets,
                bucket_side_min=args.bucket_side_min,
                bucket_side_max=args.bucket_side_max
            )
            collate_fn = triplet_collate_fn
    
    # Create data loader
    if args.use_grpo:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers
        )
    else:
        # Use bucket sampler for standard training
        bucket_sampler = BucketBatchSampler(
            train_dataset, batch_size=args.train_batch_size, drop_last=True
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,
            collate_fn=collate_fn,
            num_workers=args.dataloader_num_workers
        )
    
    # Setup optimizer
    if args.use_lora:
        # Get LoRA parameters
        lora_params = list(transformer.parameters())
        router_params = list(moe_manager.router.parameters())
        trainable_params = lora_params + router_params
    else:
        trainable_params = transformer.parameters()
    
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "muon":
        optimizer = MuonWithAuxAdam(
            trainable_params,
            lr=args.learning_rate,
            momentum=0.95,
            weight_decay=args.adam_weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Prepare everything with accelerator
    if args.use_lora:
        transformer, moe_manager.router, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, moe_manager.router, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    
    # Move other components to device
    vae.to(accelerator.device, dtype=torch.bfloat16)
    flux_redux_pipe.to(accelerator.device)
    
    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Initialize wandb
    if accelerator.is_main_process and args.use_wandb:
        wandb.init(
            project="flux-moe-grpo",
            config=vars(args),
            name=f"moe-grpo-{args.num_experts}experts"
        )
    
    # Training loop
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step - (first_epoch * len(train_dataloader))
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if args.use_lora:
            moe_manager.router.train()
        
        train_loss = 0.0
        progress_bar = tqdm(
            range(0, len(train_dataloader)),
            initial=0,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        )
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(transformer):
                # Process batch based on training mode
                if args.use_grpo:
                    # GRPO training logic
                    loss_dict = train_grpo_step(
                        batch, transformer, vae, flux_redux_pipe, moe_manager,
                        noise_scheduler, args, accelerator
                    )
                else:
                    # Standard MoE training logic  
                    loss_dict = train_standard_step(
                        batch, transformer, vae, flux_redux_pipe, moe_manager,
                        noise_scheduler, args, accelerator
                    )
                
                loss = loss_dict["total_loss"]
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                if accelerator.is_main_process:
                    logs = {
                        "train_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + (step + 1) / len(train_dataloader),
                    }
                    
                    # Add GRPO specific loss components
                    if args.use_grpo and "base_loss" in loss_dict:
                        logs.update({
                            "base_loss": loss_dict["base_loss"].detach().item() if hasattr(loss_dict["base_loss"], 'detach') else loss_dict["base_loss"],
                            "policy_loss": loss_dict.get("policy_loss", 0.0).detach().item() if hasattr(loss_dict.get("policy_loss", 0.0), 'detach') else loss_dict.get("policy_loss", 0.0),
                            "diversity_loss": loss_dict.get("diversity_loss", 0.0).detach().item() if hasattr(loss_dict.get("diversity_loss", 0.0), 'detach') else loss_dict.get("diversity_loss", 0.0),
                            "mean_reward": loss_dict.get("mean_reward", 0.0).detach().item() if hasattr(loss_dict.get("mean_reward", 0.0), 'detach') else loss_dict.get("mean_reward", 0.0),
                            "mean_ratio": loss_dict.get("mean_ratio", 1.0).detach().item() if hasattr(loss_dict.get("mean_ratio", 1.0), 'detach') else loss_dict.get("mean_ratio", 1.0),
                        })
                    
                    # Add MoE routing stats if available
                    if args.use_lora and hasattr(moe_manager, 'get_routing_stats'):
                        routing_stats = moe_manager.get_routing_stats()
                        logs.update({f"routing_{k}": v for k, v in routing_stats.items() if isinstance(v, (int, float))})
                    
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                    
                    if args.use_wandb:
                        wandb.log(logs, step=global_step)
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation
                if accelerator.is_main_process and global_step % args.validation_steps == 0:
                    if args.val_json_path:
                        logger.info("Running validation...")
                        try:
                            log_infer_moe(
                                accelerator=accelerator,
                                args=args,
                                save_path=args.output_dir,
                                epoch=epoch,
                                global_step=global_step,
                                pipefill=flux_fill_pipe,
                                pipeprior=flux_redux_pipe
                            )
                            free_memory()
                        except Exception as e:
                            logger.error(f"Validation failed: {e}")
                            traceback.print_exc()
                
                if global_step >= args.max_train_steps:
                    break
        
        # End of epoch logging
        accelerator.log({"epoch_loss": train_loss / len(train_dataloader)}, step=global_step)
    
    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_output_dir = os.path.join(args.output_dir, "final_moe")
        if args.use_lora:
            save_moe_lora_state(
                final_output_dir,
                unwrap_model(transformer),
                unwrap_model(moe_manager.router) if hasattr(moe_manager, 'router') else None,
                args.expert_names
            )
            logger.info(f"MoE LoRA model saved to {final_output_dir}")
        else:
            transformer.save_pretrained(final_output_dir)
            logger.info(f"Model saved to {final_output_dir}")
    
    accelerator.end_training()
    logger.info("Training completed")


def compute_grpo_rewards(pred_images, target_images, masks, args, human_scores=None):
    """
    计算GRPO奖励信号，支持人工标签和自动计算两种模式
    
    Args:
        pred_images: 预测图像 [batch_size, 3, H, W]
        target_images: 目标图像 [batch_size, 3, H, W] 
        masks: 掩码 [batch_size, 1, H, W]
        args: 训练参数
        human_scores: 人工标签分数 dict，包含3个分数列表
                     {'removal_score': [batch], 'background_score': [batch], 'smoothness_score': [batch]}
        
    Returns:
        dict: 包含各种奖励的字典
    """
    batch_size = pred_images.shape[0]
    device = pred_images.device
    rewards = {}
    
    if args.use_human_scores and human_scores is not None:
        # 使用人工标签分数作为奖励
        logger.info("Using human-annotated scores as rewards")
        
        # 将人工分数转换为tensor并归一化到[-1, 1]范围
        removal_scores = torch.tensor(human_scores.get('removal_score', [0.5] * batch_size), device=device, dtype=torch.float32)
        background_scores = torch.tensor(human_scores.get('background_score', [0.5] * batch_size), device=device, dtype=torch.float32)
        smoothness_scores = torch.tensor(human_scores.get('smoothness_score', [0.5] * batch_size), device=device, dtype=torch.float32)
        
        # 假设人工分数范围是[0, 1]，转换为[-1, 1]的奖励范围
        rewards['removal_reward'] = (removal_scores - 0.5) * 2.0  # [0,1] -> [-1,1]
        rewards['background_reward'] = (background_scores - 0.5) * 2.0
        rewards['smoothness_reward'] = (smoothness_scores - 0.5) * 2.0
        
        # 记录人工分数统计
        logger.debug(f"Human scores - Removal: {removal_scores.mean():.3f}, Background: {background_scores.mean():.3f}, Smoothness: {smoothness_scores.mean():.3f}")
        
    else:
        # 自动计算奖励（原有逻辑）
        logger.debug("Using automatic reward computation")
        
        # 1. 移除质量奖励 (removal quality reward)
        masked_pred = pred_images * masks
        masked_target = target_images * masks
        removal_mse = F.mse_loss(masked_pred, masked_target, reduction='none').mean(dim=[1,2,3])
        rewards['removal_reward'] = -removal_mse  # 负MSE作为奖励
        
        # 2. 背景一致性奖励 (background consistency reward)
        bg_mask = 1.0 - masks
        bg_pred = pred_images * bg_mask
        bg_target = target_images * bg_mask
        bg_mse = F.mse_loss(bg_pred, bg_target, reduction='none').mean(dim=[1,2,3])
        rewards['background_reward'] = -bg_mse
        
        # 3. 边界平滑度奖励 (boundary smoothness reward)
        # 使用Sobel算子检测边界
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        
        pred_gray = pred_images.mean(dim=1, keepdim=True)  # 转灰度
        edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
        edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 在掩码边界区域计算平滑度
        mask_edges = F.conv2d(masks, sobel_x.abs() + sobel_y.abs(), padding=1) > 0.1
        boundary_smoothness = (edge_magnitude * mask_edges.float()).mean(dim=[1,2,3])
        rewards['smoothness_reward'] = -boundary_smoothness
    
    # 4. 综合奖励（两种模式都需要）
    total_reward = (
        args.removal_reward_weight * rewards['removal_reward'] +
        args.background_reward_weight * rewards['background_reward'] + 
        args.smoothness_reward_weight * rewards['smoothness_reward']
    )
    rewards['total_reward'] = total_reward
    
    return rewards


def compute_grpo_loss(model_pred, target, adapter_weights, rewards, old_log_probs, args):
    """
    计算GRPO损失函数
    
    Args:
        model_pred: 模型预测 [batch_size, seq_len, hidden_dim]
        target: 目标 [batch_size, seq_len, hidden_dim]
        adapter_weights: 适配器权重 [batch_size, num_experts]
        rewards: 奖励字典
        old_log_probs: 旧策略的log概率 [batch_size]
        args: 训练参数
        
    Returns:
        dict: 损失字典
    """
    # 1. 基础重建损失
    base_loss = F.mse_loss(model_pred, target)
    
    # 2. 策略损失 (Policy Loss)
    # 计算当前策略的log概率
    log_probs = torch.log(adapter_weights + 1e-8).sum(dim=-1)
    
    # 计算重要性采样比率
    ratio = torch.exp(log_probs - old_log_probs.detach())
    
    # PPO-style clipped loss
    advantages = rewards['total_reward'].detach()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 3. 专家多样性损失 (Expert Diversity Loss)
    # 鼓励专家权重分化
    expert_entropy = -(adapter_weights * torch.log(adapter_weights + 1e-8)).sum(dim=-1).mean()
    diversity_loss = -expert_entropy  # 负熵，鼓励多样性
    
    # 4. 总损失
    total_loss = (
        args.base_loss_weight * base_loss +
        args.policy_loss_weight * policy_loss +
        args.diversity_loss_weight * diversity_loss
    )
    
    return {
        'total_loss': total_loss,
        'base_loss': base_loss,
        'policy_loss': policy_loss,
        'diversity_loss': diversity_loss,
        'mean_reward': rewards['total_reward'].mean(),
        'mean_ratio': ratio.mean()
    }


def train_grpo_step(batch, transformer, vae, flux_redux_pipe, moe_manager, noise_scheduler, args, accelerator):
    """
    执行一步GRPO训练
    """
    weight_dtype = torch.bfloat16
    
    # 处理批次数据
    ref = batch["ref"].to(accelerator.device, dtype=weight_dtype)
    src = batch["src"].to(accelerator.device, dtype=weight_dtype) 
    mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)
    imgs = batch["result"].to(accelerator.device, dtype=weight_dtype)
    
    # 编码prompt
    prompt_embeds = encode_prompt({"ref": ref}, flux_redux_pipe, weight_dtype)
    
    # VAE编码
    with torch.no_grad():
        model_input = vae.encode(imgs).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    
    # 添加噪声
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    # 采样时间步
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
    timesteps = timesteps.long()
    
    # 添加噪声到模型输入
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
    
    # 准备输入
    packed_noisy_model_input = flux_fill_pipe._pack_latents(
        noisy_model_input,
        batch_size=bsz,
        num_channels_latents=noisy_model_input.shape[1],
        height=noisy_model_input.shape[2],
        width=noisy_model_input.shape[3],
    )
    
    # === GRPO采样阶段 ===
    with torch.no_grad():
        # 使用当前策略生成样本
        if args.use_lora and moe_manager is not None:
            old_model_pred, old_adapter_weights = moe_manager.forward_with_routing(
                features=packed_noisy_model_input,
                hidden_states=packed_noisy_model_input,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                return_dict=False
            )
            # 计算旧策略的log概率
            old_log_probs = torch.log(old_adapter_weights + 1e-8).sum(dim=-1)
        else:
            old_model_pred = transformer(
                hidden_states=packed_noisy_model_input,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                return_dict=False
            )[0]
            old_adapter_weights = None
            old_log_probs = torch.zeros(bsz, device=accelerator.device)
    
    # === GRPO训练阶段 ===
    # 前向传播
    if args.use_lora and moe_manager is not None:
        model_pred, adapter_weights = moe_manager.forward_with_routing(
            features=packed_noisy_model_input,
            hidden_states=packed_noisy_model_input,
            timestep=timesteps,
            prompt_embeds=prompt_embeds,
            return_dict=False
        )
    else:
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps,
            prompt_embeds=prompt_embeds,
            return_dict=False
        )[0]
        adapter_weights = torch.ones(bsz, 3, device=accelerator.device) / 3.0
    
    # 计算目标
    target = noise
    packed_target = flux_fill_pipe._pack_latents(
        target,
        batch_size=bsz,
        num_channels_latents=target.shape[1],
        height=target.shape[2],
        width=target.shape[3],
    )
    
    # 解码预测结果用于奖励计算
    if args.enable_grpo_rewards:
        try:
            # 解码预测和目标到像素空间
            with torch.no_grad():
                # 解包和解码预测
                unpacked_pred = flux_fill_pipe._unpack_latents(
                    model_pred, height=noisy_model_input.shape[2], width=noisy_model_input.shape[3], vae_scale_factor=8
                )
                pred_scaled = (unpacked_pred / vae.config.scaling_factor) + vae.config.shift_factor
                pred_images = vae.decode(pred_scaled).sample
                
                # 解包和解码目标
                unpacked_target = flux_fill_pipe._unpack_latents(
                    packed_target, height=noisy_model_input.shape[2], width=noisy_model_input.shape[3], vae_scale_factor=8
                )
                target_scaled = (unpacked_target / vae.config.scaling_factor) + vae.config.shift_factor
                target_images = vae.decode(target_scaled).sample
                
                # 提取人工标签分数（如果存在）
                human_scores = None
                if args.use_human_scores and 'scores' in batch:
                    # 从batch中提取3个分数
                    scores_data = batch['scores']  # 假设格式: [batch_size, 3] 或 dict
                    if isinstance(scores_data, torch.Tensor):
                        # tensor格式: [batch_size, 3] -> [removal, background, smoothness]
                        human_scores = {
                            'removal_score': scores_data[:, 0].cpu().numpy().tolist(),
                            'background_score': scores_data[:, 1].cpu().numpy().tolist(),
                            'smoothness_score': scores_data[:, 2].cpu().numpy().tolist()
                        }
                    elif isinstance(scores_data, dict):
                        # dict格式: 直接使用
                        human_scores = scores_data
                    else:
                        logger.warning(f"Unsupported scores format: {type(scores_data)}, falling back to auto computation")
                
                # 计算奖励
                rewards = compute_grpo_rewards(pred_images, target_images, mask, args, human_scores)
        except Exception as e:
            logger.warning(f"Failed to compute GRPO rewards: {e}, falling back to simple loss")
            rewards = {'total_reward': torch.zeros(bsz, device=accelerator.device)}
    else:
        rewards = {'total_reward': torch.zeros(bsz, device=accelerator.device)}
    
    # 计算损失
    if args.enable_grpo_loss and adapter_weights is not None:
        loss_dict = compute_grpo_loss(model_pred, packed_target, adapter_weights, rewards, old_log_probs, args)
    else:
        # 回退到简单MSE损失
        loss_dict = compute_simple_mse_loss(model_pred, packed_target)
    
    return loss_dict


def train_standard_step(batch, transformer, vae, flux_redux_pipe, moe_manager, noise_scheduler, args, accelerator):
    """
    Execute one standard MoE training step (copied from simple version).
    """
    weight_dtype = torch.bfloat16
    
    # Process batch
    ref = batch["ref"].to(accelerator.device, dtype=weight_dtype)
    src = batch["src"].to(accelerator.device, dtype=weight_dtype) 
    mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)
    imgs = batch["result"].to(accelerator.device, dtype=weight_dtype)
    
    # Encode prompt
    prompt_embeds = encode_prompt({"ref": ref}, flux_redux_pipe, weight_dtype)
    
    # VAE encode
    with torch.no_grad():
        model_input = vae.encode(imgs).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    
    # Add noise
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    
    # Sample timesteps
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
    timesteps = timesteps.long()
    
    # Add noise to model input
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
    
    # Prepare inputs
    packed_noisy_model_input = flux_fill_pipe._pack_latents(
        noisy_model_input,
        batch_size=bsz,
        num_channels_latents=noisy_model_input.shape[1],
        height=noisy_model_input.shape[2],
        width=noisy_model_input.shape[3],
    )
    
    # Forward pass with MoE
    if args.use_lora and moe_manager is not None:
        model_pred, adapter_weights = moe_manager.forward_with_routing(
            features=packed_noisy_model_input,
            hidden_states=packed_noisy_model_input,
            timestep=timesteps,
            prompt_embeds=prompt_embeds,
            return_dict=False
        )
    else:
        model_pred = transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps,
            prompt_embeds=prompt_embeds,
            return_dict=False
        )[0]
    
    # Compute loss
    target = noise
    packed_target = flux_fill_pipe._pack_latents(
        target,
        batch_size=bsz,
        num_channels_latents=target.shape[1],
        height=target.shape[2],
        width=target.shape[3],
    )
    
    # Use simple MSE loss
    loss_dict = compute_simple_mse_loss(model_pred, packed_target)
    
    return loss_dict


if __name__ == "__main__":
    main()
        