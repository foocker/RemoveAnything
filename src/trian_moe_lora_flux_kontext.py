#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import logging
import math
import time
import gc
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm
from optimizer.muon import MuonWithAuxAdam
from copy import deepcopy

from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory
)

from data.all_data import load_triplet_paths, load_triplet_paths_from_dir
from models.pipeline_tools import encode_images, prepare_text_input, Flux_fill_encode_masks_images
from models.image_project import image_output
from models.transformer import tranformer_forward
from data.triplet_bucket_dataset import TripletBucketDataset, triplet_collate_fn
from data.bucket_utils import BucketBatchSampler
from data.bucket_utils import (
    parse_buckets_string,
    find_nearest_bucket
)

from PIL import Image
import cv2
from data.data_utils import (
    get_bbox_from_mask,
    expand_bbox,
    expand_image_mask,
    pad_to_square,
    box2squre,
    crop_back
)

logger = get_logger(__name__, log_level="INFO")


# 实现 MoE (Mixture of Experts) LoRA Router
class LoRARouter(nn.Module):
    """Expert Router for Mixture of LoRA Adapters.
    
    This router determines which adapter or combination of adapters to use
    based on input features.
    """
    def __init__(self, input_dim, num_experts, routing_strategy="soft", temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.routing_strategy = routing_strategy  # 'soft', 'hard', or 'topk'
        self.temperature = temperature  # 控制softmax的锐度
        self.top_k = 2  # 如果使用topk策略，选择的专家数量
        
        # 路由网络，用于决定使用哪个专家
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_experts)
        )
    
    def forward(self, x):
        """前向传播，输入特征，输出每个专家的权重
        
        Args:
            x: 输入特征，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            weights: 每个专家的权重，形状为 [batch_size, num_experts]
        """
        # 为了简化，我们使用序列的平均值作为路由输入
        # 实际应用中可能需要更复杂的特征提取
        if len(x.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            routing_inputs = x.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            routing_inputs = x
        
        # 计算路由分数
        routing_logits = self.router(routing_inputs)  # [batch_size, num_experts]
        
        # 根据不同路由策略计算权重
        if self.routing_strategy == "soft":
            # 软路由：使用softmax计算每个专家的权重
            weights = F.softmax(routing_logits / self.temperature, dim=-1)
            
        elif self.routing_strategy == "hard":
            # 硬路由：只选择概率最高的专家
            indices = torch.argmax(routing_logits, dim=-1)
            weights = torch.zeros_like(routing_logits)
            weights.scatter_(-1, indices.unsqueeze(-1), 1.0)
            
        elif self.routing_strategy == "topk":
            # Top-k路由：选择概率最高的k个专家
            topk_values, topk_indices = torch.topk(routing_logits, self.top_k, dim=-1)
            weights = torch.zeros_like(routing_logits)
            weights.scatter_(-1, topk_indices, F.softmax(topk_values / self.temperature, dim=-1))
        
        return weights


# 实现MoE LoRA管理器
class MoELoRAManager:
    """管理多个LoRA适配器并实现MoE功能。
    
    该类负责初始化多个LoRA适配器，并根据路由器的输出
    动态设置适当的适配器权重。
    """
    def __init__(self, model, lora_configs, adapter_names, routing_config=None):
        """
        Args:
            model: 基础模型，LoRA将应用于此模型
            lora_configs: 列表，每个元素是一个LoRA配置字典
            adapter_names: 列表，LoRA适配器名称
            routing_config: 字典，路由器配置
        """
        self.model = model
        self.lora_configs = lora_configs
        self.adapter_names = adapter_names
        self.num_experts = len(adapter_names)
        
        # 为每个任务/数据类型添加LoRA适配器
        for i, (config, name) in enumerate(zip(lora_configs, adapter_names)):
            self.model.add_adapter(name, LoraConfig(**config))
            
        # 初始化路由器
        if routing_config is None:
            routing_config = {
                "input_dim": 2048,  # 假设隐藏层维度为2048
                "routing_strategy": "soft",
                "temperature": 1.0
            }
        
        self.router = LoRARouter(
            input_dim=routing_config["input_dim"],
            num_experts=self.num_experts,
            routing_strategy=routing_config["routing_strategy"],
            temperature=routing_config["temperature"]
        )
        
        # 确保路由器参数可以训练
        self.router.train()
        
        # 初始默认为均匀权重
        self.current_weights = torch.ones(self.num_experts) / self.num_experts
        
    def compute_adapter_weights(self, features):
        """计算当前输入应该使用的适配器权重"""
        with torch.set_grad_enabled(self.router.training):
            return self.router(features)
    
    def set_adapter_weights(self, weights):
        """设置适配器权重"""
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        self.current_weights = weights
        self.model.set_adapter(self.adapter_names, adapter_weights=weights)
        
    def forward_with_routing(self, features, **forward_kwargs):
        """使用动态路由进行前向传播"""
        # 计算适配器权重
        adapter_weights = self.compute_adapter_weights(features)
        
        # 对批次中的每个样本设置不同的权重(简化版，实际应使用函数式API或其他方法)
        batch_size = adapter_weights.shape[0]
        results = []
        
        # 这里简化处理，实际中可能需要更高效的实现
        for i in range(batch_size):
            # 为当前样本设置适配器权重
            self.set_adapter_weights(adapter_weights[i])
            
            # 单独处理这个样本
            # 注意：这里假设forward_kwargs中的所有张量都可以用索引[i:i+1]提取单个样本
            # 实际实现可能需要更复杂的处理
            sample_kwargs = {}
            for k, v in forward_kwargs.items():
                if isinstance(v, torch.Tensor):
                    sample_kwargs[k] = v[i:i+1]
                else:
                    sample_kwargs[k] = v
            
            # 调用模型前向传播
            result = self.model(**sample_kwargs)
            results.append(result)
        
        # 合并结果
        # 注意：这里假设结果是可以简单连接的，实际情况可能需要更复杂的处理
        combined_result = torch.cat(results, dim=0)
        
        return combined_result, adapter_weights


# 修改保存函数以处理MoE LoRA权重和路由器
def save_moe_lora_state(output_dir, transformer, router, adapter_names):
    """保存MoE LoRA模型的状态，包括所有适配器和路由器"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存路由器
    router_path = os.path.join(output_dir, "lora_router.pt")
    torch.save(router.state_dict(), router_path)
    logger.info(f"路由器保存至 {router_path}")
    
    # 保存每个适配器
    for adapter_name in adapter_names:
        adapter_output_dir = os.path.join(output_dir, f"adapter_{adapter_name}")
        os.makedirs(adapter_output_dir, exist_ok=True)
        
        # 获取特定适配器的权重
        adapter_state_dict = get_peft_model_state_dict(transformer, adapter_name)
        
        # 保存适配器权重
        FluxFillPipeline.save_lora_weights(
            adapter_output_dir,
            transformer_lora_layers=adapter_state_dict,
            safe_serialization=True,
        )
        logger.info(f"适配器 {adapter_name} 保存至 {adapter_output_dir}")


# 加载MoE LoRA状态的函数
def load_moe_lora_state(input_dir, transformer, router, adapter_names):
    """加载MoE LoRA模型状态，包括所有适配器和路由器"""
    # 加载路由器
    router_path = os.path.join(input_dir, "lora_router.pt")
    if os.path.exists(router_path):
        router.load_state_dict(torch.load(router_path))
        logger.info(f"从 {router_path} 加载路由器")
    else:
        logger.warning(f"未找到路由器权重: {router_path}")
    
    # 加载每个适配器
    for adapter_name in adapter_names:
        adapter_input_dir = os.path.join(input_dir, f"adapter_{adapter_name}")
        
        if os.path.exists(adapter_input_dir):
            logger.info(f"加载适配器 {adapter_name} 从 {adapter_input_dir}")
            
            try:
                lora_state_dict = FluxFillPipeline.lora_state_dict(adapter_input_dir)
                transformer_state_dict = {
                    f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() 
                    if k.startswith("transformer.")
                }
                incompatible_keys = set_peft_model_state_dict(
                    transformer, transformer_state_dict, adapter_name=adapter_name
                )
                
                if incompatible_keys is not None:
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(f"加载适配器 {adapter_name} 时发现意外键: {unexpected_keys}")
                        
            except Exception as e:
                logger.error(f"加载适配器 {adapter_name} 失败: {str(e)}")
        else:
            logger.warning(f"未找到适配器目录: {adapter_input_dir}")


# 创建训练流程中的MoE钩子函数
def create_moe_hooks(transformer, router, adapter_names):
    """创建用于MoE训练的钩子函数"""
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_ = None
            for model in models:
                if isinstance(model, type(transformer)):
                    transformer_ = model
                weights.pop()  # 防止模型被重复保存
            
            if transformer_ is not None:
                save_moe_lora_state(output_dir, transformer_, router, adapter_names)
    
    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(transformer)):
                transformer_ = model
        
        if transformer_ is not None:
            load_moe_lora_state(input_dir, transformer_, router, adapter_names)
    
    return save_model_hook, load_model_hook


def parse_args():
    parser = argparse.ArgumentParser(description="使用MoE-LoRA训练Flux模型")
    
    # 数据参数
    parser.add_argument("--train_json_path", type=str, required=True, help="训练数据的JSON路径")
    parser.add_argument("--val_json_path", type=str, required=True, help="验证数据的JSON路径")
    
    # 基本训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/moe_lora", help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="混合精度训练类型")
    
    # 模型参数
    parser.add_argument("--flux_fill_id", type=str, default="stabilityai/flux-fill", help="FluxFillPipeline模型ID")
    parser.add_argument("--flux_redux_id", type=str, default="stabilityai/flux-prior-redux", help="FluxPriorReduxPipeline模型ID")
    
    # MoE-LoRA参数
    parser.add_argument("--use_moe_lora", action="store_true", default=True, help="是否使用MoE-LoRA")
    parser.add_argument("--num_experts", type=int, default=3, help="专家数量")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA的alpha参数")
    parser.add_argument("--routing_strategy", type=str, default="soft", choices=["soft", "hard", "topk"], help="路由策略")
    parser.add_argument("--routing_temperature", type=float, default=1.0, help="路由softmax温度")
    parser.add_argument("--expert_names", type=str, default="expert1,expert2,expert3", help="专家名称，逗号分隔")
    
    # 其他参数省略，与标准训练脚本相同...
    
    args = parser.parse_args()
    return args


def main():
    # 初始化加速器
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置加速器
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs")
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps if hasattr(args, "gradient_accumulation_steps") else 1,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载模型
    logger.info(f"加载FluxFillPipeline模型: {args.flux_fill_id}")
    flux_fill_pipe = FluxFillPipeline.from_pretrained(args.flux_fill_id)
    
    logger.info(f"加载FluxPriorReduxPipeline模型: {args.flux_redux_id}")
    flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(args.flux_redux_id)
    
    # 提取transformer组件
    transformer = flux_fill_pipe.transformer
    transformer.requires_grad_(False)
    
    # 配置MoE-LoRA
    if args.use_moe_lora:
        logger.info("配置MoE-LoRA适配器...")
        
        # 解析专家名称
        expert_names = args.expert_names.split(",")
        if len(expert_names) != args.num_experts:
            logger.warning(f"专家名称数量({len(expert_names)})与专家数量({args.num_experts})不匹配，将使用默认名称")
            expert_names = [f"expert{i}" for i in range(args.num_experts)]
        
        # 配置LoRA
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        
        # 创建多个LoRA配置，可以为每个专家配置不同的参数
        lora_configs = []
        for i in range(args.num_experts):
            lora_config = {
                "r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "init_lora_weights": "gaussian",
                "target_modules": target_modules
            }
            lora_configs.append(lora_config)
        
        # 配置路由器
        routing_config = {
            "input_dim": 2048,  # 这里需要根据实际模型确定
            "routing_strategy": args.routing_strategy,
            "temperature": args.routing_temperature
        }
        
        # 创建MoE管理器
        moe_manager = MoELoRAManager(
            model=transformer,
            lora_configs=lora_configs,
            adapter_names=expert_names,
            routing_config=routing_config
        )
        
        # 设置保存和加载钩子
        save_hook, load_hook = create_moe_hooks(transformer, moe_manager.router, expert_names)
        accelerator.register_save_state_pre_hook(save_hook)
        accelerator.register_load_state_pre_hook(load_hook)
        
        # 收集需要训练的参数
        trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        trainable_params.extend(moe_manager.router.parameters())
        
        logger.info(f"MoE-LoRA模式 - 可训练参数数量: {len(trainable_params)}")
    else:
        # 标准训练模式
        logger.info("使用标准训练模式")
        transformer.requires_grad_(True)
        trainable_params = transformer.parameters()
    
    # 优化器设置
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=0.0001,  # 默认学习率
        weight_decay=0.01
    )
    
    # ... 这里是数据加载器、训练循环等，与常规训练流程类似
    # ... 只在关键部分需要修改以使用MoE功能
    
    # 示例训练循环（占位，实际需要实现）
    logger.info("开始MoE-LoRA训练...")
    logger.info("(注：这是占位代码，需要实现完整的训练循环)")
    
    # 保存最终模型
    if args.use_moe_lora and accelerator.is_main_process:
        final_output_dir = os.path.join(args.output_dir, "final_moe_lora")
        save_moe_lora_state(
            final_output_dir,
            accelerator.unwrap_model(transformer),
            moe_manager.router,
            expert_names
        )
        logger.info(f"MoE-LoRA模型已保存到 {final_output_dir}")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()
    logger.info("MoE-LoRA训练完成")
