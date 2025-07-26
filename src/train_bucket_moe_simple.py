#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import logging
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from tqdm.auto import tqdm
from peft import LoraConfig

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
    mse_loss = F.mse_loss(pred, target, reduction='none')
    
    # 如果有权重，应用权重
    if weighting is not None:
        # 确保weighting的形状与loss兼容
        if weighting.dim() < mse_loss.dim():
            # 扩展weighting的维度以匹配loss
            for _ in range(mse_loss.dim() - weighting.dim()):
                weighting = weighting.unsqueeze(-1)
        mse_loss = mse_loss * weighting
    
    # 计算平均损失
    total_loss = mse_loss.mean()
    
    return {
        'total_loss': total_loss,
        'mse_loss': total_loss
    }


class FeatureExtractor(nn.Module):
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
        if self.projection is None:
            input_dim = x.shape[-1]
            output_dim = 256  # 简化输出维度
            
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.LayerNorm(output_dim)
            ).to(x.device, x.dtype)
        
        if len(x.shape) == 2:  # [batch_size, hidden_dim]
            return self.projection(x)
        
        pooled = x.mean(dim=1)
        return self.projection(pooled)


class LearnableRouter(nn.Module):
    def __init__(self, num_experts=3, initial_weights=None, dtype=None):
        super().__init__()
        self.num_experts = num_experts
        
        if initial_weights is None:
            initial_weights = [1.0/num_experts] * num_experts  # 默认均匀权重
        
        total_weight = sum(initial_weights)
        normalized_weights = [w/total_weight for w in initial_weights]
        
        # 将权重设为可学习参数（使用logits形式避免约束）
        # 通过inverse softmax转换为logits
        logits = [math.log(w + 1e-8) for w in normalized_weights]
        
        # 使用传入的dtype，如果没有指定则使用float32作为默认值
        if dtype is None:
            dtype = torch.float32
        self.weight_logits = nn.Parameter(torch.tensor(logits, dtype=dtype))
        
        self.feature_extractor = FeatureExtractor()
    
    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            weights: 三个专家的可学习权重，形状为 [batch_size, 3]
        """
        
        # TODO 数据集的moe权重，还是当前输入的features的moe权重，待测试和实现
        batch_size = x.shape[0]
        
        # weight_logits已经使用正确的dtype初始化，不需要再转换
        weights = F.softmax(self.weight_logits, dim=-1)
        weights = weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 3]
        
        return weights
    
    def get_current_weights(self):
        """获取当前的权重值（用于监控和调试）"""
        with torch.no_grad():
            # 计算softmax权重
            weights = F.softmax(self.weight_logits, dim=-1)
            # 将权重移到CPU并转换为float32以便转numpy（避免在GPU上进行dtype转换）
            weights_cpu = weights.cpu()
            # 只有在CPU上且在no_grad下才转换dtype
            if weights_cpu.dtype == torch.bfloat16:
                weights_cpu = weights_cpu.float()
            return weights_cpu.numpy().tolist()


class MoELoRAManager:
    def __init__(self, model, lora_configs, adapter_names, initial_weights=None):
        """
        Args:
            model: 基础模型，LoRA将应用于此模型
            lora_configs: 列表，每个元素是一个LoRA配置字典
            adapter_names: 列表，LoRA适配器名称
            initial_weights: 初始权重列表，默认为均匀分配
        """
        self.model = model
        self.lora_configs = lora_configs
        self.adapter_names = adapter_names
        self.num_experts = len(adapter_names)
        
        # 为每个任务/数据类型添加LoRA适配器
        for i, (lora_config, adapter_name) in enumerate(zip(lora_configs, adapter_names)):
            if adapter_name not in getattr(self.model, "peft_config", {}):
                self.model.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)
        
        model_dtype = next(self.model.parameters()).dtype
        
        self.router = LearnableRouter(num_experts=self.num_experts, initial_weights=initial_weights, dtype=model_dtype)
        self.current_weights = None
        
        if initial_weights is None:
            self.current_weights = [1.0/self.num_experts] * self.num_experts  # 默认均匀权重
        else:
            # 确保权重和为1
            total_weight = sum(initial_weights)
            self.current_weights = [w/total_weight for w in initial_weights]
        
    def compute_adapter_weights(self, features):
        """
        计算当前输入应该使用的适配器权重（使用可学习路由器）
        
        Args:
            features: 输入特征，形状 [batch_size, ...]
            
        Returns:
            torch.Tensor: 可学习适配器权重，形状 [batch_size, num_experts]
        """
        weights = self.router(features)
        return weights
    
    def set_adapter_weights(self, weights):
        """设置适配器权重
        
        PEFT库期望weights参数是一个一维列表，例如[0.3, 0.3, 0.4]。
        输入的weights可能是二维张量[batch_size, num_experts]，
        所以需要进行处理以匹配库的期望。
        """
        if isinstance(weights, torch.Tensor):
            if len(weights.shape) > 1:
                weights = weights.mean(dim=0)  # [batch_size, num_experts] -> [num_experts]
            
            adapter_weights_list = weights.detach().cpu().tolist()  # 转换为普通的Python列表
            self.current_weights = adapter_weights_list
        else:
            self.current_weights = weights
        self.model.set_adapters(self.adapter_names, weights=self.current_weights)
        
    def get_routing_stats(self):
        """获取路由统计信息（可学习版本）"""
        current_weights = self.router.get_current_weights()
        return {
            "current_weights": current_weights,
            "num_experts": self.num_experts,
            "adapter_names": self.adapter_names,
            "weights_sum": sum(current_weights),
            "weights_entropy": self._compute_entropy(current_weights)
        }
    
    def _compute_entropy(self, weights):
        """计算权重的熵（衡量权重分布的均匀性）"""
        import math
        entropy = -sum(w * math.log(w + 1e-8) for w in weights if w > 0)
        return entropy
    
    def get_current_weights(self):
        return self.router.get_current_weights()
    
    def forward_with_routing(self, features, **forward_kwargs):
        """
        Args:
            features: 输入特征
            **forward_kwargs: 传递给模型的参数
        Returns:
            result: 模型输出
            adapter_weights: 使用的适配器权重
        """
        adapter_weights = self.compute_adapter_weights(features)
        
        avg_weights = adapter_weights.mean(dim=0).detach().cpu().tolist()
        self.set_adapter_weights(avg_weights)
        
        # TODO 模型前向传播,仅仅是 加了lora的transformer的前向传播，和整个fillpipe的推理还是不同的，整个fillpipe的推理包含了当前的模块
        result = self.model(**forward_kwargs)
        
        return result, adapter_weights


def save_moe_lora_state(output_dir, transformer, router, adapter_names):
    """保存MoE LoRA模型的状态，包括所有适配器和路由器"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        import json
        import datetime
        
        router_path = os.path.join(output_dir, "moe_router.safetensors")
        save_file(router.state_dict(), router_path)
        logger.info(f"MoE路由器保存至: {router_path}")
        
        for adapter_name in adapter_names:
            adapter_dir = os.path.join(output_dir, f"lora_{adapter_name}")
            os.makedirs(adapter_dir, exist_ok=True)
            adapter_state_dict = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
            save_file(adapter_state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))
            
            adapter_config = {
                "adapter_name": adapter_name,
                "adapter_type": "lora",
                "is_moe_expert": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, indent=2, ensure_ascii=False)
                
            logger.info(f"MoE专家适配器 {adapter_name} 已保存到: {adapter_dir}")
        
        moe_config = {
            "moe_type": "lora_experts",
            "num_experts": len(adapter_names),
            "adapter_names": adapter_names,
            "router_type": "neural_router",
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(os.path.join(output_dir, "moe_config.json"), "w", encoding="utf-8") as f:
            json.dump(moe_config, f, indent=2, ensure_ascii=False)
            
        logger.info(f"MoE配置已保存，包含 {len(adapter_names)} 个专家适配器")
        
    except Exception as e:
        logger.error(f"保存MoE LoRA状态失败: {str(e)}")
        raise


def load_moe_lora_state(input_dir, transformer, router, adapter_names):
    """加载MoE LoRA模型状态，包括所有适配器和路由器"""
    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        
        router_path = os.path.join(input_dir, "moe_router.safetensors")
        if os.path.exists(router_path):
            router_state_dict = load_file(router_path)
            router.load_state_dict(router_state_dict)
            logger.info(f"MoE路由器已从 {router_path} 加载")
        else:
            # 兼容旧格式
            old_router_path = os.path.join(input_dir, "lora_router.pt")
            if os.path.exists(old_router_path):
                router.load_state_dict(torch.load(old_router_path))
                logger.info(f"MoE路由器已从旧格式 {old_router_path} 加载")
            else:
                logger.warning(f"未找到MoE路由器权重文件")
        
        for adapter_name in adapter_names:
            adapter_dir = os.path.join(input_dir, f"lora_{adapter_name}")
            
            if os.path.exists(adapter_dir):
                try:
                    adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
                    if os.path.exists(adapter_file):
                        adapter_state_dict = load_file(adapter_file)
                        set_peft_model_state_dict(transformer, adapter_state_dict, adapter_name=adapter_name)
                        logger.info(f"MoE专家适配器 {adapter_name} 已从 {adapter_dir} 加载")
                    else:
                        logger.warning(f"MoE适配器文件不存在: {adapter_file}")
                        
                except Exception as e:
                    logger.error(f"加载MoE适配器 {adapter_name} 失败: {str(e)}")
            else:
                # 兼容旧格式
                old_adapter_dir = os.path.join(input_dir, f"adapter_{adapter_name}")
                if os.path.exists(old_adapter_dir):
                    logger.info(f"尝试从旧格式加载适配器 {adapter_name}")
                    # 这里可以添加旧格式加载逻辑
                else:
                    logger.warning(f"未找到MoE适配器: {adapter_name}")
                    
    except Exception as e:
        logger.error(f"加载MoE LoRA状态失败: {str(e)}")
        raise


def create_moe_lora_hooks(transformer, router, adapter_names, accelerator_instance):
    """创建用于MoE训练的钩子函数"""
    def save_model_hook(models, weights, output_dir):
        if accelerator_instance.is_main_process:
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
    parser = argparse.ArgumentParser(description="使用LazyBucket数据集训练MoE-LoRA模型执行移除任务")
    
    # 数据参数
    parser.add_argument("--train_json_path", type=str, required=True, 
                        help="训练数据的JSON路径")
    parser.add_argument("--val_json_path", type=str, required=True, 
                        help="验证数据的JSON路径")
    parser.add_argument("--resolution", type=int, default=768, 
                        help="训练图像分辨率")
    parser.add_argument("--center_crop", action="store_true", default=False, 
                        help="是否中心裁剪图像")
    parser.add_argument("--random_flip", action="store_true", default=False, 
                        help="是否随机翻转图像")
    # Bucket 参数
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument("--train_metadata_file", type=str, default=None,
                        help="预计算的元数据文件路径，加速数据集加载")
    # 基本训练参数
    parser.add_argument("--output_dir", type=str, default="outputs/moe_removal", 
                        help="MoE模型保存路径")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子")
    parser.add_argument("--mixed_precision", type=str, default="fp16", 
                        choices=["no", "fp16", "bf16"], help="混合精度训练类型")
    parser.add_argument("--report_to", type=str, default="tensorboard", 
                        help="使用的跟踪器，可选 'tensorboard' 或 'wandb'")
    parser.add_argument("--logging_dir", type=str, default="logs", 
                        help="日志保存路径")
    parser.add_argument("--allow_tf32", action="store_true", default=False,
                        help="是否允许使用 TF32 格式，可加速训练")
    
    # 模型参数
    parser.add_argument("--flux_fill_id", type=str, default="stabilityai/flux-fill",
                        help="FluxFillPipeline模型的路径或Hugging Face的模型ID")
    parser.add_argument("--flux_redux_id", type=str, default="stabilityai/flux-prior-redux",
                        help="FluxPriorReduxPipeline模型的路径或ID")
    parser.add_argument("--gradient_checkpointing", action="store_true", dest="gradient_checkpointing", default=True,
                        help="启用梯度检查点以节省内存")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="关闭梯度检查点")
    
    # transformer模型配置
    parser.add_argument("--union_cond_attn", action="store_true", dest="union_cond_attn", default=True,
                        help="启用union_cond_attn")
    parser.add_argument("--add_cond_attn", action="store_true", dest="add_cond_attn", default=False,
                        help="启用add_cond_attn")
    parser.add_argument("--latent_lora", action="store_true", dest="latent_lora", default=False,
                        help="启用latent_lora")
    
    # LoRA参数
    parser.add_argument("--use_lora", type=bool, default=True,
                        help="启用LoRA微调")
    parser.add_argument("--lora_r", type=int, default=256, 
                        help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=256, 
                        help="LoRA的alpha参数")
    parser.add_argument("--init_lora_weights", type=str, default="gaussian", 
                        choices=["gaussian", "constant"], 
                        help="LoRA权重初始化方法")
    parser.add_argument("--lora_dropout", type=float, default=0.0, 
                        help="LoRA的dropout率")
    parser.add_argument("--lora_bias", type=str, default="none", 
                        choices=["none", "all", "lora_only"], 
                        help="LoRA偏置项训练策略")
    parser.add_argument("--target_modules", type=str, 
                        default="(.*x_embedder|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_v|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_out\.0|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.2|.*single_transformer_blocks\.[0-9]+\.norm\.linear|.*single_transformer_blocks\.[0-9]+\.proj_mlp|.*single_transformer_blocks\.[0-9]+\.proj_out|.*single_transformer_blocks\.[0-9]+\.attn.to_k|.*single_transformer_blocks\.[0-9]+\.attn.to_q|.*single_transformer_blocks\.[0-9]+\.attn.to_v|.*single_transformer_blocks\.[0-9]+\.attn.to_out)",
                        help="LoRA目标模块的正则表达式")
    
    # MoE特定参数
    parser.add_argument("--num_experts", type=int, default=3,
                        help="MoE专家数量（LoRA适配器数量）")
    parser.add_argument("--routing_strategy", type=str, default="soft",
                        choices=["soft", "hard", "topk"],
                        help="MoE路由策略")
    parser.add_argument("--routing_temperature", type=float, default=1.0,
                        help="路由softmax温度参数")
    parser.add_argument("--expert_names", type=str, default="removal_expert,background_expert,completion_expert",
                        help="专家适配器名称，逗号分隔")
    
    # 训练超参数
    parser.add_argument("--train_batch_size", type=int, default=4, 
                        help="训练批量大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1, 
                        help="学习率")
    parser.add_argument("--lr_scheduler", type=str, default="constant", 
                        help="学习率调度器类型")
    
    # 优化器
    parser.add_argument("--optimizer_type", type=str, default="adamw", 
                        help="优化器类型: adamw, adamw8bit, prodigy, muon 等")
    parser.add_argument("--use_8bit_optimizer", type=str, default="false",
                        help="使用8-bit优化器(bitsandbytes)以节省显存")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, 
                        help="学习率预热步数")
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
                        help="Adam优化器beta1参数")
    parser.add_argument("--adam_beta2", type=float, default=0.999, 
                        help="Adam优化器beta2参数")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, 
                        help="Adam优化器权重衰减")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, 
                        help="Adam优化器epsilon值")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, 
                        help="梯度裁剪最大范数")
    parser.add_argument("--optimizer_config", type=dict, default=None, 
                        help="优化器配置")
    
    # 训练进度参数
    parser.add_argument("--max_train_steps", type=int, default=None, 
                        help="最大训练步数")
    parser.add_argument("--num_train_epochs", type=int, default=100, 
                        help="训练轮数")
    parser.add_argument("--checkpointing_steps", type=int, default=500, 
                        help="检查点保存间隔步数")
    parser.add_argument("--validation_steps", type=int, default=500, 
                        help="验证间隔步数")
                        
    # Flow模型训练的权重参数
    parser.add_argument("--weighting_scheme", type=str, default="karras", 
                        choices=["uniform", "karras", "s2", "logsnr", "logsnr_sqrt", "v_prediction"],
                        help="时间步采样和损失加权的策略")
    parser.add_argument("--logit_mean", type=float, default=-1.0,
                        help="非均匀采样的logit均值")
    parser.add_argument("--logit_std", type=float, default=2.0,
                        help="非均匀采样的logit标准差")
    parser.add_argument("--mode_scale", type=float, default=1.0,
                        help="非均匀采样的模式缩放因子")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="条件引导的强度因子")
    parser.add_argument("--validation_epochs", type=int, default=1, 
                        help="验证间隔步数")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="从检查点恢复训练状态，包括优化器和学习率等")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="从预训练模型路径加载模型权重，但不恢复训练状态")
    
    # 推理和加速参数
    parser.add_argument("--dataloader_num_workers", type=int, default=4, 
                        help="数据加载线程数")
    parser.add_argument("--enable_xformers_memory_efficient_attention", 
                        action="store_true", default=False, 
                        help="是否启用xformers内存优化注意力机制")
    
    # custom 模式推理
    parser.add_argument("--inference_custom", action="store_true", default=False, 
                        help="是否启用custom模式推理")
    
    # 解析参数
    args = parser.parse_args()
    
    # 处理max_train_steps
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        args.max_train_steps = None
        
    # 处理transformer配置
    args.transformer_config_dict = {
        "union_cond_attn": args.union_cond_attn,
        "add_cond_attn": args.add_cond_attn,
        "latent_lora": args.latent_lora
    }
    
    # 创建优化器配置字典
    if args.optimizer_config is None:
        if args.optimizer_type.lower() == "prodigy":
            args.optimizer_config = {
                "type": "Prodigy",
                "params": {
                    "lr": args.learning_rate,
                    "weight_decay": args.adam_weight_decay,
                    "use_bias_correction": True,
                    "safeguard_warmup": True
                }
            }
        elif args.optimizer_type.lower() == "muon":
            args.optimizer_config = {
                "type": "Muon",
                "params": {
                    "lr": args.learning_rate,
                    "weight_decay": args.adam_weight_decay
                }
            }
        elif args.optimizer_type.lower() == "adamw8bit" or args.use_8bit_optimizer.lower() == "true":
            args.optimizer_config = {
                "type": "AdamW8bit",
                "use_8bit": True
            }
        else:  # adamw
            args.optimizer_config = {
                "type": "AdamW"
            }
    
    # 处理专家名称
    args.expert_names = [name.strip() for name in args.expert_names.split(",")]
    if len(args.expert_names) != args.num_experts:
        logger.warning(f"专家名称数量({len(args.expert_names)})与num_experts({args.num_experts})不匹配，将使用默认名称")
        args.expert_names = [f"expert{i+1}" for i in range(args.num_experts)]
    
    return args


def configure_optimizers_with_wd_groups(parameters, lr, weight_decay):
    """配置优化器参数组，对不同类型参数应用不同的权重衰减策略。
    
    参数分为三组:
    1. 需要标准权重衰减的参数
    2. 不需要权重衰减的参数 (bias, LayerNorm, embeddings)
    3. 路由器和特征提取器参数 (使用较小的权重衰减)
    
    Args:
        parameters: 命名参数列表 (name, param)
        lr: 学习率
        weight_decay: 标准权重衰减率
        
    Returns:
        优化器参数组列表
    """
    no_decay_patterns = ["bias", "LayerNorm.weight", "layernorm.weight", "norm", "embeddings"]
    router_patterns = ["router", "feature_extractor"]
    
    # 初始化参数组
    decay_params = []
    no_decay_params = []
    router_params = []
    
    # 对参数进行分类
    for name, param in parameters:
        if any(nd in name.lower() for nd in no_decay_patterns):
            no_decay_params.append(param)
        elif any(rt in name.lower() for rt in router_patterns):
            router_params.append(param)
        else:
            decay_params.append(param)
    
    # 创建参数组
    param_groups = [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        {"params": router_params, "lr": lr, "weight_decay": weight_decay * 0.1},  # 路由器参数使用较小的权重衰减
    ]
    
    # 记录统计信息
    logger.info(f"优化器参数组统计:")
    logger.info(f"  标准权重衰减组 ({weight_decay}): {len(decay_params):,} 参数")
    logger.info(f"  零权重衰减组 (0.0): {len(no_decay_params):,} 参数")
    logger.info(f"  路由器参数组 ({weight_decay * 0.1}): {len(router_params):,} 参数")
    
    return param_groups


def main():
    from accelerate.state import PartialState
    _ = PartialState()  # TODO
    
    args = parse_args()
    
    # 输出关键配置信息
    logger.info(f"已配置参数: output_dir={args.output_dir}, lr={args.learning_rate}, batch_size={args.train_batch_size}")
    logger.info(f"MoE配置: num_experts={args.num_experts}, routing_strategy={args.routing_strategy}")
    logger.info(f"LoRA配置: use_lora={args.use_lora}, r={args.lora_r}, alpha={args.lora_alpha}")
    
    # 创建日志和输出目录
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    # 初始化accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)
    
    # 配置日志级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # 如果在Ampere GPU上，允许TF32提高训练速度
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    
    # 加载预训练模型组件
    logger.info(f"Loading FluxFillPipeline model: {args.flux_fill_id}")
    flux_fill_pipe = FluxFillPipeline.from_pretrained(
        args.flux_fill_id,
        torch_dtype=weight_dtype
    )
    
    # 强制禁用xformers
    if hasattr(flux_fill_pipe, "enable_xformers_memory_efficient_attention"):
        flux_fill_pipe._use_memory_efficient_attention_xformers = False
        
    # 确保 VAE 使用正确的数据类型
    if hasattr(flux_fill_pipe, "vae"):
        flux_fill_pipe.vae = flux_fill_pipe.vae.to(dtype=flux_fill_pipe.dtype)
    
    flux_fill_pipe.to(accelerator.device)
    
    logger.info(f"Loading FluxPriorReduxPipeline model: {args.flux_redux_id}")
    flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(
        args.flux_redux_id,
        torch_dtype=weight_dtype
    )
    
    flux_redux_pipe.to(accelerator.device)
    flux_redux_pipe.image_embedder.requires_grad_(False).eval()
    flux_redux_pipe.image_encoder.requires_grad_(False).eval()
    
    vae = flux_fill_pipe.vae
    text_encoder = flux_fill_pipe.text_encoder
    text_encoder_2 = flux_fill_pipe.text_encoder_2
    transformer = flux_fill_pipe.transformer
    
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    text_encoder_2.requires_grad_(False).eval()
    transformer.requires_grad_(False)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.flux_fill_id, subfolder="scheduler", torch_dtype=weight_dtype
    )
    
    # 设置MoE LoRA
    if args.use_lora:
        logger.info("Setting up MoE-LoRA for training transformer")
        
        # LoRA配置
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
        
        lora_config_dict = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "init_lora_weights": args.init_lora_weights,
            "target_modules": target_modules,
            "lora_dropout": args.lora_dropout,
            "bias": args.lora_bias
        }
        
        logger.info(f"创建MoE LoRA管理器: {args.num_experts}个专家, 路由策略: {args.routing_strategy}")
        
        moe_manager = MoELoRAManager(
            model=transformer,
            adapter_names=args.expert_names,
            lora_configs=[lora_config_dict] * args.num_experts,
            initial_weights=None
        )
        
        logger.info(f"简化MoE LoRA设置完成: {args.num_experts}个专家, 初始权重: {moe_manager.get_current_weights()}")
    else:
        moe_manager = None
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    if args.use_lora and moe_manager is not None:
        save_model_hook, load_model_hook = create_moe_lora_hooks(
            transformer, moe_manager.router, args.expert_names, accelerator
        )
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    if args.use_lora:
        lora_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        router_params = list(moe_manager.router.parameters()) if moe_manager is not None else []
        trainable_params = lora_params + router_params
        
        named_trainable_params = []
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                named_trainable_params.append((name, param))
        router_param_count = 0
        if moe_manager is not None and hasattr(moe_manager, 'router'):
            for name, param in moe_manager.router.named_parameters():
                if param.requires_grad:
                    named_trainable_params.append((f"moe_router.{name}", param))
                    router_param_count += param.numel()
                
        logger.info(f"MoE-LoRA模式 - LoRA参数: {sum(p.numel() for n, p in named_trainable_params if 'lora' in n.lower())}, 路由器参数: {router_param_count}")
    else:
        transformer.requires_grad_(True)
        trainable_params = transformer.parameters()
        named_trainable_params = list(transformer.named_parameters())
        logger.info("全量微调模式 - 训练所有transformer参数")
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    optimizer_type = args.optimizer_config.get("type", "AdamW")
    optimizer_params = args.optimizer_config.get("params", {})
    use_8bit = args.optimizer_config.get("use_8bit", False)

    grouped_parameters = configure_optimizers_with_wd_groups(
        parameters=named_trainable_params,
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay
    )
    
    if optimizer_type.lower() == "adamw8bit" or use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                grouped_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon
            )
            logger.info(f"使用8-bit AdamW优化器，节省显存约50%")
        except ImportError:
            logger.warning("无法导入bitsandbytes，请安装: pip install bitsandbytes")
            logger.warning("回退到标准AdamW优化器")
            optimizer = torch.optim.AdamW(
                grouped_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon
            )
    elif optimizer_type.lower() == "prodigy":
        try:
            from prodigyopt import Prodigy
            optimizer = Prodigy(
                grouped_parameters,  # 使用分组参数
                lr=optimizer_params.get("lr", 1),
                use_bias_correction=optimizer_params.get("use_bias_correction", True),
                safeguard_warmup=optimizer_params.get("safeguard_warmup", True)
            )
            logger.info(f"使用Prodigy优化器，参数: {optimizer_params}")
        except ImportError:
            logger.warning("无法导入Prodigy优化器，回退到AdamW")
            optimizer = torch.optim.AdamW(
                grouped_parameters,  # 使用分组参数
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon
            )
    elif optimizer_type.lower() == "muon":
        try:
            # 尝试导入和使用MUON优化器
            optimizer = MuonWithAuxAdam(
                grouped_parameters,  # 使用分组参数
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                # 权重衰减在分组参数中设置
            )
            logger.info(f"使用Muon优化器")
        except ImportError:
            logger.warning("无法导入Muon优化器，回退到AdamW")
            optimizer = torch.optim.AdamW(
                grouped_parameters,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon
            )
    else:
        optimizer = torch.optim.AdamW(
            grouped_parameters,  # 使用分组参数
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon
        )
        logger.info(f"使用AdamW优化器(带权重衰减区分)")
        
    if accelerator.is_main_process:
        total_params = sum(p.numel() for group in grouped_parameters for p in group['params'] if p.requires_grad)
        logger.info(f"参与训练的总参数数量: {total_params:,}")
        if args.use_lora and moe_manager is not None:
            if moe_manager.router is not None:
                router_params = sum(p.numel() for p in moe_manager.router.parameters())
                logger.info(f"MoE路由器参数数量: {router_params:,}")
            else:
                logger.info("MoE路由器将在第一次前向传播时懒加载初始化")
    
    buckets = parse_buckets_string(args.aspect_ratio_buckets)
    
    logger.info("加载训练数据集...")
    try:
        train_dataset = TripletBucketDataset(json_path=args.train_json_path, 
                                        buckets=buckets,
                                        metadata_file=args.train_metadata_file,
                                        custom=args.inference_custom)
        
        if len(train_dataset) == 0:
            logger.error("训练数据集为空，请检查数据路径和文件格式")
            raise ValueError("训练数据集为空")
    except Exception as e:
        logger.error(f"加载训练数据集时发生错误: {str(e)}")
        raise
    
    logger.info(f"加载了 {len(train_dataset)} 个训练样本")
    
    batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=triplet_collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    if moe_manager is not None:
        moe_manager.router = accelerator.prepare(moe_manager.router)
    
    # 重新计算 num_update_steps_per_epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    if accelerator.is_main_process:
        accelerator.init_trackers("moe_removal_training")
    
    # 定义get_sigmas函数来计算对应的sigma值
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        """Calculate sigma values for the given timesteps.
        
        Args:
            timesteps: The timestep values to find sigmas for
            n_dim: The number of dimensions for the output tensor
            dtype: The dtype of the output tensor
            
        Returns:
            The sigma values shaped appropriately for multiplying with model inputs
        """
        # 获取噪声调度器的sigmas和时间步
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        
        # 将时间步映射到噪声调度器的索引
        # 直接使用线性映射，避免精确匹配
        # 将[0,1]范围的timesteps映射到sigmas的索引范围
        step_indices = []
        for t in timesteps:
            # 使用线性插值映射
            idx = t.item() * (len(sigmas) - 1)
            idx = max(0, min(len(sigmas) - 1, int(round(idx))))
            step_indices.append(idx)
        
        
        # 获取相应的sigma值并展平
        sigma = sigmas[step_indices].flatten()
        
        # 调整形状以适配图像输入
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
            
        return sigma
    
    def get_sigmas_kontext(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        accelerator.print(f"从检查点恢复训练: {path}")
        accelerator.load_state(path)
        try:
            global_step = int(os.path.basename(path).split("-")[-1])
            logger.info(f"恢复训练到步骤 {global_step}")
            first_epoch = global_step // num_update_steps_per_epoch
            initial_global_step = global_step
        except ValueError:
            logger.warning(f"无法从路径获取步骤信息，从步骤0开始")
            global_step = 0
            first_epoch = 0
            initial_global_step = 0
    else:
        initial_global_step = 0
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps"
    )
    
    logger.info("***** 开始MoE训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  每设备批量大小 = {args.train_batch_size}")
    logger.info(f"  总批量大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    logger.info(f"  MoE专家数量 = {args.num_experts}")
    logger.info(f"  路由策略 = {args.routing_strategy}")
    
    # MoE训练循环
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if moe_manager is not None:
            moe_manager.router.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                loss = 0.0
                ref = batch["ref"].to(accelerator.device, dtype=weight_dtype)  # 参考图像
                src = batch["src"].to(accelerator.device, dtype=weight_dtype)  # 源图像
                mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)  # mask
                imgs = batch["result"].to(accelerator.device, dtype=weight_dtype)  # 结果图像
                
                # 编码图像信息 - 使用ref图像生成prompt embeddings
                prompt_embeds = []
                pooled_prompt_embeds = []
                
                for i in range(ref.shape[0]):
                    image_tensor = ref[i].cpu()
                    # 转换为float32，因为numpy不支持BFloat16
                    image_tensor = image_tensor.to(torch.float32)
                    image_tensor = image_tensor.permute(1, 2, 0)
                    image_numpy = image_tensor.numpy()
                    pil_image = Image.fromarray((image_numpy * 255).astype('uint8'))
                    
                    with torch.no_grad():
                        # 使用image_output函数编码图像
                        prompt_embed, pooled_prompt_embed = image_output(flux_redux_pipe, pil_image, device=accelerator.device)
                        prompt_embeds.append(prompt_embed.squeeze(1))
                        pooled_prompt_embeds.append(pooled_prompt_embed.squeeze(1))
                
                prompt_embeds = torch.cat(prompt_embeds, dim=0)
                pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0)
                
                # 准备文本输入,这里是object图的embedding(通过flux_redux得到传入flux_fill_pipe)
                prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                    flux_fill_pipe, prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds
                )

                with torch.no_grad():
                    x_0, img_ids = encode_images(flux_fill_pipe, imgs)
                    x_0 = x_0.to(device=accelerator.device, dtype=weight_dtype)
                    img_ids = img_ids.to(device=accelerator.device)
                    
                    # 使用compute_density_for_timestep_sampling进行优化的时间步采样
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=imgs.shape[0],
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    # 转换为时间步索引，限制在合理范围内
                    indices = torch.clamp((u * noise_scheduler.config.num_train_timesteps).long(), 
                                         0, noise_scheduler.config.num_train_timesteps - 1)
                    timesteps = indices.clone().detach().float().to(accelerator.device) / noise_scheduler.config.num_train_timesteps
                    t = timesteps
                    
                    x_1 = torch.randn_like(x_0, device=accelerator.device, dtype=weight_dtype)
                    t_ = t.unsqueeze(1).unsqueeze(1).to(device=accelerator.device, dtype=weight_dtype)  # 扩展维度并确保设备一致
                    x_t = ((1 - t_) * x_0 + t_ * x_1)
                    
                    # 计算sigmas用于后续损失加权
                    # 使用与Flux模型一致的get_sigmas函数
                    sigmas = get_sigmas(timesteps=t, dtype=weight_dtype)
                    
                    src_latents, mask_latents = Flux_fill_encode_masks_images(flux_fill_pipe, src, mask)
                    condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

                    guidance = None
                    if transformer.config.guidance_embeds:
                        guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                        guidance = guidance.expand(t.shape[0])
                
                if args.use_lora and moe_manager is not None:
                    moe_manager.set_adapter_weights(moe_manager.current_weights)
                transformer_out = tranformer_forward(
                    transformer,
                    model_config=args.transformer_config_dict if hasattr(args, 'transformer_config_dict') else None,
                    hidden_states=torch.cat((x_t, condition_latents), dim=2),
                    timestep=t,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )

                pred = transformer_out[0]
                        
                # 对于Flow模型，直接使用(x_1 - x_0)作为目标
                # 使用compute_loss_weighting_for_sd3优化损失计算
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                # 保持与模型相同的dtype，避免在混合精度训练中出现dtype不匹配
                target = (x_1 - x_0)
                
                loss_dict = compute_simple_mse_loss(
                    pred=pred, 
                    target=target, 
                    weighting=weighting
                )
                
                loss = loss_dict['total_loss']
                
                # 记录各组件损失（用于监控和调试）
                if accelerator.is_main_process and global_step % 500 == 0:
                    component_losses = {
                        f"loss/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                        for k, v in loss_dict.items()
                    }
                    accelerator.log(component_losses, step=global_step)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients and args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                    
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
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"已保存检查点到 {save_path}")
                
                # debug 用，快速验证validation 
                if accelerator.is_main_process:
                    if global_step > 1 and global_step % args.validation_steps == 0:
                        logger.info("Running validation...")
                        try:
                            if args.val_json_path:
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
                            else:
                                logger.warning("val_json_path not provided, skipping validation.")
                        except Exception as e:
                            logger.error(f"Inference failed with error: {e}")
                            traceback.print_exc()
                            logger.info("Inference failed, but training will continue.")

                if global_step >= args.max_train_steps:
                    break
        
        if accelerator.is_main_process:
            if epoch > 0 and epoch % args.validation_epochs == 0:
                logger.info("Running validation...")
                try:
                    if args.val_json_path:
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
                    else:
                        logger.warning("val_json_path not provided, skipping validation.")
                except Exception as e:
                    logger.error(f"Inference failed with error: {e}")
                    traceback.print_exc()
                    logger.info("Inference failed, but training will continue.")

        accelerator.log({"epoch_loss": train_loss / len(train_dataloader)}, step=global_step)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_output_dir = os.path.join(args.output_dir, "final_moe")
        if args.use_lora:
            save_moe_lora_state(
                final_output_dir,
                unwrap_model(transformer),
                router,
                args.expert_names
            )
            logger.info(f"MoE LoRA模型已保存到 {final_output_dir}")
        else:
            transformer.save_pretrained(final_output_dir)
            logger.info(f"模型已保存到 {final_output_dir}")
    
    accelerator.end_training()
    logger.info("MoE训练完成")


if __name__ == "__main__":
    main()
