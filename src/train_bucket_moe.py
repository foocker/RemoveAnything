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
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2.0)
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
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


def compute_removal_task_loss(pred, target, mask_info=None, weighting=None, loss_config=None, vae=None, vae_scale_factor=8):
    """
    为Remove任务设计的专门损失函数。
    
    结合了以下组件：
    1. 基础MSE损失 - 保证整体重建质量（在latent空间）
    2. 边界平滑损失 - 确保边缘自然平滑（在像素空间）
    3. 外部区域一致性损失 - 保持未遮挡区域不变（在像素空间）
    4. 高频细节保持损失 - 保持重要细节（在latent空间）
    
    Args:
        pred: 预测的latent，形状 [batch_size, channels, height, width]
        target: 目标latent，形状 [batch_size, channels, height, width] 
        mask_info: mask相关信息，包含mask、边界等
        weighting: 基础权重，用于时间步加权
        loss_config: 损失配置参数
        vae: VAE模型，用于将latent解码为像素空间（如果需要像素空间损失）
        vae_scale_factor: VAE缩放因子
        
    Returns:
        dict: 包含各组件损失的字典
    """
    
    def _decode_latents_to_pixels(latents, vae, vae_scale_factor):
        """将latent张量解码为像素空间的图像
        
        Args:
            latents: latent张量，Flux格式 [batch, seq_len, hidden_dim] (3D)
            vae: VAE解码器
            vae_scale_factor: VAE缩放因子
        
        Returns:
            image: 像素空间图像 [batch, channels, height, width]
        """
        if vae is None:
            return None
            
        try:
            with torch.no_grad():
                # 检查输入格式
                if latents.dim() != 3:
                    logger.warning(f"预期3维latent输入 [batch, seq_len, hidden], 但得到 {latents.shape}")
                    return None
                
                batch_size, num_patches, channels = latents.shape
                logger.info(f"[DECODE] 输入latents形状: batch={batch_size}, patches={num_patches}, channels={channels}")
                
                # 推测原始图像尺寸（基于典型的训练尺寸）
                # 根据num_patches推算height和width
                # Flux通常使用patch size为2x2，所以latent_height * latent_width = num_patches
                import math
                side_len = int(math.sqrt(num_patches))
                logger.info(f"[DECODE] 计算的side_len: {side_len}, side_len^2: {side_len*side_len}")
                
                if side_len * side_len != num_patches:
                    # 如果不是完全平方数，尝试常见的宽高比
                    logger.info(f"[DECODE] num_patches={num_patches}不是完全平方数，尝试推断宽高比")
                    if num_patches == 2048:
                        # 2048 = 32 * 64, 对应512x1024图像 (h,2*w约定)
                        latent_height, latent_width = 32, 64
                        logger.info(f"[DECODE] 推断为2048: {latent_height}x{latent_width} -> 512x1024图像")
                    elif num_patches == 1024:
                        # 1024 = 32 * 32, 对应512x512图像
                        latent_height, latent_width = 32, 32
                        logger.info(f"[DECODE] 推断为1024: {latent_height}x{latent_width} -> 512x512图像")
                    elif num_patches == 512:
                        # 512 = 16 * 32, 对应256x512图像
                        latent_height, latent_width = 16, 32
                        logger.info(f"[DECODE] 推断为512: {latent_height}x{latent_width} -> 256x512图像")
                    else:
                        logger.warning(f"无法推断latent尺寸，num_patches={num_patches}")
                        return None
                else:
                    latent_height = latent_width = side_len
                    logger.info(f"[DECODE] 完全平方数: {latent_height}x{latent_width}")
                
                # 计算原始图像尺寸（用于unpack）
                # 注意: Flux使用h, 2*w的约定，所以width需要*2
                height = latent_height * vae_scale_factor * 2  # *2是因为Flux的packing
                width = latent_width * vae_scale_factor * 2   # 注意：这里已经是2*w
                logger.info(f"[DECODE] 计算的原始图像尺寸: {height}x{width} (vae_scale_factor={vae_scale_factor})")
                logger.info(f"[DECODE] 注意: Flux使用h,2*w的约定")
                
                # 实现Flux的_unpack_latents逻辑将3D packed格式转为4D标准格式
                # 避免导入问题，直接实现逻辑
                def _unpack_latents_local(latents, height, width, vae_scale_factor):
                    batch_size, num_patches, channels = latents.shape
                    logger.info(f"[UNPACK] 输入: batch_size={batch_size}, num_patches={num_patches}, channels={channels}")
                    logger.info(f"[UNPACK] 目标尺寸: height={height}, width={width}, vae_scale_factor={vae_scale_factor}")
                    
                    # VAE applies 8x compression on images but we must also account for packing
                    # 确保尺寸计算正确
                    height = 2 * (int(height) // (vae_scale_factor * 2))
                    width = 2 * (int(width) // (vae_scale_factor * 2))
                    logger.info(f"[UNPACK] 调整后的尺寸: height={height}, width={width}")
                    
                    # 验证尺寸一致性
                    expected_patches = (height // 2) * (width // 2)
                    logger.info(f"[UNPACK] 验证: expected_patches={(height // 2) * (width // 2)}, actual_patches={num_patches}")
                    
                    try:
                        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
                        logger.info(f"[UNPACK] view后的形状: {latents.shape}")
                        
                        latents = latents.permute(0, 3, 1, 4, 2, 5)
                        logger.info(f"[UNPACK] permute后的形状: {latents.shape}")
                        
                        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
                        logger.info(f"[UNPACK] 最终reshape后的形状: {latents.shape}")
                        
                        return latents
                    except Exception as e:
                        logger.error(f"[UNPACK] 错误: {e}")
                        logger.error(f"[UNPACK] 尝试的参数: height={height}, width={width}, channels={channels}")
                        raise
                
                latents_4d = _unpack_latents_local(latents, height, width, vae_scale_factor)
                logger.info(f"[DECODE] unpack后的latents_4d.shape: {latents_4d.shape}")
                
                # 应用VAE的scaling和shift因子 (关键步骤!)
                # 参考fillpipe: latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                if hasattr(vae.config, 'scaling_factor'):
                    scaling_factor = vae.config.scaling_factor
                    logger.info(f"[DECODE] 使用VAE scaling_factor: {scaling_factor}")
                else:
                    scaling_factor = 0.3611  # Flux VAE默认值
                    logger.info(f"[DECODE] 使用默认scaling_factor: {scaling_factor}")
                    
                if hasattr(vae.config, 'shift_factor'):
                    shift_factor = vae.config.shift_factor
                    logger.info(f"[DECODE] 使用VAE shift_factor: {shift_factor}")
                else:
                    shift_factor = 0.1159  # Flux VAE默认值
                    logger.info(f"[DECODE] 使用默认shift_factor: {shift_factor}")
                
                # 应用scaling和shift
                latents_4d = (latents_4d / scaling_factor) + shift_factor
                logger.info(f"[DECODE] scaling和shift后的latents_4d范围: [{latents_4d.min():.4f}, {latents_4d.max():.4f}]")
                
                # 确保数据类型与VAE模型一致 (修复dtype不匹配问题)
                # 检查VAE的参数数据类型
                vae_dtype = next(vae.parameters()).dtype
                logger.info(f"[DECODE] VAE模型数据类型: {vae_dtype}, latents_4d数据类型: {latents_4d.dtype}")
                
                if latents_4d.dtype != vae_dtype:
                    logger.info(f"[DECODE] 转换latents_4d数据类型: {latents_4d.dtype} -> {vae_dtype}")
                    latents_4d = latents_4d.to(vae_dtype)
                
                # VAE解码到像素空间
                image = vae.decode(latents_4d, return_dict=False)[0]
                logger.info(f"[DECODE] 解码后的image.shape: {image.shape}")
                
                return image
                
        except Exception as e:
            logger.warning(f"Latent解码失败: {str(e)}, 返回None")
            return None
    device = pred.device
    dtype = pred.dtype
    
    # 默认损失配置
    if loss_config is None:
        loss_config = {
            'base_weight': 1.0,        # 基础MSE损失权重
            'boundary_weight': 0.3,    # 边界平滑损失权重 
            'consistency_weight': 0.5, # 外部区域一致性损失权重
            'detail_weight': 0.2       # 高频细节保持损失权重
        }
    
    losses = {}
    
    # 1. 基础MSE损失
    base_loss = F.mse_loss(pred.float(), target.float(), reduction="none")
    
    # 应用时间步加权
    if weighting is not None:
        # 确保权重形状匹配
        while len(weighting.shape) < len(base_loss.shape):
            weighting = weighting.unsqueeze(-1)
        base_loss = base_loss * weighting
    
    losses['base_loss'] = base_loss.mean() * loss_config['base_weight']
    
    # 如果没有mask信息，只返回基础损失
    if mask_info is None:
        losses['total_loss'] = losses['base_loss']
        return losses
    
    # DEBUG: 打印所有输入tensor的shape信息
    logger.info(f"=== 损失函数调试信息 ===")
    logger.info(f"pred.shape: {pred.shape}")
    logger.info(f"target.shape: {target.shape}")
    if mask_info and 'mask_latent' in mask_info:
        logger.info(f"mask_latent.shape: {mask_info['mask_latent'].shape}")
    logger.info(f"loss_config: {loss_config}")
    logger.info(f"vae available: {vae is not None}")
    
    # 需要像素空间损失时，解码latent为像素
    pred_pixels = None
    target_pixels = None
    if vae is not None and ((loss_config['boundary_weight'] > 0) or (loss_config['consistency_weight'] > 0)):
        logger.info("[DECODE] 解码latent为像素空间...")
        pred_pixels = _decode_latents_to_pixels(pred, vae, vae_scale_factor)
        target_pixels = _decode_latents_to_pixels(target, vae, vae_scale_factor)
        if pred_pixels is not None:
            logger.info(f"[DECODE] pred_pixels.shape: {pred_pixels.shape}")
        if target_pixels is not None:
            logger.info(f"[DECODE] target_pixels.shape: {target_pixels.shape}")
    
    try:
        # 2. 边界平滑损失 - 在像素空间保证边缘过渡自然性
        if 'mask' in mask_info and loss_config['boundary_weight'] > 0 and pred_pixels is not None and target_pixels is not None:
            try:
                # 使用原始mask（像素空间）
                mask = mask_info['mask'].to(device, dtype=pred_pixels.dtype)
                logger.info(f"[BOUNDARY] 原始mask.shape: {mask.shape}")
                logger.info(f"[BOUNDARY] pred_pixels.shape: {pred_pixels.shape}")
                
                # 确保mask与像素图像尺寸匹配
                if mask.shape[2:] != pred_pixels.shape[2:]:
                    logger.info(f"[BOUNDARY] 调整mask尺寸: {mask.shape[2:]} -> {pred_pixels.shape[2:]}")
                    mask = F.interpolate(mask, size=pred_pixels.shape[2:], mode='nearest')
                    logger.info(f"[BOUNDARY] 调整后mask.shape: {mask.shape}")
                
                # 在像素空间计算梯度用于检测边缘
                if pred_pixels.shape[2] > 1 and pred_pixels.shape[3] > 1:
                    pred_grad_x = pred_pixels[:, :, :, 1:] - pred_pixels[:, :, :, :-1]
                    pred_grad_y = pred_pixels[:, :, 1:, :] - pred_pixels[:, :, :-1, :]
                    target_grad_x = target_pixels[:, :, :, 1:] - target_pixels[:, :, :, :-1]
                    target_grad_y = target_pixels[:, :, 1:, :] - target_pixels[:, :, :-1, :]
                    
                    # 调整mask尺寸以匹配梯度
                    mask_x = mask[:, :, :, 1:]
                    mask_y = mask[:, :, 1:, :]
                    
                    # 边界平滑损失 - 只在mask边界附近计算
                    boundary_loss_x = F.mse_loss(pred_grad_x * mask_x, target_grad_x * mask_x, reduction="mean")
                    boundary_loss_y = F.mse_loss(pred_grad_y * mask_y, target_grad_y * mask_y, reduction="mean")
                    
                    losses['boundary_loss'] = (boundary_loss_x + boundary_loss_y) * loss_config['boundary_weight']
                    logger.info(f"[BOUNDARY] boundary_loss: {losses['boundary_loss'].item():.6f}")
                else:
                    losses['boundary_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
            except Exception as e:
                logger.warning(f"Boundary loss computation failed: {e}")
                losses['boundary_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            losses['boundary_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
            
        # 3. 外部区域一致性损失 - 在像素空间保持未遮挡区域不变
        if 'mask' in mask_info and loss_config['consistency_weight'] > 0 and pred_pixels is not None and target_pixels is not None:
            try:
                mask = mask_info['mask'].to(device, dtype=pred_pixels.dtype)
                
                # 确保mask维度与像素图像匹配
                if mask.shape[2:] != pred_pixels.shape[2:]:
                    mask = F.interpolate(mask, size=pred_pixels.shape[2:], mode='nearest')
                
                # 外部区域 (1 - mask) - 在像素空间计算
                external_mask = 1.0 - mask
                
                # 在外部区域计算一致性损失 - 使用像素空间的pred和target
                external_loss = F.mse_loss(pred_pixels * external_mask, target_pixels * external_mask, reduction="mean")
                losses['consistency_loss'] = external_loss * loss_config['consistency_weight']
            except Exception as e:
                logger.warning(f"Consistency loss computation failed: {e}")
                losses['consistency_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            losses['consistency_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
            
        # 4. 高频细节保持损失 - 在像素空间保持重要细节
        if loss_config['detail_weight'] > 0:
            # 必须在像素空间计算，因为latent是3D格式无法直接使用2D卷积
            if pred_pixels is not None and target_pixels is not None:
                logger.info(f"[DETAIL] 在像素空间计算高频细节损失")
                logger.info(f"[DETAIL] pred_pixels.shape: {pred_pixels.shape}, target_pixels.shape: {target_pixels.shape}")
                
                try:
                    # 使用Laplacian算子检测高频细节
                    laplacian_kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                                                   device=device, dtype=pred_pixels.dtype)
                    # 为每个通道创建卷积核
                    laplacian_kernel = laplacian_kernel.repeat(pred_pixels.shape[1], 1, 1, 1)
                    logger.info(f"[DETAIL] laplacian_kernel.shape: {laplacian_kernel.shape}")
                    
                    # 在像素空间应用Laplacian算子
                    pred_detail = F.conv2d(pred_pixels, laplacian_kernel, padding=1, groups=pred_pixels.shape[1])
                    target_detail = F.conv2d(target_pixels, laplacian_kernel, padding=1, groups=target_pixels.shape[1])
                    
                    detail_loss = F.mse_loss(pred_detail, target_detail, reduction="mean")
                    losses['detail_loss'] = detail_loss * loss_config['detail_weight']
                    logger.info(f"[DETAIL] detail_loss: {losses['detail_loss'].item():.6f}")
                except Exception as e:
                    logger.warning(f"Detail loss computation failed: {e}")
                    losses['detail_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
            else:
                logger.info(f"[DETAIL] 跳过细节损失：需要VAE解码但pred_pixels/target_pixels为None")
                losses['detail_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            losses['detail_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
            
    except Exception as e:
        # 如果额外损失计算失败，回退到基础损失
        logger.warning(f"Additional loss computation failed: {e}, falling back to base loss only")
        losses['boundary_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
        losses['consistency_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
        losses['detail_loss'] = torch.tensor(0.0, device=device, dtype=dtype)
    
    # 计算总损失
    losses['total_loss'] = (losses['base_loss'] + 
                           losses['boundary_loss'] + 
                           losses['consistency_loss'] + 
                           losses['detail_loss'])
    
    return losses


def extract_mask_info_from_batch(batch, device, dtype):
    """
    从batch中提取mask相关信息，用于损失计算。
    
    Args:
        batch: 训练batch
        device: 设备
        dtype: 数据类型
        
    Returns:
        dict: mask相关信息
    """
    mask_info = {}
    
    try:
        if "mask" in batch:
            mask = batch["mask"].to(device, dtype=dtype)
            
            # 如果是3通道mask，转换为单通道
            if mask.dim() == 4 and mask.shape[1] == 3:
                # 取第一个通道或转换为灰度
                mask = mask[:, 0:1, :, :]
            elif mask.dim() == 3:
                # 添加通道维度
                mask = mask.unsqueeze(1)
            
            # 确保mask值在[0,1]范围内
            mask = torch.clamp(mask, 0, 1)
            
            # 将mask编码到latent空间
            with torch.no_grad():
                # 简单的下采样到latent尺寸
                mask_latent = F.interpolate(mask, scale_factor=1/8, mode='nearest')
                mask_info['mask_latent'] = mask_latent
                
                # 计算mask面积比例
                mask_area = mask.sum(dim=[2, 3]) / (mask.shape[2] * mask.shape[3])
                mask_info['mask_area_ratio'] = mask_area
                
                # 计算边缘复杂度 - 使用Sobel算子
                sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                     device=device, dtype=dtype).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                     device=device, dtype=dtype).view(1, 1, 3, 3)
                
                edge_x = F.conv2d(mask, sobel_x, padding=1)
                edge_y = F.conv2d(mask, sobel_y, padding=1)
                edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
                
                # 平均边缘强度作为复杂度指标
                edge_complexity = edge_magnitude.mean(dim=[2, 3])
                mask_info['edge_complexity'] = edge_complexity
                
                # 简单的遮挡检测 - 如果mask中有孔洞或复杂形状
                # 这里简化处理，实际中可以使用更复杂的检测算法
                has_occlusion = (edge_complexity > 0.1).any(dim=1)  # 简单阈值判断
                mask_info['has_occlusion'] = has_occlusion
                
    except Exception as e:
        logger.warning(f"Failed to extract mask info: {e}")
        # 返回空的mask_info
        pass
    
    return mask_info


class MaskAwareFeatureExtractor(nn.Module):
    """针对Remove任务优化的mask感知特征提取模块。
    
    专门为三专家架构设计，减少对Redux text embedding的依赖，
    专注于视觉特征驱动的路由决策。
    
    支持多种特征提取模式：
    - mask_focused: 专注于mask区域的特征提取
    - boundary_aware: 边缘感知的特征提取  
    - content_adaptive: 内容自适应的特征提取
    """
    def __init__(self, feature_mode="mask_focused"):
        super().__init__()
        self.feature_mode = feature_mode  # "mask_focused", "boundary_aware", "content_adaptive"
        self.key = None  # 将在第一次前向传播时初始化
        self.projection = None  # 将在第一次前向传播时初始化
        
        # mask处理相关组件
        self.mask_processor = None
        self.boundary_detector = None
    
    def forward(self, x, mask_info=None):
        """前向传播，提取用于Remove任务路由的特征
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_dim] 或 [batch_size, hidden_dim]
            mask_info: mask相关信息，包含mask区域位置、边界信息等
            
        Returns:
            features: 提取的特征，形状为 [batch_size, output_dim]
        """
        # 延迟初始化网络层 - 针对Remove任务优化
        if self.projection is None:
            # 确定输入维度
            input_dim = x.shape[-1]
            # 设置输出维度为512，提供更丰富的特征表示
            output_dim = 512
            
            # 初始化投影层 - 使用更复杂的网络
            self.projection = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, output_dim),
                nn.LayerNorm(output_dim)
            ).to(x.device, x.dtype)
            
            # 针对不同模式初始化相关组件
            if self.feature_mode == "mask_focused":
                # mask处理器
                self.mask_processor = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1), 
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(8)
                ).to(x.device, x.dtype)
                
            elif self.feature_mode == "boundary_aware":
                # 边界检测器
                self.boundary_detector = nn.Sequential(
                    nn.Conv2d(4, 32, 3, padding=1),  # RGB + mask
                    nn.ReLU(),
                    nn.Conv2d(32, 16, 3, padding=1),
                    nn.ReLU(), 
                    nn.AdaptiveAvgPool2d(4)
                ).to(x.device, x.dtype)
                
            elif self.feature_mode == "content_adaptive":
                # 内容自适应注意力
                seq_len = x.shape[1] if len(x.shape) == 3 else 1
                self.key = nn.Linear(input_dim, 1).to(x.device, x.dtype)
        
        # 处理不同的输入形状 - Remove任务优化
        if len(x.shape) == 2:  # [batch_size, hidden_dim]
            # 对于已经池化的特征，直接投影
            base_features = self.projection(x)
            
            # 如果有mask信息，尝试融合
            if mask_info is not None and self.mask_processor is not None:
                try:
                    if 'mask' in mask_info:
                        mask_features = self.mask_processor(mask_info['mask'])
                        mask_features = mask_features.flatten(1)
                        # 将mask特征与基础特征连接
                        return torch.cat([base_features, mask_features], dim=1)
                except:
                    pass  # 如果mask处理失败，回退到基础特征
            
            return base_features
        
        # 对于序列输入 [batch_size, seq_len, hidden_dim] - Remove任务优化
        if self.feature_mode == "mask_focused":
            # 专注于mask相关的token位置
            seq_len = x.shape[1]
            if seq_len >= 7:
                # 选择更多与mask相关的关键位置
                key_positions = [0, seq_len//8, seq_len//4, seq_len//2, 
                               3*seq_len//4, 7*seq_len//8, seq_len-1]
                key_tokens = torch.stack([x[:, i, :] for i in key_positions], dim=1)
                pooled = key_tokens.mean(dim=1)
            else:
                pooled = x.mean(dim=1)
                
        elif self.feature_mode == "boundary_aware":
            # 边缘感知的特征提取
            seq_len = x.shape[1]
            if seq_len >= 6:
                boundary_positions = [0, seq_len//6, seq_len//3, seq_len//2,
                                    2*seq_len//3, 5*seq_len//6, seq_len-1]
                boundary_tokens = torch.stack([x[:, i, :] for i in boundary_positions], dim=1)
                pooled = boundary_tokens.mean(dim=1)
            else:
                pooled = x.mean(dim=1)
                
        elif self.feature_mode == "content_adaptive":
            # 内容自适应的注意力机制
            if self.key is not None:
                attention_weights = F.softmax(self.key(x).squeeze(-1), dim=1)
                pooled = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
            else:
                pooled = x.mean(dim=1)
        else:
            # 默认模式
            pooled = x.mean(dim=1)
        
        # 基础特征投影
        base_features = self.projection(pooled)
        
        # 融合mask相关特征
        if mask_info is not None:
            additional_features = []
            
            # mask特征处理
            if (self.feature_mode == "mask_focused" and 
                'mask' in mask_info and self.mask_processor is not None):
                try:
                    mask_features = self.mask_processor(mask_info['mask'])
                    mask_features = mask_features.flatten(1)
                    additional_features.append(mask_features)
                except:
                    pass
                    
            # 边界特征处理
            if (self.feature_mode == "boundary_aware" and 
                'boundary' in mask_info and self.boundary_detector is not None):
                try:
                    boundary_features = self.boundary_detector(mask_info['boundary'])
                    boundary_features = boundary_features.flatten(1)
                    additional_features.append(boundary_features)
                except:
                    pass
            
            # 将所有特征连接
            if additional_features:
                all_features = [base_features] + additional_features
                return torch.cat(all_features, dim=1)
        
        return base_features


class ThreeExpertRemovalRouter(nn.Module):
    """三专家Remove路由器，专门为RemoveAnything任务设计。
    
    三个专家分别专注于：
    1. 消除专家 (removal_expert): 强化消除能力，彻底去除目标对象
    2. 背景生成专家 (background_expert): 强化背景生成，创造自然连贯的背景
    3. 物体补全专家 (completion_expert): 强化被遮挡物体的补全能力
    
    路由策略针对Remove任务优化，减少对Redux text embedding的依赖。
    """
    def __init__(self, num_experts=3, routing_strategy="soft", temperature=1.0):
        super().__init__()
        # 强制使用3个专家
        self.num_experts = 3  
        self.expert_names = ["removal_expert", "background_expert", "completion_expert"]
        self.routing_strategy = routing_strategy  # 'soft', 'hard', or 'topk'
        self.temperature = temperature  # 控制softmax的锋度
        self.top_k = 2  # 如果使用topk策略，选择的专家数量
        
        # 使用针对Remove任务优化的特征提取器
        self.feature_extractor = MaskAwareFeatureExtractor(feature_mode="mask_focused")
        
        # 路由网络将在第一次前向传播时初始化
        self.router = None
        
        # 专家特性分析网络 - 分析输入特征以确定任务类型
        self.task_analyzer = None
    
    def forward(self, x, mask_info=None, task_hint=None):
        """前向传播，基于Remove任务特征输出三专家权重
        
        Args:
            x: 输入特征，形状为 [batch_size, seq_len, hidden_dim]
            mask_info: mask相关信息，用于增强路由决策
            task_hint: 任务提示，用于两阶段训练时的专家选择
            
        Returns:
            weights: 三个专家的权重，形状为 [batch_size, 3]
                    [removal_weight, background_weight, completion_weight]
        """
        # 使用Remove任务优化的特征提取器
        routing_inputs = self.feature_extractor(x, mask_info=mask_info)  # [batch_size, feature_dim]
        
        # 懒加载初始化路由器网络 - Remove任务优化
        if self.router is None:
            input_dim = routing_inputs.shape[-1]
            mid_dim = max(64, input_dim // 2)  # 确保中间层有足够容量
            
            # 创建专门的三专家路由网络
            self.router = nn.Sequential(
                nn.Linear(input_dim, mid_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid_dim, mid_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(mid_dim // 2, 3)  # 强制输出3个专家权重
            ).to(routing_inputs.device, routing_inputs.dtype)
            
            # 任务分析器 - 分析当前样本更适合哪种专家处理
            self.task_analyzer = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(), 
                nn.Linear(64, 3),  # 输出三种任务的置信度
                nn.Sigmoid()
            ).to(routing_inputs.device, routing_inputs.dtype)
            
            # 调试信息: 确认路由器网络结构
            router_param_count = sum(p.numel() for p in self.router.parameters())
            logger.info(f"[ROUTER] 路由器网络已初始化 - 参数数量: {router_param_count}")
            logger.info(f"[ROUTER] 输入维度: {input_dim}, 中间维度: {mid_dim}")
            
            # 测试路由器输出是否正常
            with torch.no_grad():
                test_output = self.router(routing_inputs[:1])  # 测试第一个样本
                logger.info(f"[ROUTER] 初始化后测试输出 (raw): {test_output[0].cpu().tolist()}")
        
        # 计算基础路由分数
        logits = self.router(routing_inputs)  # [batch_size, 3]
        
        # 如果提供了任务提示（用于两阶段训练），结合任务分析
        if task_hint is not None and self.task_analyzer is not None:
            task_confidence = self.task_analyzer(routing_inputs)
            # 根据任务提示调整logits
            if task_hint == "removal_only":
                logits[:, 0] += 2.0  # 增强消除专家权重
            elif task_hint == "background_only":
                logits[:, 1] += 2.0  # 增强背景生成专家权重
            elif task_hint == "completion_only":
                logits[:, 2] += 2.0  # 增强物体补全专家权重
        
        # 基于mask特征进行路由决策优化
        if mask_info is not None:
            # mask面积分析 - 大面积mask更需要背景生成
            if 'mask_area_ratio' in mask_info:
                area_ratio = mask_info['mask_area_ratio']
                # 大面积(>0.3)增强背景生成专家
                if area_ratio > 0.3:
                    logits[:, 1] += 1.0  # background_expert
            
            # 边缘复杂度分析 - 复杂边缘更需要消除专家
            if 'edge_complexity' in mask_info:
                edge_complexity = mask_info['edge_complexity']
                # 高复杂度边缘增强消除专家
                if edge_complexity > 0.5:
                    logits[:, 0] += 1.0  # removal_expert
                    
            # 遮挡分析 - 有明显遮挡物需要补全专家
            if 'has_occlusion' in mask_info:
                if mask_info['has_occlusion']:
                    logits[:, 2] += 1.0  # completion_expert
        
        # 应用路由策略
        if self.routing_strategy == "soft":
            # Softmax路由，三个专家都参与
            weights = F.softmax(logits / self.temperature, dim=-1)
        elif self.routing_strategy == "hard":
            # 硬路由，只选择一个专家
            hard_weights = torch.zeros_like(logits)
            max_indices = torch.argmax(logits, dim=-1)
            hard_weights.scatter_(1, max_indices.unsqueeze(1), 1.0)
            weights = hard_weights
        elif self.routing_strategy == "topk":
            # Top-K路由，选择K个最好的专家（通常选择2个）
            top_k = min(self.top_k, 3)
            topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
            
            # 创建mask
            mask = torch.zeros_like(logits)
            mask.scatter_(1, topk_indices, 1.0)
            
            # 应用mask并归一化
            masked_logits = logits * mask
            weights = F.softmax(masked_logits / self.temperature, dim=-1)
        else:
            raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")
        
        return weights


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
        
        # 为每个任务/数据类型添加LoRA适配器 (如果还不存在)
        for i, (lora_config, adapter_name) in enumerate(zip(lora_configs, adapter_names)):
            # 检查适配器是否已存在
            if adapter_name not in getattr(self.model, "peft_config", {}):
                self.model.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)
        
        # 路由器配置先不初始化，等待第一个特征输入时再创建
        if routing_config is None:
            routing_config = {"routing_strategy": "soft", "temperature": 1.0}
        self.router = None
        self.routing_config = routing_config
        
        # 设备和数据类型信息
        self.sample_param = next(self.model.parameters())
        
        # 初始化路由统计信息
        self._routing_stats = {"total_examples": 0, "num_experts": self.num_experts}
        
        # 初始默认为均匀权重
        self.current_weights = torch.ones(self.num_experts) / self.num_experts
        
    def compute_adapter_weights(self, features, mask_info=None, task_hint=None):
        """
        计算当前输入应该使用的适配器权重
        
        Args:
            features: 输入特征，形状 [batch_size, ...]
            mask_info: mask相关信息，用于mask感知路由
            task_hint: 任务提示，用于两阶段训练
            
        Returns:
            torch.Tensor: 适配器权重，形状 [batch_size, num_experts]
        """
        # 懒惰初始化路由器
        if self.router is None:
            self.router = ThreeExpertRemovalRouter(
                num_experts=self.num_experts,
                routing_strategy=self.routing_config["routing_strategy"],
                temperature=self.routing_config["temperature"]
            )
            self.router.to(device=self.sample_param.device, dtype=self.sample_param.dtype)
            self.router.train()
        
        # 先调用路由器的forward以触发懒惰初始化，然后检查参数
        try:
            # 进行一次前向传播以确保路由器网络被初始化
            with torch.no_grad():
                _ = self.router(features[:1], mask_info=mask_info, task_hint=task_hint)
        except Exception as e:
            logger.warning(f"[ROUTER] 路由器初始化失败: {e}")
            # 返回默认均匀权重
            if isinstance(self.current_weights, list):
                weights_tensor = torch.tensor(self.current_weights, device=features.device)
                return weights_tensor.unsqueeze(0).expand(features.shape[0], -1)
            else:
                return self.current_weights.unsqueeze(0).expand(features.shape[0], -1).to(features.device)
        
        # 现在检查路由器参数状态
        router_params = list(self.router.parameters())
        logger.info(f"[ROUTER] 路由器参数数量: {len(router_params)}")
        
        if not router_params:
            # 仍然没有参数，返回默认均匀权重
            logger.warning(f"[ROUTER] 路由器仍没有参数，返回默认均匀权重")
            if isinstance(self.current_weights, list):
                weights_tensor = torch.tensor(self.current_weights, device=features.device)
                return weights_tensor.unsqueeze(0).expand(features.shape[0], -1)
            else:
                return self.current_weights.unsqueeze(0).expand(features.shape[0], -1).to(features.device)
            
        sample_param = next(self.router.parameters())
        router_device = sample_param.device
        router_dtype = sample_param.dtype
        
        if features.device != router_device or features.dtype != router_dtype:
            features = features.to(device=router_device, dtype=router_dtype)
        
        # 处理mask_info设备兼容性    
        if mask_info is not None:
            # 确保 mask_info 中的 tensor 都在正确的设备上
            for key, value in mask_info.items():
                if isinstance(value, torch.Tensor):
                    mask_info[key] = value.to(device=router_device, dtype=router_dtype)
            
        with torch.set_grad_enabled(self.router.training):
            adapter_weights = self.router(features, mask_info=mask_info, task_hint=task_hint)
            
            # 调试信息: 输出路由器的实际输出
            logger.info(f"[ROUTER] 路由器原始输出 shape: {adapter_weights.shape}")
            logger.info(f"[ROUTER] 路由器权重样例 [0]: {adapter_weights[0].detach().cpu().tolist()}")
            if adapter_weights.shape[0] > 1:
                logger.info(f"[ROUTER] 路由器权重样例 [1]: {adapter_weights[1].detach().cpu().tolist()}")
            logger.info(f"[ROUTER] 批次平均权重: {adapter_weights.mean(dim=0).detach().cpu().tolist()}")
            
            return adapter_weights
    
    def set_adapter_weights(self, weights):
        """设置适配器权重
        
        PEFT库期望weights参数是一个一维列表，例如[0.3, 0.3, 0.4]。
        输入的weights可能是二维张量[batch_size, num_experts]，
        所以需要进行处理以匹配库的期望。
        """
        if isinstance(weights, torch.Tensor):
            # 如果是二维张量，取平均值作为适配器权重
            if len(weights.shape) > 1:
                # 这里取batch的平均值作为适配器权重
                weights = weights.mean(dim=0)  # [batch_size, num_experts] -> [num_experts]
            
            # 保持为cpu列表格式，与多LoRA行为一致
            adapter_weights_list = weights.detach().cpu().tolist()  # 转换为普通的Python列表
            self.current_weights = adapter_weights_list
        else:
            # 如果已经是列表格式，直接使用
            self.current_weights = weights
        
        # 设置适配器权重，使用一维列表格式
        self.model.set_adapters(self.adapter_names, weights=self.current_weights)
        
    def get_routing_stats(self):
        """获取路由的统计信息"""
        stats = {}
        
        # 检查路由器是否已初始化
        if self.router is None:
            stats["router_status"] = "not_initialized"
        else:
            stats["router_status"] = "initialized"
            
            # 统计路由器参数数量
            if list(self.router.parameters()):
                router_params = sum(p.numel() for p in self.router.parameters())
                stats["router_params"] = router_params
        
        # 统计当前适配器权重信息
        if hasattr(self, "current_weights") and self.current_weights is not None:
            if isinstance(self.current_weights, list):
                stats["current_weights"] = self.current_weights
            else:
                stats["current_weights"] = self.current_weights.detach().cpu().tolist()
            
        # 如果有路由统计信息，则添加
        if hasattr(self, "_routing_stats"):
            stats.update(self._routing_stats)
            
        return stats
        
    def forward_with_routing(self, features, **forward_kwargs):
        """基于路由计算前向传播结果
        
        Args:
            features: 用于路由决策的特征
            forward_kwargs: 模型前向传播的参数
            
        Returns:
            combined_result: 各专家结果的加权组合
            adapter_weights: 使用的适配器权重
        """
        # 使用路由器计算每个样本的适配器权重
        adapter_weights = self.compute_adapter_weights(features)  # [batch_size, num_experts]
        batch_size = adapter_weights.shape[0]
        
        # 更新路由统计信息
        self._routing_stats["total_examples"] += batch_size
        
        # 保存当前批次路由权重的平均值供调试和统计使用
        # 这可以反映路由器的真实训练状态
        self.current_weights = adapter_weights.mean(dim=0).detach()
        
        # 记录批次内每个专家的使用频率统计
        expert_usage = torch.argmax(adapter_weights, dim=1)  # 每个样本使用的主要专家
        for expert_idx in range(self.num_experts):
            count = (expert_usage == expert_idx).sum().item()
            self._routing_stats[f"expert_{expert_idx}_usage"] += count
        
        results = []
        # 分别处理每个样本，不同样本可能使用不同的适配器组合
        for i in range(batch_size):
            # 为这个样本设置适配器权重
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
    parser.add_argument("--task_hint", type=str, default="",
                        help="任务提示，用于两阶段训练（先单个专家训练，再联合训练）")
    
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
    
    # 损失函数控制参数 - 用于消融实验
    parser.add_argument("--enable_base_loss", action="store_true", default=True,
                        help="启用基础MSE损失")
    parser.add_argument("--enable_boundary_loss", action="store_true", default=False,
                        help="启用边界平滑损失")
    parser.add_argument("--enable_consistency_loss", action="store_true", default=False,
                        help="启用外部区域一致性损失")
    parser.add_argument("--enable_detail_loss", action="store_true", default=False,
                        help="启用细节保持损失")
    parser.add_argument("--enable_mask_info", action="store_true", default=False,
                        help="启用mask信息提取（用于非base损失计算）")
    
    # 损失权重参数
    parser.add_argument("--base_loss_weight", type=float, default=1.0,
                        help="基础损失权重")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.0,
                        help="边界平滑损失权重")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0,
                        help="外部区域一致性损失权重")
    parser.add_argument("--detail_loss_weight", type=float, default=0.0,
                        help="细节保持损失权重")
    
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
    # Initialize PartialState before any logging
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
    
    # 设置权重数据类型
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
    
    # Move model to the appropriate device
    flux_fill_pipe.to(accelerator.device)
    
    logger.info(f"Loading FluxPriorReduxPipeline model: {args.flux_redux_id}")
    flux_redux_pipe = FluxPriorReduxPipeline.from_pretrained(
        args.flux_redux_id,
        torch_dtype=weight_dtype
    )
    # Move models to the appropriate device
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
    
    # 初始化噪声调度器
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
        
        # 创建MoE LoRA管理器
        logger.info(f"创建MoE LoRA管理器: {args.num_experts}个专家, 路由策略: {args.routing_strategy}")
        
        # 创建MoE路由器
        routing_config = {
            "input_dim": transformer.config.joint_attention_dim,
            "routing_strategy": args.routing_strategy,
            "temperature": args.routing_temperature
        }
        
        if args.num_experts > 1:
            router = ThreeExpertRemovalRouter(
                num_experts=args.num_experts,
                routing_strategy=args.routing_strategy,
                temperature=args.routing_temperature
            ).to(accelerator.device)
        else:
            router = None
        
        # 创建moe_manager并添加必需属性
        moe_manager = MoELoRAManager(
            model=transformer,
            adapter_names=args.expert_names,
            lora_configs=[lora_config_dict] * args.num_experts,
            routing_config=routing_config
        )
        
        # 确保设置必需的sample_param属性
        moe_manager.sample_param = next(transformer.parameters())
        logger.info(f"MoE LoRA设置完成: {args.num_experts}个专家, 路由策略: {args.routing_strategy}")
    else:
        router = None
        moe_manager = None
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # 创建MoE保存和加载钩子
    if args.use_lora:
        save_model_hook, load_model_hook = create_moe_lora_hooks(
            transformer, router, args.expert_names, accelerator
        )
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
    
    # 设置可训练参数
    if args.use_lora:
        # MoE-LoRA模式：训练LoRA参数和路由器参数
        lora_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
        router_params = list(router.parameters()) if router is not None else []
        trainable_params = lora_params + router_params
        
        # 创建命名参数列表用于优化器参数分组
        named_trainable_params = []
        # 添加LoRA参数
        for name, param in transformer.named_parameters():
            if param.requires_grad:
                named_trainable_params.append((name, param))
        # 添加路由器参数
        if router is not None:
            for name, param in router.named_parameters():
                named_trainable_params.append((f"router.{name}", param))
                
        logger.info(f"MoE-LoRA模式 - LoRA参数: {len(lora_params)}, 路由器参数: {len(router_params)}")
    else:
        # 全量微调模式：训练所有transformer参数
        transformer.requires_grad_(True)
        trainable_params = transformer.parameters()
        named_trainable_params = list(transformer.named_parameters())
        logger.info("全量微调模式 - 训练所有transformer参数")
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # 优化器设置
    optimizer_type = args.optimizer_config.get("type", "AdamW")
    optimizer_params = args.optimizer_config.get("params", {})
    use_8bit = args.optimizer_config.get("use_8bit", False)

    # 准备带有权重衰减区分的参数组
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
                # 权重衰减在分组参数中设置，这里不需要
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
                # 权重衰减在分组参数中设置，这里不需要再指定
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
        
    # 打印优化器统计信息
    if accelerator.is_main_process:
        total_params = sum(p.numel() for group in grouped_parameters for p in group['params'] if p.requires_grad)
        logger.info(f"参与训练的总参数数量: {total_params:,}")
        if args.use_lora and moe_manager is not None:
            # 路由器可能尚未初始化，此时只打印占位符消息
            if moe_manager.router is not None:
                router_params = sum(p.numel() for p in moe_manager.router.parameters())
                logger.info(f"MoE路由器参数数量: {router_params:,}")
            else:
                logger.info("MoE路由器将在第一次前向传播时懒加载初始化")
    
    # 数据加载器设置
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
    
    # 准备MoE路由器（如果有）
    if router is not None:
        router = accelerator.prepare(router)
    
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
        if router is not None:
            router.train()
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
                    # 确保编码的latents在正确设备上
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
                    # 使用clone().detach()替代torch.tensor()，并确保在正确设备上
                    timesteps = indices.clone().detach().float().to(accelerator.device) / noise_scheduler.config.num_train_timesteps
                    t = timesteps
                    
                    # 生成随机噪声，确保在正确设备上
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
                        # 使用args.guidance_scale参数而不是固定值
                        guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                        guidance = guidance.expand(t.shape[0])
                
                # MoE路由逻辑：只在启用额外损失时才进行路由计算
                # 基础MSE损失时使用传统LoRA，专家分化损失时使用MoE路由
                use_moe_routing = (
                    args.use_lora and moe_manager is not None and 
                    (args.enable_boundary_loss or args.enable_consistency_loss or args.enable_detail_loss)
                )
                
                if use_moe_routing:
                    # 准备路由特征输入
                    hidden_states = torch.cat((x_t, condition_latents), dim=2)
                    
                    # 提取mask信息用于路由决策
                    routing_mask_info = extract_mask_info_from_batch(batch, accelerator.device, weight_dtype)
                    
                    # 计算mask感知的适配器权重
                    adapter_weights = moe_manager.compute_adapter_weights(
                        hidden_states, 
                        mask_info=routing_mask_info,
                        task_hint=args.task_hint if args.task_hint else None  # 可选的任务提示
                    )
                    # 设置适配器权重
                    moe_manager.set_adapter_weights(adapter_weights)
                    
                    logger.info(f"[MoE] 启用路由模式 - 当前权重: {moe_manager.current_weights}")
                elif args.use_lora and moe_manager is not None:
                    # 基础MSE损失模式：使用均匀权重，相当于传统LoRA的平均效果
                    uniform_weights = [1.0/moe_manager.num_experts] * moe_manager.num_experts
                    moe_manager.set_adapter_weights(uniform_weights)
                    
                    if global_step % 50 == 0:  # 减少日志频率
                        logger.info(f"[MoE] 基础损失模式 - 使用均匀权重: {uniform_weights}")
                
                # 统一使用tranformer_forward函数进行前向传播
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
                target = (x_1 - x_0).float()
                
                # DEBUG: 输出关键形状信息
                if global_step % 50 == 0:
                    logger.info(f"=== 训练步骤 {global_step} 的形状信息 ===")
                    logger.info(f"pred.shape: {pred.shape}")
                    logger.info(f"target.shape: {target.shape}")
                    logger.info(f"weighting.shape: {weighting.shape}")
                    
                    # 只在真正使用MoE路由时才显示路由统计
                    if use_moe_routing:
                        logger.info(f"MoE 路由统计: {moe_manager.get_routing_stats()}")
                    elif moe_manager is not None:
                        logger.info(f"MoE 模式: 基础损失(均匀权重)模式")
                
                # 根据args控制是否提取mask信息
                # detail_loss也需要VAE解码，所以也需要mask_info
                mask_info = None
                if args.enable_mask_info and (args.enable_boundary_loss or args.enable_consistency_loss or args.enable_detail_loss):
                    mask_info = extract_mask_info_from_batch(batch, accelerator.device, weight_dtype)
                
                # 根据args参数构建损失配置
                loss_config = {
                    'base_weight': args.base_loss_weight if args.enable_base_loss else 0.0,
                    'boundary_weight': args.boundary_loss_weight if args.enable_boundary_loss else 0.0,
                    'consistency_weight': args.consistency_loss_weight if args.enable_consistency_loss else 0.0,
                    'detail_weight': args.detail_loss_weight if args.enable_detail_loss else 0.0
                }
                
                loss_dict = compute_removal_task_loss(
                    pred=pred, 
                    target=target, 
                    mask_info=mask_info,  # 根据args控制是否传入mask_info
                    weighting=weighting, 
                    loss_config=loss_config,
                    vae=vae,
                    vae_scale_factor=args.vae_scale_factor if hasattr(args, 'vae_scale_factor') else 8
                )
                
                loss = loss_dict['total_loss']
                
                # 记录各组件损失（用于监控和调试）
                if accelerator.is_main_process and global_step % 50 == 0:
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
                    
                    # 只在使用MoE路由时才记录路由统计，避免基础损失模式下的错误信息
                    if 'use_moe_routing' in locals() and use_moe_routing and moe_manager is not None:
                        routing_stats = moe_manager.get_routing_stats()
                        logs.update(routing_stats)
                    elif moe_manager is not None:
                        # 基础损失模式，只记录模式信息
                        logs['moe_mode'] = 'uniform_weights'
                    
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                # 保存检查点
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
        
        # if accelerator.is_main_process:
        #     if epoch > 0 and epoch % args.validation_epochs == 0:
        #         logger.info("Running validation...")
        #         try:
        #             if args.val_json_path:
        #                 log_infer_moe(
        #                     accelerator=accelerator, 
        #                     args=args, 
        #                     save_path=args.output_dir, 
        #                     epoch=epoch, 
        #                     global_step=global_step, 
        #                     pipefill=flux_fill_pipe, 
        #                     pipeprior=flux_redux_pipe
        #                 )
        #                 free_memory()
        #             else:
        #                 logger.warning("val_json_path not provided, skipping validation.")
        #         except Exception as e:
        #             logger.error(f"Inference failed with error: {e}")
        #             traceback.print_exc()
        #             logger.info("Inference failed, but training will continue.")

        # 每轮结束后记录平均损失
        accelerator.log({"epoch_loss": train_loss / len(train_dataloader)}, step=global_step)
    
    # 保存最终模型
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
