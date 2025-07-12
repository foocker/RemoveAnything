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

from data.all_data import  load_triplet_paths, load_triplet_paths_from_dir
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


def log_infer(accelerator, args, save_path, epoch, global_step, 
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
    logger.info(f"Running inference... \nEpoch: {epoch}, Step: {global_step}")
    save_dir = os.path.join(save_path, f"infer_seed{args.seed}")
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
        
        # 确保pipeprior也使用正确的数据类型
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
        
        edited_image_save_path = os.path.join(save_dir, f"seed{args.seed}_epoch_{epoch}_step_{global_step}_{file_name}")
        composite_image.save(edited_image_save_path)
        
    del pipefill, pipeprior
    torch.cuda.empty_cache()
    gc.collect()
    

def log_infer_custom(accelerator, args, save_path, epoch, global_step, 
               pipefill: "FluxFillPipeline", pipeprior: "FluxPriorReduxPipeline"):
    model_dtype = torch.bfloat16
    logger.info(f"Setting all model components to use {model_dtype}")
    
    if hasattr(pipefill, "vae") and hasattr(pipefill.vae, "_decode"):
        original_decode = pipefill.vae._decode
        
        class DecoderPatch:
            def __init__(self, pipe, orig_decode, dtype):
                self.pipe = pipe
                self.original_decode = orig_decode
                self.dtype = dtype
                
            def __call__(self, z):
                z = z.to(dtype=self.dtype)
                with torch.no_grad():
                    for module in self.pipe.vae.decoder.named_modules():
                        if hasattr(module, "weight") and module.weight is not None:
                            if module.weight.dtype != self.dtype:
                                module.weight.data = module.weight.data.to(self.dtype)
                        if hasattr(module, "bias") and module.bias is not None:
                            if module.bias.dtype != self.dtype:
                                module.bias.data = module.bias.data.to(self.dtype)
                    
                    try:
                        return self.original_decode(z)
                    except Exception as e:
                        logger.warning(f"Error in patched decode: {e}")
                        logger.info("Falling back to float32 for this operation")
                        z_float = z.to(dtype=torch.float32)
                        self.pipe.vae.decoder = self.pipe.vae.decoder.to(dtype=torch.float32)
                        result = self.original_decode(z_float)
                        self.pipe.vae.decoder = self.pipe.vae.decoder.to(dtype=self.dtype)
                        return result
        
        pipefill.vae._decode = DecoderPatch(pipefill, original_decode, model_dtype)
        logger.info("Successfully patched VAE decoder method")
    
    pipefill.to(dtype=model_dtype)
    pipeprior.to(dtype=model_dtype)

    set_seed(args.seed)
    logger.info(f"Running inference with custom processing... \nEpoch: {epoch}, Step: {global_step}")
    save_dir = os.path.join(save_path, f"infer_custom_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.isdir(args.val_json_path):
        triplet_paths = load_triplet_paths_from_dir(args.val_json_path)
    else:
        triplet_paths = load_triplet_paths(args.val_json_path)
    buckets = parse_buckets_string(args.aspect_ratio_buckets)

    for paths in triplet_paths:
        source_image_path = paths["input_image"]
        mask_image_path = paths["mask"]
        file_name = os.path.basename(source_image_path)
        removed_image_path = paths["edited_image"] if os.path.exists(paths["edited_image"]) else source_image_path

        input_image = cv2.imread(source_image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        removed_image = cv2.imread(removed_image_path)
        removed_image = cv2.cvtColor(removed_image, cv2.COLOR_BGR2RGB)
        removed_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]
        
        orig_input_image = input_image.copy()
        orig_mask = removed_mask.copy()
        
        remove_box_yyxx = get_bbox_from_mask(removed_mask)
        
        ref_mask_3 = np.stack([removed_mask, removed_mask, removed_mask], -1)
        
        masked_ref_image = input_image * (1 - ref_mask_3) + np.ones_like(input_image) * 255 * ref_mask_3
        
        expand_remove_box_yyxx = expand_bbox(removed_mask, remove_box_yyxx, ratio=[1.1, 1.5])
        
        y1, y2, x1, x2 = expand_remove_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        
        h, w = input_image.shape[:2]
        target_size = buckets[find_nearest_bucket(h, w, buckets)]
        
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=0, random=False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), target_size).astype(np.uint8)
        
        masked_task_image = input_image[y1:y2, x1:x2, :]
        masked_task_image = pad_to_square(masked_task_image, pad_value=0, random=False).astype(np.uint8)
        masked_task_image = cv2.resize(masked_task_image.astype(np.uint8), target_size).astype(np.uint8)
        
        tar_image = removed_image[y1:y2, x1:x2, :]
        tar_image = pad_to_square(tar_image, pad_value=0, random=False).astype(np.uint8)
        tar_image = cv2.resize(tar_image.astype(np.uint8), target_size).astype(np.uint8)
        
        tar_mask = ref_mask_3[y1:y2, x1:x2, :] * 255
        tar_mask = pad_to_square(tar_mask, pad_value=0, random=False).astype(np.uint8)
        tar_mask = cv2.resize(tar_mask.astype(np.uint8), target_size).astype(np.uint8)
        
        mask_black = np.ones_like(tar_image) * 0
        diptych_ref_tar = np.concatenate([masked_ref_image, masked_task_image], axis=1)
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
        
        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych = Image.fromarray(mask_diptych)
        
        with torch.no_grad():
            pipe_prior_output = pipeprior(Image.fromarray(masked_ref_image))
        
        for key, value in pipe_prior_output.items():
            if isinstance(value, torch.Tensor):
                pipe_prior_output[key] = value.to(dtype=model_dtype)
        
        generator = torch.Generator(accelerator.device).manual_seed(args.seed)
        edited_image = pipefill(
            image=diptych_ref_tar,
            mask_image=mask_diptych,
            height=mask_diptych.size[1],
            width=mask_diptych.size[0],
            max_sequence_length=512,
            num_inference_steps=30,
            generator=generator,
            **pipe_prior_output,
        ).images[0]
        
        # 裁剪右半部分（编辑后的图像）
        t_width, t_height = edited_image.size
        start_x = t_width // 2
        edited_image = edited_image.crop((start_x, 0, t_width, t_height))
        
        edited_image = np.array(edited_image)
        edited_pil = Image.fromarray(edited_image)
        edited_resized = edited_pil.resize((x2-x1, y2-y1))
        final_output = orig_input_image.copy()
        final_output[y1:y2, x1:x2, :] = np.array(edited_resized)
        final_output = Image.fromarray(final_output)

        original_image = Image.fromarray(orig_input_image)

        visible_mask = Image.fromarray(
            np.where(orig_mask[:, :, None] > 0, 
                    np.ones_like(orig_input_image) * 255, 
                    orig_input_image).astype(np.uint8)
        )
        

        total_width = original_image.width + visible_mask.width + final_output.width
        composite_image = Image.new('RGB', (total_width, original_image.height))
        
        composite_image.paste(original_image, (0, 0))
        composite_image.paste(visible_mask, (original_image.width, 0))
        composite_image.paste(final_output, (original_image.width + visible_mask.width, 0))
        
        edited_image_save_path = os.path.join(save_dir, f"custom_seed{args.seed}_epoch_{epoch}_step_{global_step}_{file_name}")
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

def parse_args():
    parser = argparse.ArgumentParser(description="使用LazyBucket数据集训练模型执行移除任务")
    
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
    parser.add_argument("--output_dir", type=str, default="outputs/removal", 
                        help="模型保存路径")
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
                        help="优化器类型, Prodigy, adamw, muon, etc.")    
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
    
    # 多个lora
    parser.add_argument("--lora_adapters", type=str, default="adapter1,adapter2", help="LoRA adapter names, comma separated")
    parser.add_argument("--adapter_weights", type=str, default="0.7,0.3", help="LoRA adapter weights, comma separated")
    
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
        else:  # adamw
            args.optimizer_config = {
                "type": "AdamW"
            }
    
    return args


def main():
    # Initialize PartialState before any logging
    from accelerate.state import PartialState
    _ = PartialState()  # TODO
    
    args = parse_args()
    
    # 输出关键配置信息
    logger.info(f"已配置参数: output_dir={args.output_dir}, lr={args.learning_rate}, batch_size={args.train_batch_size}")
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
        
    # 确保 VAE 使用正确的数据类型, 训练的稳定性来说，vae应该使用float32 # TODO 后续验证
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
    
    # 增加LoRA模块
    if args.use_lora:
        logger.info("Using LoRA for training transformer")
        # TODO from flux kontext``
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
            "target_modules": target_modules
        }
        
        logger.info("添加多个LoRA适配器")
        # # 添加第一个LoRA适配器
        # transformer.add_adapter("adapter1", LoraConfig(**lora_config_dict))
        
        # # 添加第二个LoRA适配器（可选择不同配置）
        # lora_config_dict2 = lora_config_dict.copy()  # 可以修改此配置用于第二个适配器
        # transformer.add_adapter("adapter2", LoraConfig(**lora_config_dict2))
        
        # # 设置活动适配器，可以同时激活多个适配器
        # transformer.set_adapter(["adapter1", "adapter2"], adapter_weights=[0.7, 0.3])
        # # 或者只激活一个用于训练
        # # transformer.set_adapter("adapter1")
        
        # 解析适配器名称和权重
        adapter_names = args.lora_adapters.split(",")
        adapter_weights = [float(w) for w in args.adapter_weights.split(",")]
        assert len(adapter_names) == len(adapter_weights), "适配器名称和权重数量必须一致"

        # 添加多个适配器
        for adapter_name in adapter_names:
            transformer.add_adapter(adapter_name, LoraConfig(**lora_config_dict))

        # 设置活动适配器和权重
        transformer.set_adapter(adapter_names, adapter_weights=adapter_weights)
        
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = {}
                    for adapter_name in model.peft_config.keys():
                        transformer_lora_layers_to_save[adapter_name] = get_peft_model_state_dict(model, adapter_name)
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
                
            for adapter_name, state_dict in transformer_lora_layers_to_save.items():
                adapter_output_dir = os.path.join(output_dir, f"adapter_{adapter_name}")
                os.makedirs(adapter_output_dir, exist_ok=True)
                FluxFillPipeline.save_lora_weights(
                    adapter_output_dir,
                    transformer_lora_layers={adapter_name: state_dict},
                    **_collate_lora_metadata(modules_to_save),
                )
            
    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")
        
        adapter_dirs = [d for d in os.listdir(input_dir) if d.startswith("adapter_") and os.path.isdir(os.path.join(input_dir, d))]
        
        if adapter_dirs:
            for adapter_dir in adapter_dirs:
                adapter_path = os.path.join(input_dir, adapter_dir)
                adapter_name = adapter_dir.replace("adapter_", "")
                logger.info(f"从 {adapter_path} 加载适配器 {adapter_name}")
                
                try:
                    lora_state_dict = FluxFillPipeline.lora_state_dict(adapter_path)
                    transformer_state_dict = {
                        f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
                    }
                    incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name=adapter_name)
                    
                    if incompatible_keys is not None:
                        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                        if unexpected_keys:
                            logger.warning(
                                f"加载适配器 {adapter_name} 权重时发现意外的键: {unexpected_keys}"
                            )
                except Exception as e:
                    logger.error(f"加载适配器 {adapter_name} 失败: {str(e)}")
        else:
            lora_state_dict = FluxFillPipeline.lora_state_dict(input_dir)
            transformer_state_dict = {
                f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }
            
            adapter_name = "default"
            if hasattr(args, "lora_adapters") and args.lora_adapters:
                adapter_names = args.lora_adapters.split(",")
                if adapter_names:
                    adapter_name = adapter_names[0]
                    
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name=adapter_name)
            if incompatible_keys is not None:
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        
        # Make sure the trainable params are in float32
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)
        
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # 设置可训练参数（在所有检查点恢复和LoRA权重加载之后）
    if args.use_lora:
        # LoRA模式：只训练LoRA参数
        trainable_params = list(filter(
            lambda p: p.requires_grad, transformer.parameters()
        ))
        logger.info(f"LoRA模式 - 可训练参数数量: {len(trainable_params)}")
    else:
        # 全量微调模式：训练所有transformer参数
        transformer.requires_grad_(True)
        trainable_params = transformer.parameters()
        logger.info("全量微调模式 - 训练所有transformer参数")
    
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         transformer.enable_xformers_memory_efficient_attention()  # TODO 推理报错
    #     else:
    #         logger.warning("xformers not available, memory efficient attention not enabled")
    
    optimizer_type = args.optimizer_config.get("type", "AdamW")
    optimizer_params = args.optimizer_config.get("params", {})

    if optimizer_type.lower() == "prodigy":
        try:
            from prodigyopt import Prodigy
            optimizer = Prodigy(
                list(trainable_params),
                lr=optimizer_params.get("lr", 1),
                weight_decay=optimizer_params.get("weight_decay", 0.01),
                use_bias_correction=optimizer_params.get("use_bias_correction", True),
                safeguard_warmup=optimizer_params.get("safeguard_warmup", True)
            )
            logger.info(f"使用Prodigy优化器，参数: {optimizer_params}")
        except ImportError:
            logger.warning("无法导入Prodigy优化器，回退到AdamW")
            optimizer = torch.optim.AdamW(
                list(trainable_params),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon
            )
    elif optimizer_type.lower() == "muon":
        try:
            # 为Muon优化器分组参数
            muon_groups = [param for name, param in transformer.named_parameters() 
                          if param.requires_grad and param.ndim >= 2 and 
                          "pos_embed" not in name and "final_layer" not in name and 
                          "x_embedder" not in name and "t_embedder" not in name and "y_embedder" not in name]
            
            adamw_groups = [param for name, param in transformer.named_parameters() 
                           if param.requires_grad and not (param.ndim >= 2 and 
                           "pos_embed" not in name and "final_layer" not in name and 
                           "x_embedder" not in name and "t_embedder" not in name and "y_embedder" not in name)]
            
            optimizer = MuonWithAuxAdam([
                {"params": muon_groups, "use_muon": True, "lr": args.learning_rate, "weight_decay": args.adam_weight_decay},
                {"params": adamw_groups, "use_muon": False, "lr": args.learning_rate * 0.1, "weight_decay": args.adam_weight_decay}
            ])
            logger.info(f"使用Muon优化器，主参数lr={args.learning_rate}，辅助参数lr={args.learning_rate * 0.1}")
            logger.info(f"Muon参数组数量: {len(muon_groups)}，AdamW参数组数量: {len(adamw_groups)}")
        except ImportError:
            logger.warning("无法导入Muon优化器，回退到AdamW")
            optimizer = torch.optim.AdamW(
                list(trainable_params),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon
            )
    else:
        optimizer = torch.optim.AdamW(
            list(trainable_params),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon
        )
        logger.info(f"使用AdamW优化器，参数: {{在args中的参数: lr={args.learning_rate}, betas=({args.adam_beta1}, {args.adam_beta2}), weight_decay={args.adam_weight_decay}, eps={args.adam_epsilon}}}")
    
    buckets = parse_buckets_string(args.aspect_ratio_buckets)
    
    logger.info("加载训练数据集...")
    try:
        train_dataset = TripletBucketDataset(json_path=args.train_json_path, 
                                        buckets=buckets,
                                        metadata_file=args.train_metadata_file,
                                        custom=args.inference_custom)
        
        if len(train_dataset) == 0:
            logger.error("训练数据集为空，请检查数据路径和文件格式")
            raise ValueError("训练数据集为空，请检查日志中的路径和文件格式信息")
    except Exception as e:
        logger.error(f"加载训练数据集时发生错误: {str(e)}")
        raise
    
    logger.info(f"加载了 {len(train_dataset)} 个训练样本")
    
    batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=False)
    
    logger.info(f"加载了 {len(train_dataset)} 个训练样本")
    
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
    
    # prepare 需要重新计算 num_update_steps_per_epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    if accelerator.is_main_process:
        accelerator.init_trackers("removal_training")
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        accelerator.print(f"从检查点恢复训练，主模型结构已改变，只计算global_step: {path}")
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
    
    # 使用预训练的LoRA权重进行初始化。
    if args.use_lora and args.pretrained_model_path:
        # 加载独立的LoRA权重文件
        logger.info("加载预训练LoRA权重")
        pretrained_model_path = None
        pretrained_weight_name = None
        try:
            pretrained_model_path = args.pretrained_model_path
            logger.info(f"将使用预训练LoRA权重: {pretrained_model_path}")
            # 检查权重文件是否存在
            safetensors_path = os.path.join(pretrained_model_path, "pytorch_lora_weights.safetensors")
            bin_path = os.path.join(pretrained_model_path, "pytorch_lora_weights.bin")
            
            if os.path.exists(safetensors_path):
                logger.info(f"找到LoRA权重文件: {safetensors_path}")
                pretrained_weight_name = "pytorch_lora_weights.safetensors"
            elif os.path.exists(bin_path):
                logger.info(f"找到LoRA权重文件: {bin_path}")
                pretrained_weight_name = "pytorch_lora_weights.bin"
            else:
                logger.warning(f"在 {pretrained_model_path} 中找不到标准LoRA权重文件")
                pretrained_model_path = None
        except Exception as e:
            logger.error(f"检查预训练权重时发生错误: {str(e)}")
            logger.info("继续使用初始化权重训练")
            pretrained_model_path = None
        
        if pretrained_model_path:
            try:
                logger.info(f"在pipeline中加载预训练LoRA权重: {pretrained_model_path}")
                if pretrained_weight_name:
                    logger.info(f"使用指定的权重文件: {pretrained_weight_name}")
                    flux_fill_pipe.load_lora_weights(pretrained_model_path, weight_name=pretrained_weight_name)
                else:
                    flux_fill_pipe.load_lora_weights(pretrained_model_path)
                logger.info("成功加载LoRA权重")
            except Exception as e:
                logger.error(f"加载LoRA权重时出错: {str(e)}")
                logger.info("继续使用初始化权重训练")
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
        desc="Steps"
    )
    
    logger.info("***** 开始训练 *****")
    logger.info(f"  样本数量 = {len(train_dataset)}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  每设备批量大小 = {args.train_batch_size}")
    logger.info(f"  总批量大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    
    logger.info(f"当前训练配置:")
    logger.info(f"  - 学习率: {args.learning_rate}")
    logger.info(f"  - 批次大小: {args.train_batch_size}")
    logger.info(f"  - 梯度累积步数: {args.gradient_accumulation_steps}")
    logger.info(f"  - 是否使用预训练权重: {'是' if args.pretrained_model_path else '否'}")
    if args.pretrained_model_path:
        logger.info(f"  - 预训练权重路径: {args.pretrained_model_path}")
    logger.info(f"  - 是否从检查点恢复训练状态: {'是' if args.resume_from_checkpoint else '否'}")
    if args.resume_from_checkpoint:
        logger.info(f"  - 恢复检查点路径: {args.resume_from_checkpoint}")
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                start_time = time.time()
                
                ref = batch["ref"].to(accelerator.device, dtype=weight_dtype)  # 参考图像
                src = batch["src"].to(accelerator.device, dtype=weight_dtype)  # 源图像
                mask = batch["mask"].to(accelerator.device, dtype=weight_dtype)  # mask
                imgs = batch["result"].to(accelerator.device, dtype=weight_dtype)  # 结果图像
                
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
                        # dual change here TODO pil_image
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
                    
                    t = torch.sigmoid(torch.randn((imgs.shape[0],), device=accelerator.device))
                    x_1 = torch.randn_like(x_0).to(accelerator.device)
                    t_ = t.unsqueeze(1).unsqueeze(1)
                    x_t = ((1 - t_) * x_0 + t_ * x_1).to(weight_dtype)
                    
                    src_latents, mask_latents = Flux_fill_encode_masks_images(flux_fill_pipe, src, mask)
                    condition_latents = torch.cat((src_latents, mask_latents), dim=-1)

                    guidance = (
                        torch.ones_like(t).to(accelerator.device)
                        if transformer.config.guidance_embeds
                        else None
                    )
                
                # 可以在此处对attention做一些training free 的优化 TODO 
                transformer_out = tranformer_forward(
                    transformer,
                    model_config=args.transformer_config_dict,  # TODO attn_forward 中没被使用 ?
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
                
                loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                step_time = time.time() - start_time
            
            # 同步梯度后更新进度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                train_loss += loss.detach().item()
                
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                    
                    if args.use_lora:
                        unwrapped_transformer = accelerator.unwrap_model(transformer)
                        FluxFillPipeline.save_lora_weights(
                            save_directory=os.path.join(args.output_dir, f"lora-{global_step}"),
                            transformer_lora_layers=get_peft_model_state_dict(unwrapped_transformer),
                            safe_serialization=True,
                        )
                        logger.info(f"LoRA weights saved to {os.path.join(args.output_dir, f'lora-{global_step}')}")
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step_time": step_time}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step >= args.max_train_steps:
                break
            
            # # debug 用，快速验证validation 
            # if accelerator.is_main_process:
            #     if global_step > 1 and global_step % args.validation_steps == 0:
            #         logger.info("Running validation...")
            #         try:
            #             if args.val_json_path:
            #                 log_infer(accelerator, args, args.output_dir, epoch, global_step, flux_fill_pipe, flux_redux_pipe)
            #                 free_memory()
            #             else:
            #                 logger.warning("val_json_path not provided, skipping validation.")
            #         except Exception as e:
            #             logger.error(f"Inference failed with error: {e}")
            #             traceback.print_exc()
            #             logger.info("Inference failed, but training will continue.")
                
        if accelerator.is_main_process:
            if epoch > 0 and epoch % args.validation_epochs == 0:
                logger.info("Running validation...")
                try:
                    if args.val_json_path:
                        if args.inference_custom:
                            log_infer_custom(accelerator, args, args.output_dir, epoch, global_step, flux_fill_pipe, flux_redux_pipe)
                        else:
                            log_infer(accelerator, args, args.output_dir, epoch, global_step, flux_fill_pipe, flux_redux_pipe)
                        free_memory()
                    else:
                        logger.warning("val_json_path not provided, skipping validation.")
                except Exception as e:
                    logger.error(f"Inference failed with error: {e}")
                    traceback.print_exc()
                    logger.info("Inference failed, but training will continue.")
                    
        # 每轮结束后记录平均损失
        accelerator.log({"epoch_loss": train_loss / len(train_dataloader)}, step=global_step)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_lora:
            unwrapped_transformer = accelerator.unwrap_model(transformer)
            lora_save_path = os.path.join(args.output_dir, "final_lora")
            os.makedirs(lora_save_path, exist_ok=True)
            
            FluxFillPipeline.save_lora_weights(
                save_directory=lora_save_path,
                transformer_lora_layers=get_peft_model_state_dict(unwrapped_transformer),
                safe_serialization=True,
            )
            logger.info(f"\n训练完成！最终LoRA权重保存至 {lora_save_path}")
        else:
            unwrapped_transformer = accelerator.unwrap_model(transformer)
            final_save_path = os.path.join(args.output_dir, "final_model")
            os.makedirs(final_save_path, exist_ok=True)
            
            flux_pipe_new = FluxFillPipeline.from_pretrained(
                args.flux_fill_id,
                transformer=unwrapped_transformer,
                torch_dtype=weight_dtype
            )
            flux_pipe_new.save_pretrained(final_save_path)
            logger.info(f"\n训练完成！最终模型保存至 {final_save_path}")
            
    accelerator.end_training()


if __name__ == "__main__":
    main()
    logger.info("\n\n训练脚本执行完毕，用于移除任务的LoRA微调已完成")