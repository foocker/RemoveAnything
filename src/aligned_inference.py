#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import gc
import logging
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from peft import LoraConfig

from accelerate.utils import set_seed
from data.data_utils import get_bbox_from_mask, expand_bbox, expand_image_mask, pad_to_square, box2squre, crop_back
from data.all_data import  load_triplet_paths, load_triplet_paths_from_dir
from data.bucket_utils import parse_buckets_string, find_nearest_bucket

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

def load_models_and_weights(args, device):
    """加载模型和权重，与训练脚本中的方式一致"""
    model_dtype = torch.bfloat16
    logger.info(f"设置所有模型组件使用 {model_dtype}")
    
    # 加载FluxFill模型
    logger.info(f"加载FluxFill模型: {args.flux_fill_path}")
    pipefill = FluxFillPipeline.from_pretrained(
        args.flux_fill_path,
        torch_dtype=model_dtype,
        use_safetensors=True
    )
    
    # 加载FluxPriorRedux模型
    logger.info(f"加载FluxPriorRedux模型: {args.flux_redux_path}")
    pipeprior = FluxPriorReduxPipeline.from_pretrained(
        args.flux_redux_path,
        torch_dtype=model_dtype
    )
    
    # 移动到设备
    pipefill.to(device)
    pipeprior.to(device)
    
    # 从不同部分提取组件（与训练脚本一致）
    vae = pipefill.vae
    text_encoder = pipefill.text_encoder
    text_encoder_2 = pipefill.text_encoder_2
    transformer = pipefill.transformer
    
    # 设置模型状态
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    text_encoder_2.requires_grad_(False).eval()
    transformer.requires_grad_(False).eval()
    pipeprior.image_embedder.requires_grad_(False).eval()
    pipeprior.image_encoder.requires_grad_(False).eval()
    
    # 添加LoRA适配器配置
    logger.info("添加LoRA适配器配置")
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
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=target_modules
    )
    
    transformer.add_adapter(lora_config)
    
    # 加载LoRA权重
    logger.info(f"加载LoRA权重: {args.lora_weights_path}")
    try:
        # 检查权重文件
        safetensors_path = os.path.join(args.lora_weights_path, "pytorch_lora_weights.safetensors")
        bin_path = os.path.join(args.lora_weights_path, "pytorch_lora_weights.bin")
        
        if os.path.exists(safetensors_path):
            weight_name = "pytorch_lora_weights.safetensors"
            logger.info(f"找到LoRA权重文件: {safetensors_path}")
        elif os.path.exists(bin_path):
            weight_name = "pytorch_lora_weights.bin"
            logger.info(f"找到LoRA权重文件: {bin_path}")
        else:
            weight_name = None
            logger.warning(f"在 {args.lora_weights_path} 中找不到标准LoRA权重文件")
        
        if weight_name:
            pipefill.load_lora_weights(args.lora_weights_path, weight_name=weight_name)
        else:
            pipefill.load_lora_weights(args.lora_weights_path)
        
        logger.info("成功加载LoRA权重")
    except Exception as e:
        logger.error(f"加载LoRA权重出错: {str(e)}")
        raise e
    
    # 修补VAE解码器方法
    if hasattr(pipefill, "vae") and hasattr(pipefill.vae, "_decode"):
        original_decode = pipefill.vae._decode
        pipefill.vae._decode = DecoderPatch(pipefill, original_decode, model_dtype)
        logger.info("成功修补VAE解码器方法")
    
    return pipefill, pipeprior

def run_aligned_inference(args):
    """执行与训练脚本中一致的推理流程"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    pipefill, pipeprior = load_models_and_weights(args, device)
    model_dtype = torch.bfloat16
    
    # 设置随机种子
    set_seed(args.seed)
    logger.info(f"执行推理... 种子: {args.seed}")
    
    # 创建输出目录
    save_dir = os.path.join(args.output_dir, f"aligned_infer_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据路径
    if os.path.isdir(args.val_json_path):
        triplet_paths = load_triplet_paths_from_dir(args.val_json_path)
    else:
        triplet_paths = load_triplet_paths(args.val_json_path)
    
    # 解析桶
    buckets = parse_buckets_string(args.aspect_ratio_buckets)
    
    # 对每个样本进行推理
    for i, paths in enumerate(triplet_paths):
        if i >= args.max_samples:
            break
            
        logger.info(f"处理样本 {i+1}/{min(len(triplet_paths), args.max_samples)}")
        
        source_image_path = paths["input_image"]  # 被消除
        mask_image_path = paths["mask"]  # 待消除区域
        file_name = os.path.basename(source_image_path)
        removed_image_path = paths["edited_image"] if os.path.exists(paths["edited_image"]) else source_image_path
        
        ref_image_path = removed_image_path
        ref_mask_path = mask_image_path
        
        # 读取图像和掩码
        ref_image = cv2.imread(ref_image_path)
        h, w = ref_image.shape[:2]
        target_size = buckets[find_nearest_bucket(h, w, buckets)]
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        tar_image = cv2.imread(source_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:, :, 0]
        tar_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]
        
        # 确保掩码和图像大小一致
        if tar_mask.shape != tar_image.shape[:2]:
            tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))
        
        # 处理参考图像和掩码
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)
        y1, y2, x1, x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = ref_mask[y1:y2, x1:x2]
        ratio = 1.3
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
        
        # 处理目标图像和掩码
        kernel = np.ones((7, 7), np.uint8)
        iterations = 2
        tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
        tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=2.0)
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
        y1, y2, x1, x2 = tar_box_yyxx_crop
        
        old_tar_image = tar_image.copy()
        
        tar_image = tar_image[y1:y2, x1:x2, :]
        tar_mask = tar_mask[y1:y2, x1:x2]
        
        H1, W1 = tar_image.shape[0], tar_image.shape[1]
        
        # 调整大小
        tar_mask = pad_to_square(tar_mask, pad_value=0)
        tar_mask = cv2.resize(tar_mask, target_size)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), target_size).astype(np.uint8)
        
        # 获取先验模型输出
        with torch.no_grad():
            pipe_prior_output = pipeprior(Image.fromarray(masked_ref_image))
        
        # 转换类型
        for key, value in pipe_prior_output.items():
            if isinstance(value, torch.Tensor):
                pipe_prior_output[key] = value.to(dtype=model_dtype)
        
        # 最终处理目标图像
        tar_image = pad_to_square(tar_image, pad_value=255)
        H2, W2 = tar_image.shape[0], tar_image.shape[1]
        tar_image = cv2.resize(tar_image, target_size)
        
        # 拼接参考和目标图像
        diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
        
        # 准备掩码
        tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
        mask_black = np.ones_like(tar_image) * 0
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
        
        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych = Image.fromarray(mask_diptych)
        
        # 生成图像
        generator = torch.Generator(device).manual_seed(args.seed)
        edited_image = pipefill(
            image=diptych_ref_tar,
            mask_image=mask_diptych,
            height=mask_diptych.size[1],
            width=mask_diptych.size[0],
            max_sequence_length=512,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            **pipe_prior_output,
        ).images[0]
        
        # 处理输出
        t_width, t_height = edited_image.size
        start_x = t_width // 2
        edited_image = edited_image.crop((start_x, 0, t_width, t_height))
        
        edited_image = np.array(edited_image)
        edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop))
        edited_image = Image.fromarray(edited_image)
        
        # 创建复合图像
        original_image = Image.fromarray(old_tar_image)
        # 确保掩码是RGB格式
        if len(np.array(tar_mask).shape) == 3:
            visible_mask_array = np.array(tar_mask)
        else:
            visible_mask_array = np.stack([np.array(tar_mask) * 255, np.array(tar_mask) * 255, np.array(tar_mask) * 255], axis=-1)
        
        # 使用原始的掩码直接覆盖在原始图像上，保持完全一致的位置和尺寸
        original_mask = (cv2.imread(mask_image_path) > 128).astype(np.uint8)[:, :, 0]
        
        # 确保掩码与原始图像尺寸匹配
        if original_mask.shape[:2] != old_tar_image.shape[:2]:
            original_mask = cv2.resize(original_mask, (old_tar_image.shape[1], old_tar_image.shape[0]))
        
        # 创建三通道掩码
        original_mask_3ch = np.stack([original_mask, original_mask, original_mask], axis=-1)
        
        # 创建掩码可视化 - 在掩码区域显示白色
        visible_mask_array = np.where(original_mask_3ch > 0, np.ones_like(old_tar_image) * 255, old_tar_image).astype(np.uint8)
        visible_mask = Image.fromarray(visible_mask_array)
        
        # 创建复合图像
        total_width = original_image.width + visible_mask.width + edited_image.width
        max_height = max(original_image.height, visible_mask.height, edited_image.height)
        composite_image = Image.new('RGB', (total_width, max_height))
        
        composite_image.paste(original_image, (0, 0))
        composite_image.paste(visible_mask, (original_image.width, 0))
        composite_image.paste(edited_image, (original_image.width + visible_mask.width, 0))
        
        # 保存结果
        output_filename = f"aligned_infer_seed{args.seed}_{file_name}"
        output_path = os.path.join(save_dir, output_filename)
        composite_image.save(output_path)
        logger.info(f"保存结果到 {output_path}")
        
    # 清理内存
    del pipefill, pipeprior
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_dir

def parse_args():
    parser = argparse.ArgumentParser(description="与训练脚本中对齐的推理代码")
    
    # 模型路径
    parser.add_argument("--flux_fill_path", type=str, required=True, help="FluxFill模型路径")
    parser.add_argument("--flux_redux_path", type=str, required=True, help="FluxPriorRedux模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="LoRA权重路径")
    
    # 数据路径
    parser.add_argument("--val_json_path", type=str, required=True, help="验证数据JSON文件或目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 推理参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--aspect_ratio_buckets", type=str, default='512,512;', help="宽高桶，分号分隔的列表")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="推理步数")
    parser.add_argument("--max_samples", type=int, default=10, help="最大处理样本数")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        output_dir = run_aligned_inference(args)
        logger.info(f"推理完成，结果保存在 {output_dir}")
    except Exception as e:
        logger.error(f"推理过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
