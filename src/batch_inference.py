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
from data.all_data import load_triplet_paths_from_dir

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# VAE解码器补丁，保证类型一致性
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
    """加载模型和权重"""
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
    
    # 提取组件
    transformer = pipefill.transformer
    
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

def process_single_image(pipefill, pipeprior, source_path, mask_path, reference_path, args, output_dir, device, model_dtype):
    """处理单张图像"""
    logger.info(f"处理图像: {source_path}")
    
    # 读取源图像和掩码
    source_image = Image.open(source_path).convert("RGB")
    source_mask = Image.open(mask_path).convert("L")
    
    # 如果提供了参考图像，使用参考图像；否则使用源图像
    if reference_path and os.path.exists(reference_path):
        reference_image = Image.open(reference_path).convert("RGB")
    else:
        reference_image = source_image
        
    # 确保图像尺寸匹配
    source_width, source_height = source_image.size
    # aspect_ratio = source_width / source_height
    
    # # 调整图像大小，保持纵横比，确保是64的倍数
    # if aspect_ratio > 1:
    #     new_width = max(512, min(1024, source_width))
    #     new_height = int(new_width / aspect_ratio)
    # else:
    #     new_height = max(512, min(1024, source_height))
    #     new_width = int(new_height * aspect_ratio)
    
    # # 将调整大小设置为64的倍数
    # new_width = ((new_width + 63) // 64) * 64
    # new_height = ((new_height + 63) // 64) * 64
    new_width = 512
    new_height = 512
    
    logger.info(f"调整图像大小到: {new_width}x{new_height}")
    
    # 调整图像大小
    source_image_resized = source_image.resize((new_width, new_height))
    source_mask_resized = source_mask.resize((new_width, new_height))
    reference_image_resized = reference_image.resize((new_width, new_height))
    
    # 创建一个干净的参考图像（掩码部分设为白色）
    reference_array = np.array(reference_image_resized)
    mask_array = np.array(source_mask_resized) > 128
    
    # 创建3通道掩码
    mask_3ch = np.stack([mask_array, mask_array, mask_array], axis=-1)
    
    # 将掩码区域在参考图像中设置为白色
    clean_reference = reference_array.copy()
    clean_reference[mask_3ch] = 255
    clean_reference = Image.fromarray(clean_reference)
    
    # 创建双联图 (左边是参考图像，右边是源图像)
    total_width = new_width * 2
    diptych = Image.new('RGB', (total_width, new_height))
    diptych.paste(clean_reference, (0, 0))
    diptych.paste(source_image_resized, (new_width, 0))
    
    # 创建双联掩码 (左边是黑色，右边是掩码)
    mask_black = Image.new('RGB', (new_width, new_height), color=0)
    mask_white = Image.fromarray((mask_3ch * 255).astype(np.uint8))
    diptych_mask = Image.new('RGB', (total_width, new_height))
    diptych_mask.paste(mask_black, (0, 0))
    diptych_mask.paste(mask_white, (new_width, 0))
    
    # 获取先验模型输出
    with torch.no_grad():
        pipe_prior_output = pipeprior(clean_reference)
    
    # 转换类型
    for key, value in pipe_prior_output.items():
        if isinstance(value, torch.Tensor):
            pipe_prior_output[key] = value.to(dtype=model_dtype)
    
    # 执行推理
    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    
    # 生成图像
    result = pipefill(
        image=diptych,
        mask_image=diptych_mask,
        height=new_height,
        width=total_width,
        max_sequence_length=512,
        num_inference_steps=args.num_steps,
        generator=generator,
        **pipe_prior_output,
    ).images[0]
    
    # 裁剪结果，只保留右半部分（处理后的图像）
    edited_image = result.crop((new_width, 0, total_width, new_height))
    
    # 如果需要，将结果调整回原始大小
    if args.resize_to_original:
        edited_image = edited_image.resize((source_width, source_height))
    
    # 获取文件名用于保存
    file_name = os.path.basename(source_path)
    name, ext = os.path.splitext(file_name)
    
    output_filename = f"{name}_result_seed{args.seed}{ext}"
    output_path = os.path.join(output_dir, output_filename)
    
    # 创建比较图像
    comparison = Image.new('RGB', (source_width * 3, source_height))
    
    # 调整尺寸
    source_image_display = source_image.resize((source_width, source_height))
    # 确保掩码数组使用正确的类型
    mask_array = np.array(source_mask.resize((source_width, source_height))) > 128
    mask_display = Image.fromarray(mask_array.astype(np.uint8) * 255).convert("RGB")
    edited_display = edited_image.resize((source_width, source_height))
    
    # 粘贴图像
    comparison.paste(source_image_display, (0, 0))
    comparison.paste(mask_display, (source_width, 0))
    comparison.paste(edited_display, (source_width * 2, 0))
    
    comparison.save(output_path)
    
    logger.info(f"结果已保存至: {output_path}")
    return output_path

def run_batch_inference(args):
    """批量处理目录中的所有图像"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    
    # 加载模型
    pipefill, pipeprior = load_models_and_weights(args, device)
    model_dtype = torch.bfloat16
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果提供了数据目录，使用load_triplet_paths_from_dir加载路径
    if args.data_dir:
        logger.info(f"从目录加载图像: {args.data_dir}")
        triplet_paths = load_triplet_paths_from_dir(args.data_dir)
        logger.info(f"找到 {len(triplet_paths)} 个图像/掩码对")
        
        # 限制处理数量
        if args.max_images > 0:
            triplet_paths = triplet_paths[:args.max_images]
        
        for i, paths in enumerate(triplet_paths):
            logger.info(f"处理图像 {i+1}/{len(triplet_paths)}: {paths['input_image']}")
            source_path = paths["input_image"]
            mask_path = paths["mask"]
            reference_path = paths["edited_image"] if paths["edited_image"] and os.path.exists(paths["edited_image"]) else None
            
            process_single_image(
                pipefill, pipeprior, 
                source_path, mask_path, reference_path,
                args, args.output_dir, device, model_dtype
            )
    
    # 如果提供了单独的图像路径，处理单张图像
    elif args.source_image and args.source_mask:
        source_path = args.source_image
        mask_path = args.source_mask
        reference_path = args.reference_image
        
        process_single_image(
            pipefill, pipeprior, 
            source_path, mask_path, reference_path,
            args, args.output_dir, device, model_dtype
        )
    else:
        logger.error("必须提供data_dir参数或同时提供source_image和source_mask参数")
        return None
    
    # 清理内存
    del pipefill, pipeprior
    torch.cuda.empty_cache()
    gc.collect()
    
    return args.output_dir

def parse_args():
    parser = argparse.ArgumentParser(description="批量RemoveAnything推理")
    
    # 模型路径
    parser.add_argument("--flux_fill_path", type=str, required=True, help="FluxFill模型路径")
    parser.add_argument("--flux_redux_path", type=str, required=True, help="FluxPriorRedux模型路径")
    parser.add_argument("--lora_weights_path", type=str, required=True, help="LoRA权重路径")
    
    # 数据路径 - 两种模式：目录或单张图像
    parser.add_argument("--data_dir", type=str, help="包含input和mask子目录的数据目录")
    
    # 单张图像模式的参数
    parser.add_argument("--source_image", type=str, help="源图像路径")
    parser.add_argument("--source_mask", type=str, help="源图像掩码路径")
    parser.add_argument("--reference_image", type=str, help="参考图像路径，不提供则使用源图像")
    
    # 输出设置
    parser.add_argument("--output_dir", type=str, default="./batch_results", help="输出目录")
    parser.add_argument("--resize_to_original", action="store_true", help="将结果调整回原始大小")
    
    # 推理参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_steps", type=int, default=30, help="推理步数")
    parser.add_argument("--max_images", type=int, default=-1, help="最大处理图像数，-1表示处理所有图像")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        output_dir = run_batch_inference(args)
        if output_dir:
            logger.info(f"批处理完成，结果保存在 {output_dir}")
    except Exception as e:
        logger.error(f"推理过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
