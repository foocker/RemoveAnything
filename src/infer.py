#!/usr/bin/env python
# coding=utf-8
"""
RemoveAnything LoRA 权重推理脚本
用于加载训练好的 LoRA 权重进行物体移除推理
"""

from PIL import Image
import torch
import os
import numpy as np
import cv2
import time
import argparse
import logging
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from data.data_utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask
from data.all_data import load_triplet_paths_from_dir

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def parser_data_dir(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录 {data_dir} 不存在")
    img_dir = os.path.join(data_dir, "input")
    mask_dir = os.path.join(data_dir, "mask")
    image_list = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)])
    mask_list = sorted([os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir)])
    
    return image_list, mask_list

def parse_args():
    parser = argparse.ArgumentParser(description="RemoveAnything LoRA 权重推理脚本")
    
    # 模型路径
    parser.add_argument("--flux_fill_path", type=str, 
                        required=True,
                        help="FluxFill模型的路径")
    parser.add_argument("--lora_weights_path", type=str, 
                        required=True,
                        help="LoRA权重路径")
    parser.add_argument("--flux_redux_path", type=str, 
                        required=True,
                        help="FluxRedux模型路径")
    
    # 单图片模式
    parser.add_argument("--source_image", type=str, 
                        help="源图像路径")
    parser.add_argument("--source_mask", type=str, 
                        help="源蒙版路径")
    parser.add_argument("--ref_image", type=str, 
                        help="参考图像路径，默认与源图像相同", default=None)
    parser.add_argument("--ref_mask", type=str, 
                        help="参考蒙版路径，默认与源蒙版相同", default=None)
    
    # 目录模式
    parser.add_argument("--input_dir", type=str,
                        default=None,
                        help="输入目录，包含源图像和蒙版")
    
    # 输出路径
    parser.add_argument("--output_dir", type=str, 
                        default="./output",
                        help="结果保存目录")
    
    # 推理参数
    parser.add_argument("--seed", type=int, default=666, help="随机种子")
    parser.add_argument("--size", type=int, default=768, 
                        choices=[512, 768, 1024],
                        help="处理图像的大小")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="推理步数")
    parser.add_argument("--repeat", type=int, default=1,
                        help="推理重复次数")
    parser.add_argument("--expansion_ratio", type=float, default=2.0,
                        help="遮罩扩展比例，值越大裁剪区域越大")
    
    return parser.parse_args()

def infer_single_image(pipe, redux, args, source_image_path, mask_image_path, 
                       ref_image_path=None, ref_mask_path=None, device="cuda", dtype=torch.bfloat16):
    """处理单个图像和蒙版对
    
    Args:
        pipe: FluxFillPipeline模型
        redux: FluxPriorReduxPipeline模型
        args: 命令行参数
        source_image_path: 源图像路径
        mask_image_path: 源蒙版路径
        ref_image_path: 参考图像路径，默认与源图像相同
        ref_mask_path: 参考蒙版路径，默认与源蒙版相同
        device: 设备
        dtype: 数据类型
    """
    # 设置参考图像，如果没有提供则使用源图像
    ref_image_path = ref_image_path if ref_image_path else source_image_path
    ref_mask_path = ref_mask_path if ref_mask_path else mask_image_path
    
    logger.info(f"处理图像: {source_image_path} 和蒙版: {mask_image_path}")
    
    # 加载图像和蒙版
    ref_image = cv2.imread(ref_image_path)
    if ref_image is None:
        logger.error(f"无法加载参考图像: {ref_image_path}")
        return
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    
    tar_image = cv2.imread(source_image_path)
    if tar_image is None:
        logger.error(f"无法加载源图像: {source_image_path}")
        return
    tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)
    
    ref_mask = cv2.imread(ref_mask_path)
    if ref_mask is None:
        logger.error(f"无法加载参考蒙版: {ref_mask_path}")
        return
    ref_mask = (ref_mask > 128).astype(np.uint8)[:, :, 0]
    
    tar_mask = cv2.imread(mask_image_path)
    if tar_mask is None:
        logger.error(f"无法加载源蒙版: {mask_image_path}")
        return
    tar_mask = (tar_mask > 128).astype(np.uint8)[:, :, 0]
    
    # 确保蒙版和图像尺寸匹配
    if tar_mask.shape[:2] != (tar_image.shape[0], tar_image.shape[1]):
        logger.info("调整蒙版尺寸以匹配图像")
        tar_mask = cv2.resize(tar_mask, (tar_image.shape[1], tar_image.shape[0]))

    # 处理参考图像
    logger.info("处理参考图像...")
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    
    # 提取参考图像中对象的区域
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :] 
    ref_mask = ref_mask[y1:y2, x1:x2] 
    
    # 扩展参考图像，与 infer.py 保持一致使用 1.3 的比例
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    
    # 扩展目标蒙版，与 infer.py 保持一致
    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)
    
    # 处理目标图像
    logger.info("处理目标图像...")
    # 获取目标边界框，与 infer.py 保持一致
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    # 先扩展蒙版区域
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
    # 再扩展到更大的裁剪区域
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=args.expansion_ratio)   
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop
    
    # 保存原始图像用于后处理
    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]
    
    # 记录尺寸信息用于后处理
    H1, W1 = tar_image.shape[0], tar_image.shape[1]
    
    # 调整目标蒙版尺寸
    tar_mask = pad_to_square(tar_mask, pad_value=0)
    size = (args.size, args.size)
    tar_mask = cv2.resize(tar_mask, size)
    
    # 提取参考图像特征
    logger.info("提取参考图像特征...")
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    
    # 获取先验输出
    with torch.no_grad():
        pipe_prior_output = redux(Image.fromarray(masked_ref_image))
    
    # 确保所有张量类型一致
    for key, value in pipe_prior_output.items():
        if isinstance(value, torch.Tensor):
            pipe_prior_output[key] = value.to(device=device, dtype=dtype)
    
    # 调整目标图像尺寸
    tar_image = pad_to_square(tar_image, pad_value=255)
    H2, W2 = tar_image.shape[0], tar_image.shape[1]
    tar_image = cv2.resize(tar_image, size)
    
    # 创建双图像和蒙版
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)
    tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
    mask_black = np.ones_like(tar_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
    
    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)
    
    # 显示处理过程
    logger.info("准备推理...")
    
    # 推理
    seeds = [args.seed]
    repeat = args.repeat
    num_inference_steps = args.num_inference_steps
    
    results = []
    
    for seed_idx, seed in enumerate(seeds):
        logger.info(f"使用种子 {seed} 进行推理 ({seed_idx + 1}/{len(seeds)})")
        generator = torch.Generator(device).manual_seed(seed)
        
        for i in range(repeat):
            logger.info(f"推理 {i + 1}/{repeat}")
            start_time = time.time()
            
            # 执行推理
            edited_image = pipe(
                image=diptych_ref_tar,
                mask_image=mask_diptych,
                height=mask_diptych.size[1],
                width=mask_diptych.size[0],
                max_sequence_length=512,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **pipe_prior_output,
            ).images[0]
            
            end_time = time.time()
            logger.info(f"推理耗时: {end_time - start_time:.2f} 秒")
            
            # 裁剪结果
            width, height = edited_image.size
            left = width // 2
            right = width
            top = 0
            bottom = height
            edited_image = edited_image.crop((left, top, right, bottom))
            
            # 将结果放回原始图像
            edited_image = np.array(edited_image)
            edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
            edited_image = Image.fromarray(edited_image)
            
            # 保存结果
            source_filename = os.path.splitext(os.path.basename(source_image_path))[0]
            mask_filename = os.path.splitext(os.path.basename(mask_image_path))[0]
            output_filename = f"{source_filename}_mask_{mask_filename}_seed{seed}_steps{num_inference_steps}_size{args.size}_repeat{i}.png"
            output_path = os.path.join(args.output_dir, output_filename)
            edited_image.save(output_path)
            logger.info(f"结果已保存到: {output_path}")
            results.append(output_path)
    
    return results


def detect_lora_type(lora_path):
    """检测LoRA权重的类型
    
    Args:
        lora_path: LoRA权重路径
        
    Returns:
        str: 'moe_experts', 'multi_adapter', 'single_lora', 或 'unknown'
    """
    if not os.path.exists(lora_path):
        return 'unknown'
    
    # 检查是否为MoE专家型
    moe_config_path = os.path.join(lora_path, 'moe_config.json')
    if os.path.exists(moe_config_path):
        return 'moe_experts'
    
    # 检查是否为多适配器型
    multi_config_path = os.path.join(lora_path, 'multi_adapter_config.json')
    if os.path.exists(multi_config_path):
        return 'multi_adapter'
    
    # 检查是否为单LoRA型
    single_lora_path = os.path.join(lora_path, 'pytorch_lora_weights.safetensors')
    if os.path.exists(single_lora_path):
        return 'single_lora'
    
    return 'unknown'


def load_lora_weights_unified(pipe, lora_path, target_modules):
    """统一的LoRA权重加载函数，支持三种格式
    
    Args:
        pipe: FluxFillPipeline实例
        lora_path: LoRA权重路径
        target_modules: 目标模块列表
        
    Returns:
        bool: 是否成功加载
    """
    import json
    from peft import LoraConfig
    
    lora_type = detect_lora_type(lora_path)
    logger.info(f"检测到LoRA类型: {lora_type}")
    
    transformer = pipe.transformer
    
    try:
        if lora_type == 'moe_experts':
            # MoE专家型：加载多个专家适配器
            moe_config_path = os.path.join(lora_path, 'moe_config.json')
            with open(moe_config_path, 'r') as f:
                moe_config = json.load(f)
            
            adapter_names = moe_config.get('adapter_names', [])
            logger.info(f"发现MoE专家: {adapter_names}")
            
            # 为每个专家添加适配器
            for i, adapter_name in enumerate(adapter_names):
                expert_dir = os.path.join(lora_path, f'lora_{adapter_name}')
                if os.path.exists(expert_dir):
                    logger.info(f"加载专家适配器: {adapter_name}")
                    
                    # 创建适配器配置
                    lora_config = LoraConfig(
                        r=64,
                        lora_alpha=64,
                        init_lora_weights="gaussian",
                        target_modules=target_modules
                    )
                    
                    # 添加适配器
                    adapter_id = f"expert_{i}"
                    transformer.add_adapter(lora_config, adapter_name=adapter_id)
                    
                    # 加载权重
                    pipe.load_lora_weights(expert_dir, adapter_name=adapter_id)
                    logger.info(f"成功加载专家 {adapter_name} 权重")
                else:
                    logger.warning(f"专家目录不存在: {expert_dir}")
            
            # 设置默认权重（均匀分布）
            num_experts = len(adapter_names)
            if num_experts > 0:
                equal_weight = 1.0 / num_experts
                adapter_weights = [equal_weight] * num_experts
                pipe.set_adapters([f"expert_{i}" for i in range(num_experts)], adapter_weights)
                logger.info(f"设置MoE专家权重: {adapter_weights}")
            
        elif lora_type == 'multi_adapter':
            # 多适配器型：加载多个适配器并按配置权重混合
            multi_config_path = os.path.join(lora_path, 'multi_adapter_config.json')
            with open(multi_config_path, 'r') as f:
                multi_config = json.load(f)
            
            adapters = multi_config.get('adapters', [])
            adapter_weights = multi_config.get('adapter_weights', [])
            logger.info(f"发现多适配器: {adapters}, 权重: {adapter_weights}")
            
            # 为每个适配器添加配置
            loaded_adapters = []
            for i, adapter_name in enumerate(adapters):
                adapter_dir = os.path.join(lora_path, f'lora_{adapter_name}')
                if os.path.exists(adapter_dir):
                    logger.info(f"加载适配器: {adapter_name}")
                    
                    # 创建适配器配置
                    lora_config = LoraConfig(
                        r=64,
                        lora_alpha=64,
                        init_lora_weights="gaussian",
                        target_modules=target_modules
                    )
                    
                    # 添加适配器
                    adapter_id = f"adapter_{i}"
                    transformer.add_adapter(lora_config, adapter_name=adapter_id)
                    
                    # 加载权重
                    pipe.load_lora_weights(adapter_dir, adapter_name=adapter_id)
                    loaded_adapters.append(adapter_id)
                    logger.info(f"成功加载适配器 {adapter_name} 权重")
                else:
                    logger.warning(f"适配器目录不存在: {adapter_dir}")
            
            # 设置适配器权重
            if loaded_adapters and adapter_weights:
                # 确保权重数量匹配
                weights = adapter_weights[:len(loaded_adapters)]
                if len(weights) < len(loaded_adapters):
                    # 如果权重不足，用均匀分布补充
                    remaining = len(loaded_adapters) - len(weights)
                    equal_weight = (1.0 - sum(weights)) / remaining if remaining > 0 else 0
                    weights.extend([equal_weight] * remaining)
                
                pipe.set_adapters(loaded_adapters, weights)
                logger.info(f"设置多适配器权重: {weights}")
            
        elif lora_type == 'single_lora':
            # 单LoRA型：传统单一适配器
            logger.info("加载单一LoRA适配器")
            
            # 创建适配器配置
            lora_config = LoraConfig(
                r=64,
                lora_alpha=64,
                init_lora_weights="gaussian",
                target_modules=target_modules
            )
            
            # 添加适配器
            transformer.add_adapter(lora_config)
            
            # 加载权重
            pipe.load_lora_weights(lora_path)
            logger.info("成功加载单一LoRA权重")
            
        else:
            logger.error(f"未知的LoRA类型: {lora_type}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"LoRA权重加载失败: {str(e)}")
        logger.error(f"错误详情: {type(e).__name__}")
        return False


def load_weights(args, device, dtype):
    # 加载预训练模型
    logger.info("加载 FluxFill 模型...")
    pipe = FluxFillPipeline.from_pretrained(
        args.flux_fill_path,
        torch_dtype=dtype,
        use_safetensors=True
    )
    
    # 禁用 xformers 以确保兼容性
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe._use_memory_efficient_attention_xformers = False
        logger.info("已禁用 xformers 内存优化")
    
    # 定义目标模块
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
    
    # 使用统一的LoRA加载函数
    logger.info(f"加载 LoRA 权重: {args.lora_weights_path}")
    success = load_lora_weights_unified(pipe, args.lora_weights_path, target_modules)
    
    if not success:
        logger.error("LoRA权重加载失败")
        return None, None
    
    # 加载 FluxPriorRedux 模型
    logger.info("加载 FluxPriorRedux 模型...")
    redux = FluxPriorReduxPipeline.from_pretrained(
        args.flux_redux_path,
        torch_dtype=dtype,
    )
    
    # 将模型移至GPU
    pipe.to(device=device, dtype=dtype)
    redux.to(device=device, dtype=dtype)
    
    logger.info("模型已加载到设备")
    return pipe, redux


def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备和数据类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    logger.info(f"使用设备: {device}, 数据类型: {dtype}")
    
    # 加载模型权重
    pipe, redux = load_weights(args, device, dtype)
    
    # 检查模型是否成功加载
    if pipe is None or redux is None:
        logger.error("模型加载失败，退出程序")
        return
        
    # 设置随机种子
    set_seed(args.seed)
        
    input_dir = args.input_dir if args.input_dir is not None else ""
    
    if os.path.isdir(input_dir):
        logger.info(f"使用目录模式，解析目录: {input_dir}")
        triplet_paths = load_triplet_paths_from_dir(input_dir)
        
        for i, paths in enumerate(triplet_paths):
            logger.info(f"处理图像 {i+1}/{len(triplet_paths)}: {paths['input_image']}")
            source_path = paths["input_image"]
            mask_path = paths["mask"]
            reference_path = paths["edited_image"] if paths["edited_image"] and os.path.exists(paths["edited_image"]) else None
            infer_single_image(
                pipe=pipe,
                redux=redux,
                args=args,
                source_image_path=source_path,
                mask_image_path=mask_path,
                ref_image_path=reference_path,
                ref_mask_path=mask_path,
                device=device,
                dtype=dtype
            )
    else:
        # 单图片模式
        logger.info("使用单图片模式")
        source_image_path = args.source_image
        mask_image_path = args.source_mask
            
        # 检查必要的参数
        if not source_image_path or not mask_image_path:
            logger.error("单图片模式下必须提供source_image和source_mask参数")
            return
            
        # 设置参考图像，如果没有提供则使用源图像
        ref_image_path = args.ref_image if args.ref_image else source_image_path
        ref_mask_path = args.ref_mask if args.ref_mask else mask_image_path
            
        # 调用单图片处理函数
        infer_single_image(
            pipe=pipe,
            redux=redux,
            args=args,
            source_image_path=source_image_path,
            mask_image_path=mask_image_path,
            ref_image_path=ref_image_path,
            ref_mask_path=ref_mask_path,
            device=device,
            dtype=dtype
        )
            
    logger.info("推理完成!")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
