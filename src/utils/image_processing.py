#!/usr/bin/env python
# coding=utf-8
"""
共享的图像处理功能
"""

import numpy as np
import cv2
import torch
import logging
from PIL import Image
from ..data.data_utils import get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask

# 设置日志记录
logger = logging.getLogger(__name__)

def process_image_for_removal(pipe, redux, source_image, source_mask, size=768, num_inference_steps=30, 
                             seed=666, expansion_ratio=2.0, device="cuda", dtype=torch.bfloat16):
    """
    处理图像和蒙版以进行物体移除
    
    Args:
        pipe: FluxFillPipeline模型
        redux: FluxPriorReduxPipeline模型
        source_image: 源图像，可以是PIL图像、numpy数组或文件路径
        source_mask: 源蒙版，可以是PIL图像、numpy数组或文件路径
        size: 推理尺寸
        num_inference_steps: 推理步数
        seed: 随机种子
        expansion_ratio: 边界框扩展比例
        device: 设备
        dtype: 数据类型
    
    Returns:
        PIL图像: 处理后的图像
    """
    # 转换输入图像格式
    if isinstance(source_image, str):
        source_image = cv2.imread(source_image)
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    elif isinstance(source_image, Image.Image):
        source_image = np.array(source_image)
    
    # 转换蒙版格式
    if isinstance(source_mask, str):
        source_mask = cv2.imread(source_mask)
        source_mask = (source_mask > 128).astype(np.uint8)[:, :, 0]
    elif isinstance(source_mask, Image.Image):
        source_mask = np.array(source_mask)
        if len(source_mask.shape) == 3 and source_mask.shape[2] > 1:
            source_mask = source_mask[:, :, 0]
        source_mask = (source_mask > 128).astype(np.uint8)
    elif isinstance(source_mask, np.ndarray) and len(source_mask.shape) == 3:
        source_mask = (source_mask > 128).astype(np.uint8)[:, :, 0]
    
    # 确保蒙版和图像尺寸匹配
    if source_mask.shape[:2] != (source_image.shape[0], source_image.shape[1]):
        logger.info("调整蒙版尺寸以匹配图像")
        source_mask = cv2.resize(source_mask, (source_image.shape[1], source_image.shape[0]))
    
    # 引用图像和蒙版（与源图像相同）
    ref_image = source_image.copy()
    ref_mask = source_mask.copy()
    
    # 保存原始图像用于后处理
    old_source_image = source_image.copy()
    
    # 处理参考图像
    logger.info("处理参考图像...")
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    
    # 提取参考图像中对象的区域
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :] 
    ref_mask = ref_mask[y1:y2, x1:x2] 
    
    # 扩展参考图像
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio) 
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    
    # 扩展目标蒙版
    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(source_mask, kernel, iterations=iterations)
    
    # 处理目标图像
    logger.info("处理目标图像...")
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)
    tar_box_yyxx_crop = expand_bbox(source_image, tar_box_yyxx, ratio=expansion_ratio)   
    tar_box_yyxx_crop = box2squre(source_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop
    
    source_image = source_image[y1:y2, x1:x2, :]
    tar_mask = tar_mask[y1:y2, x1:x2]
    
    # 记录尺寸信息用于后处理
    H1, W1 = source_image.shape[0], source_image.shape[1]
    
    # 调整目标蒙版尺寸
    tar_mask = pad_to_square(tar_mask, pad_value=0)
    img_size = (size, size)
    tar_mask = cv2.resize(tar_mask, img_size)
    
    # 提取参考图像特征
    logger.info("提取参考图像特征...")
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), img_size).astype(np.uint8)
    
    # 获取先验输出
    with torch.no_grad():
        pipe_prior_output = redux(Image.fromarray(masked_ref_image))
    
    # 确保所有张量类型一致
    for key, value in pipe_prior_output.items():
        if isinstance(value, torch.Tensor):
            pipe_prior_output[key] = value.to(device=device, dtype=dtype)
    
    # 调整目标图像尺寸
    source_image = pad_to_square(source_image, pad_value=255)
    H2, W2 = source_image.shape[0], source_image.shape[1]
    source_image = cv2.resize(source_image, img_size)
    
    # 创建双图像和蒙版
    diptych_ref_tar = np.concatenate([masked_ref_image, source_image], axis=1)
    tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
    mask_black = np.ones_like(source_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)
    
    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)
    
    # 推理
    logger.info("开始推理...")
    start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    
    if device == "cuda":
        start_time.record()
    else:
        import time
        start_time_cpu = time.time()
    
    # 设置随机种子和生成器
    from src.infer import set_seed
    set_seed(seed)
    generator = torch.Generator(device).manual_seed(seed)
    
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
    
    if device == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        elapsed = start_time.elapsed_time(end_time) / 1000.0
        logger.info(f"推理耗时: {elapsed:.2f} 秒")
    else:
        elapsed = time.time() - start_time_cpu
        logger.info(f"推理耗时: {elapsed:.2f} 秒")
    
    # 裁剪结果
    width, height = edited_image.size
    left = width // 2
    right = width
    top = 0
    bottom = height
    edited_image = edited_image.crop((left, top, right, bottom))
    
    # 将结果放回原始图像
    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_source_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
    edited_image = Image.fromarray(edited_image)
    
    # 返回结果
    return edited_image
