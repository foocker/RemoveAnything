#!/usr/bin/env python
# coding=utf-8
"""
RemoveAnything Web Interface
基于Gradio的物体移除Web界面
"""

import os
import sys
import time
import torch
import numpy as np
import gradio as gr
import argparse
import logging
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.image_processing import process_image_for_removal

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RemoveAnything Web Interface")
    
    # 模型路径 - 从命令行传入
    parser.add_argument("--flux_fill_path", type=str, 
                        required=True,
                        help="FluxFill模型的路径")
    parser.add_argument("--lora_weights_path", type=str, 
                        required=True,
                        help="LoRA权重路径")
    parser.add_argument("--flux_redux_path", type=str, 
                        required=True,
                        help="FluxRedux模型路径")
    
    # 模型参数
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["auto", "cuda", "cpu"], 
                        help="运行设备，'auto'会自动选择CUDA（如果可用）或CPU")
    parser.add_argument("--dtype", type=str, default="bf16", 
                        choices=["bf16", "fp16", "fp32"], 
                        help="模型精度类型，设置为fp32可使用更少显存")
    parser.add_argument("--offload_modules", action="store_true", 
                        help="启用模型组件卸载以节省显存")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="是否公开分享")
    
    # 输出路径
    parser.add_argument("--output_dir", type=str, default="./output", help="结果保存目录")
    
    return parser.parse_args()


def pil_to_numpy(pil_image):
    """将PIL图像转换为numpy数组"""
    return np.array(pil_image)

def numpy_to_pil(numpy_image):
    """将numpy数组转换为PIL图像"""
    return Image.fromarray(numpy_image.astype(np.uint8))

def mask_to_numpy(mask_image):
    """将蒙版图像转换为二值numpy数组"""
    if isinstance(mask_image, Image.Image):
        mask_array = np.array(mask_image)
    else:
        mask_array = mask_image
        
    if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
        mask_array = mask_array[:, :, 0]  # 取第一个通道
    
    mask_array = (mask_array > 128).astype(np.uint8)
    
    return mask_array

def load_models(flux_fill_path, lora_weights_path, flux_redux_path, device_arg="auto", dtype_arg="bf16", offload_modules=False):
    """加载模型和权重"""
    logger.info("正在加载模型...")
    
    # 设置设备
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    # 设置数据类型
    if dtype_arg == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        logger.info("使用 bfloat16 精度")
    elif dtype_arg == "fp16" and device.type == "cuda":
        dtype = torch.float16
        logger.info("使用 float16 精度")
    else:
        dtype = torch.float32
        logger.info("使用 float32 精度")
    
    # 记录CUDA内存情况
    if device.type == "cuda":
        try:
            logger.info(f"加载模型前 CUDA 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception:
            pass
    
    # 尝试设置更大的分割大小来减少内存碎片
    if device.type == "cuda":
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 使用低内存模式加载模型
    try:
        # 加载FluxFill模型
        pipe = FluxFillPipeline.from_pretrained(
            flux_fill_path,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        
        # 内存优化设置
        pipe.enable_attention_slicing()
        
        if device.type == "cuda":
            # 禁用xformers以确保兼容性
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe._use_memory_efficient_attention_xformers = False
                logger.info("已禁用xformers内存优化")
            
            # 启用梯度检查点优化
            pipe.enable_vae_slicing()
        
        # 加载LoRA权重
        try:
            pipe.load_lora_weights(lora_weights_path)
            logger.info(f"成功加载LoRA权重: {lora_weights_path}")
        except Exception as e:
            logger.error(f"加载LoRA权重出错: {str(e)}")
            return None, None, None, None
        
        # 加载FluxPriorRedux模型
        redux = FluxPriorReduxPipeline.from_pretrained(
            flux_redux_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        # 根据卸载选项决定如何处理模型
        if offload_modules and device.type == "cuda":
            # 使用CPU卸载来节省GPU内存
            logger.info("使用模型组件CPU卸载来节省GPU内存")
            pipe.enable_model_cpu_offload()
            redux.enable_model_cpu_offload()
        else:
            # 正常将模型移到设备
            logger.info(f"将模型移至设备: {device}")
            
            # 分模块移动模型到设备以减少内存占用峰值
            if device.type == "cuda":
                # 分别移动模型组件以减少内存峰值
                for name, module in pipe.components.items():
                    if hasattr(module, "to") and callable(module.to):
                        module.to(device=device, dtype=dtype if name != "safety_checker" else None)
                
                # 将Redux模型移到设备
                redux.to(device=device, dtype=dtype)
            else:
                # 在CPU上直接移动整个模型
                pipe.to(device=device)
                redux.to(device=device)
        
        # 记录内存使用
        if device.type == "cuda":
            try:
                logger.info(f"加载模型后 CUDA 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            except Exception:
                pass
        
        return pipe, redux, device, dtype
    
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def process_image(pipe, redux, source_image, source_mask, size=768, num_inference_steps=30, 
                 seed=666, expansion_ratio=2.0, device="cuda", dtype=torch.bfloat16):
    """处理图像和蒙版 - 使用共享的图像处理函数"""
    return process_image_for_removal(
        pipe=pipe, 
        redux=redux, 
        source_image=source_image, 
        source_mask=source_mask, 
        size=size, 
        num_inference_steps=num_inference_steps, 
        seed=seed, 
        expansion_ratio=expansion_ratio, 
        device=device, 
        dtype=dtype
    )

models = None
args = None

def gradio_process_image(input_image, input_mask, size, steps, seed, expansion_ratio):
    """Gradio接口函数"""
    global models, args
    
    if input_image is None:
        return None, "请上传源图像"
    
    if input_mask is None:
        return None, "请上传或绘制蒙版"
    
    try:
        result = process_image(
            pipe=models["pipe"],
            redux=models["redux"],
            source_image=input_image,
            source_mask=input_mask,
            size=size,
            num_inference_steps=steps,
            seed=seed,
            expansion_ratio=expansion_ratio,
            device=models["device"],
            dtype=models["dtype"]
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = int(time.time())
        output_filename = f"result_seed{seed}_steps{steps}_size{size}_{timestamp}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        result.save(output_path)
        
        return result, f"处理成功! 结果保存在: {output_path}"
        
    except Exception as e:
        logger.error(f"处理图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"处理失败: {str(e)}"

def create_ui(args):
    """创建Gradio界面"""
    with gr.Blocks(title="RemoveAnything - 物体移除工具") as app:
        gr.Markdown("""
        # RemoveAnything - 物体移除工具
        
        使用FLUX模型和LoRA微调的物体移除工具。上传图像并绘制要移除的物体蒙版。
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入部分
                gr.Markdown("### 输入")
                input_image = gr.Image(label="源图像", type="pil")
                input_mask = gr.Image(label="蒙版 (白色区域表示要移除的物体)", 
                                       type="numpy",
                                       tool="sketch", 
                                       brush_radius=20,
                                       brush_color="white")
                
                # 参数设置
                with gr.Accordion("高级设置", open=False):
                    size = gr.Slider(label="图像大小", minimum=512, maximum=1024, value=768, step=128)
                    steps = gr.Slider(label="推理步数", minimum=10, maximum=50, value=30, step=1)
                    seed = gr.Slider(label="随机种子", minimum=0, maximum=2147483647, value=666, step=1)
                    expansion_ratio = gr.Slider(label="遮罩扩展比例", minimum=1.0, maximum=4.0, value=2.0, step=0.1)
                
                # 处理按钮
                process_btn = gr.Button("处理图像", variant="primary")
                
            with gr.Column(scale=1):
                # 输出部分
                gr.Markdown("### 输出")
                output_image = gr.Image(label="结果图像", type="pil")
                output_message = gr.Textbox(label="处理信息")
        
        # 设置事件处理
        process_btn.click(
            fn=gradio_process_image,
            inputs=[input_image, input_mask, size, steps, seed, expansion_ratio],
            outputs=[output_image, output_message]
        )
        
        # 添加示例
        gr.Examples(
            [
                ["examples/image/000004.png", "examples/mask/000004_mask.png", 768, 30, 666, 2.0],
            ],
            inputs=[input_image, input_mask, size, steps, seed, expansion_ratio],
        )
    
    return app

def main():
    global args, models
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    dtype_arg = args.dtype
    
    logger.info(f"使用设备: {args.device}, 数据类型: {dtype_arg}, 卸载模块: {args.offload_modules}")
    
    pipe, redux, device, dtype = load_models(
        args.flux_fill_path,
        args.lora_weights_path,
        args.flux_redux_path,
        device_arg=args.device,
        dtype_arg=dtype_arg,
        offload_modules=args.offload_modules
    )
    
    if pipe is None or redux is None:
        logger.error("加载模型失败，无法启动应用")
        return
    
    models = {
        "pipe": pipe,
        "redux": redux,
        "device": device,
        "dtype": dtype
    }
    
    app = create_ui(args)
    
    logger.info(f"启动Gradio服务器: http://{args.host}:{args.port}")
    logger.info("使用方法: 上传图像、绘制蒙版，然后点击'处理图像'")
    
    if device.type == "cpu":
        logger.warning("在CPU上运行模型，推理速度将很慢，请耐心等待")
    
    app.launch(server_name=args.host, server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
