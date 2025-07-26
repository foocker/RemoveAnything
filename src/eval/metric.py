import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None
import open_clip
from dataclasses import dataclass
from typing import Optional, List
import yaml
import json
import torchvision


@dataclass
class MetricConfig:
    """Configuration class for specifying which metrics to evaluate"""
    # Basic image quality metrics
    enable_psnr: bool = True
    enable_ssim: bool = True
    enable_lpips: bool = False  # 禁用LPIPS避免下载VGG16模型 
    
    # Deep learning based metrics
    enable_clip: bool = False 
    enable_dino: bool = False 
    enable_fid: bool = True
    
    # Model configurations
    lpips_net: str = 'vgg'  # 'alex', 'vgg', 'squeeze'
    clip_model: str = 'ViT-B/32'  # OpenCLIP model name
    clip_model_path: Optional[str] = None
    dino_model: str = 'dino_vits16'  # DINO model variant
    dino_model_path: Optional[str] = None
    fid_weights_path: Optional[str] = '/aicamera-mlp/fq_proj/weights/pt_inception-2015-12-05-6726825d.pth'  # Custom FID weights
    
    # Input/output paths
    gt_dir: Optional[str] = None    # 目标图像目录
    pred_dir: Optional[str] = None  # 预测图像目录
    text_dir: Optional[str] = None  # 文本提示目录
    
    # Device and performance
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    batch_size: int = 1
    
    # Output configuration
    output_precision: int = 4  # decimal places for results
    save_individual_results: bool = True
    
    # Text prompt for CLIP evaluation (if needed)
    default_text_prompt: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'MetricConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'MetricConfig':
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = self.__dict__.copy()
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def to_json(self, output_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = self.__dict__.copy()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def get_enabled_metrics(self) -> List[str]:
        """Get list of enabled metric names"""
        enabled = []
        if self.enable_psnr:
            enabled.append('psnr')
        if self.enable_ssim:
            enabled.append('ssim')
        if self.enable_lpips:
            enabled.append('lpips')
        if self.enable_clip:
            enabled.extend(['clip_sim_img1', 'clip_sim_img2'])
        if self.enable_dino:
            enabled.append('dino_sim')
        if self.enable_fid:
            enabled.append('fid')
        return enabled


class ImageEvaluator:
    def __init__(self, config: MetricConfig = None):
        if config is None:
            config = MetricConfig()  # 使用默认配置
        
        self.config = config
        self.device = config.device if config.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 初始化LPIPS
        if config.enable_lpips:
            print(f"Loading LPIPS model: {config.lpips_net}")
            self.lpips_fn = lpips.LPIPS(net=config.lpips_net).to(self.device)
        else:
            self.lpips_fn = None
        
        # 初始化CLIP
        self.clip_model = None
        self.clip_preprocess = None
        if config.enable_clip:
            try:
                if config.clip_model_path:
                    # 从本地路径加载CLIP模型
                    print(f"Loading CLIP model from local path: {config.clip_model_path}")
                    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                        config.clip_model, pretrained=config.clip_model_path
                    )
                else:
                    # 使用默认在线下载方式
                    print(f"Loading CLIP model online: {config.clip_model}")
                    self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(config.clip_model)
                
                self.clip_model = self.clip_model.to(self.device)
                print("CLIP model loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load CLIP model: {e}")
                self.clip_model = None
        
        # 初始化DINO
        self.dino_model = None
        if config.enable_dino:
            try:
                if config.dino_model_path:
                    # 从本地路径加载DINO模型
                    print(f"Loading DINO model from local path: {config.dino_model_path}")
                    self.dino_model = torch.load(config.dino_model_path, map_location='cpu')
                    # 如果加载的是状态字典，需要创建模型结构
                    if isinstance(self.dino_model, dict):
                        # 先创建模型结构，再加载权重
                        model_structure = torch.hub.load('facebookresearch/dino:main', config.dino_model, pretrained=False)
                        model_structure.load_state_dict(self.dino_model)
                        self.dino_model = model_structure
                else:
                    # 使用默认在线下载方式
                    print(f"Loading DINO model online: {config.dino_model}")
                    self.dino_model = torch.hub.load('facebookresearch/dino:main', config.dino_model)
                
                self.dino_model = self.dino_model.to(self.device)
                self.dino_model.eval()
                print("DINO model loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load DINO model: {e}")
                self.dino_model = None
        
        # DINO图像预处理
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def calculate_psnr(self, img1, img2):
        """计算PSNR"""
        img1_np = np.array(img1) / 255.0
        img2_np = np.array(img2) / 255.0
        return peak_signal_noise_ratio(img1_np, img2_np)
    
    def calculate_ssim(self, img1, img2):
        """计算SSIM"""
        img1_np = np.array(img1) / 255.0
        img2_np = np.array(img2) / 255.0
        return structural_similarity(img1_np, img2_np, channel_axis=-1, data_range=1.0)
    
    def calculate_lpips(self, img1, img2):
        """计算LPIPS"""
        if not self.config.enable_lpips or self.lpips_fn is None:
            return None
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1_tensor, img2_tensor)
            raw_value = lpips_value.item()
            print(f"LPIPS raw value: {raw_value}")
        return raw_value
    
    def calculate_clip_similarity(self, img, text):
        """计算CLIP相似度"""
        if not self.config.enable_clip or self.clip_model is None:
            return None
            
        img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        text_token = open_clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            img_features = self.clip_model.encode_image(img_tensor)
            text_features = self.clip_model.encode_text(text_token)
            
            # 归一化特征
            img_features /= img_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (img_features @ text_features.T).item()
        
        return similarity
    
    def calculate_dino_similarity(self, img1, img2):
        """计算DINO特征相似度"""
        if not self.config.enable_dino or self.dino_model is None:
            return None
            
        img1_tensor = self.dino_transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.dino_transform(img2).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat1 = self.dino_model(img1_tensor)
            feat2 = self.dino_model(img2_tensor)
            
            # 归一化特征
            feat1 /= feat1.norm(dim=-1, keepdim=True)
            feat2 /= feat2.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = torch.cosine_similarity(feat1, feat2).item()
        
        return similarity
    
    @staticmethod
    def calculate_fid(dir1, dir2, custom_weights_path=None, batch_size=16, device='auto'):
        """计算FID (需要目录路径) - 支持自定义权重"""
        try:
            # 检查两个目录中的图像文件
            files1 = [f for f in os.listdir(dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            files2 = [f for f in os.listdir(dir2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if len(files1) == 0 or len(files2) == 0:
                print(f"警告: 目录中没有找到图像文件: {dir1}({len(files1)}张图), {dir2}({len(files2)}张图)")
                return None
                
            print(f"计算FID: 目录1有{len(files1)}张图像, 目录2有{len(files2)}张图像")
            
            # 优先使用自定义FID实现
            if custom_weights_path and os.path.exists(custom_weights_path):
                print(f"使用自定义权重计算FID: {custom_weights_path}")
                return ImageEvaluator._calculate_fid_with_custom_weights(
                    dir1, dir2, custom_weights_path, batch_size, device
                )
            
            # 回退到torch_fidelity（如果可用）
            if calculate_metrics is not None:
                print("使用torch_fidelity计算FID")
                metrics = calculate_metrics(
                    input1=dir1,
                    input2=dir2,
                    cuda=torch.cuda.is_available(),
                    fid=True,
                    batch_size=batch_size,
                    num_workers=1,
                    verbose=True,
                    cache=False
                )
                return metrics.get('frechet_inception_distance', metrics.get('fid', None))
            else:
                print("torch_fidelity不可用，使用默认FID实现")
                return ImageEvaluator._calculate_fid_with_custom_weights(
                    dir1, dir2, None, batch_size, device
                )
                
        except Exception as e:
            print(f"FID计算错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _calculate_fid_with_custom_weights(dir1, dir2, custom_weights_path=None, batch_size=16, device='auto'):
        """使用自定义权重计算FID的内部实现"""
        try:
            # 导入必要的模块
            current_dir = os.path.dirname(__file__)
            pytorch_fid_path = os.path.join(current_dir, 'pytorch-fid')
            
            if not os.path.exists(pytorch_fid_path):
                raise ImportError(f"pytorch-fid目录不存在: {pytorch_fid_path}")
            
            # 临时添加到路径
            if pytorch_fid_path not in sys.path:
                sys.path.insert(0, pytorch_fid_path)
            
            # 动态导入模块
            import importlib.util
            
            # 加载inception模块
            inception_spec = importlib.util.spec_from_file_location(
                "inception", os.path.join(pytorch_fid_path, "inception.py")
            )
            inception_module = importlib.util.module_from_spec(inception_spec)
            inception_spec.loader.exec_module(inception_module)
            
            # 加载fid_score模块
            fid_score_spec = importlib.util.spec_from_file_location(
                "fid_score", os.path.join(pytorch_fid_path, "fid_score.py")
            )
            fid_score_module = importlib.util.module_from_spec(fid_score_spec)
            fid_score_spec.loader.exec_module(fid_score_module)
            
            # 如果有自定义权重，修改inception模块
            if custom_weights_path and os.path.exists(custom_weights_path):
                def custom_fid_inception_v3():
                    """使用自定义权重的FID Inception模型"""
                    inception = inception_module._inception_v3(num_classes=1008, aux_logits=False, weights=None)
                    inception.Mixed_5b = inception_module.FIDInceptionA(192, pool_features=32)
                    inception.Mixed_5c = inception_module.FIDInceptionA(256, pool_features=64)
                    inception.Mixed_5d = inception_module.FIDInceptionA(288, pool_features=64)
                    inception.Mixed_6b = inception_module.FIDInceptionC(768, channels_7x7=128)
                    inception.Mixed_6c = inception_module.FIDInceptionC(768, channels_7x7=160)
                    inception.Mixed_6d = inception_module.FIDInceptionC(768, channels_7x7=160)
                    inception.Mixed_6e = inception_module.FIDInceptionC(768, channels_7x7=192)
                    inception.Mixed_7b = inception_module.FIDInceptionE_1(1280)
                    inception.Mixed_7c = inception_module.FIDInceptionE_2(2048)
                    
                    # 加载自定义权重
                    print(f"加载自定义FID权重: {custom_weights_path}")
                    state_dict = torch.load(custom_weights_path, map_location='cpu')
                    inception.load_state_dict(state_dict)
                    return inception
                
                # 替换原始函数
                inception_module.fid_inception_v3 = custom_fid_inception_v3
            
            if device == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(device)
            
            print(f"使用batch_size=1计算FID（避免不同尺寸图像问题）")
            fid_score = fid_score_module.calculate_fid_given_paths(
                [dir1, dir2],
                batch_size=1,  
                device=device,
                dims=2048,
                num_workers=0 
            )
            
            return fid_score
            
        except Exception as e:
            print(f"自定义FID计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_single_pair(self, img1_path, img2_path, text=None):
        """评估单对图像"""
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 调整图像大小以确保可比较
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.BICUBIC)
        
        results = {
            'psnr': self.calculate_psnr(img1, img2),
            'ssim': self.calculate_ssim(img1, img2),
            'lpips': self.calculate_lpips(img1, img2)
        }
        
        # 只有在DINO可用时才计算DINO相似度
        if self.config.enable_dino and self.dino_model is not None:
            results['dino_sim'] = self.calculate_dino_similarity(img1, img2)
        
        # 只有在CLIP可用且提供了文本时计算CLIP相似度
        if self.config.enable_clip and text:
            results['clip_sim_img1'] = self.calculate_clip_similarity(img1, text)
            results['clip_sim_img2'] = self.calculate_clip_similarity(img2, text)
        
        return results
    
    def evaluate(self):
        """评估指标"""
        if not self.config.pred_dir or not self.config.gt_dir:
            print("缺少目标目录或预测目录，无法评估")
            return
        
        # 列出两个目录中的所有图像文件
        gt_files = sorted([f for f in os.listdir(self.config.gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        pred_files = sorted([f for f in os.listdir(self.config.pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # 确保两个目录中的文件数量相同
        common_files = set(gt_files).intersection(set(pred_files))
        if len(common_files) == 0:
            print(f"没有找到共同的文件名，尝试按顺序匹配...")
            common_count = min(len(gt_files), len(pred_files))
            gt_files = gt_files[:common_count]
            pred_files = pred_files[:common_count]
        else:
            gt_files = sorted(list(common_files))
            pred_files = sorted(list(common_files))
        
        # 收集所有指标
        all_metrics = []
        
        # 计算FID（如果启用）
        fid = None
        if self.config.enable_fid:
            fid = self.calculate_fid(
                self.config.gt_dir, 
                self.config.pred_dir,
                custom_weights_path=self.config.fid_weights_path,
                batch_size=self.config.batch_size,
                device=self.config.device
            )
            if fid is not None:
                print(f"最终FID结果: {fid:.4f}")
        
        # 逐对计算其他指标
        for i, (gt_name, pred_name) in enumerate(zip(gt_files, pred_files)):
            gt_path = os.path.join(self.config.gt_dir, gt_name)
            pred_path = os.path.join(self.config.pred_dir, pred_name)
            
            # 如果存在提示文本
            text = None
            if self.config.text_dir:
                text_file = os.path.join(self.config.text_dir, os.path.splitext(gt_name)[0] + '.txt')
                if os.path.exists(text_file):
                    with open(text_file, 'r') as f:
                        text = f.read().strip()
            
            # 计算单对指标
            metrics = self.evaluate_single_pair(gt_path, pred_path, text)
            metrics['filename'] = gt_name
            all_metrics.append(metrics)
        
        # 计算平均指标
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            if metric != 'filename':
                values = [m[metric] for m in all_metrics if m.get(metric) is not None]
                avg_metrics[metric] = sum(values) / len(values) if values else None
        
        # 添加FID
        avg_metrics['fid'] = fid
        
        # 调试输出 - 显示FID值
        print(f"DEBUG - FID值: {fid}")
        print(f"DEBUG - FID类型: {type(fid)}")
        
        # 打印平均指标
        print("Average Metrics:")
        for metric, value in avg_metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: N/A (disabled)")
        
        return avg_metrics
    
    def evaluate_directory(self, gt_dir, pred_dir, output_file=None, text=None):
        """评估目录中的所有图像对"""
        # 保存当前配置
        orig_gt_dir = self.config.gt_dir
        orig_pred_dir = self.config.pred_dir
        
        # 设置新目录
        self.config.gt_dir = gt_dir
        self.config.pred_dir = pred_dir
        
        # 评估
        avg_metrics = self.evaluate()
        
        # 恢复原始配置
        self.config.gt_dir = orig_gt_dir
        self.config.pred_dir = orig_pred_dir
        
        return avg_metrics, None  # 兼容原来的返回值格式


if __name__ == '__main__':
    # 测试配置 - 测试所有指标
    config = MetricConfig(
        enable_psnr=True,    # 启用PSNR
        enable_ssim=True,    # 启用SSIM
        enable_lpips=True,   # 启用LPIPS
        enable_fid=True,     # 启用FID
        enable_clip=False,   # 禁用CLIP
        enable_dino=False,   # 禁用DINO
        gt_dir="/aicamera-mlp/fq_proj/codes/RemoveAnything/sampled_v3/gt",
        pred_dir="/aicamera-mlp/fq_proj/codes/RemoveAnything/sampled_v3/pred"
    )
    print("配置已加载，开始评估...")
    
    evaluator = ImageEvaluator(config)
    avg_metrics = evaluator.evaluate()