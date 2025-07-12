import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from torch_fidelity import calculate_metrics
import open_clip

class ImageEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 初始化LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # 初始化CLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B/32')
        self.clip_model = self.clip_model.to(device)
        
        # 初始化DINO
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.dino_model = self.dino_model.to(device)
    
    def calculate_psnr(self, img1, img2):
        """计算PSNR"""
        img1_np = np.array(img1) / 255.0
        img2_np = np.array(img2) / 255.0
        return peak_signal_noise_ratio(img1_np, img2_np)
    
    def calculate_ssim(self, img1, img2):
        """计算SSIM"""
        img1_np = np.array(img1) / 255.0
        img2_np = np.array(img2) / 255.0
        return structural_similarity(img1_np, img2_np, multichannel=True, data_range=1.0)
    
    def calculate_lpips(self, img1, img2):
        """计算LPIPS"""
        img1_tensor = self.transform(img1).unsqueeze(0).to(self.device)
        img2_tensor = self.transform(img2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1_tensor, img2_tensor)
        return lpips_value.item()
    
    def calculate_clip_similarity(self, img, text):
        """计算CLIP相似度"""
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
        img1_tensor = self.clip_preprocess(img1).unsqueeze(0).to(self.device)  # 使用CLIP的预处理
        img2_tensor = self.clip_preprocess(img2).unsqueeze(0).to(self.device)
        
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
    def calculate_fid(dir1, dir2):
        """计算FID (需要目录路径)"""
        metrics = calculate_metrics(
            input1=dir1,
            input2=dir2,
            cuda=torch.cuda.is_available(),
            fid=True
        )
        return metrics['fid']
    
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
            'lpips': self.calculate_lpips(img1, img2),
            'dino_sim': self.calculate_dino_similarity(img1, img2)
        }
        
        if text:
            results['clip_sim_img1'] = self.calculate_clip_similarity(img1, text)
            results['clip_sim_img2'] = self.calculate_clip_similarity(img2, text)
        
        return results
    
    def evaluate_directory(self, gt_dir, pred_dir, output_file=None, text=None):
        """评估目录中的所有图像对"""
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        common_files = set([os.path.splitext(f)[0] for f in gt_files]) & set([os.path.splitext(f)[0] for f in pred_files])
        
        all_results = []
        for base_name in common_files:
            gt_path = os.path.join(gt_dir, base_name + '.png')  # 假设文件扩展名为.png
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, base_name + '.jpg')  # 尝试.jpg
            
            pred_path = os.path.join(pred_dir, base_name + '.png')
            if not os.path.exists(pred_path):
                pred_path = os.path.join(pred_dir, base_name + '.jpg')
            
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                result = self.evaluate_single_pair(gt_path, pred_path, text)
                result['filename'] = base_name
                all_results.append(result)
        
        # 计算平均值
        avg_results = {metric: np.mean([r[metric] for r in all_results]) 
                      for metric in all_results[0] if metric != 'filename'}
        
        # 计算FID
        avg_results['fid'] = self.calculate_fid(gt_dir, pred_dir)
        
        # 写入结果
        if output_file:
            with open(output_file, 'w') as f:
                f.write('Individual Results:\n')
                for r in all_results:
                    f.write(f"{r['filename']}:\n")
                    for k, v in r.items():
                        if k != 'filename':
                            f.write(f"  {k}: {v:.4f}\n")
                    f.write('\n')
                
                f.write('Average Results:\n')
                for k, v in avg_results.items():
                    f.write(f"{k}: {v:.4f}\n")
        
        return avg_results, all_results


if __name__ == '__main__':
    evaluator = ImageEvaluator()
    avg_metrics, all_metrics = evaluator.evaluate_directory(
        gt_dir='path/to/ground_truth',
        pred_dir='path/to/predictions',
        output_file='evaluation_results.txt',
        text='a high quality image'
    )
    print("Average Metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")