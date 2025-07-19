import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import bezier
import random
import torchvision.transforms as T

class BaseDataset(Dataset):
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass

    def __getitem__(self, idx):
        item = self.get_sample(idx)
        return item
 

    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass
    
    def process_triplets(self, input_image, removed_mask, removed_image, size=(768, 768)):
        '''
        input_image: input image 需要被擦除的图
        removed_mask: removed object 被擦除对象对应的mask
        removed_image: 擦除后的效果
        '''
        remove_box_yyxx = get_bbox_from_mask(removed_mask)
        
        ref_mask_3 = np.stack([removed_mask, removed_mask, removed_mask], -1)
        masked_ref_image = input_image * ref_mask_3 + np.ones_like(input_image) * 255 * (1-ref_mask_3)
        
        y1, y2, x1, x2 = remove_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
        ref_mask = removed_mask[y1:y2, x1:x2]
        
        ratio = np.random.randint(11, 15) / 10
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
        
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
        
        # ========= 处理训练目标 ===========
        # 对被擦除的mask做贝塞尔操作，生成一个新的mask
        expand_remove_box_yyxx = expand_bbox(removed_mask, remove_box_yyxx, ratio=[1.1, 1.2])
        
        y1, y2, x1, x2 = remove_box_yyxx
        e_y1, e_y2, e_x1, e_x2 = expand_remove_box_yyxx
        
        # TODO 不同类型，概率不同
        prob_bezier = 0.4
        prob_box = 0.8
            
        prob = random.uniform(0, 1)
        
        if prob <= prob_bezier:
            tar_mask = Image.new('RGB', (input_image.shape[1], input_image.shape[0]), (0, 0, 0))
            top_nodes = np.asfortranarray([
                [x1, (x1+x2)/2, x2],
                [y1, e_y1, y1],
            ])
            down_nodes = np.asfortranarray([
                [x2, (x1+x2)/2, x1],
                [y2, e_y2, y2],
            ])
            left_nodes = np.asfortranarray([
                [x1, e_x1, x1],
                [y2, (y1+y2)/2, y1],
            ])
            right_nodes = np.asfortranarray([
                [x2, e_x2, x2],
                [y1, (y1+y2)/2, y2],
            ])
            
            top_curve = bezier.Curve(top_nodes, degree=2)
            right_curve = bezier.Curve(right_nodes, degree=2)
            down_curve = bezier.Curve(down_nodes, degree=2)
            left_curve = bezier.Curve(left_nodes, degree=2)
            
            curve_list = [top_curve, right_curve, down_curve, left_curve]
            pt_list = []
            random_width = 40
            
            for curve in curve_list:
                x_list = []
                y_list = []
                for i in range(1, 19):
                    x_original = curve.evaluate(i * 0.05)[0][0]
                    y_original = curve.evaluate(i * 0.05)[1][0]
                    
                    random_x_offset = random.randint(-random_width, random_width)
                    random_y_offset = random.randint(-random_width, random_width)
                    
                    x = x_original + random_x_offset
                    y = y_original + random_y_offset
                    
                    x_in_range = x < x1 or x > x2
                    y_in_range = y < y1 or y > y2
                    
                    if not x_in_range:
                        x = x_original
                    if not y_in_range:
                        y = y_original
                        
                    if (x, y) not in zip(x_list, y_list):
                        pt_list.append((x, y))
                        x_list.append(x)
                        y_list.append(y)
            
            tar_mask_draw = ImageDraw.Draw(tar_mask)
            tar_mask_draw.polygon(pt_list, fill=(255, 255, 255))
            
            tar_mask = np.array(tar_mask)
            
        elif prob > prob_bezier and prob <= prob_box:
            tar_mask = np.zeros_like(input_image, dtype=np.uint8)
            tar_mask[e_y1:e_y2, e_x1:e_x2] = 255
            
        else:
            tar_mask = removed_mask.copy()
            tar_mask[tar_mask == 1] = 255
            
            kernel = np.ones((7, 7), np.uint8)
            iterations = 2
            tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)
            tar_mask = np.stack([tar_mask, tar_mask, tar_mask], -1)
        
        masked_task_image = input_image * (1-(tar_mask == 255))  # 
        masked_task_image = pad_to_square(masked_task_image, pad_value=255, random=False).astype(np.uint8)
        masked_task_image = cv2.resize(masked_task_image.astype(np.uint8), size).astype(np.uint8)
        
        tar_image = pad_to_square(removed_image, pad_value=255, random=False).astype(np.uint8)
        tar_image = cv2.resize(tar_image.astype(np.uint8), size).astype(np.uint8)
        
        tar_mask = pad_to_square(tar_mask, pad_value=0, random=False).astype(np.uint8)
        tar_mask = cv2.resize(tar_mask.astype(np.uint8), size).astype(np.uint8)
        
        mask_black = np.ones_like(tar_image) * 0
        # 纯黑+消除物体扩张后的mask
        diptych_mask = self.to_tensor(np.concatenate([mask_black, tar_mask], axis=1)) 
        # 被消除物体扩张填白裁剪后的图像+带mask的原始图像）
        diptych_src = self.to_tensor(np.concatenate([masked_ref_image, masked_task_image], axis=1)) 
        # 被消除物体扩张填白裁剪后的图像 + 擦除后的图像
        diptych_result = self.to_tensor(np.concatenate([masked_ref_image, tar_image], axis=1))
        # 被消除物体扩张填白裁剪后的图像
        masked_ref_image = self.to_tensor(masked_ref_image)
        
        item = dict(
            ref=masked_ref_image,   # 被擦除物体的参考图像
            src=diptych_src,        # 源图像（参考物体 + 带消除物体的mask的原始图像）
            result=diptych_result,  # 结果图像（参考物体 + 擦除后的图像）
            mask=diptych_mask,      # mask（黑色 + 被擦除物体的mask）
        )
        return item
    
    def process_custom(self, input_image, removed_mask, removed_image, size=(768, 768)):
        '''
        数据不一定是三元组，或者是三元组，是否使用diptych可以自定义
        input_image: input image 需要被擦除的图
        removed_mask: removed object 被擦除对象对应的mask
        removed_image: 擦除后的效果
        较triplets有两个改动：
            1. masked_ref_img 做inverse local 改动，被消除区域外切box扩展后box做crop,然后被消除区域填白255
            2. masked_task_img 做input crop 改动，原始图中被消除的区域外切box扩展后box做crop
        '''
        remove_box_yyxx = get_bbox_from_mask(removed_mask)
        
        ref_mask_3 = np.stack([removed_mask, removed_mask, removed_mask], -1)
        masked_ref_image = input_image * (1 - ref_mask_3) + np.ones_like(input_image) * 255 * ref_mask_3
        
        # 在扩展mask之前，可以先对mask做一个贝塞尔操作,这里实验想法验证，先从简
        expand_remove_box_yyxx = expand_bbox(removed_mask, remove_box_yyxx, ratio=[1.1, 1.5])
        
        y1, y2, x1, x2 = expand_remove_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]  # 第一条改动
        
        # 255 means re generate 0 means keep original pixel
        masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)  
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
        
        # 第二条改动
        masked_task_image = input_image[y1:y2, x1:x2, :]
        masked_task_image = pad_to_square(masked_task_image, pad_value=255, random=False).astype(np.uint8)
        masked_task_image = cv2.resize(masked_task_image.astype(np.uint8), size).astype(np.uint8)
        
        tar_image = removed_image[y1:y2, x1:x2, :]
        tar_image = pad_to_square(tar_image, pad_value=255, random=False).astype(np.uint8)
        tar_image = cv2.resize(tar_image.astype(np.uint8), size).astype(np.uint8)
        
        tar_mask = ref_mask_3[y1:y2, x1:x2, :]* 255
        tar_mask = pad_to_square(tar_mask, pad_value=0, random=False).astype(np.uint8)
        tar_mask = cv2.resize(tar_mask.astype(np.uint8), size).astype(np.uint8)
        
        mask_black = np.ones_like(tar_image) * 0
        diptych_mask = self.to_tensor(np.concatenate([mask_black, tar_mask], axis=1)) 
        diptych_src = self.to_tensor(np.concatenate([masked_ref_image, masked_task_image], axis=1)) 
        diptych_result = self.to_tensor(np.concatenate([masked_ref_image, tar_image], axis=1))
        masked_ref_image = self.to_tensor(masked_ref_image)
        
        item = dict(
            ref=masked_ref_image,   
            src=diptych_src,        
            result=diptych_result,  
            mask=diptych_mask,      
        )
        return item
    
    def process_simple(self, input_image, removed_mask, removed_image, size=(768, 768), 
                      enable_soft_mask=False, soft_mask_params=None):
        '''
        input_image: input image 需要被擦除的图
        removed_mask: removed object 被擦除对象对应的mask
        removed_image: 擦除后的效果
        size: 桶大小
        enable_soft_mask: 是否启用软mask预处理
        soft_mask_params: 软mask参数 {"dilate_kernel_size": 3, "blur_kernel_size": 5, "sigma": 1.0}
        此为flux_kontext_inpaint 数据
        '''
        input_image = self.to_tensor(input_image)
        mask = removed_mask * 255  # add some crop expand ... TODO 
        mask = self.to_tensor(mask) 
        result = self.to_tensor(removed_image)
        
        # 可选的软mask预处理
        soft_mask = None
        if enable_soft_mask:
            soft_mask = self._create_soft_mask(mask, soft_mask_params or {})
        
        item = dict(
            input_image=input_image,  # input image
            mask=mask,   # conditional mask
            result=result,  # result image
        )
        
        # 只有启用时才添加软mask
        if soft_mask is not None:
            item['soft_mask'] = soft_mask
            
        return item
    
    def _create_soft_mask(self, mask, params):
        '''
        创建软边界mask
        '''
        import torch
        import torch.nn.functional as F
        
        # 默认参数
        dilate_kernel_size = params.get('dilate_kernel_size', 3)
        blur_kernel_size = params.get('blur_kernel_size', 5)
        sigma = params.get('sigma', 1.0)
        
        # 确保mask是4D tensor [B, C, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # 膨胀操作
        padding = dilate_kernel_size // 2
        dilate_kernel = torch.ones(1, 1, dilate_kernel_size, dilate_kernel_size, 
                                 device=mask.device, dtype=mask.dtype)
        expanded_mask = F.conv2d(mask, dilate_kernel, padding=padding)
        expanded_mask = torch.clamp(expanded_mask, 0, 1)
        
        # 高斯模糊
        padding_blur = blur_kernel_size // 2
        x = torch.arange(blur_kernel_size, dtype=mask.dtype, device=mask.device)
        x = x - blur_kernel_size // 2
        gauss_kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_kernel_1d = gauss_kernel_1d / gauss_kernel_1d.sum()
        gauss_kernel_2d = gauss_kernel_1d[:, None] * gauss_kernel_1d[None, :]
        gauss_kernel_2d = gauss_kernel_2d[None, None, :, :]
        
        soft_mask = F.conv2d(expanded_mask, gauss_kernel_2d, padding=padding_blur)
        
        # 恢复原始维度
        if soft_mask.dim() == 4 and soft_mask.shape[0] == 1:
            soft_mask = soft_mask.squeeze(0)
            
        return soft_mask
        