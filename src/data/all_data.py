import cv2
import numpy as np
import os
import json
from .data_utils import * 
from .base import BaseDataset

def load_triplet_paths(json_path):
    """从JSON文件加载三元组的文件路径，但不加载实际图像
    参数:
        json_path: JSON文件路径
    返回:
        列表，每个元素是一个包含文件路径的字典
    """ 
    assert os.path.exists(json_path), f"{json_path} does not exist"
    
    try:
        import ijson
        use_ijson = True
    except ImportError:
        print("警告: 未安装ijson库，将使用标准json加载。建议 pip install ijson 以优化内存使用。")
        use_ijson = False
    
    root_path = os.path.dirname(json_path)
    triplet_paths = []
    invalid_paths = []
    processed_count = 0
    
    # 检查目录是否存在
    gt_dir = os.path.join(root_path, "gt")
    input_dir = os.path.join(root_path, "input")
    mask_dir = os.path.join(root_path, "mask")
    
    print(f"数据路径检查: \n - gt目录: {'存在' if os.path.isdir(gt_dir) else '不存在！'} ({gt_dir}) \n - input目录: {'存在' if os.path.isdir(input_dir) else '不存在！'} ({input_dir}) \n - mask目录: {'存在' if os.path.isdir(mask_dir) else '不存在！'} ({mask_dir})")
    
    if use_ijson:
        with open(json_path, 'rb') as f:
            # 流式解析mapping部分
            for gt_image, mask_input_files in ijson.kvitems(f, 'mapping'):
                processed_count += 1
                gt_path = os.path.join(root_path, "gt", os.path.basename(gt_image))
                
                if not os.path.exists(gt_path):
                    invalid_paths.append(f"GT图像不存在: {gt_path}")
                    continue
                    
                for mask_input_file in mask_input_files:
                    input_path = os.path.join(root_path, "input", os.path.basename(mask_input_file))
                    mask_path = os.path.join(root_path, "mask", os.path.basename(mask_input_file))
                    
                    # 验证路径是否存在
                    all_valid = True
                    if not os.path.exists(input_path):
                        invalid_paths.append(f"输入图像不存在: {input_path}")
                        all_valid = False
                    if not os.path.exists(mask_path):
                        invalid_paths.append(f"掩码图像不存在: {mask_path}")
                        all_valid = False
                        
                    if all_valid:
                        triplet_paths.append({
                            "input_image": input_path,
                            "edited_image": gt_path,
                            "mask": mask_path
                        })
    else:
        with open(json_path, "r") as f:
            data = json.load(f)
            if "mapping" not in data:
                print(f"错误: JSON文件 {json_path} 中没有'mapping'字段。文件包含以下字段: {list(data.keys())}")
                return []
                
            mapping_data = data["mapping"]
            processed_count = len(mapping_data)
        
        for gt_image, mask_input_files in mapping_data.items():
            gt_path = os.path.join(root_path, "gt", os.path.basename(gt_image))
            
            if not os.path.exists(gt_path):
                invalid_paths.append(f"GT图像不存在: {gt_path}")
                continue
                
            for mask_input_file in mask_input_files:
                input_path = os.path.join(root_path, "input", os.path.basename(mask_input_file))
                mask_path = os.path.join(root_path, "mask", os.path.basename(mask_input_file))
                
                # 验证路径是否存在
                all_valid = True
                if not os.path.exists(input_path):
                    invalid_paths.append(f"输入图像不存在: {input_path}")
                    all_valid = False
                if not os.path.exists(mask_path):
                    invalid_paths.append(f"掩码图像不存在: {mask_path}")
                    all_valid = False
                    
                if all_valid:
                    triplet_paths.append({
                        "input_image": input_path,
                        "edited_image": gt_path,
                        "mask": mask_path
                    })
    
    # 打印统计信息
    print(f"处理了 {processed_count} 条映射项，找到 {len(triplet_paths)} 条有效的三元组路径")
    if invalid_paths:
        print(f"发现 {len(invalid_paths)} 条无效路径:")
        for i, path in enumerate(invalid_paths[:10]):  # 只打印前10个错误
            print(f" - {path}")
        if len(invalid_paths) > 10:
            print(f" - ...以及其他 {len(invalid_paths)-10} 条错误（已省略）")
    
    return triplet_paths


class TripletsData(BaseDataset):
    def __init__(self, json_path, size=(768, 768)):
        super().__init__()
        self.size = size
        self.json_path = json_path
        self.root_path = os.path.dirname(json_path)
        self.triplet_paths = load_triplet_paths(json_path)
                
    def __len__(self):
        return len(self.triplet_paths)
    
    def get_sample(self, idx):
        paths = self.triplet_paths[idx]
        
        input_image = cv2.imread(paths["input_image"])
        if input_image is None:
            raise ValueError(f"无法读取图像: {paths['input_image']}")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
        mask_img = cv2.imread(paths["mask"])
        if mask_img is None:
            raise ValueError(f"无法读取mask: {paths['mask']}")
        removed_mask = (mask_img > 128).astype(np.uint8)[:,:,0]
        
        removed_image = cv2.imread(paths["edited_image"])
        if removed_image is None:
            raise ValueError(f"无法读取擦除后图像: {paths['edited_image']}")
        removed_image = cv2.cvtColor(removed_image, cv2.COLOR_BGR2RGB)
        
        item = self.process_triplets(input_image, removed_mask, removed_image, self.size)
        return item

