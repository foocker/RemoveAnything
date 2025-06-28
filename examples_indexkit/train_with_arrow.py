"""
使用Arrow格式数据集进行RORem模型训练的示例。
这个示例展示了如何将train_RORem.py中的数据加载逻辑替换为使用IndexKits的Arrow格式数据集。
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径，确保能导入项目模块
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myutils.arrow_dataset import create_ror_arrow_dataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RORem训练 - 使用Arrow数据集")
    
    # 数据集参数
    parser.add_argument(
        "--arrow_path", 
        type=str, 
        nargs="+",
        required=True, 
        help="Arrow数据文件路径，可以是单个文件或多个文件"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=512, 
        help="图像分辨率"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="训练批次大小"
    )
    parser.add_argument(
        "--random_flip", 
        action="store_true", 
        help="是否随机翻转图像"
    )
    parser.add_argument(
        "--multireso", 
        action="store_true", 
        help="是否使用多分辨率模式"
    )
    parser.add_argument(
        "--world_size", 
        type=int, 
        default=1, 
        help="分布式训练的进程数"
    )
    
    # 加入原train_RORem.py中可能需要的其他参数
    # ... 根据需要添加 ...
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建Arrow数据集
    print("创建Arrow数据集...")
    dataset = create_ror_arrow_dataset(args)
    
    # 对数据集进行随机打乱
    dataset.shuffle(42)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # 示例：获取一个批次并打印信息
    print(f"数据集大小: {len(dataset)}")
    print(f"加载器批次数: {len(dataloader)}")
    
    for batch_idx, batch in enumerate(dataloader):
        print("\n批次:", batch_idx)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: 形状{value.shape}, 类型{value.dtype}")
            else:
                print(f"{key}: {type(value)}")
        
        # 只打印第一个批次
        if batch_idx == 0:
            break
    
    print("\n在train_RORem.py中集成Arrow数据集的步骤:")
    print("1. 导入RORArrowDataset: from myutils.arrow_dataset import create_ror_arrow_dataset")
    print("2. 添加命令行参数: --arrow_path 和 --use_arrow_dataset")
    print("3. 条件替换数据加载代码:") 
    print("""
    if args.use_arrow_dataset:
        # 使用Arrow数据集
        dataset = create_ror_arrow_dataset(args)
    else:
        # 原有的数据加载逻辑
        dataset_dict = meta_to_inpaint_dataset_format_custom(args.meta_path, meta_folder)
        dataset = Dataset.from_dict(dataset_dict).cast_column("input_image", Image()).cast_column("edited_image", Image()).cast_column("mask", Image())
    """)


if __name__ == "__main__":
    main()
