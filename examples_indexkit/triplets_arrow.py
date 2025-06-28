import os
import hashlib
import pandas as pd
import pyarrow as pa
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import gc
import json
import argparse

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False


def stream_triplets_from_json(json_path, batch_size=10000):
    """使用流式处理从JSON文件生成三元组数据，减少内存使用
    参数:
        json_path: JSON文件路径
        batch_size: 每批处理的样本数
    返回:
        生成器，每次产生一批三元组数据
    """ 
    assert os.path.exists(json_path), f"{json_path} does not exist"
    
    try:
        import ijson
        use_ijson = True
    except ImportError:
        print("警告: 未安装ijson库，将使用标准json加载。建议 pip install ijson 以优化内存使用。")
        use_ijson = False
    
    root_path = os.path.dirname(json_path)
    
    batch_data = {
        "input_image": [],
        "edited_image": [],
        "edit_prompt": [],
        "mask": []
    }
    
    count = 0
    
    if use_ijson:
        with open(json_path, 'rb') as f:
            # 流式解析mapping部分
            for gt_image, mask_input_files in ijson.kvitems(f, 'mapping'):
                gt_path = os.path.join(root_path, "gt", os.path.basename(gt_image))
                
                for mask_input_file in mask_input_files:
                    input_path = os.path.join(root_path, "input", os.path.basename(mask_input_file))
                    mask_path = os.path.join(root_path, "mask", os.path.basename(mask_input_file))
                    
                    batch_data["input_image"].append(input_path)
                    batch_data["edited_image"].append(gt_path)
                    batch_data["edit_prompt"].append("")
                    batch_data["mask"].append(mask_path)
                    
                    count += 1
                    if count >= batch_size:
                        yield batch_data
                        # 重置批量数据
                        batch_data = {
                            "input_image": [],
                            "edited_image": [],
                            "edit_prompt": [],
                            "mask": []
                        }
                        count = 0
    else:
        with open(json_path, "r") as f:
            data = json.load(f)
            mapping_data = data["mapping"]
        
        for gt_image, mask_input_files in mapping_data.items():
            gt_path = os.path.join(root_path, "gt", os.path.basename(gt_image))
            
            for mask_input_file in mask_input_files:
                input_path = os.path.join(root_path, "input", os.path.basename(mask_input_file))
                mask_path = os.path.join(root_path, "mask", os.path.basename(mask_input_file))
                
                batch_data["input_image"].append(input_path)
                batch_data["edited_image"].append(gt_path)
                batch_data["edit_prompt"].append("")
                batch_data["mask"].append(mask_path)
                
                count += 1
                if count >= batch_size:
                    yield batch_data
                    # 重置批量数据
                    batch_data = {
                        "input_image": [],
                        "edited_image": [],
                        "edit_prompt": [],
                        "mask": []
                    }
                    count = 0
    
    # 返回剩余的批次
    if count > 0:
        yield batch_data


def triplets_from_json(json_path):
    """兼容旧版接口，但提醒用户使用新的流式接口
    参数:
        json_path: JSON文件路径
    返回:
        包含所有三元组数据的字典
    """
    print("警告: triplets_from_json 加载整个数据集到内存。对于大数据集，请使用 stream_triplets_from_json 以优化内存使用。")
    
    assert os.path.exists(json_path), f"{json_path} does not exist"
    
    with open(json_path, "r") as f:
        data = json.load(f)
        mapping_data = data["mapping"]
        
    output_data = {
        "input_image": [],
        "edited_image": [],
        "edit_prompt": [],
        "mask": []
    }
    
    root_path = os.path.dirname(json_path)
    
    for gt_image, mask_input_files in mapping_data.items():
        gt_path = os.path.join(root_path, "gt", os.path.basename(gt_image))
        
        for mask_input_file in mask_input_files:
            input_path = os.path.join(root_path, "input", os.path.basename(mask_input_file))
            mask_path = os.path.join(root_path, "mask", os.path.basename(mask_input_file))
            
            output_data["input_image"].append(input_path)
            output_data["edited_image"].append(gt_path)
            output_data["edit_prompt"].append("")
            output_data["mask"].append(mask_path)
            
    return output_data


def parse_triplet_data(triplet_paths, max_image_size=None):
    """处理单个三元组数据，支持图像大小限制
    
    参数:
        triplet_paths: 包含 (input_path, mask_path, gt_path) 的元组
        max_image_size: 可选，图像的最大尺寸限制（宽或高的最大值）
    """
    try:
        input_path, mask_path, gt_path = triplet_paths

        with Image.open(input_path) as f:
            width, height = f.size
            
            # 如果指定了最大尺寸且图像超过这个尺寸，则跳过
            if max_image_size and (width > max_image_size or height > max_image_size):
                print(f"跳过过大的图像 {input_path}: {width}x{height} > {max_image_size}")
                return None
        
        with open(input_path, "rb") as fp:
            input_image = fp.read()
            md5 = hashlib.md5(input_image).hexdigest()

        with open(mask_path, "rb") as fp:
            mask_image = fp.read()

        with open(gt_path, "rb") as fp:
            gt_image = fp.read()

        return [md5, width, height, input_image, mask_image, gt_image]

    except Exception as e:
        print(f"处理文件出错 {input_path if 'input_path' in locals() else triplet_paths}: {e}")
        return None


def create_arrow_files(json_path, output_dir, num_processes=8, samples_per_file=10000, memory_efficient=True):
    """从源json文件创建 Arrow 文件，支持内存优化模式
    
    参数:
        json_path: JSON文件路径
        output_dir: 输出目录
        num_processes: 多进程处理的进程数
        samples_per_file: 每个Arrow文件包含的样本数
        memory_efficient: 是否使用内存优化模式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if memory_efficient:
        # 使用流式处理模式
        triplet_generator = stream_triplets_from_json(json_path, batch_size=samples_per_file)
        batch_idx = 0
        
        for batch_triplets in triplet_generator:
            num_samples = len(batch_triplets['input_image'])
            if num_samples == 0:
                continue
                
            arrow_path = os.path.join(output_dir, f"{batch_idx:05d}.arrow")
            if os.path.exists(arrow_path):
                print(f"跳过已存在的 {arrow_path}")
                batch_idx += 1
                continue
            
            print(f"处理批次 {batch_idx}, 样本数: {num_samples}")
            
            # 准备三元组路径
            triplet_paths = []
            for i in range(num_samples):
                triplet_paths.append(
                    ( 
                     batch_triplets['input_image'][i], 
                     batch_triplets['mask'][i],
                     batch_triplets['edited_image'][i]
                    )
                )
            
            # 使用多进程处理数据
            with Pool(num_processes) as pool:
                # 使用chunksize参数优化大批量处理
                chunksize = max(1, num_samples // (num_processes * 4))
                results = list(tqdm(
                    pool.imap(parse_triplet_data, triplet_paths, chunksize=chunksize), 
                    total=len(triplet_paths)
                ))
            
            # 过滤掉None结果
            initial_count = len(results)
            results = [r for r in results if r is not None]
            filtered_count = initial_count - len(results)
            if filtered_count > 0:
                print(f"过滤了 {filtered_count} 个无效结果")
            print(f"成功处理 {len(results)} 个样本")
            
            if len(results) > 0:
                # 创建 DataFrame 和 Arrow 表
                columns = ["md5", "width", "height",  "input_image", "mask", "edited_image",]
                df = pd.DataFrame(results, columns=columns)
                table = pa.Table.from_pandas(df)
                
                # 写入 Arrow 文件
                with pa.OSFile(arrow_path, "wb") as sink:
                    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)
                
                print(f"已保存到 {arrow_path}")
                
                # 清理内存
                del df, table, results
                gc.collect()
                
            batch_idx += 1
            
    else:
        # 传统处理方式，加载所有数据到内存
        triplets = triplets_from_json(json_path)
        print(f"找到 {len(triplets['input_image'])} 个完整的三元组")
        
        # 将三元组分成多个批次
        num_batches = (len(triplets['input_image']) + samples_per_file - 1) // samples_per_file
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * samples_per_file
            end_idx = min((batch_idx + 1) * samples_per_file, len(triplets['input_image']))
            
            # 准备三元组路径
            triplet_paths = []
            for i in range(start_idx, end_idx):
                triplet_paths.append(
                    (triplets['input_image'][i], 
                     triplets['mask'][i],
                     triplets['edited_image'][i])
                )
            
            arrow_path = os.path.join(output_dir, f"{batch_idx:05d}.arrow")
            if os.path.exists(arrow_path):
                print(f"跳过已存在的 {arrow_path}")
                continue
            
            print(f"处理批次 {batch_idx+1}/{num_batches}, 样本数: {len(triplet_paths)}")
            
            # 使用多进程处理数据
            with Pool(num_processes) as pool:
                # 使用chunksize参数优化大批量处理
                chunksize = max(1, len(triplet_paths) // (num_processes * 4))
                results = list(tqdm(
                    pool.imap(parse_triplet_data, triplet_paths, chunksize=chunksize), 
                    total=len(triplet_paths)
                ))
            
            # 过滤掉None结果
            initial_count = len(results)
            results = [r for r in results if r is not None]
            filtered_count = initial_count - len(results)
            if filtered_count > 0:
                print(f"过滤了 {filtered_count} 个无效结果")
            print(f"成功处理 {len(results)} 个样本")
            
            if len(results) > 0:
                # 创建 DataFrame 和 Arrow 表
                columns = ["md5", "width", "height", "input_image",  "mask", "edited_image"]
                df = pd.DataFrame(results, columns=columns)
                table = pa.Table.from_pandas(df)
                
                # 写入 Arrow 文件
                with pa.OSFile(arrow_path, "wb") as sink:
                    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)
                
                print(f"已保存到 {arrow_path}")
                
                # 清理内存
                del df, table, results
                gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从源json文件创建 Arrow 文件")
    parser.add_argument("--json_path", type=str, required=True, help="JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--num_processes", type=int, default=8, help="多进程处理的进程数")
    parser.add_argument("--samples_per_file", type=int, default=10000, help="每个Arrow文件包含的样本数")
    parser.add_argument("--memory_efficient", type=str, default="true", help="是否使用内存优化模式 (true/false)")
    
    args = parser.parse_args()
    
    # 将字符串转换为布尔值
    memory_efficient = args.memory_efficient.lower() == "true"
    
    create_arrow_files(args.json_path, args.output_dir, args.num_processes, args.samples_per_file, memory_efficient)