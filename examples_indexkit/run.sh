#!/bin/bash
# 这个脚本用于从JSON映射文件生成Arrow格式的数据文件
# 使用方法: ./run.sh <json_path> <output_dir> [num_processes] [samples_per_file]

set -e

# 设置默认值
JSON_PATH=${1:-"/aicamera-mlp/fq_proj/datasets/Eraser/flickr_commodity_stuff/gt_added_mapping.json"}
OUTPUT_DIR=${2:-"/aicamera-mlp/fq_proj/datasets/Eraser/flickr_commodity_stuff/arrow"}
NUM_PROCESSES=${3:-8}
SAMPLES_PER_FILE=${4:-2000}

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 输出运行信息
echo "生成Arrow格式数据..."
echo "JSON映射文件: $JSON_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "进程数: $NUM_PROCESSES"
echo "每个文件样本数: $SAMPLES_PER_FILE"

# 检查ijson库
python -c "import ijson" 2>/dev/null && echo "Using ijson for memory-efficient streaming" || echo "ijson not found, using standard JSON loading (may require more memory)"

# 运行Python脚本生成Arrow文件
python triplets_arrow.py \
    --json_path $JSON_PATH \
    --output_dir $OUTPUT_DIR \
    --num_processes $NUM_PROCESSES \
    --samples_per_file $SAMPLES_PER_FILE \
    --memory_efficient true

echo "Arrow数据生成完成！输出目录: $OUTPUT_DIR"