#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 数据路径
TRAIN_JSON_PATH="/xx/data/removal/train.json"
VAL_JSON_PATH="/xx/data/removal/val.json"
TRAIN_DATA_DIR="/xx/data/removal"

# 输出路径
OUTPUT_DIR="/xx/outputs/removal"

# 模型参数
FLUX_FILL_ID="stabilityai/flux-fill"
FLUX_REDUX_ID="stabilityai/flux-prior-redux"
RESOLUTION=768

# 训练参数
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=4
MAX_TRAIN_STEPS=20000
GRADIENT_ACCUMULATION_STEPS=1
VALIDATION_STEPS=500
CHECKPOINT_STEPS=1000

# LoRA参数
USE_LORA=true
LORA_R=32
LORA_ALPHA=32


# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行训练脚本
python src/train.py \
  --train_json_path $TRAIN_JSON_PATH \
  --val_json_path $VAL_JSON_PATH \
  --output_dir $OUTPUT_DIR \
  --resolution $RESOLUTION \
  --flux_fill_id $FLUX_FILL_ID \
  --flux_redux_id $FLUX_REDUX_ID \
  --learning_rate $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --max_train_steps $MAX_TRAIN_STEPS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --validation_steps $VALIDATION_STEPS \
  --checkpointing_steps $CHECKPOINT_STEPS \
  --use_lora $USE_LORA \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --target_modules $TARGET_MODULES \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing

echo "训练完成！"
