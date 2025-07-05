#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 数据路径
TRAIN_JSON_PATH="/xx/xx.json"
VAL_JSON_PATH="/xx/data/removal/val.json"

# 输出路径
OUTPUT_DIR="/xx/removal_flux_lora"

# 模型参数
FLUX_FILL_ID="/xx/FLUX.1-Fill-dev"
FLUX_REDUX_ID="/xx/FLUX.1-Redux-dev"
RESOLUTION='512,512'
# aspect_ratio_buckets='1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'
aspect_ratio_buckets='512,512'

# 训练参数
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=4
MAX_TRAIN_STEPS=20000
GRADIENT_ACCUMULATION_STEPS=1
VALIDATION_STEPS=500
CHECKPOINT_STEPS=1000

# LoRA参数
USE_LORA=true
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.0
TARGET_MODULES="(.*x_embedder|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_v|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_out\.0|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.2|.*single_transformer_blocks\.[0-9]+\.norm\.linear|.*single_transformer_blocks\.[0-9]+\.proj_mlp|.*single_transformer_blocks\.[0-9]+\.proj_out|.*single_transformer_blocks\.[0-9]+\.attn.to_k|.*single_transformer_blocks\.[0-9]+\.attn.to_q|.*single_transformer_blocks\.[0-9]+\.attn.to_v|.*single_transformer_blocks\.[0-9]+\.attn.to_out)"


# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行训练脚本
accelerate launch src/train_bucket.py \
  --train_json_path $TRAIN_JSON_PATH \
  --val_json_path $VAL_JSON_PATH \
  --output_dir $OUTPUT_DIR \
  --aspect_ratio_buckets $aspect_ratio_buckets \
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
  --mixed_precision bf16 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing

echo "训练完成！"
