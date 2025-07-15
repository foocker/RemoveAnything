#!/bin/bash

# RemoveAnything Three-Expert MoE LoRA Training Script (适配本地环境)
# 基于用户提供的实际配置参数

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# 数据路径配置 (使用用户的实际路径)
# =============================================================================
TRAIN_JSON_PATH="/xx/gt_added_mapping_sample_1000.json"
# VAL_JSON_PATH="/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO_v2/gt_added_mapping_val.json"
VAL_JSON_PATH="/xxx"  # 也可以是路径：input,mask配的
TRAIN_METADATA_FILE="/xx/train_metadata.json"

# =============================================================================
# 输出路径配置
# =============================================================================
OUTPUT_DIR="/xx/removal_flux_moe_three_experts"

# =============================================================================
# 模型路径配置 (使用用户的实际路径)
# =============================================================================
FLUX_FILL_ID="/xx/FLUX.1-Fill-dev"
FLUX_REDUX_ID="/xx/FLUX.1-Redux-dev"

# =============================================================================
# 分辨率和桶配置
# =============================================================================
RESOLUTION='512,512'
ASPECT_RATIO_BUCKETS='512,512'
# aspect_ratio_buckets='1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'

# =============================================================================
# MoE 专家配置
# =============================================================================
NUM_EXPERTS=3
EXPERT_NAMES="removal_expert,background_expert,completion_expert"
ROUTING_STRATEGY="soft"  # Options: soft, hard, topk
ROUTING_TEMPERATURE=1.0

# =============================================================================
# 8-bit优化器配置 (节省显存约50%)
# =============================================================================
# 设置为true启用8-bit优化器，false使用标准优化器
USE_8BIT_OPTIMIZER=true
# USE_8BIT_OPTIMIZER=false  # 取消注释这行来禁用8-bit优化器

if [ "$USE_8BIT_OPTIMIZER" = "true" ]; then
    OPTIMIZER_TYPE="adamw8bit"
    echo "⚙️  已启用8-bit优化器，可节省显存约50%"
    echo "🚨 请确保已安装bitsandbytes: pip install bitsandbytes"
else
    OPTIMIZER_TYPE="adamw"
    echo "📊 使用标准AdamW优化器"
fi

# =============================================================================
# 两阶段训练配置 (可选)
# =============================================================================
# 取消注释以下行来进行专家特化训练:
# TASK_HINT="removal_only"      # 专注训练移除专家
# TASK_HINT="background_only"   # 专注训练背景专家  
# TASK_HINT="completion_only"   # 专注训练补全专家
# 留空表示联合训练所有专家
TASK_HINT=""

# =============================================================================
# 损失函数控制 - 消融实验配置
# =============================================================================
# 基础设置 - 只启用base loss（默认安全模式）
ENABLE_BASE_LOSS=true
ENABLE_MASK_INFO=false
ENABLE_BOUNDARY_LOSS=false
ENABLE_CONSISTENCY_LOSS=false
ENABLE_DETAIL_LOSS=false

# 损失权重配置
BASE_LOSS_WEIGHT=1.0
BOUNDARY_LOSS_WEIGHT=0.3
CONSISTENCY_LOSS_WEIGHT=0.5
DETAIL_LOSS_WEIGHT=0.2

# =============================================================================
# 训练超参数配置 (基于用户原始配置)
# =============================================================================
LEARNING_RATE=5e-5
TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=20
GRADIENT_ACCUMULATION_STEPS=4  # 增加梯度累积，减少峰值显存
VALIDATION_STEPS=1000            # 减少验证频率，节省显存
CHECKPOINT_STEPS=1000           # 减少检查点频率

# =============================================================================
# LoRA配置 (基于用户原始配置)
# =============================================================================
USE_LORA=true
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.0
TARGET_MODULES="(.*x_embedder|.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_v|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_out\.0|.*(?<!single_)transformer_blocks\.[0-9]+\.ff\.net\.2|.*single_transformer_blocks\.[0-9]+\.norm\.linear|.*single_transformer_blocks\.[0-9]+\.proj_mlp|.*single_transformer_blocks\.[0-9]+\.proj_out|.*single_transformer_blocks\.[0-9]+\.attn.to_k|.*single_transformer_blocks\.[0-9]+\.attn.to_q|.*single_transformer_blocks\.[0-9]+\.attn.to_v|.*single_transformer_blocks\.[0-9]+\.attn.to_out)"

# =============================================================================
# 其他配置
# =============================================================================
SEED=233
MIXED_PRECISION="bf16"

# =============================================================================
# 创建输出目录
# =============================================================================
mkdir -p $OUTPUT_DIR

echo "========================================================================"
echo "RemoveAnything Three-Expert MoE LoRA Training"
echo "========================================================================"
echo "配置信息:"
echo "- 专家数量: $NUM_EXPERTS"
echo "- 专家名称: $EXPERT_NAMES"
echo "- 路由策略: $ROUTING_STRATEGY"
echo "- 路由温度: $ROUTING_TEMPERATURE"
echo "- 任务提示: ${TASK_HINT:-'无 (联合训练)'}"
echo "- 输出目录: $OUTPUT_DIR"
echo "- 学习率: $LEARNING_RATE"
echo "- 优化器: ${OPTIMIZER_TYPE:-adamw}${USE_8BIT_OPTIMIZER:+ (8-bit)}"
echo "- 批量大小: $TRAIN_BATCH_SIZE"
echo "- LoRA秩: $LORA_R"
echo "========================================================================"

# =============================================================================
# 运行三专家MoE训练
# =============================================================================
accelerate launch src/train_bucket_moe.py \
  --train_json_path "$TRAIN_JSON_PATH" \
  --val_json_path "$VAL_JSON_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS" \
  --flux_fill_id "$FLUX_FILL_ID" \
  --flux_redux_id "$FLUX_REDUX_ID" \
  --learning_rate $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --validation_steps $VALIDATION_STEPS \
  --checkpointing_steps $CHECKPOINT_STEPS \
  --use_lora $USE_LORA \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --target_modules "$TARGET_MODULES" \
  --mixed_precision $MIXED_PRECISION \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --seed $SEED \
  --optimizer_type "$OPTIMIZER_TYPE" \
  --use_8bit_optimizer $USE_8BIT_OPTIMIZER \
  --num_experts $NUM_EXPERTS \
  --expert_names "$EXPERT_NAMES" \
  --routing_strategy "$ROUTING_STRATEGY" \
  --routing_temperature $ROUTING_TEMPERATURE \
  --task_hint "$TASK_HINT" \
  --enable_base_loss \
  --base_loss_weight $BASE_LOSS_WEIGHT \
  --boundary_loss_weight $BOUNDARY_LOSS_WEIGHT \
  --consistency_loss_weight $CONSISTENCY_LOSS_WEIGHT \
  --detail_loss_weight $DETAIL_LOSS_WEIGHT \
  --report_to "tensorboard" \
  --logging_dir "$OUTPUT_DIR/logs" \
#   --enable_mask_info \
#   --enable_boundary_loss \
#   --enable_consistency_loss \
#   --enable_detail_loss \

echo "========================================================================"
echo "训练完成！"
echo "检查点保存在: $OUTPUT_DIR"
echo "日志保存在: $OUTPUT_DIR/logs"
echo "========================================================================"

# =============================================================================
# 使用说明
# =============================================================================
cat << 'EOF'

🎯 两阶段训练策略 (推荐):

1. 专家特化训练阶段:
   # 训练移除专家
   TASK_HINT="removal_only" bash train_moe_local.sh
   
   # 训练背景专家
   TASK_HINT="background_only" bash train_moe_local.sh
   
   # 训练补全专家
   TASK_HINT="completion_only" bash train_moe_local.sh

2. 联合优化阶段:
   # 联合训练所有专家
   TASK_HINT="" bash train_moe_local.sh

🚀 单阶段联合训练:
   # 直接运行
   bash train_moe_local.sh

📊 监控训练过程:
   # 查看TensorBoard
   tensorboard --logdir=/xx/removal_flux_moe_three_experts/logs
   
   # 查看专家使用统计
   grep "Expert Usage" /xx/removal_flux_moe_three_experts/logs/training.log

EOF
