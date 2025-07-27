#!/bin/bash

# RemoveAnything Three-Expert MoE LoRA Training Script (适配本地环境)
# 基于用户提供的实际配置参数

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
# 显存优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# =============================================================================
# 数据路径配置 (使用用户的实际路径)
# =============================================================================
TRAIN_JSON_PATH="/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO_v2/gt_added_mapping_sample_1000.json"
# VAL_JSON_PATH="/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO_v2/gt_added_mapping_val.json"
VAL_JSON_PATH="/aicamera-mlp/fq_proj/datasets/Eraser/user_remove_test_v1"  # 也可以是路径：input,mask配的
TRAIN_METADATA_FILE="/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO_v2/train_metadata.json"

# =============================================================================
# 输出路径配置
# =============================================================================
OUTPUT_DIR="/aicamera-mlp/fq_proj/results/removal_flux_three_experts_fixed_grpo"

# =============================================================================
# 恢复训练选项 (两者选一)
# =============================================================================

# =============================================================================
# 模型路径配置 (使用用户的实际路径)
# =============================================================================
FLUX_FILL_ID="/aicamera-mlp/fq_proj/weights/modelscope/FLUX.1-Fill-dev"
FLUX_REDUX_ID="/aicamera-mlp/fq_proj/weights/modelscope/FLUX.1-Redux-dev"

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
GRADIENT_ACCUMULATION_STEPS=8  # 进一步增加梯度累积，减少峰值显存
VALIDATION_STEPS=1000           # 减少验证频率，节省显存
CHECKPOINT_STEPS=1000          # 减少检查点频率

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
# accelerate launch src/train_bucket_moe.py \
#   --train_json_path "$TRAIN_JSON_PATH" \
#   --val_json_path "$VAL_JSON_PATH" \
#   --output_dir "$OUTPUT_DIR" \
#   --aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS" \
#   --flux_fill_id "$FLUX_FILL_ID" \
#   --flux_redux_id "$FLUX_REDUX_ID" \
#   --learning_rate $LEARNING_RATE \
#   --train_batch_size $TRAIN_BATCH_SIZE \
#   --num_train_epochs $NUM_TRAIN_EPOCHS \
#   --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#   --validation_steps $VALIDATION_STEPS \
#   --checkpointing_steps $CHECKPOINT_STEPS \
#   --use_lora $USE_LORA \
#   --lora_r $LORA_R \
#   --lora_alpha $LORA_ALPHA \
#   --lora_dropout $LORA_DROPOUT \
#   --target_modules "$TARGET_MODULES" \
#   --mixed_precision $MIXED_PRECISION \
#   --enable_xformers_memory_efficient_attention \
#   --gradient_checkpointing \
#   --seed $SEED \
#   --optimizer_type "$OPTIMIZER_TYPE" \
#   --use_8bit_optimizer $USE_8BIT_OPTIMIZER \
#   --num_experts $NUM_EXPERTS \
#   --expert_names removal_expert background_expert completion_expert \
#   --routing_strategy "$ROUTING_STRATEGY" \
#   --routing_temperature $ROUTING_TEMPERATURE \
#   --task_hint "$TASK_HINT" \
#   --enable_base_loss \
#   --base_loss_weight $BASE_LOSS_WEIGHT \
#   --boundary_loss_weight $BOUNDARY_LOSS_WEIGHT \
#   --consistency_loss_weight $CONSISTENCY_LOSS_WEIGHT \
#   --detail_loss_weight $DETAIL_LOSS_WEIGHT \
#   --report_to "tensorboard" \
#   --logging_dir "$OUTPUT_DIR/logs" \
# #   --enable_mask_info \
# #   --enable_boundary_loss \
# #   --enable_consistency_loss \
# #   --enable_detail_loss \

# accelerate launch src/train_bucket_moe_simple.py \
#   --train_json_path "$TRAIN_JSON_PATH" \
#   --val_json_path "$VAL_JSON_PATH" \
#   --output_dir "$OUTPUT_DIR" \
#   --aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS" \
#   --flux_fill_id "$FLUX_FILL_ID" \
#   --flux_redux_id "$FLUX_REDUX_ID" \
#   --learning_rate $LEARNING_RATE \
#   --train_batch_size $TRAIN_BATCH_SIZE \
#   --num_train_epochs $NUM_TRAIN_EPOCHS \
#   --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#   --validation_steps $VALIDATION_STEPS \
#   --checkpointing_steps $CHECKPOINT_STEPS \
#   --use_lora $USE_LORA \
#   --lora_r $LORA_R \
#   --lora_alpha $LORA_ALPHA \
#   --lora_dropout $LORA_DROPOUT \
#   --target_modules "$TARGET_MODULES" \
#   --mixed_precision $MIXED_PRECISION \
#   --enable_xformers_memory_efficient_attention \
#   --gradient_checkpointing \
#   --seed $SEED \
#   --optimizer_type "$OPTIMIZER_TYPE" \
#   --use_8bit_optimizer $USE_8BIT_OPTIMIZER \
#   --num_experts $NUM_EXPERTS \
#   --expert_names removal_expert background_expert completion_expert \
#   --routing_strategy "$ROUTING_STRATEGY" \
#   --routing_temperature $ROUTING_TEMPERATURE \
#   --report_to "tensorboard" \
#   --logging_dir "$OUTPUT_DIR/logs" \


# accelerate launch src/train_bucket_moe_fixed_grpo.py \
#   --train_json_path "$TRAIN_JSON_PATH" \
#   --val_json_path "$VAL_JSON_PATH" \
#   --output_dir "$OUTPUT_DIR" \
#   --aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS" \
#   --flux_fill_id "$FLUX_FILL_ID" \
#   --flux_redux_id "$FLUX_REDUX_ID" \
#   --learning_rate $LEARNING_RATE \
#   --train_batch_size $TRAIN_BATCH_SIZE \
#   --num_train_epochs $NUM_TRAIN_EPOCHS \
#   --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#   --validation_steps $VALIDATION_STEPS \
#   --checkpointing_steps $CHECKPOINT_STEPS \
#   --use_lora $USE_LORA \
#   --lora_r $LORA_R \
#   --lora_alpha $LORA_ALPHA \
#   --lora_dropout $LORA_DROPOUT \
#   --target_modules "$TARGET_MODULES" \
#   --mixed_precision $MIXED_PRECISION \
#   --enable_xformers_memory_efficient_attention \
#   --gradient_checkpointing \
#   --seed $SEED \
#   --optimizer_type "$OPTIMIZER_TYPE" \
#   --use_8bit_optimizer $USE_8BIT_OPTIMIZER \
#   --num_experts $NUM_EXPERTS \
#   --expert_names removal_expert background_expert completion_expert \
#   --routing_strategy "$ROUTING_STRATEGY" \
#   --routing_temperature $ROUTING_TEMPERATURE \
#   --report_to "tensorboard" \
#   --logging_dir "$OUTPUT_DIR/logs" \
#   --fixed_expert_weights 0.6 0.3 0.1 \
#   --enable_grpo_loss \
#   --use_grpo \
#   --base_loss_weight 1.0 \
#   --policy_loss_weight 0.1 \
#   --clip_range 0.2

# echo "========================================================================"
# echo "训练完成！"
# echo "检查点保存在: $OUTPUT_DIR"
# echo "日志保存在: $OUTPUT_DIR/logs"
# echo "========================================================================"


# =============================================================================
# GRPO 动态路由训练 (推荐用于真正的强化学习)
# =============================================================================

echo "========================================================================"
echo "GRPO动态路由训练完成！"
echo "检查点保存在: ${OUTPUT_DIR}_dynamic"
echo "日志保存在: ${OUTPUT_DIR}_dynamic/logs"
echo "========================================================================"

# # 简化GRPO训练（基于原始MoE脚本的最小化集成）
# echo "开始简化GRPO训练（基于原始MoE脚本）..."
# echo "输出目录: ${OUTPUT_DIR}_simple_grpo"

# accelerate launch src/train_bucket_moe_simple_grpo.py \
#   --train_json_path $TRAIN_JSON_PATH \
#   --val_json_path "$VAL_JSON_PATH" \
#   --output_dir "${OUTPUT_DIR}_simple_grpo" \
#   --aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS" \
#   --flux_fill_id $FLUX_FILL_ID \
#   --flux_redux_id $FLUX_REDUX_ID \
#   --learning_rate $LEARNING_RATE \
#   --train_batch_size $TRAIN_BATCH_SIZE \
#   --num_train_epochs $NUM_TRAIN_EPOCHS \
#   --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#   --validation_steps $VALIDATION_STEPS \
#   --checkpointing_steps $CHECKPOINT_STEPS \
#   --use_lora \
#   --lora_r $LORA_R \
#   --lora_alpha $LORA_ALPHA \
#   --lora_dropout $LORA_DROPOUT \
#   --target_modules "$TARGET_MODULES" \
#   --mixed_precision $MIXED_PRECISION \
#   --enable_xformers_memory_efficient_attention \
#   --gradient_checkpointing \
#   --seed $SEED \
#   --optimizer_type "$OPTIMIZER_TYPE" \
#   --use_8bit_optimizer \
#   --num_experts $NUM_EXPERTS \
#   --expert_names removal_expert,background_expert,completion_expert \
#   --routing_strategy "soft" \
#   --routing_temperature $ROUTING_TEMPERATURE \
#   --report_to "tensorboard" \
#   --logging_dir "${OUTPUT_DIR}_simple_grpo/logs" \
#   --enable_grpo_loss \
#   --use_grpo \
#   --base_loss_weight 1.0 \
#   --policy_loss_weight 0.1 \
#   --diversity_loss_weight 0.05 \
#   --clip_range 0.2 \
#   --removal_reward_weight 1.0 \
#   --background_reward_weight 0.5 \
#   --smoothness_reward_weight 0.3

# echo "========================================================================"
# echo "简化GRPO训练完成！"
# echo "检查点保存在: ${OUTPUT_DIR}_simple_grpo"
# echo "日志保存在: ${OUTPUT_DIR}_simple_grpo/logs"
# echo "========================================================================"

# =============================================================================
# 固定权重GRPO训练测试 (基于train_multi_lora_fixed_grpo.py)
# =============================================================================
echo "开始固定权重GRPO训练测试..."
echo "输出目录: ${OUTPUT_DIR}_fixed_grpo_test"

accelerate launch src/train_multi_lora_fixed_grpo.py \
  --train_json_path $TRAIN_JSON_PATH \
  --val_json_path "$VAL_JSON_PATH" \
  --output_dir "${OUTPUT_DIR}_fixed_grpo_test" \
  --aspect_ratio_buckets "$ASPECT_RATIO_BUCKETS" \
  --flux_fill_id $FLUX_FILL_ID \
  --flux_redux_id $FLUX_REDUX_ID \
  --learning_rate $LEARNING_RATE \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --validation_steps $VALIDATION_STEPS \
  --checkpointing_steps $CHECKPOINT_STEPS \
  --use_8bit_optimizer \
  --use_lora True \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --target_modules "$TARGET_MODULES" \
  --mixed_precision $MIXED_PRECISION \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --seed $SEED \
  --optimizer_type "$OPTIMIZER_TYPE" \
  --lora_adapters "adapter1,adapter2,adapter3" \
  --adapter_weights "0.6,0.3,0.1" \
  --report_to "tensorboard" \
  --logging_dir "${OUTPUT_DIR}_fixed_grpo_test/logs" \
  --enable_grpo_loss \
  --use_grpo \
  --base_loss_weight 1.0 \
  --policy_loss_weight 0.1 \
  --diversity_loss_weight 0.05 \
  --clip_range 0.2 \
  --removal_reward_weight 1.0 \
  --background_reward_weight 0.5 \
  --smoothness_reward_weight 0.3

echo "========================================================================"
echo "固定权重GRPO训练测试完成！"
echo "检查点保存在: ${OUTPUT_DIR}_fixed_grpo_test"
echo "日志保存在: ${OUTPUT_DIR}_fixed_grpo_test/logs"
echo "========================================================================"
