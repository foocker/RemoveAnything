# GRPO训练支持人工标签数据使用指南

## 概述

优化后的 `train_bucket_moe_grpo.py` 现在完全支持使用包含人工标注3个分数的数据进行GRPO训练。这些分数用作奖励信号来指导MoE专家的学习。

## 数据格式

### 1. Dict格式（推荐）
```json
{
  "ref": "path/to/reference_image.jpg",
  "src": "path/to/source_image.jpg", 
  "mask": "path/to/mask.jpg",
  "result": "path/to/result_image.jpg",
  "scores": {
    "removal_score": 0.85,      // 移除质量分数 [0, 1]
    "background_score": 0.72,   // 背景一致性分数 [0, 1]  
    "smoothness_score": 0.91    // 边界平滑度分数 [0, 1]
  }
}
```

### 2. Tensor格式
```json
{
  "ref": "path/to/reference_image.jpg",
  "src": "path/to/source_image.jpg",
  "mask": "path/to/mask.jpg", 
  "result": "path/to/result_image.jpg",
  "scores": [0.85, 0.72, 0.91]  // [removal, background, smoothness]
}
```

## 分数含义

1. **removal_score (0-1)**: 移除质量分数
   - 1.0 = 完美移除，目标物体完全消失
   - 0.0 = 移除失败，目标物体仍然可见

2. **background_score (0-1)**: 背景一致性分数  
   - 1.0 = 背景完全一致，无明显痕迹
   - 0.0 = 背景不一致，有明显修复痕迹

3. **smoothness_score (0-1)**: 边界平滑度分数
   - 1.0 = 边界非常平滑自然
   - 0.0 = 边界生硬，有明显接缝

## 使用方法

### 基本GRPO训练（使用人工标签）
```bash
python src/train_bucket_moe_grpo.py \
    --train_json_path /path/to/labeled_data.json \
    --output_dir ./output/grpo_human_labels \
    --use_grpo \
    --use_human_scores \
    --enable_grpo_rewards \
    --enable_grpo_loss \
    --score_format dict \
    --use_lora \
    --num_experts 3 \
    --removal_reward_weight 0.4 \
    --background_reward_weight 0.3 \
    --smoothness_reward_weight 0.3 \
    --base_loss_weight 1.0 \
    --policy_loss_weight 0.1 \
    --diversity_loss_weight 0.05 \
    --clip_range 0.2 \
    --train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_train_steps 10000
```

### 标准MoE训练（自动计算奖励）
```bash
python src/train_bucket_moe_grpo.py \
    --train_json_path /path/to/standard_data.json \
    --output_dir ./output/moe_standard \
    --use_lora \
    --num_experts 3 \
    --train_batch_size 4 \
    --learning_rate 1e-4 \
    --max_train_steps 10000
```

## 关键参数说明

### GRPO控制参数
- `--use_grpo`: 启用GRPO训练模式
- `--use_human_scores`: 使用人工标注分数作为奖励
- `--enable_grpo_rewards`: 启用GRPO奖励计算
- `--enable_grpo_loss`: 启用GRPO损失计算
- `--score_format`: 分数格式，"dict"或"tensor"

### 奖励权重参数
- `--removal_reward_weight`: 移除质量奖励权重（默认0.4）
- `--background_reward_weight`: 背景一致性奖励权重（默认0.3）
- `--smoothness_reward_weight`: 边界平滑度奖励权重（默认0.3）

### 损失权重参数
- `--base_loss_weight`: 基础重建损失权重（默认1.0）
- `--policy_loss_weight`: 策略损失权重（默认0.1）
- `--diversity_loss_weight`: 专家多样性损失权重（默认0.05）

### PPO参数
- `--clip_range`: PPO裁剪范围（默认0.2）

## 训练模式对比

| 模式 | 数据要求 | 奖励来源 | 适用场景 |
|------|----------|----------|----------|
| 标准MoE | 标准triplet数据 | 无奖励信号 | 基础MoE训练 |
| GRPO自动 | 标准triplet数据 | 自动计算奖励 | 无标签数据的策略优化 |
| GRPO人工 | 包含3个分数的标签数据 | 人工标注奖励 | 有标签数据的精确优化 |

## 监控指标

训练过程中会记录以下关键指标：

### 基础指标
- `train_loss`: 总训练损失
- `lr`: 学习率

### GRPO特有指标
- `base_loss`: 基础重建损失
- `policy_loss`: 策略损失
- `diversity_loss`: 专家多样性损失
- `mean_reward`: 平均奖励值
- `mean_ratio`: 平均重要性采样比率

### MoE路由指标
- `routing_expert_0_usage`: 专家0使用频率
- `routing_expert_1_usage`: 专家1使用频率  
- `routing_expert_2_usage`: 专家2使用频率

## 数据准备建议

1. **标注质量**: 确保人工标注的一致性和准确性
2. **分数分布**: 尽量保证3个分数有合理的分布，避免全部集中在某个范围
3. **数据平衡**: 包含不同难度和类型的样本
4. **格式验证**: 使用提供的数据集类验证数据格式正确性

## 故障排除

### 常见问题

1. **数据格式错误**
   ```
   ValueError: Expected dict format for scores, got <class 'list'>
   ```
   解决：检查 `--score_format` 参数是否与数据格式匹配

2. **缺少分数字段**
   ```
   ValueError: use_human_scores=True but no 'scores' field found in data
   ```
   解决：确保数据中包含 `scores` 字段

3. **奖励计算失败**
   ```
   WARNING: Failed to compute GRPO rewards, falling back to simple loss
   ```
   解决：检查VAE解码是否正常，可能是显存不足

## 性能优化建议

1. **批次大小**: 根据显存调整，GRPO需要额外的VAE解码开销
2. **奖励计算频率**: 可以考虑不是每步都计算奖励以节省计算
3. **混合精度**: 使用bfloat16可以节省显存和加速训练
4. **梯度累积**: 在显存受限时使用梯度累积增加有效批次大小

这个实现为RemoveAnything项目提供了完整的GRPO训练支持，能够充分利用人工标注的质量评分来指导模型学习。
