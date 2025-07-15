# RemoveAnything Intelligent MoE LoRA Training

This document describes the **intelligent three-expert Mixture of Experts (MoE) LoRA architecture** for the RemoveAnything project, featuring **adaptive routing strategies** based on training tasks.

## 🧠 Intelligent Routing Strategy

### 🎯 Core Innovation: Task-Aware MoE Activation

Our MoE system now intelligently switches between two modes based on the training configuration:

```python
# Smart routing decision
use_moe_routing = (
    args.use_lora and moe_manager is not None and 
    (args.enable_boundary_loss or args.enable_consistency_loss or args.enable_detail_loss)
)
```

### 📊 Two Operating Modes

#### 🔄 **Mode 1: MoE Routing Mode** (Expert Specialization)
**Trigger**: When specialized losses are enabled
```bash
# Enable expert differentiation
--enable_boundary_loss      # Edge refinement signals
--enable_consistency_loss   # Background preservation signals  
--enable_detail_loss       # Detail preservation signals
```

**Behavior**:
- ✅ Dynamic routing weights based on mask analysis
- ✅ Expert specialization through distinct loss signals
- ✅ Routing statistics monitoring
- ✅ Task-aware weight distribution

#### ⚖️ **Mode 2: Uniform Weight Mode** (Traditional LoRA Effect)
**Trigger**: When only base MSE loss is enabled
```bash
# Basic training mode
--enable_base_loss          # Only MSE reconstruction loss
# (no additional specialized losses)
```

**Behavior**:
- ✅ Fixed uniform weights [1/3, 1/3, 1/3]
- ✅ Equivalent to traditional LoRA ensemble
- ✅ Computational efficiency
- ✅ No routing overhead

## 🏗️ Expert Architecture

### Three Specialized Experts

1. **🎯 Removal Expert** (`removal_expert`)
   - **Specialization**: Complex object removal with occlusion handling
   - **Activated by**: High edge complexity, medium-large masks
   - **Optimized for**: Clean removal with seamless integration

2. **🌄 Background Expert** (`background_expert`)
   - **Specialization**: Natural background generation and completion
   - **Activated by**: Large area masks, low detail complexity
   - **Optimized for**: Contextual background reconstruction

3. **🔍 Completion Expert** (`completion_expert`)
   - **Specialization**: Fine detail reconstruction and partial completion
   - **Activated by**: Small masks, high detail preservation needs
   - **Optimized for**: Texture and detail preservation

## 🚀 Quick Start

```bash
# Basic training with uniform weights (current config)
bash train_moe.sh
```

## 📊 Advanced Loss System

### 🔄 Adaptive Loss Computation

Our system dynamically computes losses in different spaces for optimal performance:

```python
# Intelligent space selection
- base_loss: Computed in latent space (efficient)
- boundary_loss: Computed in pixel space (mask-aligned)
- consistency_loss: Computed in pixel space (mask-aligned) 
- detail_loss: Computed in pixel space (texture-aligned)
```

### 🎯 Loss Components & Expert Signals

#### 1. **Base MSE Loss** (weight: 1.0)
- **Space**: Latent space
- **Purpose**: Overall reconstruction quality
- **Expert Signal**: Uniform (no differentiation)

#### 2. **Boundary Smoothness Loss** (weight: 0.3)
- **Space**: Pixel space (post-VAE decode)
- **Purpose**: Natural edge transitions
- **Expert Signal**: 🎯 **Removal Expert** specialization

#### 3. **External Consistency Loss** (weight: 0.5)
- **Space**: Pixel space (post-VAE decode)
- **Purpose**: Background preservation
- **Expert Signal**: 🌄 **Background Expert** specialization

#### 4. **High-Frequency Detail Loss** (weight: 0.2)
- **Space**: Pixel space (post-VAE decode)
- **Purpose**: Texture and detail preservation
- **Expert Signal**: 🔍 **Completion Expert** specialization

### ⚙️ Loss Configuration Strategies

#### Strategy A: Expert Differentiation Training
```bash
# Enable all losses for maximum expert specialization
--enable_base_loss \
--enable_boundary_loss --boundary_loss_weight=0.3 \
--enable_consistency_loss --consistency_loss_weight=0.5 \
--enable_detail_loss --detail_loss_weight=0.2 \
--enable_mask_info
```

#### Strategy B: Baseline Training  
```bash
# Only base loss for uniform weight training
--enable_base_loss --base_loss_weight=1.0
# (other losses disabled)
```

#### Strategy C: Custom Specialization
```bash
# Focus on specific aspects
--enable_base_loss \
--enable_boundary_loss --boundary_loss_weight=0.8  # High edge focus
--enable_mask_info
```

## 🎯 Intelligent Mask-Aware Routing

### 🧠 Router Intelligence System

The router analyzes both **visual features** and **task requirements** to make routing decisions:

#### Input Features
```python
# Multi-modal routing input
hidden_states = cat([x_t, condition_latents], dim=2)  # Visual context
mask_info = {                                         # Task context
    'mask_latent': mask_in_latent_space,
    'area_ratio': mask_coverage_percentage,
    'edge_complexity': sobel_edge_analysis,
    'boundary_smoothness': edge_regularity_score
}
```

#### Routing Network Architecture
```python
class ThreeExpertRemovalRouter(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # 3 experts
        )
```

### 📊 Routing Decision Logic

#### 🔄 **Active Routing Mode**
```python
# When specialized losses provide clear signals
if boundary_loss or consistency_loss or detail_loss:
    weights = router.forward(hidden_states, mask_info)
    # Result: [0.2, 0.6, 0.2] - Background expert dominates
```

#### ⚖️ **Uniform Mode**
```python
# When only base loss is used
if only_base_loss:
    weights = [0.333, 0.333, 0.333]  # Equal contribution
    # Result: Traditional LoRA ensemble behavior
```

### 🎛️ Routing Configuration

```bash
# Router hyperparameters
--routing_temperature=1.0    # Temperature for softmax (0.5-2.0)
--task_hint="removal_focus"  # Optional task guidance
```

### 📈 Routing Statistics Monitoring

```python
# Training logs show:
"expert_0_usage": 45,    # Removal expert usage count
"expert_1_usage": 78,    # Background expert usage count  
"expert_2_usage": 23,    # Completion expert usage count
"current_weights": [0.25, 0.55, 0.20]  # Current batch weights
```

## ⚙️ Key Parameters

```bash
# Essential MoE settings
--use_lora                 # Enable LoRA training
--num_experts=3           # Three expert architecture
--lora_r=256 --lora_alpha=256 --lora_dropout=0.0
--routing_temperature=1.0  # Routing decision sharpness

# Loss configuration (choose based on strategy)
--enable_base_loss --base_loss_weight=1.0         # Always enabled
--enable_boundary_loss --boundary_loss_weight=0.3  # For routing mode
--enable_consistency_loss --consistency_loss_weight=0.5 # For routing mode
--enable_detail_loss --detail_loss_weight=0.2      # For routing mode
--enable_mask_info                                 # Required for routing
```


## 📋 Data Format

### 🎯 Required Batch Structure

Your dataset must provide the following format:

```python
batch = {
    "ref": reference_image,      # [B, C, H, W] Reference image for Redux encoding
    "src": source_image,         # [B, C, H, W] Source image with object to remove  
    "mask": mask,               # [B, 1, H, W] Binary mask of region to remove
    "result": target_image      # [B, C, H, W] Target result image
}
```

### 📐 Data Specifications

- **Image Resolution**: 512×1024, 1024×512, or 512×512 (following Flux h,2*w convention)
- **Channels**: RGB images (3 channels)
- **Mask Format**: Single channel binary mask (0=keep, 1=remove)
- **Data Type**: float32 or float16
- **Normalization**: [0, 1] range (handled by VAE)

### 🔄 Data Processing Pipeline

```python
# Training pipeline processing
1. ref → flux_redux_pipe.image_encoder → condition_latents
2. src → vae.encode → source_latents  
3. result → vae.encode → target_latents
4. mask → vae.encode → mask_latent (for loss computation)
```

## 🔧 Advanced Usage

### 🎛️ Loss Weight Tuning

根据数据集特点调整损失权重：

```bash
# 细节丰富的数据集
--boundary_loss_weight=0.8 --detail_loss_weight=0.5

# 复杂背景的数据集  
--consistency_loss_weight=0.8 --boundary_loss_weight=0.3

# 不规则形状的数据集
--boundary_loss_weight=0.6 --consistency_loss_weight=0.5
```

### 📊 训练监控

**路由统计** (仅在路由模式下):
- `expert_0/1/2_usage`: 专家使用次数
- `current_weights`: 当前路由权重 (应该会分化)

**损失组件**:
- `loss/base`: 基础重建损失
- `loss/boundary`: 边界平滑损失  
- `loss/consistency`: 背景保持损失
- `loss/detail`: 细节保持损失

### 🔄 训练模式切换

可以在训练过程中切换模式：

```bash
# 阶段1: 均匀权重训练 (0-1000步)
--enable_base_loss

# 阶段2: 路由模式训练 (1000+步)
--enable_base_loss --enable_boundary_loss --enable_consistency_loss --enable_detail_loss --enable_mask_info
```

## 🎮 Training Strategies

### 🎯 Strategy 1: Progressive Specialization (Recommended)

#### Phase 1: Foundation Training (500-1000 steps)
```bash
# Build solid foundation with uniform weights
--enable_base_loss --base_loss_weight=1.0
# MoE operates in uniform weight mode [0.33, 0.33, 0.33]
```

#### Phase 2: Expert Differentiation (1000+ steps)
```bash
# Enable specialized losses for expert differentiation
--enable_base_loss --base_loss_weight=1.0 \
--enable_boundary_loss --boundary_loss_weight=0.3 \
--enable_consistency_loss --consistency_loss_weight=0.5 \
--enable_detail_loss --detail_loss_weight=0.2 \
--enable_mask_info
# MoE switches to active routing mode
```

### 🚀 Strategy 2: Direct Specialization

```bash
# Start with full specialization immediately
--enable_base_loss --base_loss_weight=1.0 \
--enable_boundary_loss --boundary_loss_weight=0.3 \
--enable_consistency_loss --consistency_loss_weight=0.5 \
--enable_detail_loss --detail_loss_weight=0.2 \
--enable_mask_info
```

### ⚖️ Strategy 3: Conservative Baseline

```bash
# Traditional LoRA-like training with uniform weights
--enable_base_loss --base_loss_weight=1.0
# Keep other losses disabled throughout training
```

## 📈 Monitoring and Debugging

### Key Metrics to Monitor

1. **Total Loss**: Overall training progress
2. **Component Losses**: Individual loss contributions
3. **Expert Usage**: Routing distribution across experts
4. **Boundary Quality**: Edge transition smoothness
5. **Consistency**: Background preservation quality

### Logging Output

```
INFO - Epoch 1/100, Step 100/5000
INFO - Total Loss: 0.1234, Base: 0.0800, Boundary: 0.0200, Consistency: 0.0150, Detail: 0.0084
INFO - Expert Usage: removal=0.45, background=0.35, completion=0.20
INFO - Routing Temperature: 1.0, Strategy: soft
```

## 🔍 Troubleshooting

### Common Issues

1. **Expert Imbalance**: One expert dominates routing
   - **Solution**: Adjust routing temperature or use two-stage training

2. **Poor Boundary Quality**: Visible seams at mask edges
   - **Solution**: Increase boundary_weight in loss function

3. **Background Artifacts**: Unwanted changes outside mask
   - **Solution**: Increase consistency_weight in loss function

4. **Loss of Details**: Blurry or smoothed results
   - **Solution**: Increase detail_weight in loss function

### Debug Commands

```bash
# Check routing distribution
grep "Expert Usage" logs/training.log

# Monitor loss components  
grep "Component Losses" logs/training.log

# Validate model weights
python -c "
import torch
model = torch.load('outputs/checkpoint-1000/model.safetensors')
print([k for k in model.keys() if 'lora' in k])
"
```

## 📚 Technical Details

### Architecture Changes

1. **Replaced** `FeatureExtractor` → `MaskAwareFeatureExtractor`
2. **Replaced** `LoRARouter` → `ThreeExpertRemovalRouter`  
3. **Added** `compute_removal_task_loss()` specialized loss function
4. **Added** `extract_mask_info_from_batch()` mask analysis
5. **Enhanced** MoE routing with mask awareness


## 🎯 Expected Results

With proper training, you should see:

- **Specialized Expert Behavior**: Each expert handling its designated task type
- **Improved Boundary Quality**: Smoother transitions at removal edges  
- **Better Background Preservation**: Minimal artifacts outside mask regions
- **Enhanced Detail Retention**: Preserved texture and fine details
- **Intelligent Routing**: Appropriate expert selection based on mask characteristics


## 🔧 Troubleshooting

### 🐛 常见问题

**路由权重不变化**: 仅启用了base_loss，需要启用其他损失来激活路由
**内存不足**: 降低`--batch_size=1`，启用`--gradient_checkpointing`
**损失组件不显示**: 检查是否启用`--enable_mask_info`

### 📈 性能提示

- **Batch Size**: 从2开始，根据显存调整
- **Learning Rate**: 保守的1e-4保证MoE稳定性
- **Checkpointing**: 每500步保存检查点

### 🔍 调试模式

```bash
# 启用详细日志
--logging_dir="./logs" --report_to="tensorboard"

# 查看 tensorboard 日志
tensorboard --logdir=./logs
```

## 📚 资源指南

- **训练脚本**: `train_moe.sh` (路由模式)
- **日志查看**: `tensorboard --logdir=./logs`
- **模型保存**: 每个专家适配器分别保存

---

**祖传遇げ🚀 祥禄与模型MoE的训练！**
