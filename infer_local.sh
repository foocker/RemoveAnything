#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Model paths - adjust these to your local paths
FLUX_FILL_PATH="/aicamera-mlp/fq_proj/weights/modelscope/FLUX.1-Fill-dev"
LORA_WEIGHTS_PATH="/aicamera-mlp/fq_proj/weights/Remove-Anything"
FLUX_REDUX_PATH="/aicamera-mlp/fq_proj/weights/modelscope/FLUX.1-Redux-dev"

# Image and mask paths
SOURCE_IMAGE="examples/source_image/000004.png"
SOURCE_MASK="examples/source_mask/000004_aligned_mask.png"

# Output directory
OUTPUT_DIR="./results"

# Inference parameters
SIZE=768  # Options: 512, 768, 1024
NUM_INFERENCE_STEPS=30
SEED=666
REPEAT=1

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference script
python src/infer.py \
  --flux_fill_path $FLUX_FILL_PATH \
  --lora_weights_path $LORA_WEIGHTS_PATH \
  --flux_redux_path $FLUX_REDUX_PATH \
  --source_image $SOURCE_IMAGE \
  --source_mask $SOURCE_MASK \
  --output_dir $OUTPUT_DIR \
  --size $SIZE \
  --num_inference_steps $NUM_INFERENCE_STEPS \
  --seed $SEED \
  --repeat $REPEAT

echo "Inference completed!"