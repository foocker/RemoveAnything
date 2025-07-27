#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Model paths - adjust these to your local paths
FLUX_FILL_PATH="/aicamera-mlp/fq_proj/weights/modelscope/FLUX.1-Fill-dev"
LORA_WEIGHTS_PATH="/aicamera-mlp/fq_proj/results/removal_flux_lora/checkpoint-3000"
# LORA_WEIGHTS_PATH="/aicamera-mlp/fq_proj/results/removal_flux_lora_mutil/checkpoint-4000"
# LORA_WEIGHTS_PATH="/aicamera-mlp/fq_proj/results/removal_flux_moe_three_experts/checkpoint-4000"
FLUX_REDUX_PATH="/aicamera-mlp/fq_proj/weights/modelscope/FLUX.1-Redux-dev"

# Image and mask paths
# SOURCE_IMAGE="examples/image/000004.png"
# SOURCE_MASK="examples/mask/000004_mask.png"

# SOURCE_IMAGE=/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO/input/Bench_1.png
# SOURCE_MASK=/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO/mask/Bench_1.png

SOURCE_IMAGE=/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO/input/COCO_110.png
SOURCE_MASK=/aicamera-mlp/fq_proj/datasets/Eraser/open_ROREM_RORD_COCO/mask/COCO_110.png
# INPUT_DIR=/aicamera-mlp/fq_proj/datasets/Eraser/user_remove_test_v1
# INPUT_DIR=/aicamera-mlp/fq_proj/codes/RemoveAnything/split_results
INPUT_DIR=/aicamera-mlp/fq_proj/codes/RemoveAnything/sampled_v3

# Output directory
OUTPUT_DIR="./results_3000_single_sampled_v3"
# OUTPUT_DIR="./results_4000_mutil"
# OUTPUT_DIR="./results_4000_moe"

# Inference parameters
SIZE=512  # Options: 512, 768, 1024
NUM_INFERENCE_STEPS=20
SEED=42
REPEAT=1

# Create output directory
mkdir -p $OUTPUT_DIR

# # Run inference script
python src/infer.py \
  --flux_fill_path $FLUX_FILL_PATH \
  --lora_weights_path $LORA_WEIGHTS_PATH \
  --flux_redux_path $FLUX_REDUX_PATH \
  --output_dir $OUTPUT_DIR \
  --size $SIZE \
  --num_inference_steps $NUM_INFERENCE_STEPS \
  --seed $SEED \
  --repeat $REPEAT \
  --input_dir $INPUT_DIR \
  # --source_image $SOURCE_IMAGE \
  # --source_mask $SOURCE_MASK \

# echo "Inference completed!"

