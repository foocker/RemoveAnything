#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Model paths - adjust these to your local paths
FLUX_FILL_PATH="/xx/FLUX.1-Fill-dev"
LORA_WEIGHTS_PATH="/xx/Remove-Anything"
FLUX_REDUX_PATH="/xx/FLUX.1-Redux-dev"

# Image and mask paths
SOURCE_IMAGE="examples/source_image/xx.png"
SOURCE_MASK="examples/source_mask/xx_mask.png"

# Output directory
OUTPUT_DIR="./results"

# Inference parameters
SIZE=768  # Options: 512, 768, 1024
NUM_INFERENCE_STEPS=30
SEED=666
REPEAT=1#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Model paths - adjust these to your local paths
FLUX_FILL_PATH="xx/FLUX.1-Fill-dev"
LORA_WEIGHTS_PATH="xx/Remove-Anything"
FLUX_REDUX_PATH="xx/FLUX.1-Redux-dev"

# Image and mask paths
# SOURCE_IMAGE="examples/image/000004.png"
# SOURCE_MASK="examples/mask/000004_mask.png"

# SOURCE_IMAGE=xx/input/Bench_1.png
# SOURCE_MASK=xx/mask/Bench_1.png

SOURCE_IMAGE=xx/input/COCO_110.png
SOURCE_MASK=xx/mask/COCO_110.png
INPUT_DIR=xx/remove_test_examples

# Output directory
OUTPUT_DIR="./results_3000_single"
# OUTPUT_DIR="./results_4000_mutil"
# OUTPUT_DIR="./results_4000_moe"

# Inference parameters
SIZE=512  # Options: 512, 768, 1024
NUM_INFERENCE_STEPS=20
SEED=42
REPEAT=1

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference script
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

echo "Inference completed!"
