#!/bin/bash

# Set local model paths
FLUX_FILL_PATH="/xx/FLUX.1-Fill-dev"
LORA_WEIGHTS_PATH="/xx/removal_flux_lora/lora-7400"
FLUX_REDUX_PATH="/xx/FLUX.1-Redux-dev"

# Set memory management options
# Options: auto, cuda, cpu
DEVICE="cpu"  
# Options: bf16, fp16, fp32
DTYPE="bf16"  
# Enable CPU offloading to save GPU memory
OFFLOAD="--offload_modules"  # Remove this flag to disable offloading

# Run the app with model paths from this script
python app.py \
  --flux_fill_path "$FLUX_FILL_PATH" \
  --lora_weights_path "$LORA_WEIGHTS_PATH" \
  --flux_redux_path "$FLUX_REDUX_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  $OFFLOAD