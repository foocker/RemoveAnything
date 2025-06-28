# RemoveAnything

A deep learning tool for removing objects in images using FLUX model with LoRA fine-tuning.

## Overview

RemoveAnything allows you to:
- Remove objects from images by providing a mask
- Fine-tune the FLUX model on your own dataset using LoRA


## Training

1. Prepare your dataset with source images and masks
2. Configure the training parameters in `train.sh` 
3. Run the training script:

```bash
./train.sh
```

The training script uses Accelerate to enable distributed training and mixed precision.

## Inference

1. Prepare your source image and mask
2. Configure the inference parameters in `infer.sh`
3. Run the inference script:

```bash
./infer.sh
```

## Command Line Arguments

### Training

```
--train_json_path: Path to training JSON file
--val_json_path: Path to validation JSON file
--output_dir: Directory to save model checkpoints
--resolution: Image resolution for training
--flux_fill_id: Path to FLUX Fill model
--flux_redux_id: Path to FLUX Redux model
--learning_rate: Learning rate
--train_batch_size: Training batch size
--max_train_steps: Maximum number of training steps
--gradient_accumulation_steps: Gradient accumulation steps
--validation_steps: Steps between validations
--checkpointing_steps: Steps between checkpoints
--use_lora: Whether to use LoRA
--lora_r: LoRA rank
--lora_alpha: LoRA alpha
--lora_dropout: LoRA dropout
--target_modules: LoRA target modules
```

### Inference

```
--flux_fill_path: Path to FLUX Fill model
--lora_weights_path: Path to LoRA weights
--flux_redux_path: Path to FLUX Redux model
--source_image: Path to source image
--source_mask: Path to source mask
--output_dir: Directory to save results
--size: Image size for processing (512, 768, or 1024)
--num_inference_steps: Number of inference steps
--seed: Random seed for generation
--repeat: Number of times to repeat inference
```

## References

This project uses the [Insert Anything](https://github.com/song-wensong/insert-anything) model architecture with LoRA fine-tuning.
