# RemoveAnything

A deep learning tool for removing objects from images using the FLUX model with LoRA fine-tuning.

## Features

- **Object Removal**: Remove objects from images by providing a corresponding mask.
- **Fine-tuning**: Fine-tune the FLUX model on a custom dataset using LoRA.

## Usage

### Training

1.  Prepare your dataset with source images and masks.
2.  Configure the training parameters in a shell script (e.g., `train.sh`).
3.  Run the training script:

    ```bash
    ./train.sh
    ```

### Inference

1.  Prepare your source image and mask.
2.  Configure the inference parameters in a shell script (e.g., `infer.sh`).
3.  Run the inference script:

    ```bash
    ./infer.sh
    ```

## Command-Line Arguments

### Training Arguments

| Argument                                   | Description                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| `--train_json_path`                        | Path to the training JSON file.                                                |
| `--val_json_path`                          | Path to the validation JSON file or directory.                                 |
| `--output_dir`                             | Directory to save model checkpoints.                                           |
| `--aspect_ratio_buckets`                   | A semicolon-separated list of resolutions for multi-scale training.            |
| `--flux_fill_id`                           | Path or ID of the FLUX Fill model.                                             |
| `--flux_redux_id`                          | Path or ID of the FLUX Redux model.                                            |
| `--learning_rate`                          | The learning rate for training.                                                |
| `--train_batch_size`                       | The training batch size.                                                       |
| `--num_train_epochs`                       | Total number of training epochs to perform.                                    |
| `--gradient_accumulation_steps`            | Number of gradient accumulation steps.                                         |
| `--validation_steps`                       | Steps between validation runs.                                                 |
| `--checkpointing_steps`                    | Steps between saving checkpoints.                                              |
| `--use_lora`                               | Flag to enable LoRA training.                                                  |
| `--lora_r`                                 | LoRA rank.                                                                     |
| `--lora_alpha`                             | LoRA alpha.                                                                    |
| `--lora_dropout`                           | LoRA dropout rate.                                                             |
| `--target_modules`                         | LoRA target modules to apply fine-tuning to (regex pattern).                   |
| `--mixed_precision`                        | Mixed precision training (`no`, `fp16`, `bf16`).                               |
| `--enable_xformers_memory_efficient_attention` | Flag to enable xformers memory-efficient attention.                            |
| `--gradient_checkpointing`                 | Flag to enable gradient checkpointing.                                         |
| `--pretrained_model_path`                  | Path to pre-trained LoRA weights (optional).                                   |
| `--resume_from_checkpoint`                 | Path to a checkpoint to resume training from (optional).                       |

### Inference Arguments

| Argument                | Description                               |
| ----------------------- | ----------------------------------------- |
| `--flux_fill_path`      | Path to the FLUX Fill model.              |
| `--lora_weights_path`   | Path to the trained LoRA weights.         |
| `--flux_redux_path`     | Path to the FLUX Redux model.             |
| `--source_image`        | Path to the source image.                 |
| `--source_mask`         | Path to the source mask.                  |
| `--output_dir`          | Directory to save the output results.     |
| `--size`                | Image size for processing (e.g., 1024).   |
| `--num_inference_steps` | Number of inference steps.                |
| `--seed`                | Random seed for generation.               |
| `--repeat`              | Number of times to repeat inference.      |

## Acknowledgements

This project is based on the [Insert Anything](https://github.com/song-wensong/insert-anything) model architecture.
