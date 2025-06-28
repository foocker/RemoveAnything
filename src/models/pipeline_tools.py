from diffusers.pipelines import FluxPipeline, FluxFillPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor
import torch


def Flux_fill_encode_masks_images(pipeline: FluxFillPipeline, images, masks):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)

    batch_size, num_channels_latents, height, width = images.shape

    masks = pipeline.mask_processor.preprocess(masks)
    masks = masks.to(pipeline.device).to(pipeline.dtype)
    masks = masks[:, 0, :, :] 
    masks = masks.view(
        batch_size, height, pipeline.vae_scale_factor, width, pipeline.vae_scale_factor
    )  
    masks = masks.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
    masks = masks.reshape(
        batch_size, pipeline.vae_scale_factor * pipeline.vae_scale_factor, height, width
    ) 

    masks_tokens = pipeline._pack_latents(
        masks,
        batch_size,
        pipeline.vae_scale_factor * pipeline.vae_scale_factor,
        height,
        width,
    )


    return images_tokens, masks_tokens


def encode_images(pipeline, images):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline: FluxFillPipeline, prompt=None, prompt_embeds=None, pooled_prompt_embeds=None, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def prepare_text_input_catvton(pipeline, prompt, max_sequence_length=512):
    
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids


from PIL import Image, ImageOps # Add if not present
import torchvision.transforms.functional as TF # Add if not present
import numpy as np # Add if not present
# Ensure other existing imports like torch, etc., are maintained

def extract_pil_from_masked_result(result_tensor_batch, mask_tensor_batch, ref_image_size=(256, 256)):
    """
    Extracts the masked region from each result_tensor in the batch and returns a list of PIL Images.
    The extracted patch is placed on a white background, cropped to its bounding box, 
    padded to a square, and resized.

    Args:
        result_tensor_batch (torch.Tensor): Batch of result images (B, C, H, W), normalized e.g. [-1, 1] or [0, 1].
        mask_tensor_batch (torch.Tensor): Batch of mask images (B, 1, H, W), with values typically 0 or 1.
        ref_image_size (tuple): The target size (width, height) for the output PIL image.

    Returns:
        list[PIL.Image]: A list of PIL images, one for each item in the batch.
    """
    pil_images = []
    is_normalized_neg_one_to_one = result_tensor_batch.min() < -0.1 # Heuristic for [-1, 1] normalization

    for i in range(result_tensor_batch.shape[0]):
        result_tensor = result_tensor_batch[i]
        mask_tensor = mask_tensor_batch[i]

        if is_normalized_neg_one_to_one:
            result_tensor_pil = TF.to_pil_image((result_tensor * 0.5 + 0.5).clamp(0, 1).cpu())
        else:
            result_tensor_pil = TF.to_pil_image(result_tensor.clamp(0, 1).cpu())
        
        mask_pil = TF.to_pil_image(mask_tensor.cpu()).convert("L")

        result_np = np.array(result_tensor_pil.convert("RGB"))
        mask_np = np.array(mask_pil) 

        binary_mask_np = (mask_np > 128).astype(np.uint8) 
        binary_mask_rgb_np = np.stack([binary_mask_np] * 3, axis=-1)

        white_background_np = np.ones_like(result_np) * 255
        ref_np = np.where(binary_mask_rgb_np == 1, result_np, white_background_np)
        ref_pil = Image.fromarray(ref_np.astype(np.uint8))

        true_points = np.argwhere(binary_mask_np)
        if true_points.size > 0:
            top_left = true_points.min(axis=0)
            bottom_right = true_points.max(axis=0)
            crop_box = (
                max(0, int(top_left[1])), 
                max(0, int(top_left[0])), 
                min(ref_pil.width, int(bottom_right[1]) + 1), 
                min(ref_pil.height, int(bottom_right[0]) + 1)
            )
            if crop_box[0] < crop_box[2] and crop_box[1] < crop_box[3]:
                cropped_ref_pil = ref_pil.crop(crop_box)
            else: 
                cropped_ref_pil = Image.new("RGB", ref_image_size, (255,255,255))
        else:
            cropped_ref_pil = Image.new("RGB", ref_image_size, (255,255,255))

        width, height = cropped_ref_pil.size
        if width == 0 or height == 0:
             cropped_ref_pil = Image.new("RGB", ref_image_size, (255,255,255))
        else:
            padding_left = (max(width, height) - width) // 2
            padding_right = max(width, height) - width - padding_left
            padding_top = (max(width, height) - height) // 2
            padding_bottom = max(width, height) - height - padding_top
            
            cropped_ref_pil = ImageOps.expand(cropped_ref_pil, 
                                              border=(padding_left, padding_top, padding_right, padding_bottom), 
                                              fill=(255,255,255))

        resized_ref_pil = cropped_ref_pil.resize(ref_image_size, Image.LANCZOS)
        pil_images.append(resized_ref_pil)

    return pil_images