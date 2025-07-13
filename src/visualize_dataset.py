import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data.triplet_bucket_dataset import TripletBucketDataset

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a numpy image (H, W, C)"""
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert from (C, H, W) to (H, W, C)
    img = tensor.permute(1, 2, 0).numpy()
    
    # Scale from [0, 1] to [0, 255]
    img = (img * 255).astype(np.uint8)
    
    return img

def visualize_dataset_item(item, save_dir, filename):
    """Visualize and save the four images from a dataset item"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy images
    ref_img = tensor_to_image(item['ref'])
    src_img = tensor_to_image(item['src'])
    result_img = tensor_to_image(item['result'])
    mask_img = tensor_to_image(item['mask'])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot images with titles
    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(src_img)
    axes[0, 1].set_title('Source Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(result_img)
    axes[1, 0].set_title('Result Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_img)
    axes[1, 1].set_title('Mask')
    axes[1, 1].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}_grid.png"))

    
    plt.close(fig)
    print(f"Images saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize dataset images')
    parser.add_argument('--data_json_path', type=str, required=True, help='Path to the dataset json')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--custom', action='store_true', help='Use custom dataset')
    args = parser.parse_args()
    
    # Create dataset
    dataset = TripletBucketDataset(args.data_json_path, buckets=[(512, 512)], custom=args.custom)
    
    # Visualize samples
    for i in range(min(args.num_samples, len(dataset))):
        item = dataset[i]
        visualize_dataset_item(item, args.save_dir, f"sample_{i}")

if __name__ == '__main__':
    main()
