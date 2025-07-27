#!/usr/bin/env python3
"""
FID test script using existing pytorch-fid implementation with custom weights.
Modifies the inception model loading to use custom pretrained weights.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add current directory to path to fix relative imports
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

# Modify the FID_WEIGHTS_URL to use our custom weights
import pytorch_fid.inception as inception_module

def patch_inception_for_custom_weights(custom_weights_path):
    """Patch the inception module to use custom weights"""
    
    def custom_fid_inception_v3():
        """Build pretrained Inception model for FID computation with custom weights"""
        # Use the original function but load our custom weights
        inception = inception_module._inception_v3(num_classes=1008, aux_logits=False, weights=None)
        inception.Mixed_5b = inception_module.FIDInceptionA(192, pool_features=32)
        inception.Mixed_5c = inception_module.FIDInceptionA(256, pool_features=64)
        inception.Mixed_5d = inception_module.FIDInceptionA(288, pool_features=64)
        inception.Mixed_6b = inception_module.FIDInceptionC(768, channels_7x7=128)
        inception.Mixed_6c = inception_module.FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6d = inception_module.FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6e = inception_module.FIDInceptionC(768, channels_7x7=192)
        inception.Mixed_7b = inception_module.FIDInceptionE_1(1280)
        inception.Mixed_7c = inception_module.FIDInceptionE_2(2048)

        # Load our custom weights instead of downloading
        if os.path.exists(custom_weights_path):
            print(f"Loading custom FID weights from: {custom_weights_path}")
            state_dict = torch.load(custom_weights_path, map_location='cpu')
            inception.load_state_dict(state_dict)
            print("Custom FID weights loaded successfully!")
        else:
            print(f"Custom weights not found: {custom_weights_path}")
            print("Falling back to default FID weights...")
            state_dict = inception_module.load_state_dict_from_url(inception_module.FID_WEIGHTS_URL, progress=True)
            inception.load_state_dict(state_dict)
        
        return inception
    
    # Replace the original function
    inception_module.fid_inception_v3 = custom_fid_inception_v3


def test_fid_with_custom_weights(custom_weights_path, path1, path2, batch_size=50, device='auto'):
    """Test FID calculation with custom weights"""
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    print(f"Custom weights: {custom_weights_path}")
    print(f"Batch size: {batch_size}")
    
    # Patch inception module to use custom weights
    patch_inception_for_custom_weights(custom_weights_path)
    
    # Import fid_score after patching
    from pytorch_fid.fid_score import calculate_fid_given_paths
    
    # Calculate FID
    print(f"\nCalculating FID between:")
    print(f"  Path 1: {path1}")
    print(f"  Path 2: {path2}")
    
    try:
        fid_score = calculate_fid_given_paths(
            [path1, path2], 
            batch_size=batch_size, 
            device=device, 
            dims=2048, 
            num_workers=1
        )
        
        print(f"\nFID Score: {fid_score:.4f}")
        return fid_score
        
    except Exception as e:
        print(f"FID calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_loading(custom_weights_path):
    """Test if custom weights can be loaded"""
    print("=" * 60)
    print("TESTING CUSTOM WEIGHTS LOADING")
    print("=" * 60)
    
    try:
        # Patch and test loading
        patch_inception_for_custom_weights(custom_weights_path)
        
        from pytorch_fid.inception import InceptionV3
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = InceptionV3([3], resize_input=True, normalize_input=True, 
                           requires_grad=False, use_fid_inception=True).to(device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 299, 299).to(device)
        with torch.no_grad():
            features = model(dummy_input)
        
        print(f"✓ Model loading successful! Output shape: {features[0].shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test FID with custom Inception weights')
    parser.add_argument('--weights', type=str, 
                       default='/aicamera-mlp/fq_proj/weights/pt_inception-2015-12-05-6726825d.pth',
                       help='Path to custom Inception weights')
    parser.add_argument('--path1', type=str, 
                       default='/aicamera-mlp/fq_proj/codes/RemoveAnything/sampled_v3/gt',
                       help='First image directory')
    parser.add_argument('--path2', type=str,
                       default='/aicamera-mlp/fq_proj/codes/RemoveAnything/sampled_v3/pred', 
                       help='Second image directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--test_loading_only', action='store_true', 
                       help='Only test model loading, skip FID calculation')
    
    args = parser.parse_args()
    
    print("FID Test with Custom Weights")
    print("=" * 60)
    
    # Test 1: Model loading
    success = test_model_loading(args.weights)
    if not success:
        print("Model loading test failed. Exiting.")
        return
    
    if args.test_loading_only:
        print("Model loading test completed successfully!")
        return
    
    # Test 2: FID calculation
    if not os.path.exists(args.path1):
        print(f"Path 1 does not exist: {args.path1}")
        return
    
    if not os.path.exists(args.path2):
        print(f"Path 2 does not exist: {args.path2}")
        return
    
    print("\n" + "=" * 60)
    print("TESTING FID CALCULATION")
    print("=" * 60)
    
    fid_score = test_fid_with_custom_weights(
        args.weights, args.path1, args.path2, 
        batch_size=args.batch_size, device=args.device
    )
    
    if fid_score is not None:
        print(f"\n✓ FID calculation successful!")
        print(f"Final FID Score: {fid_score:.4f}")
    else:
        print("\n✗ FID calculation failed!")
    
    print("\nTest completed!")


if __name__ == '__main__':
    main()
