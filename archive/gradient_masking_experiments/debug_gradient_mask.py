#!/usr/bin/env python3
"""
Debug script to visualize gradient masks and identify noise/static issues
"""

import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Add guided-diffusion to path
sys.path.append('./guided-diffusion')

try:
    from guided_diffusion.tfg_data_util import create_gradient_mask, tfg_add_cond_inputs
    print("✅ Successfully imported TFG utilities")
except ImportError as e:
    print(f"❌ Failed to import TFG utilities: {e}")
    sys.exit(1)

def visualize_masks():
    """Create and visualize different mask configurations"""
    print("🔍 Debugging Gradient Mask Implementation")
    print("=" * 50)
    
    # Test parameters
    B, H, W = 1, 128, 128
    face_hide_percentage = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"📐 Mask dimensions: {H}x{W}")
    print(f"🎯 Face hide percentage: {face_hide_percentage}")
    
    # Test different configurations
    configs = [
        {"blur": 0, "name": "Hard Masking (Original)"},
        {"blur": 5, "name": "Light Blur"},
        {"blur": 15, "name": "Medium Blur"},
        {"blur": 25, "name": "Heavy Blur"},
    ]
    
    fig, axes = plt.subplots(2, len(configs), figsize=(16, 8))
    
    for i, config in enumerate(configs):
        blur_size = config["blur"]
        name = config["name"]
        
        print(f"\n🧪 Testing {name} (blur={blur_size})")
        
        # Create mask
        mask = create_gradient_mask(B, H, W, face_hide_percentage, blur_size, device, dtype)
        mask_np = mask[0, 0].cpu().numpy()  # Get first batch, first channel
        
        # Analyze mask properties
        unique_vals = np.unique(mask_np)
        print(f"   Unique values: {len(unique_vals)} (range: {unique_vals.min():.3f} - {unique_vals.max():.3f})")
        
        # Check for noise in top region (should be all zeros)
        mask_start_idx = int(H * (1 - face_hide_percentage))
        top_region = mask_np[:mask_start_idx, :]
        top_nonzero = np.count_nonzero(top_region)
        print(f"   Top region non-zero pixels: {top_nonzero}/{top_region.size} ({top_nonzero/top_region.size*100:.1f}%)")
        
        # Check transition region
        transition_start = max(0, mask_start_idx - 10)
        transition_end = min(H, mask_start_idx + 10)
        transition_region = mask_np[transition_start:transition_end, :]
        transition_std = np.std(transition_region)
        print(f"   Transition region std: {transition_std:.3f}")
        
        # Visualize mask
        axes[0, i].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"{name}\n{len(unique_vals)} unique values")
        axes[0, i].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7, label='Mask boundary')
        axes[0, i].legend()
        
        # Show cross-section (middle column)
        mid_col = W // 2
        cross_section = mask_np[:, mid_col]
        axes[1, i].plot(cross_section, 'b-', linewidth=2)
        axes[1, i].axvline(x=mask_start_idx, color='red', linestyle='--', alpha=0.7, label='Mask boundary')
        axes[1, i].set_ylim(-0.1, 1.1)
        axes[1, i].set_xlabel('Height (pixels)')
        axes[1, i].set_ylabel('Mask value')
        axes[1, i].set_title('Cross-section (middle)')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()
        
        # Highlight problematic regions
        if top_nonzero > 0:
            print(f"   ⚠️  WARNING: Found {top_nonzero} non-zero pixels in top region!")
            
        if blur_size > 0 and transition_std < 0.01:
            print(f"   ⚠️  WARNING: Very low transition variance - blur may not be working!")
    
    plt.tight_layout()
    plt.savefig('debug_gradient_masks.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved as: debug_gradient_masks.png")
    
    return configs

def test_blur_implementation():
    """Test the Gaussian blur implementation specifically"""
    print("\n🔍 Testing Gaussian Blur Implementation")
    print("=" * 50)
    
    # Create a simple test mask
    H, W = 128, 128
    test_mask = np.zeros((H, W), dtype=np.float32)
    mask_start_idx = int(H * 0.5)  # 50% hide
    test_mask[mask_start_idx:, :] = 1.0
    
    print(f"📐 Original mask: {np.count_nonzero(test_mask == 0)} zeros, {np.count_nonzero(test_mask == 1)} ones")
    
    # Test different blur kernel sizes
    blur_sizes = [5, 15, 25]
    
    fig, axes = plt.subplots(1, len(blur_sizes) + 1, figsize=(16, 4))
    
    # Show original
    axes[0].imshow(test_mask, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Hard Mask')
    axes[0].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7)
    
    for i, blur_size in enumerate(blur_sizes):
        print(f"\n🧪 Testing blur size: {blur_size}")
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(test_mask, (blur_size, blur_size), 0)
        
        # Check for issues
        min_val, max_val = blurred.min(), blurred.max()
        print(f"   Value range: {min_val:.3f} - {max_val:.3f}")
        
        # Check top region
        top_region = blurred[:mask_start_idx, :]
        top_nonzero = np.count_nonzero(top_region > 1e-6)  # Small threshold for floating point
        print(f"   Top region affected pixels: {top_nonzero}/{top_region.size} ({top_nonzero/top_region.size*100:.1f}%)")
        
        # Check if we have values outside [0,1]
        outside_range = np.count_nonzero((blurred < 0) | (blurred > 1))
        if outside_range > 0:
            print(f"   ⚠️  WARNING: {outside_range} pixels outside [0,1] range!")
        
        # Visualize
        axes[i+1].imshow(blurred, cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f'Blur {blur_size}\nRange: {min_val:.3f}-{max_val:.3f}')
        axes[i+1].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7)
        
        # Show cross-section
        mid_col = W // 2
        cross_section = blurred[:, mid_col]
        
        # Find transition region
        transition_pixels = np.where((cross_section > 0.01) & (cross_section < 0.99))[0]
        if len(transition_pixels) > 0:
            print(f"   Transition region: {len(transition_pixels)} pixels ({transition_pixels[0]} to {transition_pixels[-1]})")
        else:
            print(f"   ⚠️  WARNING: No smooth transition found!")
    
    plt.tight_layout()
    plt.savefig('debug_blur_implementation.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Blur test saved as: debug_blur_implementation.png")

def test_realistic_scenario():
    """Test with realistic batch scenario like in inference"""
    print("\n🔍 Testing Realistic Batch Scenario")
    print("=" * 50)
    
    # Simulate real inference conditions
    B, C, H, W = 4, 3, 128, 128  # Realistic batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy image batch
    img_batch = torch.randn(B, C, H, W, device=device) * 0.5  # Normalized to roughly [-1,1]
    model_kwargs = {}
    
    print(f"📦 Batch size: {B}, Channels: {C}, Image size: {H}x{W}")
    print(f"📱 Device: {device}")
    
    # Test both hard and gradient masking
    test_cases = [
        {"use_gradient": False, "blur": 0, "name": "Hard Masking"},
        {"use_gradient": True, "blur": 15, "name": "Gradient Masking"},
    ]
    
    for case in test_cases:
        print(f"\n🧪 Testing {case['name']}")
        
        result_kwargs = tfg_add_cond_inputs(
            img_batch,
            model_kwargs.copy(),
            face_hide_percentage=0.5,
            use_gradient_mask=case['use_gradient'],
            blur_kernel_size=case['blur']
        )
        
        mask = result_kwargs['mask']
        cond_img = result_kwargs['cond_img']
        
        # Analyze first batch item
        mask_sample = mask[0, 0].cpu().numpy()  # First batch, first channel
        cond_sample = cond_img[0].cpu().numpy()  # First batch, all channels
        
        # Check mask properties
        unique_vals = len(np.unique(mask_sample))
        mask_start_idx = int(H * 0.5)
        top_region = mask_sample[:mask_start_idx, :]
        top_affected = np.count_nonzero(top_region > 1e-6)
        
        print(f"   Mask unique values: {unique_vals}")
        print(f"   Top region affected: {top_affected}/{top_region.size} ({top_affected/top_region.size*100:.1f}%)")
        print(f"   Mask range: {mask_sample.min():.3f} - {mask_sample.max():.3f}")
        
        # Check conditional image
        print(f"   Cond image range: {cond_sample.min():.3f} - {cond_sample.max():.3f}")
        
        # Look for noise patterns in top region of conditional image
        top_cond = cond_sample[:, :mask_start_idx, :]  # All channels, top region
        bottom_cond = cond_sample[:, mask_start_idx:, :]  # All channels, bottom region
        
        top_std = np.std(top_cond)
        bottom_std = np.std(bottom_cond)
        
        print(f"   Top region std: {top_std:.3f}")
        print(f"   Bottom region std: {bottom_std:.3f}")
        
        if case['use_gradient'] and top_affected > top_region.size * 0.01:  # More than 1% affected
            print(f"   ⚠️  WARNING: Gradient mask affecting too much of top region!")
        
        # Save sample visualization
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(mask_sample, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'{case["name"]} - Mask')
        axes[0].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7)
        
        # Show RGB conditional image (convert from CHW to HWC)
        cond_rgb = np.transpose(cond_sample, (1, 2, 0))
        cond_rgb = (cond_rgb - cond_rgb.min()) / (cond_rgb.max() - cond_rgb.min())  # Normalize for display
        axes[1].imshow(cond_rgb)
        axes[1].set_title(f'{case["name"]} - Conditional Image')
        axes[1].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7)
        
        # Show difference from original (just first channel)
        orig_sample = img_batch[0, 0].cpu().numpy()
        diff = np.abs(cond_sample[0] - orig_sample)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'{case["name"]} - Difference from Original')
        axes[2].axhline(y=mask_start_idx, color='cyan', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'debug_realistic_{case["name"].lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        
        print(f"   📊 Saved: debug_realistic_{case['name'].lower().replace(' ', '_')}.png")

def main():
    """Run all debug tests"""
    print("🚀 Gradient Mask Debugging Suite")
    print("=" * 60)
    
    try:
        # Create output directory for debug images
        os.makedirs('debug_output', exist_ok=True)
        os.chdir('debug_output')
        
        # Run debug tests
        visualize_masks()
        test_blur_implementation()
        test_realistic_scenario()
        
        print(f"\n✅ All debug tests completed!")
        print(f"📁 Debug images saved in: debug_output/")
        print(f"💡 Please check the images to identify the noise/static issues")
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
