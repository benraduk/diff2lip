#!/usr/bin/env python3
"""
Detailed comparison of hard masking vs gradient masking to identify edge artifacts
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

def create_original_hard_mask(B, H, W, face_hide_percentage, device, dtype):
    """Create the original hard mask for comparison"""
    mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    mask_start_idx = int(H * (1 - face_hide_percentage))
    mask[:, :, mask_start_idx:, :] = 1.0
    return mask

def create_gaussian_blur_mask(B, H, W, face_hide_percentage, blur_kernel_size, device, dtype):
    """Create mask using the original Gaussian blur approach (problematic but for comparison)"""
    mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    mask_start_idx = int(H * (1 - face_hide_percentage))
    mask[:, :, mask_start_idx:, :] = 1.0
    
    if blur_kernel_size > 0:
        # Convert to numpy for cv2 operations
        mask_np = mask.cpu().numpy()
        for b in range(B):
            mask_np[b, 0] = cv2.GaussianBlur(
                mask_np[b, 0], 
                (blur_kernel_size, blur_kernel_size), 
                0
            )
        mask = torch.from_numpy(mask_np).to(device=device, dtype=dtype).clamp(0, 1)
    
    return mask

def create_improved_gradient_mask(B, H, W, face_hide_percentage, blur_kernel_size, device, dtype):
    """Create an improved gradient mask that should be smoother than hard masking"""
    mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    mask_start_idx = int(H * (1 - face_hide_percentage))
    
    if blur_kernel_size > 0:
        # Create a linear gradient instead of sigmoid
        # This should be much smoother and more predictable
        
        transition_width = min(blur_kernel_size, H - mask_start_idx)
        
        for h in range(H):
            if h < mask_start_idx:
                # Top region: completely zero
                mask_value = 0.0
            elif h >= mask_start_idx + transition_width:
                # Bottom region: completely one
                mask_value = 1.0
            else:
                # Transition region: linear gradient
                progress = (h - mask_start_idx) / transition_width
                mask_value = progress
            
            mask[:, :, h, :] = mask_value
    else:
        # Hard masking fallback
        mask[:, :, mask_start_idx:, :] = 1.0
    
    return mask

def analyze_mask_properties(mask, name, mask_start_idx):
    """Analyze mask properties for debugging"""
    mask_np = mask[0, 0].cpu().numpy()  # First batch, first channel
    
    # Basic properties
    unique_vals = np.unique(mask_np)
    min_val, max_val = mask_np.min(), mask_np.max()
    
    # Top region analysis (should be all zeros)
    top_region = mask_np[:mask_start_idx, :]
    top_nonzero = np.count_nonzero(top_region > 1e-6)
    top_mean = np.mean(top_region)
    
    # Bottom region analysis (should be all ones for hard mask)
    bottom_region = mask_np[mask_start_idx + 10:, :]  # Skip transition area
    bottom_mean = np.mean(bottom_region)
    
    # Transition region analysis
    transition_start = max(0, mask_start_idx - 5)
    transition_end = min(mask_np.shape[0], mask_start_idx + 15)
    transition_region = mask_np[transition_start:transition_end, :]
    transition_std = np.std(transition_region)
    
    # Edge analysis - look for sharp transitions
    # Calculate gradient magnitude to detect sharp edges
    grad_y, grad_x = np.gradient(mask_np)
    grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    max_gradient = np.max(grad_magnitude)
    mean_gradient = np.mean(grad_magnitude)
    
    print(f"\n📊 {name} Analysis:")
    print(f"   Unique values: {len(unique_vals)} (range: {min_val:.3f} - {max_val:.3f})")
    print(f"   Top region: {top_nonzero}/{top_region.size} non-zero ({top_nonzero/top_region.size*100:.1f}%), mean: {top_mean:.3f}")
    print(f"   Bottom region mean: {bottom_mean:.3f}")
    print(f"   Transition std: {transition_std:.3f}")
    print(f"   Edge gradient: max={max_gradient:.3f}, mean={mean_gradient:.3f}")
    
    if top_nonzero > 0:
        print(f"   ⚠️  WARNING: Top region contamination!")
    
    if max_gradient > 0.5:
        print(f"   ⚠️  WARNING: Very sharp edges detected!")
    
    return {
        'unique_vals': len(unique_vals),
        'top_contamination': top_nonzero/top_region.size,
        'bottom_mean': bottom_mean,
        'transition_std': transition_std,
        'max_gradient': max_gradient,
        'mean_gradient': mean_gradient
    }

def test_mask_on_realistic_image():
    """Test masks on a realistic scenario similar to inference"""
    print("\n🔍 Testing Masks on Realistic Image Scenario")
    print("=" * 60)
    
    # Create a test image similar to face processing
    B, C, H, W = 1, 3, 128, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    face_hide_percentage = 0.5
    blur_kernel_size = 15
    
    # Create a synthetic face-like image (gradient from top to bottom)
    test_image = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    for h in range(H):
        # Create a gradient that simulates face colors
        intensity = h / H  # 0 at top, 1 at bottom
        test_image[:, 0, h, :] = intensity * 0.8 - 0.4  # Red channel
        test_image[:, 1, h, :] = intensity * 0.6 - 0.3  # Green channel  
        test_image[:, 2, h, :] = intensity * 0.4 - 0.2  # Blue channel
    
    # Add some texture/noise to make it more realistic
    noise = torch.randn_like(test_image) * 0.1
    test_image += noise
    
    mask_start_idx = int(H * (1 - face_hide_percentage))
    
    # Test different masking approaches
    masks = {
        'Hard Masking': create_original_hard_mask(B, H, W, face_hide_percentage, device, dtype),
        'Gaussian Blur': create_gaussian_blur_mask(B, H, W, face_hide_percentage, blur_kernel_size, device, dtype),
        'Current Sigmoid': create_gradient_mask(B, H, W, face_hide_percentage, blur_kernel_size, device, dtype),
        'Linear Gradient': create_improved_gradient_mask(B, H, W, face_hide_percentage, blur_kernel_size, device, dtype),
    }
    
    fig, axes = plt.subplots(3, len(masks), figsize=(20, 12))
    
    results = {}
    
    for i, (name, mask) in enumerate(masks.items()):
        # Analyze mask properties
        results[name] = analyze_mask_properties(mask, name, mask_start_idx)
        
        # Create conditional image (simulate what happens in inference)
        noise_for_cond = torch.randn_like(test_image)
        cond_image = test_image * (1. - mask) + mask * noise_for_cond
        
        # Visualize mask
        mask_np = mask[0, 0].cpu().numpy()
        axes[0, i].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{name}\nMask')
        axes[0, i].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7)
        
        # Visualize conditional image (RGB)
        cond_rgb = cond_image[0].permute(1, 2, 0).cpu().numpy()
        cond_rgb = (cond_rgb - cond_rgb.min()) / (cond_rgb.max() - cond_rgb.min())  # Normalize for display
        axes[1, i].imshow(cond_rgb)
        axes[1, i].set_title(f'{name}\nConditional Image')
        axes[1, i].axhline(y=mask_start_idx, color='red', linestyle='--', alpha=0.7)
        
        # Show cross-section of mask
        mid_col = W // 2
        cross_section = mask_np[:, mid_col]
        axes[2, i].plot(cross_section, 'b-', linewidth=2, label='Mask values')
        axes[2, i].axvline(x=mask_start_idx, color='red', linestyle='--', alpha=0.7, label='Boundary')
        axes[2, i].set_ylim(-0.1, 1.1)
        axes[2, i].set_xlabel('Height (pixels)')
        axes[2, i].set_ylabel('Mask value')
        axes[2, i].set_title(f'{name}\nCross-section')
        axes[2, i].grid(True, alpha=0.3)
        axes[2, i].legend()
        
        # Highlight transition region
        transition_start = mask_start_idx
        transition_end = mask_start_idx + blur_kernel_size
        axes[2, i].axvspan(transition_start, transition_end, alpha=0.2, color='yellow', label='Transition')
    
    plt.tight_layout()
    plt.savefig('debug_output/mask_comparison_detailed.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Detailed comparison saved: debug_output/mask_comparison_detailed.png")
    
    # Print summary comparison
    print(f"\n📋 Summary Comparison:")
    print(f"{'Method':<15} {'TopContam%':<10} {'MaxGrad':<8} {'MeanGrad':<8} {'BottomMean':<10}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<15} {stats['top_contamination']*100:<10.1f} {stats['max_gradient']:<8.3f} {stats['mean_gradient']:<8.3f} {stats['bottom_mean']:<10.3f}")
    
    return results

def main():
    """Run detailed mask comparison"""
    print("🚀 Detailed Mask Comparison Analysis")
    print("=" * 60)
    
    try:
        # Create output directory
        os.makedirs('debug_output', exist_ok=True)
        
        # Run comprehensive analysis
        results = test_mask_on_realistic_image()
        
        print(f"\n💡 Recommendations:")
        
        # Find the best approach based on metrics
        best_gradient = float('inf')
        best_method = None
        
        for name, stats in results.items():
            if name != 'Hard Masking':  # Compare gradient methods
                if stats['max_gradient'] < best_gradient and stats['top_contamination'] < 0.01:
                    best_gradient = stats['max_gradient']
                    best_method = name
        
        if best_method:
            print(f"   ✅ Best gradient method: {best_method}")
        else:
            print(f"   ❌ All gradient methods have issues - need to fix implementation")
        
        hard_gradient = results['Hard Masking']['max_gradient']
        print(f"   📊 Hard masking gradient: {hard_gradient:.3f}")
        print(f"   🎯 Target: Gradient methods should have LOWER max gradient than hard masking")
        
        # Check if current sigmoid is worse
        current_gradient = results['Current Sigmoid']['max_gradient']
        if current_gradient > hard_gradient:
            print(f"   ⚠️  PROBLEM CONFIRMED: Current sigmoid ({current_gradient:.3f}) > Hard masking ({hard_gradient:.3f})")
            print(f"   💡 Need to implement smoother gradient approach")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
