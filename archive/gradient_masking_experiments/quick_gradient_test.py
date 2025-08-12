#!/usr/bin/env python3
"""
Quick test to validate gradient masking functionality
Tests the TFG utilities directly without full inference
"""

import sys
import os
import torch
import numpy as np

# Add guided-diffusion to path
sys.path.append('./guided-diffusion')

try:
    from guided_diffusion.tfg_data_util import create_gradient_mask, tfg_add_cond_inputs
    print("✅ Successfully imported TFG utilities")
except ImportError as e:
    print(f"❌ Failed to import TFG utilities: {e}")
    sys.exit(1)

def test_gradient_mask_creation():
    """Test the gradient mask creation function"""
    print("\n🧪 Testing gradient mask creation...")
    
    # Test parameters
    B, H, W = 2, 128, 128
    face_hide_percentage = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    print(f"📱 Device: {device}")
    print(f"📐 Dimensions: B={B}, H={H}, W={W}")
    
    # Test different blur kernel sizes
    blur_sizes = [0, 5, 15, 25]
    
    for blur_size in blur_sizes:
        try:
            mask = create_gradient_mask(B, H, W, face_hide_percentage, blur_size, device, dtype)
            
            # Validate mask properties
            assert mask.shape == (B, 1, H, W), f"Wrong mask shape: {mask.shape}"
            assert mask.device.type == device.type, f"Wrong device type: {mask.device} vs {device}"
            assert mask.dtype == dtype, f"Wrong dtype: {mask.dtype}"
            assert torch.all(mask >= 0) and torch.all(mask <= 1), "Mask values outside [0,1]"
            
            # Check gradient properties
            mask_start_idx = int(H * (1 - face_hide_percentage))
            top_region = mask[:, :, :mask_start_idx, :].mean()
            bottom_region = mask[:, :, mask_start_idx:, :].mean()
            
            print(f"  Blur {blur_size:2d}: ✅ Shape {mask.shape}, Top={top_region:.3f}, Bottom={bottom_region:.3f}")
            
            # For blurred masks, check that there's a gradient
            if blur_size > 0:
                # There should be some intermediate values between 0 and 1
                unique_values = torch.unique(mask).numel()
                assert unique_values > 2, f"Expected gradient values, got {unique_values} unique values"
                
        except Exception as e:
            print(f"  Blur {blur_size:2d}: ❌ Error - {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✅ Gradient mask creation tests passed!")
    return True

def test_tfg_add_cond_inputs():
    """Test the tfg_add_cond_inputs function with gradient masking"""
    print("\n🧪 Testing tfg_add_cond_inputs with gradient masking...")
    
    # Create test data
    B, C, H, W = 2, 3, 128, 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy image batch (normalized to [-1, 1])
    img_batch = torch.randn(B, C, H, W, device=device)
    model_kwargs = {}
    
    print(f"📱 Device: {device}")
    print(f"📐 Image batch shape: {img_batch.shape}")
    
    # Test both hard and gradient masking
    test_cases = [
        {"use_gradient_mask": False, "blur_kernel_size": 0, "name": "Hard masking"},
        {"use_gradient_mask": True, "blur_kernel_size": 5, "name": "Gradient blur 5"},
        {"use_gradient_mask": True, "blur_kernel_size": 15, "name": "Gradient blur 15"},
    ]
    
    for case in test_cases:
        try:
            result_kwargs = tfg_add_cond_inputs(
                img_batch, 
                model_kwargs.copy(), 
                face_hide_percentage=0.5,
                use_gradient_mask=case["use_gradient_mask"],
                blur_kernel_size=case["blur_kernel_size"]
            )
            
            # Validate results
            assert "mask" in result_kwargs, "Missing mask in result"
            assert "cond_img" in result_kwargs, "Missing cond_img in result"
            
            mask = result_kwargs["mask"]
            cond_img = result_kwargs["cond_img"]
            
            # Check shapes
            assert mask.shape == (B, 1, H, W), f"Wrong mask shape: {mask.shape}"
            assert cond_img.shape == img_batch.shape, f"Wrong cond_img shape: {cond_img.shape}"
            
            # Check devices (allow for device index differences like cuda vs cuda:0)
            assert mask.device.type == device.type, f"Wrong mask device type: {mask.device} vs {device}"
            assert cond_img.device.type == device.type, f"Wrong cond_img device type: {cond_img.device} vs {device}"
            
            # Analyze mask properties
            unique_values = torch.unique(mask).numel()
            mask_mean = mask.mean().item()
            
            print(f"  {case['name']:15}: ✅ Unique values: {unique_values}, Mean: {mask_mean:.3f}")
            
        except Exception as e:
            print(f"  {case['name']:15}: ❌ Error - {e}")
            return False
    
    print("✅ tfg_add_cond_inputs tests passed!")
    return True

def test_memory_usage():
    """Test memory usage of gradient masking"""
    print("\n🧪 Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return True
    
    device = torch.device("cuda")
    
    # Clear cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Test with larger batch
    B, C, H, W = 8, 3, 128, 128
    img_batch = torch.randn(B, C, H, W, device=device)
    
    # Test gradient masking memory usage
    try:
        model_kwargs = {}
        result_kwargs = tfg_add_cond_inputs(
            img_batch, 
            model_kwargs, 
            face_hide_percentage=0.5,
            use_gradient_mask=True,
            blur_kernel_size=15
        )
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = (peak_memory - initial_memory) / (1024**2)  # MB
        
        print(f"  Memory used: {memory_used:.1f}MB for batch size {B}")
        
        # Cleanup
        del img_batch, result_kwargs
        torch.cuda.empty_cache()
        
        print("✅ Memory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def main():
    """Run all quick tests"""
    print("🚀 Quick Gradient Masking Validation")
    print("=" * 50)
    
    # Check environment
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"🐍 Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
    else:
        print("⚠️  Conda environment not detected")
    
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    # Run tests
    tests = [
        test_gradient_mask_creation,
        test_tfg_add_cond_inputs,
        test_memory_usage
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! Gradient masking implementation is working correctly.")
        print("💡 You can now run the full A/B test: python test_phase1_gradient_masking.py")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
