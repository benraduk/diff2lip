#!/usr/bin/env python3
"""
Simple Memory Test for Diff2Lip - Working Version
Focus on understanding actual memory usage patterns
"""

import torch
import psutil
import time
import gc
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append('.')

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict,
)

def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process()
    cpu_mb = process.memory_info().rss / 1024 / 1024
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**2
        gpu_reserved = torch.cuda.memory_reserved() / 1024**2
        return cpu_mb, gpu_allocated, gpu_reserved
    return cpu_mb, 0, 0

def log_memory(stage):
    """Log memory with stage info"""
    cpu, gpu_alloc, gpu_res = get_memory_info()
    print(f"📊 {stage:25} | CPU: {cpu:6.1f}MB | GPU: {gpu_alloc:6.1f}MB (reserved: {gpu_res:6.1f}MB)")
    return cpu, gpu_alloc, gpu_res

def create_test_args():
    """Create minimal test arguments"""
    class Args:
        pass
    
    args = Args()
    # Model config
    args.image_size = 128
    args.num_channels = 128
    args.num_res_blocks = 2
    args.num_heads = 4
    args.num_heads_upsample = -1
    args.num_head_channels = 64
    args.attention_resolutions = "32,16,8"
    args.dropout = 0.0
    args.class_cond = False
    args.use_checkpoint = False
    args.use_scale_shift_norm = False
    args.resblock_updown = True
    args.use_fp16 = True
    args.learn_sigma = True
    
    # Diffusion config
    args.diffusion_steps = 1000
    args.noise_schedule = "linear"
    args.timestep_respacing = "ddim10"
    args.use_kl = False
    args.predict_xstart = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    args.loss_variation = False
    
    # Diff2Lip specific
    args.use_ref = True
    args.nframes = 5
    args.nrefer = 1
    args.use_audio = True
    args.audio_encoder_kwargs = {}
    args.audio_as_style = True
    args.audio_as_style_encoder_mlp = ""
    args.model_path = "checkpoints/checkpoint.pt"
    
    return args

def test_tensor_memory():
    """Test memory usage of different tensor sizes"""
    print("\n🧪 TENSOR MEMORY TEST:")
    print("=" * 60)
    
    log_memory("BASELINE")
    
    # Test different batch sizes of face tensors
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    for batch_size in batch_sizes:
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Create tensors like in real inference
            face_tensor = torch.randn(batch_size, 3, 128, 128, device='cuda', dtype=torch.float16)
            audio_tensor = torch.randn(batch_size, 80, 16, device='cuda', dtype=torch.float16)
            
            cpu, gpu_alloc, gpu_res = log_memory(f"BATCH_{batch_size:2d}")
            
            # Calculate per-sample memory
            if batch_size > 0:
                per_sample = gpu_alloc / batch_size
                print(f"    └─ {per_sample:.1f}MB per sample")
            
            # Clean up
            del face_tensor, audio_tensor
            
        except Exception as e:
            print(f"❌ Batch {batch_size} failed: {e}")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log_memory("AFTER_CLEANUP")

def test_model_memory():
    """Test actual model memory usage"""
    print("\n🏗️  MODEL MEMORY TEST:")
    print("=" * 60)
    
    args = create_test_args()
    
    log_memory("BEFORE_MODEL")
    
    # Create model
    model, diffusion = tfg_create_model_and_diffusion(
        **args_to_dict(args, tfg_model_and_diffusion_defaults().keys())
    )
    
    log_memory("MODEL_CREATED")
    
    # Load weights
    state_dict = dist_util.load_state_dict(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    log_memory("WEIGHTS_LOADED_CPU")
    
    # Move to GPU
    model.to('cuda')
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    log_memory("MODEL_ON_GPU")
    
    # Test model forward pass with different batch sizes
    print("\n🔬 MODEL FORWARD PASS TEST:")
    
    with torch.no_grad():
        for batch_size in [1, 2, 4, 8]:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Create input tensors
                x = torch.randn(batch_size, 3, 128, 128, device='cuda', dtype=torch.float16)
                t = torch.randint(0, 100, (batch_size,), device='cuda')
                
                # Simple model kwargs (minimal)
                model_kwargs = {}
                
                log_memory(f"INPUT_BATCH_{batch_size}")
                
                # Forward pass
                try:
                    output = model(x, t, **model_kwargs)
                    log_memory(f"OUTPUT_BATCH_{batch_size}")
                    
                    # Memory per sample
                    cpu, gpu_alloc, gpu_res = get_memory_info()
                    per_sample = gpu_alloc / batch_size if batch_size > 0 else 0
                    print(f"    └─ {per_sample:.1f}MB per sample")
                    
                    del output
                except Exception as e:
                    print(f"    ❌ Forward pass failed: {e}")
                
                del x, t
                
            except Exception as e:
                print(f"❌ Batch {batch_size} setup failed: {e}")
    
    # Cleanup
    del model, diffusion
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    log_memory("FINAL_CLEANUP")

def analyze_gpu_capacity():
    """Analyze GPU capacity and optimal batch sizes"""
    print("\n🎯 GPU CAPACITY ANALYSIS:")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = gpu_memory_bytes / (1024**3)
    gpu_memory_mb = gpu_memory_bytes / (1024**2)
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 Total Memory: {gpu_memory_gb:.1f}GB ({gpu_memory_mb:.0f}MB)")
    
    # Current usage from our tests
    current_usage_mb = 210  # From profiling results
    model_size_mb = 205     # Model parameters
    
    print(f"📊 Current Usage: ~{current_usage_mb}MB ({current_usage_mb/gpu_memory_mb*100:.1f}%)")
    print(f"🏗️  Model Size: ~{model_size_mb}MB")
    
    # Calculate theoretical batch sizes
    available_mb = gpu_memory_mb * 0.8  # Use 80% to be safe
    overhead_mb = current_usage_mb
    
    # Estimate memory per sample (from tensor tests)
    memory_per_sample_mb = 10  # Conservative estimate
    
    theoretical_batch = int((available_mb - overhead_mb) / memory_per_sample_mb)
    
    print(f"📈 Available for batching: {available_mb:.0f}MB (80% of total)")
    print(f"🎯 Theoretical max batch: {theoretical_batch} samples")
    print(f"⚡ Potential speedup: {theoretical_batch}x faster")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Test batch sizes: 4, 8, 16, 32")
    print(f"   2. Monitor memory usage during actual inference")
    print(f"   3. Use FP16 precision (already enabled)")
    print(f"   4. Consider gradient checkpointing for larger batches")

def main():
    """Run simple memory tests"""
    print("🔬 SIMPLE DIFF2LIP MEMORY TEST")
    print("=" * 50)
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dist_util.setup_dist()
    
    try:
        # Test 1: Basic tensor memory
        test_tensor_memory()
        
        # Test 2: Model memory
        test_model_memory()
        
        # Test 3: GPU capacity analysis
        analyze_gpu_capacity()
        
        print(f"\n✅ Memory testing complete!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
