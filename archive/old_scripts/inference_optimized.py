#!/usr/bin/env python3
"""
Optimized Diff2Lip Inference Script
Uses FaceFusion-inspired components for memory efficiency and speed
"""

import subprocess
import sys
import os
import psutil
import time

# Set environment variables for single-process execution
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MASTER_ADDR"] = "localhost"  
os.environ["MASTER_PORT"] = "12355"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

def monitor_resources():
    """Monitor system resources"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def run_optimized_inference():
    """Run optimized inference with resource monitoring"""
    
    print("🚀 Starting Optimized Diff2Lip Inference")
    print("=" * 50)
    
    # Monitor initial resources
    initial_memory = monitor_resources()
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    
    # Command arguments for optimized generation
    cmd = [
        "python", "generate_optimized.py",
        
        # Model configuration
        "--attention_resolutions", "32,16,8",
        "--class_cond", "False",
        "--learn_sigma", "True", 
        "--num_channels", "128",
        "--num_head_channels", "64",
        "--num_res_blocks", "2",
        "--resblock_updown", "True",
        "--use_fp16", "True",
        "--use_scale_shift_norm", "False",
        
        # Diffusion configuration
        "--predict_xstart", "False",
        "--diffusion_steps", "1000",
        "--noise_schedule", "linear",
        "--rescale_timesteps", "False",
        
        # Sampling configuration
        "--sampling_seed", "7",
        "--sampling_input_type", "gt",
        "--sampling_ref_type", "gt", 
        "--timestep_respacing", "ddim25",
        "--use_ddim", "True",
        
        # Model and data paths
        "--model_path", "checkpoints/checkpoint.pt",
        "--nframes", "5",
        "--nrefer", "1",
        "--image_size", "128",
        
        # Optimized batch sizes (smaller for memory efficiency)
        "--sampling_batch_size", "4",  # Reduced from 8
        "--streaming_chunk_size", "25",  # Process 25 frames at a time
        
        # Face processing
        "--face_hide_percentage", "0.5",
        "--use_ref", "True",
        "--use_audio", "True",
        "--audio_as_style", "True",
        
        # Input/output
        "--generate_from_filelist", "0",
        "--video_path", "test_media/person_short.mp4",
        "--audio_path", "test_media/speech_short.wav",
        "--out_path", "output_dir/result_optimized.mp4",
        "--save_orig", "False",
        
        # Video processing parameters
        "--pads", "0,0,0,0",
        "--is_voxceleb2", "False"
    ]
    
    print("Command:", " ".join(cmd[:10]) + "... (truncated)")
    print("Processing video in streaming chunks...")
    print()
    
    start_time = time.time()
    
    try:
        # Run the optimized generation
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        end_time = time.time()
        final_memory = monitor_resources()
        
        print("✅ SUCCESS!")
        print("=" * 50)
        print(f"Processing time: {end_time - start_time:.1f} seconds")
        print(f"Peak memory usage: {final_memory:.1f}MB")
        print(f"Memory increase: {final_memory - initial_memory:.1f}MB")
        print()
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        error_memory = monitor_resources()
        
        print("❌ ERROR OCCURRED")
        print("=" * 50)
        print(f"Processing time: {end_time - start_time:.1f} seconds")
        print(f"Memory usage at error: {error_memory:.1f}MB")
        print()
        print("STDOUT:")
        print(e.stdout if e.stdout else "No stdout")
        print()
        print("STDERR:")
        print(e.stderr if e.stderr else "No stderr")
        
        return False
    
    return True

def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        "generate_optimized.py",
        "face_detection_optimized.py", 
        "checkpoints/checkpoint.pt",
        "test_media/person.mp4",
        "test_media/speech.m4a"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print()
        print("Please ensure all files are present before running inference.")
        return False
    
    return True

def main():
    """Main function"""
    print("Optimized Diff2Lip Inference")
    print("Using FaceFusion-inspired optimizations")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Show system info
    print("System Information:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  Total RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    
    try:
        import torch
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    except ImportError:
        print("  PyTorch not available")
    
    print()
    
    # Run inference
    success = run_optimized_inference()
    
    if success:
        print("🎉 Inference completed successfully!")
        print("Check 'output_dir/result_optimized.mp4' for the result.")
    else:
        print("💥 Inference failed. Check the error messages above.")

if __name__ == "__main__":
    main()
