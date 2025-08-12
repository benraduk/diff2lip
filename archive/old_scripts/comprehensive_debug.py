#!/usr/bin/env python3
"""
Comprehensive debug script to monitor all inference processes
"""
import os
import time
import psutil
import cv2
import torch
import numpy as np
from os.path import join, basename
import subprocess
import shutil
import sys

# Add current directory to path for imports
sys.path.append('.')
from audio import audio
import face_detection

def monitor_resources():
    """Get current resource usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
    return memory_mb, cpu_percent, gpu_memory

def estimate_video_memory_usage(video_path):
    """Estimate memory usage for loading entire video"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Each frame: height × width × 3 channels × 1 byte per channel
    bytes_per_frame = height * width * 3
    total_bytes = frame_count * bytes_per_frame
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"Video Analysis: {video_path}")
    print(f"  Dimensions: {width}x{height}")
    print(f"  Frame count: {frame_count}")
    print(f"  Estimated RAM usage: {total_mb:.1f}MB ({total_mb/1024:.1f}GB)")
    print(f"  Per frame: {bytes_per_frame/1024/1024:.1f}MB")
    
    return total_mb

def test_video_loading_chunked(video_path, chunk_size=100):
    """Test loading video in chunks to reduce memory usage"""
    print("=" * 50)
    print("TESTING CHUNKED VIDEO LOADING")
    print("=" * 50)
    
    mem_before, cpu_before, gpu_before = monitor_resources()
    print(f"Before: RAM: {mem_before:.1f}MB, CPU: {cpu_before:.1f}%, GPU: {gpu_before:.1f}MB")
    
    # Extract frames using FFmpeg
    out_dir = "temp_debug_full/image"
    os.makedirs(out_dir, exist_ok=True)
    
    start_time = time.time()
    command = f"ffmpeg -loglevel error -y -i {video_path} -vf fps=25 -q:v 2 -qmin 1 {out_dir}/%05d.jpg"
    subprocess.call(command, shell=True)
    
    ffmpeg_time = time.time()
    mem_after_ffmpeg, cpu_after_ffmpeg, gpu_after_ffmpeg = monitor_resources()
    print(f"After FFmpeg: RAM: {mem_after_ffmpeg:.1f}MB, CPU: {cpu_after_ffmpeg:.1f}%, GPU: {gpu_after_ffmpeg:.1f}MB")
    print(f"FFmpeg time: {ffmpeg_time - start_time:.2f}s")
    
    # Count frames
    frame_files = sorted(os.listdir(out_dir))
    total_frames = len(frame_files)
    print(f"Total frames extracted: {total_frames}")
    
    # Load frames in chunks
    print(f"Loading frames in chunks of {chunk_size}...")
    all_frames_loaded = 0
    max_memory = mem_after_ffmpeg
    
    for chunk_start in range(0, min(total_frames, 500), chunk_size):  # Limit to first 500 frames for testing
        chunk_end = min(chunk_start + chunk_size, min(total_frames, 500))
        print(f"Loading chunk {chunk_start}-{chunk_end}...")
        
        chunk_frames = []
        for i in range(chunk_start, chunk_end):
            img_path = join(out_dir, frame_files[i])
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            chunk_frames.append(img)
        
        all_frames_loaded += len(chunk_frames)
        mem_current, cpu_current, gpu_current = monitor_resources()
        max_memory = max(max_memory, mem_current)
        print(f"  Loaded {len(chunk_frames)} frames. RAM: {mem_current:.1f}MB")
        
        # Simulate processing (then release chunk)
        del chunk_frames
    
    load_time = time.time()
    print(f"Chunked loading completed in {load_time - ffmpeg_time:.2f}s")
    print(f"Peak memory usage: {max_memory:.1f}MB")
    
    # Cleanup
    shutil.rmtree("temp_debug_full")
    
    return all_frames_loaded

def test_model_loading():
    """Test model loading process"""
    print("=" * 50)
    print("TESTING MODEL LOADING")
    print("=" * 50)
    
    mem_before, cpu_before, gpu_before = monitor_resources()
    print(f"Before: RAM: {mem_before:.1f}MB, CPU: {cpu_before:.1f}%, GPU: {gpu_before:.1f}MB")
    
    # Simulate model loading (without actually loading the full model)
    print("Checking model file size...")
    model_path = "checkpoints/checkpoint.pt"
    if os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model file size: {model_size_mb:.1f}MB")
        
        # Test loading to CPU first (like the original code)
        print("Loading model to CPU...")
        start_time = time.time()
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            cpu_load_time = time.time()
            mem_after_cpu, cpu_after_cpu, gpu_after_cpu = monitor_resources()
            print(f"CPU load time: {cpu_load_time - start_time:.2f}s")
            print(f"After CPU load: RAM: {mem_after_cpu:.1f}MB, GPU: {gpu_after_cpu:.1f}MB")
            
            # Test moving to GPU
            print("Moving model to GPU...")
            if torch.cuda.is_available():
                # Simulate moving to GPU (don't actually do it to save time)
                print("GPU available - would move model to CUDA")
                gpu_move_time = time.time()
                mem_after_gpu, cpu_after_gpu, gpu_after_gpu = monitor_resources()
                print(f"After GPU move: RAM: {mem_after_gpu:.1f}MB, GPU: {gpu_after_gpu:.1f}MB")
            
            del state_dict
            
        except Exception as e:
            print(f"Model loading error: {e}")
    else:
        print("Model file not found!")

def test_face_detection_batching():
    """Test face detection with different batch sizes"""
    print("=" * 50)
    print("TESTING FACE DETECTION BATCHING")
    print("=" * 50)
    
    # Create test frames
    print("Creating test frames (simulating full video size)...")
    test_frames = []
    for i in range(200):  # Test with 200 frames (8 seconds)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        test_frames.append(frame)
    
    mem_before, cpu_before, gpu_before = monitor_resources()
    print(f"Before: RAM: {mem_before:.1f}MB, CPU: {cpu_before:.1f}%, GPU: {gpu_before:.1f}MB")
    
    # Initialize face detector
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        start_time = time.time()
        max_memory = mem_before
        
        try:
            for i in range(0, min(len(test_frames), 50), batch_size):  # Test first 50 frames
                batch = test_frames[i:i + batch_size]
                predictions = detector.get_detections_for_batch(np.array(batch))
                
                mem_current, cpu_current, gpu_current = monitor_resources()
                max_memory = max(max_memory, mem_current)
            
            end_time = time.time()
            print(f"  Time: {end_time - start_time:.2f}s, Peak RAM: {max_memory:.1f}MB")
            
        except Exception as e:
            print(f"  Error with batch size {batch_size}: {e}")
    
    del test_frames

if __name__ == "__main__":
    print("Starting comprehensive debug...")
    print(f"Python process PID: {os.getpid()}")
    print(f"Available CPU cores: {psutil.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        # Analyze the video file
        estimate_video_memory_usage("test_media/person.mp4")
        
        # Test each component
        frames_loaded = test_video_loading_chunked("test_media/person.mp4")
        test_model_loading()
        test_face_detection_batching()
        
        print("=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)
        print("1. Consider processing video in chunks instead of loading all frames")
        print("2. Use smaller batch sizes for face detection with high-res video")
        print("3. Load model directly to GPU if possible")
        print("4. Monitor GPU memory usage during inference")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
