#!/usr/bin/env python3
"""
Test face detection with GPU acceleration
"""
import time
import psutil
import torch
import numpy as np
import sys

# Add current directory to path for imports
sys.path.append('.')
import face_detection

def monitor_resources():
    """Get current resource usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def test_gpu_face_detection():
    """Test face detection with GPU"""
    print("=" * 50)
    print("TESTING GPU FACE DETECTION")
    print("=" * 50)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Create test frames (simulating your 3-second video)
    print("Creating 75 test frames...")
    video_frames = []
    for i in range(75):  # 3 seconds at 25fps
        # Create a dummy frame with a face-like pattern
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add a simple face-like rectangle in the center
        frame[200:280, 270:370] = 200  # lighter region for face
        video_frames.append(frame)
    
    mem_before = monitor_resources()
    print(f"Before: Memory: {mem_before:.1f}MB")
    
    start_time = time.time()
    
    # Initialize face detector with GPU
    print("Initializing GPU face detector...")
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    init_time = time.time()
    mem_after_init = monitor_resources()
    print(f"After init: Memory: {mem_after_init:.1f}MB")
    print(f"Init time: {init_time - start_time:.2f}s")
    
    # Run face detection with larger batches (GPU can handle more)
    print("Running GPU face detection...")
    batch_size = 16  # Larger batch for GPU
    all_predictions = []
    
    try:
        for i in range(0, len(video_frames), batch_size):
            batch_start = time.time()
            batch = video_frames[i:i + batch_size]
            predictions = detector.get_detections_for_batch(np.array(batch))
            all_predictions.extend(predictions)
            batch_time = time.time() - batch_start
            print(f"Batch {i//batch_size + 1}/{(len(video_frames) + batch_size - 1)//batch_size}: {batch_time:.2f}s")
            
        detect_time = time.time()
        mem_after_detect = monitor_resources()
        print(f"After detection: Memory: {mem_after_detect:.1f}MB")
        print(f"Total detection time: {detect_time - init_time:.2f}s")
        print(f"Frames per second: {len(video_frames)/(detect_time - init_time):.1f} FPS")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU memory used: {gpu_memory:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"Face detection error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_face_detection()
    if success:
        print("\n🎉 GPU face detection working successfully!")
        print("This should dramatically speed up your inference!")
    else:
        print("\n❌ GPU face detection failed")
