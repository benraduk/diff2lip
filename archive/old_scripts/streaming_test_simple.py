#!/usr/bin/env python3
"""
Simple streaming test for Diff2Lip optimization
Tests the core concept without complex face detection
"""

import cv2
import numpy as np
import torch
import psutil
import time
import os
import sys
sys.path.append('.')

from audio import audio
import face_detection

def monitor_resources():
    """Monitor system resources"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
    return memory_mb, gpu_memory

def test_streaming_vs_bulk_loading():
    """
    Compare streaming vs bulk loading approach
    """
    print("🧪 Testing Streaming vs Bulk Loading")
    print("=" * 50)
    
    video_path = "test_media/person_short.mp4"
    
    # Test 1: Bulk loading (original approach)
    print("📹 Test 1: Bulk Loading (Original)")
    start_time = time.time()
    mem_before, gpu_before = monitor_resources()
    
    # Simulate original approach - load all frames
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
    
    cap.release()
    
    bulk_time = time.time() - start_time
    mem_after_bulk, gpu_after_bulk = monitor_resources()
    
    print(f"  Loaded {len(all_frames)} frames")
    print(f"  Time: {bulk_time:.2f}s")
    print(f"  Memory: {mem_before:.1f}MB → {mem_after_bulk:.1f}MB (+{mem_after_bulk-mem_before:.1f}MB)")
    
    # Test 2: Streaming approach
    print("\n🌊 Test 2: Streaming Loading")
    
    # Clear previous frames
    del all_frames
    
    start_time = time.time()
    mem_before_stream, gpu_before_stream = monitor_resources()
    
    # Streaming approach - process one frame at a time
    cap = cv2.VideoCapture(video_path)
    processed_count = 0
    max_memory = mem_before_stream
    
    for frame_idx in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Simulate processing (face detection, etc.)
            # In real implementation, this would be face detection + diffusion
            processed_frame = frame_rgb.copy()  # Placeholder processing
            processed_count += 1
            
            # Monitor peak memory
            current_mem, _ = monitor_resources()
            max_memory = max(max_memory, current_mem)
            
            # Frame is automatically garbage collected when out of scope
    
    cap.release()
    
    stream_time = time.time() - start_time
    mem_after_stream, gpu_after_stream = monitor_resources()
    
    print(f"  Processed {processed_count} frames")
    print(f"  Time: {stream_time:.2f}s")
    print(f"  Peak memory: {max_memory:.1f}MB (+{max_memory-mem_before_stream:.1f}MB)")
    print(f"  Final memory: {mem_after_stream:.1f}MB")
    
    # Comparison
    print("\n📊 Comparison:")
    print(f"  Memory savings: {(mem_after_bulk-mem_before) - (max_memory-mem_before_stream):.1f}MB")
    print(f"  Time difference: {stream_time - bulk_time:.2f}s")
    
    return mem_after_bulk-mem_before, max_memory-mem_before_stream

def test_face_detection_streaming():
    """
    Test face detection in streaming mode
    """
    print("\n👤 Testing Face Detection Streaming")
    print("=" * 50)
    
    video_path = "test_media/person_short.mp4"
    
    # Initialize original face detector
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    start_time = time.time()
    mem_before, gpu_before = monitor_resources()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    successful_detections = 0
    failed_detections = 0
    max_memory = mem_before
    
    print(f"Processing {frame_count} frames...")
    
    for frame_idx in range(min(frame_count, 20)):  # Test first 20 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face detection on single frame
        try:
            detections = detector.get_detections_for_batch(np.array([frame_rgb]))
            
            if detections and detections[0] is not None:
                successful_detections += 1
                x1, y1, x2, y2 = detections[0]
                print(f"  Frame {frame_idx}: Face detected at ({x1}, {y1}, {x2}, {y2})")
            else:
                failed_detections += 1
                print(f"  Frame {frame_idx}: No face detected")
                
        except Exception as e:
            failed_detections += 1
            print(f"  Frame {frame_idx}: Detection error: {e}")
        
        # Monitor memory
        current_mem, _ = monitor_resources()
        max_memory = max(max_memory, current_mem)
    
    cap.release()
    
    detection_time = time.time() - start_time
    mem_after, gpu_after = monitor_resources()
    
    print(f"\n📊 Face Detection Results:")
    print(f"  Successful detections: {successful_detections}")
    print(f"  Failed detections: {failed_detections}")
    print(f"  Success rate: {successful_detections/(successful_detections+failed_detections)*100:.1f}%")
    print(f"  Time: {detection_time:.2f}s ({detection_time/(successful_detections+failed_detections):.3f}s per frame)")
    print(f"  Peak memory: {max_memory:.1f}MB (+{max_memory-mem_before:.1f}MB)")

def test_audio_processing():
    """
    Test audio processing efficiency
    """
    print("\n🎵 Testing Audio Processing")
    print("=" * 50)
    
    audio_path = "test_media/speech_short.wav"
    
    start_time = time.time()
    mem_before, _ = monitor_resources()
    
    # Load audio
    sample_rate = 16000
    wav = audio.load_wav(audio_path, sample_rate)
    orig_mel = audio.melspectrogram(wav).T
    
    load_time = time.time() - start_time
    mem_after, _ = monitor_resources()
    
    print(f"  Audio length: {len(wav)/sample_rate:.1f}s")
    print(f"  Mel spectrogram shape: {orig_mel.shape}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")
    
    # Test chunk extraction
    print("\n  Testing audio chunk extraction...")
    chunk_count = 0
    for frame_idx in range(0, min(75, orig_mel.shape[0]), 5):  # Every 5th frame
        start_idx = int(80. * (frame_idx / 25.0))  # 25 fps
        end_idx = start_idx + 16  # syncnet_mel_step_size
        
        if end_idx <= orig_mel.shape[0]:
            chunk = orig_mel[start_idx:end_idx, :]
            chunk_count += 1
    
    print(f"  Extracted {chunk_count} audio chunks")

def main():
    """Main test function"""
    print("🚀 Diff2Lip Streaming Optimization Test")
    print("=" * 60)
    
    # System info
    print("System Information:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  Total RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        # Test streaming vs bulk loading
        bulk_memory, stream_memory = test_streaming_vs_bulk_loading()
        
        # Test face detection
        test_face_detection_streaming()
        
        # Test audio processing
        test_audio_processing()
        
        print("\n✅ All tests completed!")
        print(f"💾 Memory savings with streaming: {bulk_memory - stream_memory:.1f}MB")
        
        if bulk_memory > stream_memory:
            print("🎉 Streaming approach is more memory efficient!")
        else:
            print("⚠️  Streaming approach needs optimization")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
