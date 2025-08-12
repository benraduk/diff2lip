#!/usr/bin/env python3
"""
Debug script to isolate the preprocessing bottleneck
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
    return memory_mb, cpu_percent

def test_video_loading():
    """Test video frame extraction"""
    print("=" * 50)
    print("TESTING VIDEO LOADING")
    print("=" * 50)
    
    video_path = "test_media/person_short.mp4"
    out_dir = "temp_debug/image"
    os.makedirs(out_dir, exist_ok=True)
    
    # Monitor resources before
    mem_before, cpu_before = monitor_resources()
    print(f"Before: Memory: {mem_before:.1f}MB, CPU: {cpu_before:.1f}%")
    
    start_time = time.time()
    
    # Extract frames using FFmpeg (like the original code)
    print("Extracting frames with FFmpeg...")
    command = f"ffmpeg -loglevel error -y -i {video_path} -vf fps=25 -q:v 2 -qmin 1 {out_dir}/%05d.jpg"
    subprocess.call(command, shell=True)
    
    ffmpeg_time = time.time()
    mem_after_ffmpeg, cpu_after_ffmpeg = monitor_resources()
    print(f"After FFmpeg: Memory: {mem_after_ffmpeg:.1f}MB, CPU: {cpu_after_ffmpeg:.1f}%")
    print(f"FFmpeg time: {ffmpeg_time - start_time:.2f}s")
    
    # Load frames into memory (like the original code)
    print("Loading frames into memory...")
    video_frames = []
    for i, img_name in enumerate(sorted(os.listdir(out_dir))):
        img_path = join(out_dir, img_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        video_frames.append(img)
    
    load_time = time.time()
    mem_after_load, cpu_after_load = monitor_resources()
    print(f"After loading: Memory: {mem_after_load:.1f}MB, CPU: {cpu_after_load:.1f}%")
    print(f"Load time: {load_time - ffmpeg_time:.2f}s")
    print(f"Loaded {len(video_frames)} frames")
    print(f"Frame shape: {video_frames[0].shape if video_frames else 'None'}")
    
    # Cleanup
    shutil.rmtree("temp_debug")
    
    return video_frames

def test_audio_loading():
    """Test audio processing"""
    print("=" * 50)
    print("TESTING AUDIO LOADING") 
    print("=" * 50)
    
    audio_path = "test_media/speech_short.wav"
    out_dir = "temp_debug"
    os.makedirs(out_dir, exist_ok=True)
    
    # Monitor resources before
    mem_before, cpu_before = monitor_resources()
    print(f"Before: Memory: {mem_before:.1f}MB, CPU: {cpu_before:.1f}%")
    
    start_time = time.time()
    
    # Load audio (like the original code)
    print("Loading audio...")
    sample_rate = 16000
    wav = audio.load_wav(audio_path, sample_rate)
    
    load_time = time.time()
    mem_after_load, cpu_after_load = monitor_resources()
    print(f"After load: Memory: {mem_after_load:.1f}MB, CPU: {cpu_after_load:.1f}%")
    print(f"Load time: {load_time - start_time:.2f}s")
    
    # Generate mel spectrograms
    print("Generating mel spectrograms...")
    orig_mel = audio.melspectrogram(wav).T
    
    mel_time = time.time()
    mem_after_mel, cpu_after_mel = monitor_resources()
    print(f"After mel: Memory: {mem_after_mel:.1f}MB, CPU: {cpu_after_mel:.1f}%")
    print(f"Mel time: {mel_time - load_time:.2f}s")
    print(f"Mel shape: {orig_mel.shape}")
    
    # Cleanup
    shutil.rmtree("temp_debug", ignore_errors=True)
    
    return wav, orig_mel

def test_face_detection():
    """Test face detection"""
    print("=" * 50)
    print("TESTING FACE DETECTION")
    print("=" * 50)
    
    # Create dummy frames for testing
    print("Creating test frames...")
    video_frames = []
    for i in range(75):  # 3 seconds at 25fps
        # Create a dummy frame (replace with actual frames if needed)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        video_frames.append(frame)
    
    mem_before, cpu_before = monitor_resources()
    print(f"Before: Memory: {mem_before:.1f}MB, CPU: {cpu_before:.1f}%")
    
    start_time = time.time()
    
    # Initialize face detector
    print("Initializing face detector...")
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    init_time = time.time()
    mem_after_init, cpu_after_init = monitor_resources()
    print(f"After init: Memory: {mem_after_init:.1f}MB, CPU: {cpu_after_init:.1f}%")
    print(f"Init time: {init_time - start_time:.2f}s")
    
    # Run face detection with small batches
    print("Running face detection...")
    batch_size = 8
    try:
        for i in range(0, len(video_frames), batch_size):
            batch = video_frames[i:i + batch_size]
            predictions = detector.get_detections_for_batch(np.array(batch))
            print(f"Processed batch {i//batch_size + 1}/{(len(video_frames) + batch_size - 1)//batch_size}")
    except Exception as e:
        print(f"Face detection error: {e}")
        return None
    
    detect_time = time.time()
    mem_after_detect, cpu_after_detect = monitor_resources()
    print(f"After detection: Memory: {mem_after_detect:.1f}MB, CPU: {cpu_after_detect:.1f}%")
    print(f"Detection time: {detect_time - init_time:.2f}s")
    
    return detector

if __name__ == "__main__":
    print("Starting preprocessing debug...")
    print(f"Python process PID: {os.getpid()}")
    print(f"Available CPU cores: {psutil.cpu_count()}")
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Test each component
        video_frames = test_video_loading()
        wav, orig_mel = test_audio_loading()
        detector = test_face_detection()
        
        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print("All preprocessing steps completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
