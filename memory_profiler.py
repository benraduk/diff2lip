#!/usr/bin/env python3
"""
Memory Profiling Tool for Diff2Lip
Analyzes GPU/CPU memory usage during inference to identify optimization opportunities
"""

import torch
import psutil
import time
import gc
import sys
import os
import numpy as np
import cv2
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append('.')

# Import Diff2Lip components
from audio import audio
import face_detection
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict,
)

class MemoryProfiler:
    """Detailed memory profiling for Diff2Lip pipeline"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_log = []
        self.process = psutil.Process()
        
    def log_memory(self, stage: str, details: str = ""):
        """Log current memory usage"""
        # CPU Memory
        cpu_memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # GPU Memory
        gpu_allocated = 0
        gpu_cached = 0
        gpu_reserved = 0
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_cached = torch.cuda.memory_cached() / 1024**2 
            gpu_reserved = torch.cuda.memory_reserved() / 1024**2
        
        entry = {
            'timestamp': time.time(),
            'stage': stage,
            'details': details,
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_allocated_mb': gpu_allocated,
            'gpu_cached_mb': gpu_cached,
            'gpu_reserved_mb': gpu_reserved
        }
        
        self.memory_log.append(entry)
        print(f"📊 {stage}: CPU={cpu_memory_mb:.1f}MB | GPU={gpu_allocated:.1f}MB (cached={gpu_cached:.1f}MB, reserved={gpu_reserved:.1f}MB) | {details}")
        
    def get_model_memory_breakdown(self, model):
        """Analyze model parameter memory usage"""
        total_params = 0
        total_memory_mb = 0
        
        print("\n🔍 MODEL MEMORY BREAKDOWN:")
        for name, param in model.named_parameters():
            param_count = param.numel()
            param_memory_mb = param.element_size() * param_count / 1024**2
            total_params += param_count
            total_memory_mb += param_memory_mb
            
            if param_memory_mb > 1.0:  # Only show layers > 1MB
                print(f"  {name}: {param_count:,} params, {param_memory_mb:.1f}MB")
        
        print(f"📋 Total Model: {total_params:,} parameters, {total_memory_mb:.1f}MB")
        return total_params, total_memory_mb
    
    def profile_batch_sizes(self, args, test_sizes: List[int] = [1, 2, 4, 8, 16]):
        """Test different batch sizes to find optimal memory usage"""
        print("\n🧪 BATCH SIZE MEMORY TESTING:")
        print("=" * 60)
        
        # Initialize model once
        self.log_memory("INIT", "Starting batch size testing")
        
        # Load model
        model, diffusion = tfg_create_model_and_diffusion(
            **args_to_dict(args, tfg_model_and_diffusion_defaults().keys())
        )
        
        self.log_memory("MODEL_CREATED", "Model architecture created")
        
        # Load weights
        state_dict = dist_util.load_state_dict(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        
        self.log_memory("MODEL_LOADED", "Model weights loaded and moved to GPU")
        
        # Get model memory breakdown
        total_params, model_memory_mb = self.get_model_memory_breakdown(model)
        
        # Test each batch size
        batch_results = []
        
        for batch_size in test_sizes:
            print(f"\n🔬 Testing Batch Size: {batch_size}")
            print("-" * 40)
            
            try:
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                self.log_memory("BATCH_START", f"Batch size {batch_size} - cleared cache")
                
                # Create synthetic batch data
                face_batch = torch.randn(batch_size, 1, 3, args.image_size, args.image_size, 
                                       device=self.device, dtype=torch.float16 if args.use_fp16 else torch.float32)
                audio_batch = torch.randn(batch_size, 1, 1, 80, 16, 
                                        device=self.device, dtype=torch.float16 if args.use_fp16 else torch.float32)
                
                self.log_memory("BATCH_DATA", f"Created synthetic batch data")
                
                # Prepare model kwargs (simplified)
                model_kwargs = {
                    'cond_img': face_batch.squeeze(1),  # Remove frame dimension
                    'mask': torch.ones(batch_size, 1, args.image_size, args.image_size, device=self.device),
                    'indiv_mels': audio_batch.squeeze(1).squeeze(1)  # Remove frame and channel dims
                }
                
                self.log_memory("BATCH_PREPARED", f"Prepared model inputs")
                
                # Test forward pass (without full diffusion)
                with torch.no_grad():
                    # Just test the model forward pass, not full diffusion sampling
                    img_input = face_batch.squeeze(1)  # [batch, 3, H, W]
                    timesteps = torch.randint(0, 100, (batch_size,), device=self.device)
                    
                    # Forward pass
                    model_output = model(img_input, timesteps, **model_kwargs)
                    
                self.log_memory("BATCH_FORWARD", f"Completed forward pass")
                
                # Calculate memory per sample
                gpu_per_sample = (torch.cuda.memory_allocated() / 1024**2) / batch_size if torch.cuda.is_available() else 0
                
                batch_results.append({
                    'batch_size': batch_size,
                    'gpu_total_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                    'gpu_per_sample_mb': gpu_per_sample,
                    'success': True
                })
                
                print(f"✅ Batch {batch_size}: {gpu_per_sample:.1f}MB per sample")
                
                # Clean up
                del face_batch, audio_batch, model_kwargs, model_output, img_input, timesteps
                
            except Exception as e:
                print(f"❌ Batch {batch_size} FAILED: {e}")
                batch_results.append({
                    'batch_size': batch_size,
                    'gpu_total_mb': 0,
                    'gpu_per_sample_mb': 0,
                    'success': False,
                    'error': str(e)
                })
                
            # Clean up between tests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Clean up model
        del model, diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return batch_results, model_memory_mb
    
    def analyze_inference_pipeline(self, video_path: str, audio_path: str, max_frames: int = 10):
        """Analyze memory usage during actual inference pipeline"""
        print("\n🔍 INFERENCE PIPELINE MEMORY ANALYSIS:")
        print("=" * 60)
        
        # Load test media
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Load audio
        self.log_memory("AUDIO_START", "Loading audio")
        wav = audio.load_wav(audio_path, 16000)
        orig_mel = audio.melspectrogram(wav).T
        self.log_memory("AUDIO_LOADED", f"Audio: {orig_mel.shape}")
        
        # Load face detector
        self.log_memory("DETECTOR_START", "Loading face detector")
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, 
            flip_input=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.log_memory("DETECTOR_LOADED", "Face detector loaded")
        
        # Process a few frames to see memory patterns
        frame_memories = []
        
        for frame_idx in range(min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.log_memory("FRAME_LOADED", f"Frame {frame_idx}")
            
            # Face detection
            detections = detector.get_detections_for_batch(np.array([frame_rgb]))
            self.log_memory("FACE_DETECTED", f"Frame {frame_idx} - face detection")
            
            # Get audio chunk
            start_idx = int(80. * (frame_idx / 25.0))
            end_idx = start_idx + 16
            audio_chunk = orig_mel[start_idx:end_idx] if end_idx <= orig_mel.shape[0] else orig_mel[-16:]
            
            # Convert to tensors (simulate processing)
            if detections and detections[0] is not None:
                x1, y1, x2, y2 = detections[0]
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, (128, 128))
                    face_tensor = torch.FloatTensor(face_resized.astype(np.float32) / 255.0).to(self.device)
                    audio_tensor = torch.FloatTensor(audio_chunk).to(self.device)
                    
                    self.log_memory("TENSORS_CREATED", f"Frame {frame_idx} - tensors on GPU")
                    
                    # Clean up
                    del face_tensor, audio_tensor
            
            # Memory after frame processing
            cpu_mem = self.process.memory_info().rss / 1024 / 1024
            gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            frame_memories.append({'frame': frame_idx, 'cpu': cpu_mem, 'gpu': gpu_mem})
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        cap.release()
        return frame_memories
    
    def generate_report(self, batch_results: List[Dict], model_memory_mb: float):
        """Generate comprehensive memory analysis report"""
        print("\n" + "="*80)
        print("📊 MEMORY PROFILING REPORT")
        print("="*80)
        
        print(f"\n🏗️  MODEL ANALYSIS:")
        print(f"   Base Model Memory: {model_memory_mb:.1f}MB")
        print(f"   GPU Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   Total GPU Memory: {gpu_memory_gb:.1f}GB ({gpu_memory_gb*1024:.0f}MB)")
        
        print(f"\n📈 BATCH SIZE ANALYSIS:")
        successful_batches = [r for r in batch_results if r['success']]
        
        if successful_batches:
            print("   Batch Size | Total GPU | Per Sample | Status")
            print("   -----------|-----------|------------|--------")
            for result in batch_results:
                status = "✅ OK" if result['success'] else "❌ FAIL"
                if result['success']:
                    print(f"   {result['batch_size']:>9} | {result['gpu_total_mb']:>8.1f}MB | {result['gpu_per_sample_mb']:>9.1f}MB | {status}")
                else:
                    print(f"   {result['batch_size']:>9} | {'N/A':>8} | {'N/A':>9} | {status}")
            
            # Find optimal batch size
            max_successful = max(r['batch_size'] for r in successful_batches)
            optimal_memory = successful_batches[-1]['gpu_total_mb']
            
            print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
            print(f"   Current Usage: ~305MB (single frame)")
            print(f"   Maximum Batch: {max_successful} frames")
            print(f"   Optimal Memory: {optimal_memory:.1f}MB")
            
            if torch.cuda.is_available():
                gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                utilization = (optimal_memory / gpu_total_mb) * 100
                print(f"   GPU Utilization: {utilization:.1f}%")
                
                # Calculate potential speedup
                current_batch = 1
                optimal_batch = max_successful
                theoretical_speedup = optimal_batch / current_batch
                print(f"   Theoretical Speedup: {theoretical_speedup:.1f}x faster")
                
                # Memory headroom
                headroom_mb = gpu_total_mb * 0.8 - optimal_memory  # Leave 20% buffer
                if headroom_mb > 0:
                    potential_batch = int((headroom_mb / (optimal_memory / max_successful)) + max_successful)
                    print(f"   Potential Batch Size: {potential_batch} (with 80% GPU usage)")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Implement batch processing with size {max_successful if successful_batches else 2}")
        print(f"   2. Test mixed precision (FP16) for 2x memory reduction")
        print(f"   3. Enable gradient checkpointing if available")
        print(f"   4. Consider model quantization for further optimization")

def create_test_args():
    """Create test arguments"""
    class Args:
        pass
    
    args = Args()
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
    args.diffusion_steps = 1000
    args.noise_schedule = "linear"
    args.timestep_respacing = "ddim10"
    args.use_kl = False
    args.predict_xstart = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    args.loss_variation = False
    args.use_ref = True
    args.nframes = 5
    args.nrefer = 1
    args.use_audio = True
    args.audio_encoder_kwargs = {}
    args.audio_as_style = True
    args.audio_as_style_encoder_mlp = ""
    args.model_path = "checkpoints/checkpoint.pt"
    args.video_fps = 25
    args.sample_rate = 16000
    args.syncnet_mel_step_size = 16
    args.skip_timesteps = 0
    args.pads = [0, 0, 0, 0]
    args.face_hide_percentage = 0.5
    args.use_ddim = True
    
    return args

def main():
    """Run memory profiling analysis"""
    print("🔬 DIFF2LIP MEMORY PROFILER")
    print("=" * 50)
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dist_util.setup_dist()
    
    profiler = MemoryProfiler()
    args = create_test_args()
    
    try:
        # 1. Test different batch sizes
        print("Phase 1: Batch Size Memory Testing")
        batch_results, model_memory = profiler.profile_batch_sizes(args, test_sizes=[1, 2, 4, 8, 16, 32])
        
        # 2. Analyze actual inference pipeline
        print("\nPhase 2: Inference Pipeline Analysis")
        frame_memories = profiler.analyze_inference_pipeline(
            "test_media/person.mp4", 
            "test_media/speech.m4a", 
            max_frames=5
        )
        
        # 3. Generate comprehensive report
        profiler.generate_report(batch_results, model_memory)
        
        # 4. Save detailed log
        import json
        with open('memory_profile_log.json', 'w') as f:
            json.dump({
                'memory_log': profiler.memory_log,
                'batch_results': batch_results,
                'frame_memories': frame_memories,
                'model_memory_mb': model_memory
            }, f, indent=2)
        
        print(f"\n💾 Detailed log saved to: memory_profile_log.json")
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
