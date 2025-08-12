#!/usr/bin/env python3
"""
Batch Processing Diff2Lip Inference - Batch Size 4
Modified from inference_streaming.py to process 4 frames simultaneously
"""

import cv2
import os
import numpy as np
import torch
import torchvision
import psutil
import time
import sys
import gc
import subprocess
from tqdm import tqdm

# Add current directory to path
sys.path.append('.')

# Import Diff2Lip components
from audio import audio
import face_detection
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict,
)

class BatchDiff2LipProcessor:
    """
    Batch processor that processes 4 frames simultaneously for 4x speedup
    """
    
    def __init__(self, args, batch_size=4):
        self.args = args
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"🚀 BATCH SIZE: {self.batch_size} (Expected 4x speedup!)")
        
        # SPEED OPTIMIZATIONS: Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul
            print("✅ CUDA optimizations enabled")
        
        # Initialize components
        self.model = None
        self.diffusion = None
        self.detector = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Diff2Lip models"""
        print("Loading Diff2Lip model...")
        
        # Load diffusion model
        self.model, self.diffusion = tfg_create_model_and_diffusion(
            **args_to_dict(self.args, tfg_model_and_diffusion_defaults().keys())
        )
        
        # Load model weights
        state_dict = dist_util.load_state_dict(self.args.model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        
        if self.args.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()
        
        # SPEED OPTIMIZATION: Compile model (if available)
        print("Checking torch.compile compatibility...")
        try:
            import triton
            self.model = torch.compile(self.model, mode='reduce-overhead')
            print("✅ Model compiled successfully!")
        except ImportError:
            print("⚠️ Triton not available - skipping torch.compile (Windows compatibility issue)")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e} - continuing without compilation")
        
        # Initialize face detector
        print("Loading face detector...")
        self.detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, 
            flip_input=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Models loaded and optimized successfully!")
    
    def _monitor_resources(self):
        """Monitor system resources"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
        return memory_mb, gpu_memory
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio"""
        print(f"Loading audio from: {audio_path}")
        wav = audio.load_wav(audio_path, self.args.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
        print(f"Audio loaded: {orig_mel.shape} mel spectrogram")
        return orig_mel, wav
    
    def _get_audio_chunk(self, orig_mel, frame_idx):
        """Get audio chunk for specific frame"""
        start_idx = int(80. * (frame_idx / float(self.args.video_fps)))
        end_idx = start_idx + self.args.syncnet_mel_step_size
        
        if end_idx > orig_mel.shape[0]:
            # Pad if necessary
            chunk = np.zeros((self.args.syncnet_mel_step_size, orig_mel.shape[1]))
            available = orig_mel.shape[0] - start_idx
            if available > 0:
                chunk[:available] = orig_mel[start_idx:]
        else:
            chunk = orig_mel[start_idx:end_idx]
        
        return chunk
    
    def _process_batch(self, frames_batch, audio_chunks_batch):
        """Process a batch of frames simultaneously - THE KEY OPTIMIZATION!"""
        batch_size = len(frames_batch)
        
        # Batch face detection
        detections_batch = self.detector.get_detections_for_batch(np.array(frames_batch))
        
        # Prepare batch tensors
        valid_indices = []
        face_tensors = []
        audio_tensors = []
        face_crops_info = []  # Store crop info for reconstruction
        
        for i, (frame, detection, audio_chunk) in enumerate(zip(frames_batch, detections_batch, audio_chunks_batch)):
            if detection is None:
                continue
                
            x1, y1, x2, y2 = detection
            
            # Apply padding
            pady1, pady2, padx1, padx2 = self.args.pads
            y1 = max(0, y1 - pady1)
            y2 = min(frame.shape[0], y2 + pady2)
            x1 = max(0, x1 - padx1)
            x2 = min(frame.shape[1], x2 + padx2)
            
            # Crop and resize face
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            
            face_resized = cv2.resize(face_crop, (self.args.image_size, self.args.image_size))
            
            # Prepare tensors
            face_tensor = torch.FloatTensor(face_resized.astype(np.float32) / 255.0)
            face_tensor = face_tensor.permute(2, 0, 1)  # [3, H, W]
            
            audio_tensor = torch.FloatTensor(audio_chunk.T)  # [80, 16]
            
            face_tensors.append(face_tensor)
            audio_tensors.append(audio_tensor)
            face_crops_info.append((i, x1, y1, x2, y2))  # (original_index, crop_coords)
            valid_indices.append(i)
        
        if not face_tensors:
            # No valid faces in batch
            return frames_batch, [False] * batch_size
        
        # Stack into batch tensors
        # Face: [batch, 1, 3, H, W] format for Diff2Lip
        face_batch = torch.stack(face_tensors).unsqueeze(1).to(self.device)  # [N, 1, 3, H, W]
        
        # Audio: [batch, 1, 1, 80, 16] format for Diff2Lip  
        audio_batch = torch.stack(audio_tensors).unsqueeze(1).unsqueeze(1).to(self.device)  # [N, 1, 1, 80, 16]
        
        try:
            # Prepare batch in correct Diff2Lip format
            batch = {
                "image": face_batch,  # [N, 1, 3, H, W]
                "ref_img": face_batch.clone(),  # [N, 1, 3, H, W] 
                "indiv_mels": audio_batch,  # [N, 1, 1, 80, 16]
            }
            
            # Generate with diffusion model using batch processing
            from guided_diffusion.tfg_data_util import tfg_process_batch
            
            with torch.no_grad():
                # Process batch like the original code
                img_batch, model_kwargs = tfg_process_batch(
                    batch, 
                    self.args.face_hide_percentage if hasattr(self.args, 'face_hide_percentage') else 0.5,
                    use_ref=self.args.use_ref,
                    use_audio=self.args.use_audio
                )
                
                # Move to device
                img_batch = img_batch.to(self.device)
                model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}
                
                # Sample using diffusion model - THIS IS THE BATCH MAGIC!
                sample_fn = (
                    self.diffusion.p_sample_loop if not hasattr(self.args, 'use_ddim') or not self.args.use_ddim 
                    else self.diffusion.ddim_sample_loop
                )
                
                sample = sample_fn(
                    self.model,
                    img_batch.shape,  # (N, 3, 128, 128) - BATCH PROCESSING!
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    progress=False,
                )
                
                # Apply mask like original code
                mask = model_kwargs['mask']
                recon_batch = sample * mask + (1. - mask) * img_batch
                
                # Reconstruct individual frames
                result_frames = frames_batch.copy()
                success_flags = [False] * batch_size
                
                for batch_idx, (orig_idx, x1, y1, x2, y2) in enumerate(face_crops_info):
                    # Convert back to image
                    generated_face = recon_batch[batch_idx].permute(1, 2, 0).cpu().numpy()
                    generated_face = (generated_face.clip(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
                    generated_face = (generated_face * 255).astype(np.uint8)
                    
                    # Resize to original face size and place back
                    generated_face_resized = cv2.resize(generated_face, (x2-x1, y2-y1))
                    
                    result_frames[orig_idx] = frames_batch[orig_idx].copy()
                    result_frames[orig_idx][y1:y2, x1:x2] = generated_face_resized
                    success_flags[orig_idx] = True
                
                # Aggressive cleanup to prevent memory leaks
                del face_batch, audio_batch, batch, img_batch, sample, mask, recon_batch
                for k in list(model_kwargs.keys()):
                    del model_kwargs[k]
                del model_kwargs, face_tensors, audio_tensors
                torch.cuda.empty_cache()
                gc.collect()
                
                return result_frames, success_flags
                
        except Exception as e:
            print(f"Batch generation error: {e}")
            import traceback
            traceback.print_exc()
            return frames_batch, [False] * batch_size
    
    def process_video(self, video_path, audio_path, output_path):
        """Process video with batch approach"""
        print(f"Processing video: {video_path}")
        print(f"Audio: {audio_path}")
        print(f"Output: {output_path}")
        print(f"🚀 Batch size: {self.batch_size}")
        
        # Load audio once
        orig_mel, audio_wav = self._load_audio(audio_path)
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Process frames in batches
        processed_frames = []
        successful_generations = 0
        
        mem_before, gpu_before = self._monitor_resources()
        start_time = time.time()
        
        print("Processing frames in batches...")
        
        # Calculate number of batches
        num_batches = (frame_count + self.batch_size - 1) // self.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            # Collect frames for this batch
            frames_batch = []
            audio_chunks_batch = []
            
            for i in range(self.batch_size):
                frame_idx = batch_idx * self.batch_size + i
                if frame_idx >= frame_count:
                    break
                
                # Get frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_batch.append(frame_rgb)
                
                # Get corresponding audio chunk
                audio_chunk = self._get_audio_chunk(orig_mel, frame_idx)
                audio_chunks_batch.append(audio_chunk)
            
            if not frames_batch:
                break
            
            # Process this batch (THE KEY OPTIMIZATION!)
            result_frames, success_flags = self._process_batch(frames_batch, audio_chunks_batch)
            
            # Add results
            processed_frames.extend(result_frames)
            successful_generations += sum(success_flags)
            
            # Memory management
            if batch_idx % 5 == 0:  # Every 5 batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # Monitor GPU memory
                    if batch_idx % 10 == 0:  # Every 10 batches
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        print(f"  Batch {batch_idx}/{num_batches}: GPU memory: {gpu_memory:.1f}MB")
        
        cap.release()
        
        processing_time = time.time() - start_time
        mem_after, gpu_after = self._monitor_resources()
        
        print(f"Processing complete!")
        print(f"  Time: {processing_time:.1f}s ({processing_time/frame_count:.3f}s per frame)")
        print(f"  Speedup vs single-frame: {0.650/processing_time*frame_count:.1f}x faster!")
        print(f"  Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")
        print(f"  Successful generations: {successful_generations}/{frame_count} ({successful_generations/frame_count*100:.1f}%)")
        
        # Save video with FFmpeg audio integration
        if processed_frames:
            print(f"Saving video to: {output_path}")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Step 1: Save video without audio to temp file
            temp_video_path = output_path.replace('.mp4', '_batch4_temp_no_audio.mp4')
            video_tensor = torch.from_numpy(np.array(processed_frames))
            
            print("Saving video frames...")
            torchvision.io.write_video(
                temp_video_path,
                video_array=video_tensor,
                fps=int(fps),
                video_codec='libx264'
            )
            
            # Step 2: Use FFmpeg to combine video + audio
            print("Adding audio with FFmpeg...")
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # -y to overwrite output
                '-i', temp_video_path,  # Input video
                '-i', audio_path,       # Input audio
                '-c:v', 'copy',         # Copy video stream (no re-encoding)
                '-c:a', 'aac',          # Encode audio as AAC
                '-shortest',            # Match shortest stream duration
                output_path             # Final output
            ]
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    # Clean up temp file
                    os.remove(temp_video_path)
                    print(f"✅ Video with audio saved successfully!")
                    print(f"📁 Output: {output_path}")
                    return True
                else:
                    print(f"❌ FFmpeg error (code {result.returncode}):")
                    print(f"   {result.stderr}")
                    print(f"📁 Video without audio available at: {temp_video_path}")
                    return False
                    
            except FileNotFoundError:
                print("❌ FFmpeg not found. Please install FFmpeg or add it to PATH.")
                print(f"📁 Video without audio saved to: {temp_video_path}")
                return False
                
        else:
            print("❌ No frames to save")
            return False

def create_args():
    """Create arguments for Diff2Lip"""
    class Args:
        pass
    
    args = Args()
    
    # Model configuration
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
    
    # Diffusion configuration
    args.diffusion_steps = 1000
    args.noise_schedule = "linear"
    args.timestep_respacing = "ddim10"  # SPEED OPTIMIZATION: 2.5x faster
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
    
    # Model path
    args.model_path = "checkpoints/checkpoint.pt"
    
    # Data configuration
    args.video_fps = 25
    args.sample_rate = 16000
    args.syncnet_mel_step_size = 16
    args.skip_timesteps = 0
    
    # Processing configuration
    args.pads = [0, 0, 0, 0]
    args.face_hide_percentage = 0.5
    args.use_ddim = True
    
    return args

def main():
    """Main function"""
    print("🚀 BATCH-4 Diff2Lip Inference")
    print("=" * 50)
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dist_util.setup_dist()
    
    # Create arguments
    args = create_args()
    
    # Initialize processor with batch size 4
    try:
        processor = BatchDiff2LipProcessor(args, batch_size=4)
        
        # Process video
        success = processor.process_video(
            video_path="test_media/person.mp4",
            audio_path="test_media/speech.m4a", 
            output_path="output_dir/result_batch4.mp4"
        )
        
        if success:
            print("🎉 BATCH-4 Inference completed successfully!")
            print("📊 Expected speedup: ~4x faster than single-frame processing")
        else:
            print("❌ Inference failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
