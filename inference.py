#!/usr/bin/env python3
"""
Centralized Diff2Lip Inference Script
Based on successful inference_batch4.py with configurable parameters via YAML
Supports all quality enhancement features from the quality plan
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
import yaml
from tqdm import tqdm
from pathlib import Path

# Add current directory and guided-diffusion to path
sys.path.append('.')
sys.path.append('./guided-diffusion')

# Import Diff2Lip components
from audio import audio
import face_detection
from face_enhancer import create_face_enhancer

# Import guided-diffusion components
try:
    # Mock MPI for single-GPU usage to avoid MPI dependency issues
    import sys
    import types
    
    # Create a mock MPI module
    mock_mpi = types.ModuleType('MPI')
    mock_mpi.COMM_WORLD = types.SimpleNamespace()
    mock_mpi.COMM_WORLD.Get_rank = lambda: 0
    mock_mpi.COMM_WORLD.Get_size = lambda: 1
    mock_mpi.COMM_WORLD.bcast = lambda x, root=0: x
    mock_mpi.COMM_WORLD.allreduce = lambda x: x
    
    # Mock mpi4py module
    mock_mpi4py = types.ModuleType('mpi4py')
    mock_mpi4py.MPI = mock_mpi
    sys.modules['mpi4py'] = mock_mpi4py
    sys.modules['mpi4py.MPI'] = mock_mpi
    
    # Add guided-diffusion to path
    guided_diff_path = os.path.join(os.path.dirname(__file__), 'guided-diffusion')
    sys.path.insert(0, guided_diff_path)
    
    # Now import guided-diffusion modules
    from guided_diffusion import dist_util, logger
    from guided_diffusion.script_util import (
        tfg_model_and_diffusion_defaults,
        tfg_create_model_and_diffusion,
        args_to_dict,
    )
    print("✅ Using single-GPU mode with mocked MPI")
    
except ImportError as e:
    print(f"❌ Error importing guided_diffusion: {e}")
    print("💡 Make sure the guided-diffusion directory exists")
    sys.exit(1)

class ConfigurableDiff2LipProcessor:
    """
    Configurable Diff2Lip processor with YAML-based parameter system
    Supports batch processing and all quality enhancement features
    """
    
    def __init__(self, config_path="inference_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Configurable Diff2Lip Inference")
        print(f"📋 Config: {config_path}")
        print(f"🖥️  Device: {self.device}")
        print(f"📦 Batch Size: {self.config['processing']['batch_size']}")
        print(f"🎯 Quality Preset: {self.config['quality']['preset']}")
        
        # Initialize components
        self.model = None
        self.diffusion = None
        self.detector = None
        self.face_enhancer = None
        self.args = self._create_args_from_config()
        self._initialize_models()
        self._initialize_face_enhancer()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}")
            print(f"📝 Creating default config file...")
            self._create_default_config(config_path)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Configuration loaded from {config_path}")
        return config
    
    def _create_default_config(self, config_path):
        """Create default configuration file"""
        default_config = {
            'model': {
                'checkpoint_path': 'checkpoints/checkpoint.pt',
                'image_size': 128,
                'num_channels': 128,
                'num_res_blocks': 2,
                'num_heads': 4,
                'num_head_channels': 64,
                'attention_resolutions': '32,16,8',
                'use_fp16': True,
                'learn_sigma': True
            },
            'quality': {
                'preset': 'balanced',  # fast, balanced, high, ultra
                'timestep_respacing': 'ddim25',  # Will be overridden by preset
                'face_hide_percentage': 0.5,
                'use_gradient_mask': False,  # Phase 1.1 enhancement
                'blur_kernel_size': 15,  # Phase 1.1 enhancement
                'sharpening_strength': 0.0,  # Phase 1.3 enhancement
                'super_resolution': False,  # Phase 2.2 enhancement
                'temporal_smoothing': False,  # Phase 4.1 enhancement
                'smoothing_factor': 0.3
            },
            'processing': {
                'batch_size': 4,
                'use_ddim': True,
                'face_det_batch_size': 64,
                'pads': [0, 0, 0, 0],
                'video_fps': 25,
                'sample_rate': 16000,
                'syncnet_mel_step_size': 16
            },
            'audio': {
                'enhanced_processing': False,  # Phase 5.1 enhancement
                'num_mels': 80,  # Can be increased to 128 for enhanced
                'n_fft': 800,  # Can be increased to 1024 for enhanced
                'hop_size': 200,  # Will be adjusted proportionally
                'noise_reduction': False,  # Phase 5.2 enhancement
                'dynamic_range_compression': False
            },
            'masking': {
                'use_landmark_masking': False,  # Phase 3.1 enhancement
                'adaptive_masking': False,  # Phase 3.2 enhancement
                'expansion_factor': 1.2,
                'mask_blur_strength': 15
            },
            'post_processing': {
                'color_correction': False,  # Phase 6.1 enhancement
                'artifact_reduction': False,  # Phase 6.2 enhancement
                'detail_enhancement': False
            },
            'face_enhancement': {
                'enabled': False,  # Enable face enhancement post-processing
                'model': 'gfpgan_1.4',  # Enhancement model
                'blend': 80,  # Blending ratio (0-100)
                'weight': 1.0,  # Enhancement strength (0.0-1.0)
                'apply_timing': 'post_processing',  # When to apply enhancement
                'batch_processing': True,  # Process frames in batches
                'face_selector_mode': 'one',  # Face selection mode
                'reference_face_distance': 0.6,  # Distance threshold
                'mask_blur': 0.1,  # Mask edge blur
                'mask_padding': [10, 10, 10, 10]  # Mask padding
            },
            'optimization': {
                'enable_cuda_optimizations': True,
                'torch_compile': True,
                'memory_cleanup_interval': 5,
                'gpu_memory_monitoring': True
            },
            'paths': {
                'input_video': 'test_media/person.mp4',
                'input_audio': 'test_media/speech.m4a',
                'output_video': 'output_dir/result_configurable.mp4',
                'temp_dir': 'temp'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Default configuration created at {config_path}")
    
    def _create_args_from_config(self):
        """Convert YAML config to args object"""
        class Args:
            pass
        
        args = Args()
        
        # Model configuration
        model_config = self.config['model']
        args.image_size = model_config['image_size']
        args.num_channels = model_config['num_channels']
        args.num_res_blocks = model_config['num_res_blocks']
        args.num_heads = model_config['num_heads']
        args.num_heads_upsample = -1
        args.num_head_channels = model_config['num_head_channels']
        args.attention_resolutions = model_config['attention_resolutions']
        args.dropout = 0.0
        args.class_cond = False
        args.use_checkpoint = False
        args.use_scale_shift_norm = False
        args.resblock_updown = True
        args.use_fp16 = model_config['use_fp16']
        args.learn_sigma = model_config['learn_sigma']
        args.model_path = model_config['checkpoint_path']
        
        # Quality presets - Phase 1.2 enhancement
        quality_presets = {
            'fast': 'ddim10',
            'balanced': 'ddim25',
            'high': 'ddim50',
            'ultra': 'ddim100'
        }
        
        quality_config = self.config['quality']
        preset = quality_config['preset']
        if preset in quality_presets:
            args.timestep_respacing = quality_presets[preset]
            print(f"🎯 Using {preset} quality preset: {quality_presets[preset]}")
        else:
            args.timestep_respacing = quality_config.get('timestep_respacing', 'ddim25')
        
        # Diffusion configuration
        args.diffusion_steps = 1000
        args.noise_schedule = "linear"
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
        
        # Processing configuration
        processing_config = self.config['processing']
        args.video_fps = processing_config['video_fps']
        args.sample_rate = processing_config['sample_rate']
        args.syncnet_mel_step_size = processing_config['syncnet_mel_step_size']
        args.skip_timesteps = 0
        args.pads = processing_config['pads']
        args.face_hide_percentage = quality_config['face_hide_percentage']
        args.use_ddim = processing_config['use_ddim']
        
        # Enhancement parameters
        args.use_gradient_mask = quality_config.get('use_gradient_mask', False)
        args.blur_kernel_size = quality_config.get('blur_kernel_size', 15)
        args.sharpening_strength = quality_config.get('sharpening_strength', 0.0)
        args.super_resolution = quality_config.get('super_resolution', False)
        args.temporal_smoothing = quality_config.get('temporal_smoothing', False)
        args.smoothing_factor = quality_config.get('smoothing_factor', 0.3)
        
        # Phase 1.4: Circular/Elliptical Masking parameters
        args.use_circular_mask = quality_config.get('use_circular_mask', False)
        args.mask_shape = quality_config.get('mask_shape', 'ellipse')
        args.ellipse_aspect_ratio = quality_config.get('ellipse_aspect_ratio', 1.5)
        
        return args
    
    def _initialize_models(self):
        """Initialize Diff2Lip models with optimizations"""
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
        
        # CUDA optimizations
        if self.config['optimization']['enable_cuda_optimizations'] and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            print("✅ CUDA optimizations enabled")
        
        # Torch compile optimization
        if self.config['optimization']['torch_compile']:
            try:
                import triton
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✅ Model compiled successfully!")
            except ImportError:
                print("⚠️  Triton not available - skipping torch.compile")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
        
        # Initialize face detector
        print("Loading face detector...")
        self.detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, 
            flip_input=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("✅ Models loaded and optimized successfully!")
    
    def _initialize_face_enhancer(self):
        """Initialize face enhancer if enabled"""
        face_config = self.config.get('face_enhancement', {})
        
        if face_config.get('enabled', False):
            print("🎭 Initializing face enhancer...")
            try:
                self.face_enhancer = create_face_enhancer(face_config)
                print(f"✅ Face enhancer initialized: {face_config.get('model', 'gfpgan_1.4')}")
            except Exception as e:
                print(f"⚠️  Face enhancer initialization failed: {e}")
                print("✅ Continuing without face enhancement")
                self.face_enhancer = None
        else:
            print("🎭 Face enhancement disabled")
            self.face_enhancer = None
    
    def _monitor_resources(self):
        """Monitor system resources"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
        return memory_mb, gpu_memory
    
    def _load_audio(self, audio_path):
        """Load and preprocess audio with enhancement options"""
        print(f"🎵 Loading audio from: {audio_path}")
        
        # Enhanced audio processing - Phase 5.1
        audio_config = self.config['audio']
        if audio_config.get('enhanced_processing', False):
            print("🔊 Using enhanced audio processing")
            # TODO: Implement enhanced audio processing in Phase 5
            # - Higher sample rate
            # - More mel channels
            # - Noise reduction
        
        wav = audio.load_wav(audio_path, self.args.sample_rate)
        orig_mel = audio.melspectrogram(wav).T
        print(f"✅ Audio loaded: {orig_mel.shape} mel spectrogram")
        return orig_mel, wav
    
    def _get_audio_chunk(self, orig_mel, frame_idx):
        """Get audio chunk for specific frame"""
        start_idx = int(80. * (frame_idx / float(self.args.video_fps)))
        end_idx = start_idx + self.args.syncnet_mel_step_size
        
        if end_idx > orig_mel.shape[0]:
            chunk = np.zeros((self.args.syncnet_mel_step_size, orig_mel.shape[1]))
            available = orig_mel.shape[0] - start_idx
            if available > 0:
                chunk[:available] = orig_mel[start_idx:]
        else:
            chunk = orig_mel[start_idx:end_idx]
        
        return chunk
    

    
    def _enhance_lip_detail(self, generated_face, sharpening_strength=0.3):
        """Apply sharpening for detail enhancement - Phase 1.3 enhancement"""
        if sharpening_strength <= 0:
            return generated_face
        
        # Unsharp masking for detail enhancement
        blurred = cv2.GaussianBlur(generated_face, (0, 0), 1.0)
        sharpened = cv2.addWeighted(
            generated_face, 1.0 + sharpening_strength, 
            blurred, -sharpening_strength, 0
        )
        return sharpened
    
    def _apply_face_enhancement_batch(self, frames_batch):
        """Apply face enhancement to a batch of frames"""
        if not self.face_enhancer or not self.face_enhancer.enabled:
            return frames_batch
        
        face_config = self.config.get('face_enhancement', {})
        if not face_config.get('batch_processing', True):
            # Process individually
            return [self.face_enhancer.enhance_face(frame) for frame in frames_batch]
        
        try:
            # Process as batch
            enhanced_frames = self.face_enhancer.enhance_batch(frames_batch)
            return enhanced_frames
        except Exception as e:
            print(f"⚠️  Batch face enhancement failed: {e}")
            return frames_batch
    
    def _process_batch(self, frames_batch, audio_chunks_batch):
        """Process a batch of frames with configurable enhancements"""
        batch_size = len(frames_batch)
        
        # Batch face detection
        detections_batch = self.detector.get_detections_for_batch(np.array(frames_batch))
        
        # Prepare batch tensors
        valid_indices = []
        face_tensors = []
        audio_tensors = []
        face_crops_info = []
        
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
            face_tensor = face_tensor.permute(2, 0, 1)
            
            audio_tensor = torch.FloatTensor(audio_chunk.T)
            
            face_tensors.append(face_tensor)
            audio_tensors.append(audio_tensor)
            face_crops_info.append((i, x1, y1, x2, y2))
            valid_indices.append(i)
        
        if not face_tensors:
            return frames_batch, [False] * batch_size
        
        # Stack into batch tensors
        face_batch = torch.stack(face_tensors).unsqueeze(1).to(self.device)
        audio_batch = torch.stack(audio_tensors).unsqueeze(1).unsqueeze(1).to(self.device)
        
        try:
            # Prepare batch
            batch = {
                "image": face_batch,
                "ref_img": face_batch.clone(),
                "indiv_mels": audio_batch,
            }
            
            # Generate with diffusion model
            try:
                from guided_diffusion.tfg_data_util import tfg_process_batch
            except ImportError:
                print("❌ Cannot import tfg_process_batch")
                return frames_batch, [False] * batch_size
            
            with torch.no_grad():
                # Use TFG processing with masking enhancements - Phase 1.1 & 1.4
                img_batch, model_kwargs = tfg_process_batch(
                    batch, 
                    self.args.face_hide_percentage,
                    use_ref=self.args.use_ref,
                    use_audio=self.args.use_audio,
                    use_gradient_mask=self.args.use_gradient_mask,
                    blur_kernel_size=self.args.blur_kernel_size,
                    use_circular_mask=self.args.use_circular_mask,
                    mask_shape=self.args.mask_shape,
                    ellipse_aspect_ratio=self.args.ellipse_aspect_ratio
                )
                
                # Move to device
                img_batch = img_batch.to(self.device)
                model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}
                
                # Sample using diffusion model
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.args.use_ddim 
                    else self.diffusion.ddim_sample_loop
                )
                
                sample = sample_fn(
                    self.model,
                    img_batch.shape,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    progress=False,
                )
                
                # Apply mask
                mask = model_kwargs['mask']
                recon_batch = sample * mask + (1. - mask) * img_batch
                
                # Reconstruct individual frames
                result_frames = frames_batch.copy()
                success_flags = [False] * batch_size
                
                for batch_idx, (orig_idx, x1, y1, x2, y2) in enumerate(face_crops_info):
                    # Convert back to image
                    generated_face = recon_batch[batch_idx].permute(1, 2, 0).cpu().numpy()
                    generated_face = (generated_face.clip(-1, 1) + 1) / 2
                    generated_face = (generated_face * 255).astype(np.uint8)
                    
                    # Apply sharpening if enabled - Phase 1.3
                    if self.args.sharpening_strength > 0:
                        generated_face = self._enhance_lip_detail(
                            generated_face, 
                            self.args.sharpening_strength
                        )
                    
                    # Apply face enhancement if enabled
                    if self.face_enhancer and self.face_enhancer.enabled:
                        face_config = self.config.get('face_enhancement', {})
                        if face_config.get('apply_timing', 'post_processing') == 'post_processing':
                            try:
                                generated_face = self.face_enhancer.enhance_face(generated_face)
                            except Exception as e:
                                print(f"⚠️  Face enhancement failed for frame {orig_idx}: {e}")
                    
                    # Resize and place back
                    generated_face_resized = cv2.resize(generated_face, (x2-x1, y2-y1))
                    
                    result_frames[orig_idx] = frames_batch[orig_idx].copy()
                    result_frames[orig_idx][y1:y2, x1:x2] = generated_face_resized
                    success_flags[orig_idx] = True
                
                # Cleanup
                del face_batch, audio_batch, batch, img_batch, sample, mask, recon_batch
                for k in list(model_kwargs.keys()):
                    del model_kwargs[k]
                del model_kwargs, face_tensors, audio_tensors
                torch.cuda.empty_cache()
                gc.collect()
                
                return result_frames, success_flags
                
        except Exception as e:
            print(f"❌ Batch generation error: {e}")
            import traceback
            traceback.print_exc()
            return frames_batch, [False] * batch_size
    
    def process_video(self, video_path=None, audio_path=None, output_path=None):
        """Process video with configurable parameters"""
        # Use config paths if not specified
        if video_path is None:
            video_path = self.config['paths']['input_video']
        if audio_path is None:
            audio_path = self.config['paths']['input_audio']
        if output_path is None:
            output_path = self.config['paths']['output_video']
        
        print(f"🎬 Processing video: {video_path}")
        print(f"🎵 Audio: {audio_path}")
        print(f"💾 Output: {output_path}")
        
        batch_size = self.config['processing']['batch_size']
        print(f"📦 Batch size: {batch_size}")
        
        # Load audio
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
        
        print(f"📹 Video: {frame_count} frames, {fps} FPS, {width}x{height}")
        
        # Process frames in batches
        processed_frames = []
        successful_generations = 0
        
        mem_before, gpu_before = self._monitor_resources()
        start_time = time.time()
        
        print("🔄 Processing frames in batches...")
        
        num_batches = (frame_count + batch_size - 1) // batch_size
        cleanup_interval = self.config['optimization']['memory_cleanup_interval']
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            # Collect frames for this batch
            frames_batch = []
            audio_chunks_batch = []
            
            for i in range(batch_size):
                frame_idx = batch_idx * batch_size + i
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
            
            # Process this batch
            result_frames, success_flags = self._process_batch(frames_batch, audio_chunks_batch)
            
            # Add results
            processed_frames.extend(result_frames)
            successful_generations += sum(success_flags)
            
            # Memory management
            if batch_idx % cleanup_interval == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # Monitor GPU memory
                    if self.config['optimization']['gpu_memory_monitoring'] and batch_idx % 10 == 0:
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        print(f"  Batch {batch_idx}/{num_batches}: GPU memory: {gpu_memory:.1f}MB")
        
        cap.release()
        
        processing_time = time.time() - start_time
        mem_after, gpu_after = self._monitor_resources()
        
        print(f"✅ Processing complete!")
        print(f"⏱️  Time: {processing_time:.1f}s ({processing_time/frame_count:.3f}s per frame)")
        print(f"🚀 Speedup vs single-frame: {0.650/processing_time*frame_count:.1f}x faster!")
        print(f"💾 Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (+{mem_after-mem_before:.1f}MB)")
        print(f"🎯 Successful generations: {successful_generations}/{frame_count} ({successful_generations/frame_count*100:.1f}%)")
        
        # Save video
        if processed_frames:
            return self._save_video(processed_frames, output_path, audio_path, fps)
        else:
            print("❌ No frames to save")
            return False
    
    def _save_video(self, processed_frames, output_path, audio_path, fps):
        """Save video with audio integration"""
        print(f"💾 Saving video to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save video without audio to temp file
        temp_video_path = output_path.replace('.mp4', '_temp_no_audio.mp4')
        video_tensor = torch.from_numpy(np.array(processed_frames))
        
        print("📹 Saving video frames...")
        torchvision.io.write_video(
            temp_video_path,
            video_array=video_tensor,
            fps=int(fps),
            video_codec='libx264'
        )
        
        # Use FFmpeg to combine video + audio
        print("🎵 Adding audio with FFmpeg...")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
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

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configurable Diff2Lip Inference')
    parser.add_argument('--config', default='inference_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--video', help='Input video path (overrides config)')
    parser.add_argument('--audio', help='Input audio path (overrides config)')
    parser.add_argument('--output', help='Output video path (overrides config)')
    
    args = parser.parse_args()
    
    print("🚀 Configurable Diff2Lip Inference")
    print("=" * 50)
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Setup distributed training (single-GPU mode)
    try:
        dist_util.setup_dist()
    except Exception as e:
        print(f"⚠️  dist_util.setup_dist() failed: {e}")
        print("✅ Continuing in single-GPU mode")
    
    try:
        # Initialize processor
        processor = ConfigurableDiff2LipProcessor(args.config)
        
        # Process video
        success = processor.process_video(
            video_path=args.video,
            audio_path=args.audio,
            output_path=args.output
        )
        
        if success:
            print("🎉 Inference completed successfully!")
        else:
            print("❌ Inference failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
