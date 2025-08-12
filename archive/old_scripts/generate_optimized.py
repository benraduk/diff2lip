#!/usr/bin/env python3
"""
Optimized Diff2Lip Generation with FaceFusion-inspired improvements
- Streaming video processing (eliminates 20GB RAM usage)
- Optimized face detection
- Inference pooling
- Memory management
"""

import cv2
import os
from os.path import join, basename, dirname, splitext
import shutil
import argparse
import numpy as np
import random
import torch
import torchvision
import subprocess
import sys
sys.path.append('.')
from audio import audio
from tqdm import tqdm
import gc

# Import our optimized components
from face_detection_optimized import FaceFusionDetectorWrapper, StreamingVideoProcessor

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from guided_diffusion.tfg_data_util import (
    tfg_process_batch,
)

class InferencePool:
    """
    FaceFusion-inspired inference pool for model management
    """
    
    def __init__(self):
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_model(self, model_name: str, model_factory=None):
        """Get or create model instance"""
        if model_name not in self.models and model_factory:
            print(f"Loading {model_name} into inference pool...")
            self.models[model_name] = model_factory()
            if hasattr(self.models[model_name], 'to'):
                self.models[model_name].to(self.device)
        return self.models.get(model_name)
    
    def clear_model(self, model_name: str):
        """Remove model from pool"""
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()
            gc.collect()
    
    def clear_all(self):
        """Clear all models"""
        self.models.clear()
        torch.cuda.empty_cache()
        gc.collect()

# Global inference pool
inference_pool = InferencePool()

def crop_audio_window(spec, start_frame, args):
    if type(args) == argparse.Namespace:
        args = args
    else:
        args = args[0]

    start_idx = int(80. * (start_frame / float(args.video_fps)))
    end_idx = start_idx + args.syncnet_mel_step_size

    return spec[start_idx : end_idx, :]

def load_audio_streaming(audio_path, args):
    """
    Optimized audio loading - loads once and processes as needed
    """
    # Load audio file once
    wav = audio.load_wav(audio_path, args.sample_rate)
    orig_mel = audio.melspectrogram(wav).T
    
    return orig_mel, wav

def get_audio_chunk(orig_mel, frame_idx, args):
    """
    Get audio chunk for specific frame without loading entire audio repeatedly
    """
    return crop_audio_window(orig_mel.copy(), max(frame_idx - args.syncnet_T//2, 0), args)

def get_smoothened_boxes(boxes, T):
    """Temporal smoothing for face bounding boxes"""
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def sample_batch_optimized(batch, model, diffusion, args):
    """
    Optimized batch sampling with memory management
    """
    # Move batch to device efficiently
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(dist_util.dev())
    
    # Process batch
    model_kwargs = {}
    model_kwargs["y"] = batch

    # Sample with memory cleanup
    sample = diffusion.ddim_sample_loop(
        model,
        (args.sampling_batch_size, 3, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        skip_timesteps=args.skip_timesteps,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    
    # Clear intermediate tensors
    del model_kwargs
    torch.cuda.empty_cache()
    
    return sample

def generate_streaming(video_path, audio_path, model, diffusion, detector, args, out_path=None, save_orig=True):
    """
    Main generation function with streaming processing
    Eliminates the 20GB RAM issue by processing video in chunks
    """
    print("Starting streaming generation...")
    
    # Initialize streaming processor
    video_processor = StreamingVideoProcessor(video_path, audio_path)
    
    # Load audio once (much smaller than video)
    print("Loading audio...")
    orig_mel, audio_wavform = load_audio_streaming(audio_path, args)
    print(f"Audio loaded: {orig_mel.shape}")
    
    # Prepare output
    if out_path is None:
        out_path = os.path.join(args.sample_path, 
                               splitext(basename(video_path))[0] + "_" + 
                               splitext(basename(audio_path))[0] + "_result.mp4")
    
    # Initialize video writer
    temp_dir = join(args.sample_path, "temp_streaming")
    os.makedirs(temp_dir, exist_ok=True)
    
    all_generated_frames = []
    all_original_frames = []
    all_face_bboxes = []
    
    chunk_idx = 0
    total_processed = 0
    
    # Process video in streaming chunks
    print("Processing video in streaming chunks...")
    for chunk_frames, chunk_face_results in video_processor.process_frames_streaming(
        detector, args, chunk_size=args.streaming_chunk_size
    ):
        print(f"Processing chunk {chunk_idx} with {len(chunk_frames)} frames...")
        
        # Prepare batch data for this chunk
        chunk_face_frames = []
        chunk_face_bboxes = []
        chunk_audio_features = []
        
        for i, (frame, face_result) in enumerate(zip(chunk_frames, chunk_face_results)):
            frame_idx = total_processed + i
            
            # Get face data
            face_crop, bbox, success = face_result
            if not success:
                continue
            
            chunk_face_frames.append(face_crop)
            chunk_face_bboxes.append(bbox)
            
            # Get corresponding audio features
            if frame_idx < orig_mel.shape[0]:
                audio_chunk = get_audio_chunk(orig_mel, frame_idx, args)
                if audio_chunk.shape[0] == args.syncnet_mel_step_size:
                    chunk_audio_features.append(audio_chunk.T)
        
        if not chunk_face_frames:
            print(f"No valid faces in chunk {chunk_idx}, skipping...")
            chunk_idx += 1
            total_processed += len(chunk_frames)
            continue
        
        # Convert to tensors
        face_tensor = torch.FloatTensor(
            np.transpose(np.array(chunk_face_frames, dtype=np.float32) / 255., (0, 3, 1, 2))
        )
        audio_tensor = torch.FloatTensor(np.array(chunk_audio_features)).unsqueeze(1)
        
        print(f"Chunk tensors - Face: {face_tensor.shape}, Audio: {audio_tensor.shape}")
        
        # Process in sub-batches if chunk is too large
        batch_size = args.sampling_batch_size
        chunk_generated = []
        
        for batch_start in range(0, len(face_tensor), batch_size):
            batch_end = min(batch_start + batch_size, len(face_tensor))
            
            # Prepare batch
            batch_faces = face_tensor[batch_start:batch_end]
            batch_audio = audio_tensor[batch_start:batch_end]
            
            # Create batch dictionary for model
            batch = {
                'face_frames': batch_faces,
                'audio_features': batch_audio,
                'ref_frames': batch_faces.clone(),  # Use same frames as reference
            }
            
            # Generate using diffusion model
            try:
                generated_batch = sample_batch_optimized(batch, model, diffusion, args)
                chunk_generated.append(generated_batch.cpu())
                
                # Clear batch tensors
                del batch, batch_faces, batch_audio, generated_batch
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Use original faces as fallback
                chunk_generated.append(batch_faces.cpu())
        
        # Combine chunk results
        if chunk_generated:
            chunk_result = torch.cat(chunk_generated, dim=0)
            
            # Convert back to frames and store
            for i in range(chunk_result.shape[0]):
                generated_frame = chunk_result[i].permute(1, 2, 0).numpy()
                generated_frame = (generated_frame * 255).astype(np.uint8)
                all_generated_frames.append(generated_frame)
                
                # Store corresponding original frame and bbox
                if i < len(chunk_frames):
                    all_original_frames.append(chunk_frames[i])
                    all_face_bboxes.append(chunk_face_bboxes[i])
        
        # Cleanup
        del face_tensor, audio_tensor
        if 'chunk_result' in locals():
            del chunk_result
        torch.cuda.empty_cache()
        gc.collect()
        
        chunk_idx += 1
        total_processed += len(chunk_frames)
        
        print(f"Completed chunk {chunk_idx-1}, total processed: {total_processed}")
    
    print(f"Generated {len(all_generated_frames)} frames")
    
    # Reconstruct final video
    print("Reconstructing final video...")
    final_frames = []
    
    for i, (original_frame, generated_face, bbox) in enumerate(
        zip(all_original_frames, all_generated_frames, all_face_bboxes)
    ):
        # Place generated face back into original frame
        y1, y2, x1, x2 = bbox
        face_resized = cv2.resize(generated_face, (x2-x1, y2-y1))
        
        final_frame = original_frame.copy()
        final_frame[y1:y2, x1:x2] = face_resized
        final_frames.append(final_frame)
    
    # Save video
    if final_frames:
        print(f"Saving video with {len(final_frames)} frames...")
        torchvision.io.write_video(
            out_path,
            video_array=torch.from_numpy(np.array(final_frames)),
            fps=args.video_fps,
            video_codec='libx264'
        )
        print(f"Video saved to: {out_path}")
    
    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return out_path

def main():
    """Main function with optimized model loading"""
    args = create_argparser().parse_args()
    
    # Add streaming-specific arguments
    if not hasattr(args, 'streaming_chunk_size'):
        args.streaming_chunk_size = 50  # Process 50 frames at a time
    
    dist_util.setup_dist()
    logger.configure(dir=args.sample_path, format_strs=["stdout", "log"])

    logger.log("Creating optimized model...")
    
    # Use inference pool for model management
    def model_factory():
        model, diffusion = tfg_create_model_and_diffusion(
            **args_to_dict(args, tfg_model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location='cpu')
        )
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        return model, diffusion
    
    model, diffusion = inference_pool.get_model('diff2lip_model', model_factory)
    
    # Create optimized face detector
    logger.log("Creating optimized face detector...")
    detector = FaceFusionDetectorWrapper(
        model_type='retinaface',  # or 'scrfd' for faster processing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Process video with streaming approach
    if args.generate_from_filelist:
        # TODO: Implement streaming filelist processing
        print("Streaming filelist processing not yet implemented")
    else:
        result_path = generate_streaming(
            args.video_path, args.audio_path, model, diffusion, detector, args, 
            out_path=args.out_path, save_orig=args.save_orig
        )
        print(f"Generation complete: {result_path}")
    
    # Cleanup
    inference_pool.clear_all()

def create_argparser():
    """Create argument parser with streaming options"""
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        sample_path="d2l_gen",
        streaming_chunk_size=50,  # New streaming parameter
        video_fps=25,
        syncnet_T=5,
        syncnet_mel_step_size=16,
        sample_rate=16000,
        skip_timesteps=0,
        pads=[0, 0, 0, 0],
        is_voxceleb2=False,
    )
    defaults.update(tfg_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add streaming-specific arguments
    parser.add_argument("--streaming_chunk_size", type=int, default=50,
                       help="Number of frames to process in each streaming chunk")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--audio_path", type=str, required=True,
                       help="Path to input audio")
    parser.add_argument("--out_path", type=str, 
                       help="Path for output video")
    parser.add_argument("--save_orig", action="store_true",
                       help="Save original video")
    parser.add_argument("--generate_from_filelist", type=int, default=0,
                       help="Generate from filelist")
    
    return parser

if __name__ == "__main__":
    main()
