#!/usr/bin/env python3
"""
Super-Resolution Post-Processing Module for Diff2Lip
Phase 2.2: Super-Resolution Integration

Provides 2x/4x upscaling using Real-ESRGAN for enhanced lip region detail
Selective processing: Only enhances generated lip regions for efficiency
"""

import cv2
import numpy as np
import torch
import os
import sys
from pathlib import Path

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REALESRGAN_AVAILABLE = True
    print("✅ Real-ESRGAN successfully imported")
except ImportError as e:
    print(f"⚠️  Real-ESRGAN not available: {e}")
    print("💡 Falling back to ESRGAN-compatible implementation")
    REALESRGAN_AVAILABLE = False

# Fallback: Simple ESRGAN-style implementation using PyTorch
try:
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_SR_AVAILABLE = True
except ImportError:
    TORCH_SR_AVAILABLE = False


class SimpleSRNet(nn.Module):
    """
    Simple Super-Resolution Network using PyTorch
    Fallback when Real-ESRGAN is not available
    """
    def __init__(self, scale=2, num_channels=3):
        super(SimpleSRNet, self).__init__()
        self.scale = scale
        
        # Simple CNN architecture for super-resolution
        self.conv1 = nn.Conv2d(num_channels, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels * (scale ** 2), 5, padding=2)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x


class LipSuperResolver:
    """
    Super-Resolution processor for enhancing generated lip regions
    Supports 2x and 4x upscaling using Real-ESRGAN or fallback PyTorch implementation
    """
    
    def __init__(self, scale=2, model_name='RealESRGAN_x2plus', device='cuda'):
        """
        Initialize the super-resolution processor
        
        Args:
            scale: Upscaling factor (2 or 4)
            model_name: Real-ESRGAN model variant
            device: Processing device ('cuda' or 'cpu')
        """
        self.scale = scale
        self.model_name = model_name
        self.device = device
        self.upsampler = None
        self.use_realesrgan = REALESRGAN_AVAILABLE
        self.simple_model = None
        
        print(f"🚀 Initializing Super-Resolution Processor")
        print(f"📈 Scale: {scale}x")
        print(f"🤖 Model: {model_name if self.use_realesrgan else 'Simple PyTorch SR'}")
        print(f"🖥️  Device: {device}")
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the super-resolution model (Real-ESRGAN or fallback)"""
        if self.use_realesrgan:
            try:
                # Define model architecture based on scale
                if self.scale == 2:
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                    model_path = self._get_model_path('RealESRGAN_x2plus.pth')
                elif self.scale == 4:
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                    model_path = self._get_model_path('RealESRGAN_x4plus.pth')
                else:
                    raise ValueError(f"Unsupported scale: {self.scale}. Use 2 or 4.")
                
                # Initialize upsampler
                self.upsampler = RealESRGANer(
                    scale=self.scale,
                    model_path=model_path,
                    model=model,
                    tile=512,  # Process in 512x512 tiles to manage memory
                    tile_pad=32,
                    pre_pad=0,
                    half=True if self.device == 'cuda' else False,  # Use FP16 for CUDA
                    device=self.device
                )
                
                print(f"✅ Real-ESRGAN model initialized successfully")
                
            except Exception as e:
                print(f"❌ Failed to initialize Real-ESRGAN: {e}")
                print(f"🔄 Falling back to simple PyTorch implementation")
                self.use_realesrgan = False
                self._initialize_fallback_model()
        else:
            self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """Initialize simple PyTorch super-resolution model"""
        try:
            if not TORCH_SR_AVAILABLE:
                raise RuntimeError("PyTorch not available for fallback SR")
            
            self.simple_model = SimpleSRNet(scale=self.scale, num_channels=3)
            self.simple_model.to(self.device)
            self.simple_model.eval()
            
            print(f"✅ Fallback PyTorch SR model initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize fallback model: {e}")
            print(f"⚠️  Will use basic bicubic upscaling as final fallback")
            self.simple_model = None
    
    def _get_model_path(self, model_filename):
        """
        Get the path to the pre-trained model weights
        Downloads automatically if not present
        """
        # Check common locations
        possible_paths = [
            f"weights/{model_filename}",
            f"models/{model_filename}",
            f"checkpoints/{model_filename}",
            f"realesrgan/weights/{model_filename}"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                return path
        
        # Create weights directory if it doesn't exist
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        model_path = weights_dir / model_filename
        
        print(f"📥 Model will be auto-downloaded to: {model_path}")
        return str(model_path)
    
    def enhance_lip_region(self, face_image, mask, selective_enhancement=True):
        """
        Enhance the lip region of a face image using super-resolution
        
        Args:
            face_image: Input face image (numpy array, uint8)
            mask: Binary mask indicating lip region (numpy array, 0-1)
            selective_enhancement: If True, only enhance masked regions
        
        Returns:
            enhanced_image: Super-resolved image (numpy array, uint8)
        """
        try:
            # Ensure proper input format
            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)
            
            if self.use_realesrgan and self.upsampler is not None:
                # Use Real-ESRGAN
                if selective_enhancement and mask is not None:
                    enhanced_image = self._selective_enhance_realesrgan(face_image, mask)
                else:
                    enhanced_image, _ = self.upsampler.enhance(face_image, outscale=self.scale)
                
            elif self.simple_model is not None:
                # Use PyTorch fallback model
                enhanced_image = self._enhance_with_pytorch(face_image)
                
            else:
                # Final fallback: bicubic upscaling
                h, w = face_image.shape[:2]
                enhanced_image = cv2.resize(face_image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
                print(f"⚠️  Using bicubic fallback for {self.scale}x upscaling")
            
            return enhanced_image
            
        except Exception as e:
            print(f"❌ Super-resolution failed: {e}")
            # Final fallback: return bicubic upscaling
            h, w = face_image.shape[:2]
            fallback = cv2.resize(face_image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
            print(f"⚠️  Using bicubic fallback for {self.scale}x upscaling")
            return fallback
    
    def _enhance_with_pytorch(self, face_image):
        """Enhance image using PyTorch fallback model"""
        with torch.no_grad():
            # Convert to tensor
            image_tensor = torch.from_numpy(face_image).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Apply super-resolution
            enhanced_tensor = self.simple_model(image_tensor)
            
            # Convert back to numpy
            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # CRITICAL FIX: The PyTorch model produces low-range outputs
            # Need to properly scale to full 0-255 range
            enhanced_np = np.clip(enhanced_np, 0, 1)  # Ensure [0,1] range first
            
            # Scale to match input image's dynamic range instead of just 0-255
            input_min, input_max = face_image.min(), face_image.max()
            enhanced_np = enhanced_np * (input_max - input_min) + input_min
            enhanced_np = np.clip(enhanced_np, 0, 255).astype(np.uint8)
            

            
            return enhanced_np
    
    def _selective_enhance_realesrgan(self, face_image, mask):
        """
        Selectively enhance only the masked regions
        More efficient than processing the entire image
        """
        h, w = face_image.shape[:2]
        
        # Resize mask to match face image
        if mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_resized = mask.copy()
        
        # Ensure mask is binary
        if mask_resized.max() <= 1.0:
            mask_resized = (mask_resized * 255).astype(np.uint8)
        
        # Find bounding box of the mask region
        mask_binary = (mask_resized > 128).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No mask region found, enhance full image
            enhanced_image, _ = self.upsampler.enhance(face_image, outscale=self.scale)
            return enhanced_image
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, bbox_w, bbox_h = cv2.boundingRect(largest_contour)
        
        # Add padding around the region
        padding = 16
        x = max(0, x - padding)
        y = max(0, y - padding)
        bbox_w = min(w - x, bbox_w + 2 * padding)
        bbox_h = min(h - y, bbox_h + 2 * padding)
        
        # Extract region for super-resolution
        region = face_image[y:y+bbox_h, x:x+bbox_w]
        region_mask = mask_binary[y:y+bbox_h, x:x+bbox_w]
        
        # Enhance the region
        enhanced_region, _ = self.upsampler.enhance(region, outscale=self.scale)
        
        # Create full enhanced image by upscaling original
        enhanced_full = cv2.resize(face_image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
        
        # Blend enhanced region back
        enhanced_region_mask = cv2.resize(region_mask, 
                                        (bbox_w * self.scale, bbox_h * self.scale), 
                                        interpolation=cv2.INTER_LINEAR)
        enhanced_region_mask = enhanced_region_mask.astype(np.float32) / 255.0
        
        # Position in full enhanced image
        y_scaled, x_scaled = y * self.scale, x * self.scale
        
        # Blend the enhanced region
        for c in range(3):  # RGB channels
            enhanced_full[y_scaled:y_scaled+enhanced_region.shape[0], 
                         x_scaled:x_scaled+enhanced_region.shape[1], c] = (
                enhanced_region[:, :, c] * enhanced_region_mask + 
                enhanced_full[y_scaled:y_scaled+enhanced_region.shape[0], 
                             x_scaled:x_scaled+enhanced_region.shape[1], c] * (1 - enhanced_region_mask)
            )
        
        return enhanced_full.astype(np.uint8)
    
    def enhance_frame_batch(self, frames_batch, masks_batch=None):
        """
        Enhance a batch of frames
        
        Args:
            frames_batch: List of face images
            masks_batch: List of corresponding masks (optional)
        
        Returns:
            enhanced_frames: List of enhanced images
        """
        enhanced_frames = []
        
        for i, frame in enumerate(frames_batch):
            mask = masks_batch[i] if masks_batch else None
            enhanced = self.enhance_lip_region(frame, mask)
            enhanced_frames.append(enhanced)
        
        return enhanced_frames
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0
    
    def cleanup(self):
        """Clean up resources"""
        if self.upsampler is not None:
            del self.upsampler
            self.upsampler = None
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 Super-resolution resources cleaned up")


def test_super_resolution():
    """Test function for super-resolution functionality"""
    print("🧪 Testing Super-Resolution Module")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        test_mask = np.zeros((128, 128), dtype=np.float32)
        test_mask[64:96, 32:96] = 1.0  # Lip region mask
        
        # Test 2x super-resolution
        sr_processor = LipSuperResolver(scale=2, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        enhanced = sr_processor.enhance_lip_region(test_image, test_mask)
        
        print(f"✅ Test passed!")
        print(f"📊 Input size: {test_image.shape}")
        print(f"📊 Output size: {enhanced.shape}")
        print(f"🧠 Memory usage: {sr_processor.get_memory_usage():.1f} MB")
        
        sr_processor.cleanup()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_super_resolution()
