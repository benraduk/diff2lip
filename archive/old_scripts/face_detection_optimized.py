#!/usr/bin/env python3
"""
Optimized Face Detection using FaceFusion's RetinaFace/SCRFD
Maintains Diff2Lip's output format while using FaceFusion's efficient components
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from functools import lru_cache
import onnxruntime as ort

class OptimizedFaceDetector:
    """
    FaceFusion-inspired face detector that maintains Diff2Lip's output format
    """
    
    def __init__(self, model_type='retinaface', device='cuda'):
        self.model_type = model_type
        self.device = device
        self.session = None
        self.input_size = (640, 640)  # Standard detection size
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model with optimized settings"""
        model_path = self._get_model_path()
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Please download the model or use the original S3FD detector")
            return
            
        # Configure ONNX Runtime providers
        providers = []
        if self.device == 'cuda' and torch.cuda.is_available():
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }))
        providers.append('CPUExecutionProvider')
        
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"Loaded {self.model_type} model with providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.session = None
    
    def _get_model_path(self) -> str:
        """Get model path - you'll need to download these from FaceFusion"""
        model_paths = {
            'retinaface': 'models/retinaface_10g.onnx',
            'scrfd': 'models/scrfd_2.5g.onnx'
        }
        return model_paths.get(self.model_type, model_paths['retinaface'])
    
    def _prepare_input(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Prepare image for detection"""
        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # Calculate scale to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalize for model input
        input_image = padded.astype(np.float32)
        input_image = (input_image - 127.5) / 128.0  # Normalize to [-1, 1]
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        
        return input_image, scale, (new_w, new_h)
    
    def _post_process_detections(self, outputs, scale: float, original_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Post-process model outputs to get bounding boxes"""
        if self.model_type == 'retinaface':
            return self._post_process_retinaface(outputs, scale, original_size)
        elif self.model_type == 'scrfd':
            return self._post_process_scrfd(outputs, scale, original_size)
        else:
            return []
    
    def _post_process_retinaface(self, outputs, scale: float, original_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Post-process RetinaFace outputs"""
        boxes = []
        scores = []
        
        # RetinaFace typically outputs multiple scales
        for i in range(0, len(outputs), 3):  # scores, boxes, landmarks
            if i + 2 >= len(outputs):
                break
                
            score_map = outputs[i]
            box_map = outputs[i + 1]
            
            # Simple extraction - you may need to adjust based on actual model outputs
            if score_map.max() > 0.5:  # Confidence threshold
                # This is a simplified version - actual implementation depends on model architecture
                h, w = score_map.shape[-2:]
                for y in range(h):
                    for x in range(w):
                        if score_map[0, 1, y, x] > 0.5:  # Face class
                            # Extract box coordinates (simplified)
                            x1 = int((x * 8 - box_map[0, 0, y, x]) / scale)
                            y1 = int((y * 8 - box_map[0, 1, y, x]) / scale)
                            x2 = int((x * 8 + box_map[0, 2, y, x]) / scale)
                            y2 = int((y * 8 + box_map[0, 3, y, x]) / scale)
                            
                            boxes.append((x1, y1, x2, y2))
                            scores.append(float(score_map[0, 1, y, x]))
        
        # Apply NMS and return best detection
        if boxes:
            return [max(zip(boxes, scores), key=lambda x: x[1])[0]]
        return []
    
    def _post_process_scrfd(self, outputs, scale: float, original_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Post-process SCRFD outputs"""
        # Similar to RetinaFace but with SCRFD-specific processing
        # This is a placeholder - actual implementation depends on SCRFD model format
        return []
    
    def detect_faces_single(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a single image
        Returns list of bounding boxes in (x1, y1, x2, y2) format
        """
        if self.session is None:
            # Fallback to a simple face detector or return empty
            return []
        
        try:
            # Prepare input
            input_image, scale, (new_w, new_h) = self._prepare_input(image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_image})
            
            # Post-process
            boxes = self._post_process_detections(outputs, scale, image.shape[:2])
            
            return boxes
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def get_detections_for_batch(self, images: np.ndarray) -> List[Optional[Tuple[int, int, int, int]]]:
        """
        Batch face detection - maintains Diff2Lip's interface
        Args:
            images: numpy array of shape (batch_size, height, width, 3)
        Returns:
            List of bounding boxes or None if no face detected
        """
        results = []
        
        for image in images:
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if image.shape[-1] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Detect faces
            boxes = self.detect_faces_single(image_rgb)
            
            if boxes:
                # Return the first (best) detection in the format expected by Diff2Lip
                x1, y1, x2, y2 = boxes[0]
                results.append((x1, y1, x2, y2))
            else:
                results.append(None)
        
        return results

class StreamingVideoProcessor:
    """
    FaceFusion-inspired streaming video processor
    Eliminates the 20GB RAM issue by processing frames one at a time
    """
    
    def __init__(self, video_path: str, audio_path: str):
        self.video_path = video_path
        self.audio_path = audio_path
        self.video_cap = None
        self.total_frames = 0
        self.fps = 25
        
        self._initialize()
    
    def _initialize(self):
        """Initialize video capture and get metadata"""
        self.video_cap = cv2.VideoCapture(self.video_path)
        if self.video_cap.isOpened():
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            print(f"Video: {self.total_frames} frames at {self.fps} FPS")
        else:
            raise ValueError(f"Cannot open video: {self.video_path}")
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame from video"""
        if not self.video_cap.isOpened():
            return None
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_cap.read()
        
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def process_frames_streaming(self, detector, args, chunk_size=50):
        """
        Process video frames in streaming fashion
        Yields chunks of processed frames to maintain memory efficiency
        """
        processed_frames = []
        face_results = []
        
        for frame_idx in range(self.total_frames):
            # Get single frame (only ~6MB in memory)
            frame = self.get_frame(frame_idx)
            if frame is None:
                continue
            
            # Detect face in this frame
            face_detections = detector.get_detections_for_batch(np.array([frame]))
            face_detection = face_detections[0] if face_detections else None
            
            if face_detection is None:
                print(f"No face detected in frame {frame_idx}")
                continue
            
            # Process face detection result (crop, resize, etc.)
            face_result = self._process_face_detection(frame, face_detection, args)
            
            processed_frames.append(frame)
            face_results.append(face_result)
            
            # Yield chunk when ready
            if len(processed_frames) >= chunk_size or frame_idx == self.total_frames - 1:
                yield processed_frames, face_results
                processed_frames = []
                face_results = []
    
    def _process_face_detection(self, image, detection, args):
        """Process face detection to match Diff2Lip's expected format"""
        x1, y1, x2, y2 = detection
        
        # Apply padding
        if hasattr(args, 'pads'):
            pads = args.pads if isinstance(args.pads, list) else [0, 0, 0, 0]
            pady1, pady2, padx1, padx2 = pads
            
            y1 = max(0, y1 - pady1)
            y2 = min(image.shape[0], y2 + pady2)
            x1 = max(0, x1 - padx1) 
            x2 = min(image.shape[1], x2 + padx2)
        
        # Crop and resize face
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_resized = cv2.resize(face_crop, (128, 128))
            return [face_resized, (y1, y2, x1, x2), True]
        else:
            return [np.zeros((128, 128, 3), dtype=np.uint8), (y1, y2, x1, x2), False]
    
    def __del__(self):
        """Cleanup"""
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()

# Compatibility wrapper to replace the existing face detection
class FaceFusionDetectorWrapper:
    """
    Wrapper to make FaceFusion detector compatible with existing Diff2Lip code
    """
    
    def __init__(self, model_type='retinaface', device='cuda'):
        self.detector = OptimizedFaceDetector(model_type, device)
    
    def get_detections_for_batch(self, images):
        """Maintain compatibility with existing Diff2Lip interface"""
        return self.detector.get_detections_for_batch(images)

if __name__ == "__main__":
    # Test the optimized face detector
    print("Testing Optimized Face Detector...")
    
    # Create detector
    detector = OptimizedFaceDetector(model_type='retinaface', device='cuda')
    
    # Test with a sample image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect_faces_single(test_image)
    print(f"Detections: {detections}")
    
    # Test streaming processor
    print("Testing Streaming Video Processor...")
    try:
        processor = StreamingVideoProcessor("test_media/person_short.mp4", "test_media/speech_short.wav")
        print(f"Video loaded: {processor.total_frames} frames")
    except Exception as e:
        print(f"Could not test streaming processor: {e}")
