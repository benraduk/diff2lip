# Diff2Lip Quality Enhancement Plan

## Overview
This document outlines a systematic approach to enhance the quality of Diff2Lip outputs through incremental improvements. Each phase builds upon previous work while maintaining system stability and allowing for thorough testing.

## Phase 1: Foundation Improvements (Week 1-2)
*Low risk, high impact changes to establish quality baseline*

### 1.1 Enhanced Masking & Blending
**Objective**: Replace hard masking with gradient-based blending for smoother integration

**Current State**:
```python
# Hard mask in tfg_data_util.py line 58
mask[:,:,mask_start_idx:,:]=1.
```

**Implementation Steps**:
1. **Create gradient mask function** in `guided-diffusion/tfg_data_util.py`
   ```python
   def create_gradient_mask(B, H, W, face_hide_percentage, blur_kernel_size=15):
       mask = torch.zeros(B, 1, H, W)
       mask_start_idx = int(H * (1 - face_hide_percentage))
       mask[:, :, mask_start_idx:, :] = 1.0
       
       # Apply Gaussian blur for smooth transitions
       mask = gaussian_blur(mask, kernel_size=blur_kernel_size)
       return mask
   ```

2. **Add blur kernel size parameter** to argument parser in `generate.py`
   ```python
   # In create_argparser() defaults dict
   blur_kernel_size = 15  # Adjustable smoothing parameter
   ```

3. **Test with different blur values**: 5, 10, 15, 25 pixels
4. **Validation**: Compare edge artifacts before/after on test videos

**Success Metrics**:
- [ ] Reduced visible seam artifacts around mouth region
- [ ] Smooth color transitions at mask boundaries
- [ ] No performance degradation (< 5% slower)

### 1.2 Diffusion Quality Boost
**Objective**: Increase diffusion sampling steps for higher quality without major architectural changes

**Current State**:
```bash
# scripts/inference.sh line 31
--timestep_respacing ddim25
```

**Implementation Steps**:
1. **Create quality presets** in `generate.py`
   ```python
   QUALITY_PRESETS = {
       'fast': 'ddim10',      # Current optimized
       'balanced': 'ddim25',   # Current default
       'high': 'ddim50',      # Enhanced quality
       'ultra': 'ddim100'     # Maximum quality
   }
   ```

2. **Add quality parameter** to argument parser
   ```python
   quality_preset = 'balanced'  # Default preset
   ```

3. **Benchmark each preset** on identical test videos
4. **Document quality vs speed tradeoffs**

**Success Metrics**:
- [ ] Measurable quality improvement in lip detail
- [ ] Documented performance impact for each preset
- [ ] User-selectable quality levels

### 1.3 Post-Processing Sharpening
**Objective**: Add detail enhancement to generated lip regions

**Implementation Steps**:
1. **Create sharpening function** in new file `post_processing.py`
   ```python
   def enhance_lip_detail(generated_face, sharpening_strength=0.3):
       # Unsharp masking for detail enhancement
       blurred = cv2.GaussianBlur(generated_face, (0, 0), 1.0)
       sharpened = cv2.addWeighted(generated_face, 1.0 + sharpening_strength, 
                                 blurred, -sharpening_strength, 0)
       return sharpened
   ```

2. **Integrate into inference pipeline** in `generate.py` line 273
   ```python
   # After g = cv2.resize(g.astype(np.uint8), (x2 - x1, y2 - y1))
   g = enhance_lip_detail(g, args.sharpening_strength)
   ```

3. **Add parameter** to argument parser
   ```python
   sharpening_strength = 0.3  # 0.0 = no sharpening, 1.0 = maximum
   ```

**Success Metrics**:
- [ ] Increased perceived sharpness in lip region
- [ ] No over-sharpening artifacts
- [ ] Configurable enhancement level

## Phase 2: Resolution Enhancement (Week 3-4)
*Moderate risk improvements focusing on output resolution*

### 2.1 Model Resolution Scaling
**Objective**: Support higher resolution face processing (256x256)

**Current State**:
```python
# generate.py line 364
image_size=128
```

**Implementation Steps**:
1. **Verify model support** for 256x256 in `guided-diffusion/script_util.py`
   ```python
   # Line 571: channel_mult configurations already support 256x256
   elif image_size == 256:
       channel_mult = (1, 1, 2, 3, 4, 4)
   ```

2. **Create resolution-adaptive inference**
   ```python
   def determine_optimal_resolution(face_bbox):
       face_width = face_bbox[2] - face_bbox[0]
       if face_width > 200:
           return 256
       return 128
   ```

3. **Update face detection pipeline** in `generate.py`
   ```python
   # Adaptive resizing based on original face size
   optimal_size = determine_optimal_resolution(face_bbox)
   face_resized = cv2.resize(face_crop, (optimal_size, optimal_size))
   ```

4. **Test memory impact** and adjust batch sizes accordingly

**Success Metrics**:
- [ ] Support for 256x256 processing on high-resolution inputs
- [ ] Automatic resolution selection based on input quality
- [ ] Memory usage remains manageable

### 2.2 Super-Resolution Integration
**Objective**: Add post-processing super-resolution for enhanced detail

**Implementation Steps**:
1. **Install super-resolution dependencies**
   ```bash
   pip install realesrgan opencv-python-headless
   ```

2. **Create super-resolution module** `super_resolution.py`
   ```python
   from realesrgan import RealESRGANer
   
   class LipSuperResolver:
       def __init__(self, model_name='RealESRGAN_x2plus'):
           self.upsampler = RealESRGANer(
               scale=2,
               model_path=f'weights/{model_name}.pth',
               model=RealESRGAN_x2plus()
           )
       
       def enhance_lip_region(self, face_crop, lip_bbox):
           # Extract lip region, super-resolve, blend back
           pass
   ```

3. **Integrate selectively** - only apply to lip region, not entire face
4. **Add toggle parameter** for optional super-resolution

**Success Metrics**:
- [ ] 2x resolution improvement in lip detail
- [ ] Minimal processing time increase (< 50ms per frame)
- [ ] Optional feature that can be disabled

## Phase 3: Advanced Masking (Week 5-6)
*Higher complexity improvements for precise facial region control*

### 3.1 Landmark-Based Masking
**Objective**: Replace rectangular masking with precise lip boundary detection

**Implementation Steps**:
1. **Add facial landmark detection** using existing face detection infrastructure
   ```python
   # Extend face_detection/api.py
   def get_lip_landmarks(self, face_crop):
       # Use existing FAN model for 68-point landmarks
       landmarks = self.get_landmarks_from_image(face_crop)
       lip_points = landmarks[48:68]  # Standard lip landmark indices
       return lip_points
   ```

2. **Create lip-specific mask generation**
   ```python
   def create_lip_mask(face_shape, lip_landmarks, expansion_factor=1.2):
       # Create polygon mask from lip landmarks
       lip_polygon = cv2.convexHull(lip_landmarks)
       mask = cv2.fillPoly(np.zeros(face_shape), [lip_polygon], 1.0)
       
       # Expand mask slightly and blur edges
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                        (int(10*expansion_factor), int(10*expansion_factor)))
       mask = cv2.dilate(mask, kernel)
       mask = cv2.GaussianBlur(mask, (15, 15), 0)
       return mask
   ```

3. **Integrate into TFG model** in `guided-diffusion/tfg_data_util.py`
4. **Add expansion factor parameter** for mask size control

**Success Metrics**:
- [ ] Precise lip-only masking instead of rectangular regions
- [ ] Reduced artifacts on non-lip facial areas
- [ ] Configurable mask expansion for different expressions

### 3.2 Dynamic Mask Adaptation
**Objective**: Adjust mask boundaries based on mouth opening and expression

**Implementation Steps**:
1. **Implement mouth opening detection**
   ```python
   def calculate_mouth_opening(lip_landmarks):
       # Distance between upper and lower lip centers
       upper_lip = np.mean(lip_landmarks[13:16], axis=0)
       lower_lip = np.mean(lip_landmarks[19:22], axis=0)
       opening_ratio = np.linalg.norm(upper_lip - lower_lip)
       return opening_ratio
   ```

2. **Create adaptive mask sizing**
   ```python
   def adaptive_mask_size(base_expansion, mouth_opening):
       # Larger masks for open mouths, smaller for closed
       adaptive_factor = base_expansion * (0.8 + 0.4 * mouth_opening)
       return adaptive_factor
   ```

3. **Test on diverse expressions** (smiling, talking, neutral)

**Success Metrics**:
- [ ] Better mask fitting for different mouth positions
- [ ] Reduced over-generation on closed mouths
- [ ] Improved quality on wide-open expressions

## Phase 4: Temporal Consistency (Week 7-8)
*Advanced improvements for video coherence*

### 4.1 Frame-to-Frame Smoothing
**Objective**: Reduce temporal flickering and improve video consistency

**Implementation Steps**:
1. **Create temporal smoothing module** `temporal_consistency.py`
   ```python
   class TemporalSmoother:
       def __init__(self, smoothing_factor=0.3):
           self.previous_frame = None
           self.smoothing_factor = smoothing_factor
       
       def smooth_transition(self, current_frame, mask):
           if self.previous_frame is not None:
               # Weighted blend with previous frame
               smoothed = (current_frame * (1 - self.smoothing_factor) + 
                          self.previous_frame * self.smoothing_factor)
               self.previous_frame = current_frame.copy()
               return smoothed
           else:
               self.previous_frame = current_frame.copy()
               return current_frame
   ```

2. **Integrate into video processing loop** in `generate.py`
3. **Add temporal consistency parameters**

**Success Metrics**:
- [ ] Reduced frame-to-frame flickering
- [ ] Smoother lip movement transitions
- [ ] Configurable temporal smoothing strength

### 4.2 Optical Flow Guidance
**Objective**: Use motion vectors to improve temporal consistency

**Implementation Steps**:
1. **Add optical flow calculation** between consecutive frames
2. **Use flow vectors** to guide diffusion generation
3. **Implement motion-aware masking** that follows lip movement

**Success Metrics**:
- [ ] Better preservation of natural lip motion
- [ ] Reduced artifacts during rapid head movement
- [ ] Improved sync during dynamic scenes

## Phase 5: Audio Enhancement (Week 9-10)
*Improvements to audio processing for better lip-sync accuracy*

### 5.1 Higher Resolution Audio Processing
**Objective**: Improve audio representation for better lip synchronization

**Current State**:
```python
# audio/hparams.py
num_mels=80
sample_rate=16000
```

**Implementation Steps**:
1. **Create enhanced audio preset**
   ```python
   ENHANCED_AUDIO_CONFIG = {
       'num_mels': 128,        # Increased from 80
       'sample_rate': 22050,   # Increased from 16000
       'n_fft': 1024,         # Increased from 800
       'hop_size': 256,       # Adjusted proportionally
   }
   ```

2. **Add audio quality parameter** to inference scripts
3. **Test impact on lip-sync accuracy** with A/B comparisons
4. **Ensure backward compatibility** with existing models

**Success Metrics**:
- [ ] Improved lip-sync accuracy on test videos
- [ ] Support for both standard and enhanced audio processing
- [ ] Documented quality improvements

### 5.2 Audio Preprocessing Enhancement
**Objective**: Add spectral enhancement and noise reduction

**Implementation Steps**:
1. **Add spectral subtraction** for noise reduction
2. **Implement dynamic range compression** for consistent levels
3. **Add audio quality assessment** metrics

**Success Metrics**:
- [ ] Better performance on noisy audio inputs
- [ ] More consistent results across different audio qualities
- [ ] Measurable improvement in sync accuracy

## Phase 6: Advanced Post-Processing (Week 11-12)
*Final polish and optimization*

### 6.1 Color Correction and Matching
**Objective**: Match generated regions to original video color grading

**Implementation Steps**:
1. **Implement color histogram matching**
2. **Add white balance correction**
3. **Create color consistency metrics**

### 6.2 Artifact Reduction
**Objective**: Remove diffusion artifacts and enhance realism

**Implementation Steps**:
1. **Add denoising post-processing**
2. **Implement artifact detection**
3. **Create quality assessment pipeline**

## Implementation Guidelines

### Development Workflow
1. **Create feature branch** for each phase
2. **Implement changes incrementally** with git commits for each step
3. **Test thoroughly** before moving to next improvement
4. **Document performance impact** of each change
5. **Maintain backward compatibility** with existing configurations

### Testing Protocol
1. **Establish baseline metrics** with current system
2. **Use consistent test dataset** across all improvements
3. **Measure both objective and subjective quality**
4. **Document computational overhead** for each enhancement
5. **Create A/B comparison videos** for visual validation

### Quality Metrics
- **Technical**: PSNR, SSIM, LPIPS for image quality
- **Perceptual**: User studies, lip-sync accuracy scores
- **Performance**: Processing time, memory usage, GPU utilization
- **Temporal**: Frame consistency, flicker reduction

### Rollback Strategy
- **Modular implementation** allows disabling individual features
- **Configuration flags** for each enhancement
- **Performance monitoring** to detect regressions
- **Automated testing** to catch integration issues

## Success Criteria

### Phase 1 Success (Foundation)
- [ ] 20% reduction in visible artifacts
- [ ] User-selectable quality presets
- [ ] Enhanced detail in lip regions

### Phase 2 Success (Resolution)
- [ ] Support for 2x resolution improvement
- [ ] Adaptive resolution selection
- [ ] Optional super-resolution enhancement

### Phase 3 Success (Masking)
- [ ] Precise lip-only generation
- [ ] Expression-adaptive masking
- [ ] Reduced non-lip artifacts

### Phase 4 Success (Temporal)
- [ ] 50% reduction in temporal flickering
- [ ] Smooth motion preservation
- [ ] Improved video coherence

### Phase 5 Success (Audio)
- [ ] Enhanced lip-sync accuracy
- [ ] Better noise robustness
- [ ] Improved audio quality handling

### Phase 6 Success (Polish)
- [ ] Professional-quality output
- [ ] Minimal visible artifacts
- [ ] Optimized performance pipeline

## Maintenance and Monitoring

### Continuous Integration
- **Automated testing** on each commit
- **Performance regression detection**
- **Quality metric tracking**
- **Memory usage monitoring**

### Documentation
- **Parameter documentation** for all new features
- **Performance benchmarks** for each enhancement
- **User guides** for quality settings
- **Troubleshooting guides** for common issues

### Future Considerations
- **Model architecture updates** for native high-resolution support
- **Real-time processing** optimizations
- **Mobile deployment** considerations
- **Multi-language audio** support

This systematic approach ensures each improvement builds upon previous work while maintaining system stability and allowing for thorough validation at each step.
