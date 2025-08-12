# Configurable Diff2Lip Inference Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with default settings:**
   ```bash
   python inference.py
   ```

3. **Run with custom files:**
   ```bash
   python inference.py --video path/to/video.mp4 --audio path/to/audio.wav --output path/to/result.mp4
   ```

4. **Use custom configuration:**
   ```bash
   python inference.py --config my_config.yaml
   ```

## Configuration System

The `inference_config.yaml` file controls all aspects of processing. Key sections:

### Quality Settings (Phase 1 Enhancements)
```yaml
quality:
  preset: 'high'                    # fast, balanced, high, ultra
  use_gradient_mask: true           # Smooth blending
  sharpening_strength: 0.3          # Detail enhancement
```

### Processing Configuration
```yaml
processing:
  batch_size: 4                     # Frames processed simultaneously
  video_fps: 25                     # Target frame rate
```

### Enhancement Features
```yaml
# Phase 2: Resolution
model:
  image_size: 256                   # Higher resolution processing

# Phase 4: Temporal consistency  
quality:
  temporal_smoothing: true          # Frame-to-frame smoothing
  
# Phase 5: Audio enhancement
audio:
  enhanced_processing: true         # Higher quality audio
  num_mels: 128                     # More audio channels
```

## Quality Presets

| Preset | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| `fast` | 2.5x faster | Good | Testing, previews |
| `balanced` | Normal | Very Good | Default production |
| `high` | 2x slower | Excellent | High-quality output |
| `ultra` | 4x slower | Maximum | Final production |

## Phase-by-Phase Enhancement

### Phase 1: Foundation (Ready to use)
```yaml
quality:
  use_gradient_mask: true           # Smooth mask blending
  blur_kernel_size: 15              # Mask smoothing strength
  sharpening_strength: 0.3          # Detail enhancement
```

### Phase 2: Resolution (Partially ready)
```yaml
model:
  image_size: 256                   # Higher resolution faces
quality:
  super_resolution: true            # Coming soon
```

### Phase 3: Advanced Masking (Coming soon)
```yaml
masking:
  use_landmark_masking: true        # Precise lip boundaries
  adaptive_masking: true            # Expression-aware masks
```

### Phase 4: Temporal Consistency (Basic ready)
```yaml
quality:
  temporal_smoothing: true          # Frame smoothing
  smoothing_factor: 0.3             # Smoothing strength
```

### Phase 5: Audio Enhancement (Framework ready)
```yaml
audio:
  enhanced_processing: true         # Higher quality audio
  noise_reduction: true             # Audio cleaning
```

## Performance Optimization

### Memory Management
```yaml
processing:
  batch_size: 4                     # Adjust based on GPU memory
optimization:
  memory_cleanup_interval: 5        # Cleanup frequency
```

### Speed Optimization
```yaml
optimization:
  torch_compile: true               # ~1.4x speedup (requires Triton)
  enable_cuda_optimizations: true   # CUDA performance boost
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` from 4 to 2 or 1
- Increase `memory_cleanup_interval` to 3
- Disable `torch_compile` if using limited memory

### Slow Processing
- Use `preset: 'fast'` for testing
- Reduce `image_size` to 128
- Disable enhancement features for baseline speed

### Quality Issues
- Increase `preset` to 'high' or 'ultra'
- Enable `use_gradient_mask: true`
- Add `sharpening_strength: 0.3`
- Increase `blur_kernel_size` for smoother blending

## Example Configurations

### High-Speed Testing
```yaml
quality:
  preset: 'fast'
processing:
  batch_size: 8
optimization:
  torch_compile: true
```

### Maximum Quality
```yaml
quality:
  preset: 'ultra'
  use_gradient_mask: true
  sharpening_strength: 0.5
model:
  image_size: 256
  num_channels: 256
```

### Balanced Production
```yaml
quality:
  preset: 'balanced'
  use_gradient_mask: true
  sharpening_strength: 0.3
processing:
  batch_size: 4
```

## Integration with Quality Plan

This configuration system is designed to support the systematic implementation of enhancements from `quality_plan.md`:

- **Phase 1** features are ready to use
- **Phase 2-6** features have configuration placeholders
- Each phase can be enabled/disabled independently
- Backward compatibility is maintained throughout

## Monitoring and Debugging

The script provides detailed logging:
- Resource usage monitoring
- Processing time per frame
- Success rate tracking
- GPU memory monitoring
- Quality preset confirmation

Use these metrics to optimize your configuration for your specific hardware and quality requirements.
