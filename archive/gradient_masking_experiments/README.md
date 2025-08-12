# Gradient Masking Experiments Archive

This folder contains all the experimental code and configurations from our Phase 1.1 gradient masking attempts.

## What Was Tried

### 1. Gaussian Blur-Based Masking
- **File**: `debug_gradient_mask.py`
- **Approach**: Applied `cv2.GaussianBlur` to hard masks
- **Result**: ❌ Caused static/noise in untouched regions
- **Issue**: Blur spread into areas that should remain unchanged

### 2. Sigmoid-Based Directional Gradient  
- **Implementation**: Used sigmoid function for smooth transitions
- **Result**: ❌ Created sharper edges than hard masking
- **Issue**: Despite mathematical smoothness, visually worse than hard boundaries

### 3. Linear Gradient Transitions
- **Approach**: Row-by-row linear interpolation in transition zones
- **Result**: ❌ Still inferior to hard masking
- **Issue**: Smooth gradients don't necessarily mean better visual results

### 4. Cosine-Based Feathered Edges
- **Approach**: Professional compositing-style feathered transitions
- **Result**: ❌ Implementation errors and still worse than hard masking
- **Issue**: Complex approaches don't always yield better results

## Key Files

- `debug_gradient_mask.py` - Visualization and analysis of gradient masks
- `debug_mask_comparison.py` - Comprehensive comparison of all approaches  
- `quick_gradient_test.py` - Fast validation testing
- `test_*gradient*.yaml` - Various test configurations
- `test_256_resolution.yaml` - Failed 256x256 resolution attempt
- `debug_output/` - Visual outputs from testing

## Lessons Learned

1. **Hard masking is superior** - Despite theoretical advantages of smooth transitions, hard boundaries produced cleaner, artifact-free results
2. **User feedback is crucial** - Technical metrics don't always align with visual quality perception
3. **Simple can be better** - Complex gradient approaches consistently underperformed the simple hard mask
4. **Checkpoint compatibility matters** - Cannot arbitrarily change model architecture without compatible weights

## Final Decision

**Abandoned gradient masking in favor of hard masking** after extensive testing proved hard boundaries are visually superior for this specific use case.

This archive preserves the experimental work for future reference and demonstrates the thorough testing approach used in the quality enhancement project.
