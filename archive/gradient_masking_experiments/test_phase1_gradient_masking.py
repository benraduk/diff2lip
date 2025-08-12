#!/usr/bin/env python3
"""
Phase 1.1 Testing: Gradient Masking Validation
Tests the gradient masking implementation against hard masking
"""

import os
import yaml
import time
import shutil
from pathlib import Path

def create_test_configs():
    """Create test configurations for A/B testing"""
    
    test_configs = {
        'hard_masking': {
            'quality': {
                'preset': 'fast',  # Use fast for quicker testing
                'use_gradient_mask': False,
                'blur_kernel_size': 0,
                'sharpening_strength': 0.0
            },
            'processing': {'batch_size': 2},  # Small batch for testing
            'paths': {
                'input_video': 'test_media/speech.m4a',  # Will need to find a test video
                'input_audio': 'test_media/speech.m4a',
                'output_video': 'output_dir/test_hard_masking.mp4'
            }
        },
        
        'gradient_blur_5': {
            'quality': {
                'preset': 'fast',
                'use_gradient_mask': True,
                'blur_kernel_size': 5,
                'sharpening_strength': 0.0
            },
            'processing': {'batch_size': 2},
            'paths': {
                'input_video': 'test_media/speech.m4a',
                'input_audio': 'test_media/speech.m4a', 
                'output_video': 'output_dir/test_gradient_blur5.mp4'
            }
        },
        
        'gradient_blur_15': {
            'quality': {
                'preset': 'fast',
                'use_gradient_mask': True,
                'blur_kernel_size': 15,
                'sharpening_strength': 0.0
            },
            'processing': {'batch_size': 2},
            'paths': {
                'input_video': 'test_media/speech.m4a',
                'input_audio': 'test_media/speech.m4a',
                'output_video': 'output_dir/test_gradient_blur15.mp4'
            }
        },
        
        'gradient_blur_25': {
            'quality': {
                'preset': 'fast',
                'use_gradient_mask': True,
                'blur_kernel_size': 25,
                'sharpening_strength': 0.0
            },
            'processing': {'batch_size': 2},
            'paths': {
                'input_video': 'test_media/speech.m4a',
                'input_audio': 'test_media/speech.m4a',
                'output_video': 'output_dir/test_gradient_blur25.mp4'
            }
        }
    }
    
    return test_configs

def create_base_config():
    """Create base configuration template"""
    base_config = {
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
            'preset': 'fast',
            'timestep_respacing': 'ddim10',
            'face_hide_percentage': 0.5,
            'use_gradient_mask': False,
            'blur_kernel_size': 15,
            'sharpening_strength': 0.0,
            'super_resolution': False,
            'temporal_smoothing': False,
            'smoothing_factor': 0.3
        },
        'processing': {
            'batch_size': 2,
            'use_ddim': True,
            'face_det_batch_size': 64,
            'pads': [0, 0, 0, 0],
            'video_fps': 25,
            'sample_rate': 16000,
            'syncnet_mel_step_size': 16
        },
        'audio': {
            'enhanced_processing': False,
            'num_mels': 80,
            'n_fft': 800,
            'hop_size': 200,
            'noise_reduction': False,
            'dynamic_range_compression': False
        },
        'masking': {
            'use_landmark_masking': False,
            'adaptive_masking': False,
            'expansion_factor': 1.2,
            'mask_blur_strength': 15
        },
        'post_processing': {
            'color_correction': False,
            'artifact_reduction': False,
            'detail_enhancement': False
        },
        'optimization': {
            'enable_cuda_optimizations': True,
            'torch_compile': False,  # Disable for testing to avoid Triton issues
            'memory_cleanup_interval': 5,
            'gpu_memory_monitoring': True
        },
        'paths': {
            'input_video': 'test_media/person.mp4',
            'input_audio': 'test_media/speech.m4a',
            'output_video': 'output_dir/result_test.mp4',
            'temp_dir': 'temp'
        }
    }
    return base_config

def merge_configs(base_config, test_config):
    """Merge test configuration with base configuration"""
    merged = base_config.copy()
    
    for section, values in test_config.items():
        if section in merged:
            merged[section].update(values)
        else:
            merged[section] = values
    
    return merged

def run_test_configuration(config_name, config, test_video_path=None):
    """Run a single test configuration"""
    print(f"\n🧪 Testing {config_name}")
    print("=" * 50)
    
    # Create config file
    config_path = f"test_config_{config_name}.yaml"
    
    # Update video path if provided
    if test_video_path and os.path.exists(test_video_path):
        config['paths']['input_video'] = test_video_path
        print(f"📹 Using test video: {test_video_path}")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"📝 Config created: {config_path}")
    print(f"🎯 Gradient masking: {config['quality']['use_gradient_mask']}")
    print(f"🔵 Blur kernel size: {config['quality']['blur_kernel_size']}")
    
    # Record start time
    start_time = time.time()
    
    # Run inference
    try:
        import subprocess
        result = subprocess.run([
            'python', 'inference.py', 
            '--config', config_path
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        processing_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Test completed successfully!")
            print(f"⏱️  Processing time: {processing_time:.1f}s")
            print(f"📁 Output: {config['paths']['output_video']}")
            
            # Check if output file exists
            if os.path.exists(config['paths']['output_video']):
                file_size = os.path.getsize(config['paths']['output_video']) / (1024*1024)
                print(f"📦 Output size: {file_size:.1f}MB")
            else:
                print("⚠️  Output file not found")
                
        else:
            print(f"❌ Test failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Test timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Test error: {e}")
    
    # Cleanup config file
    if os.path.exists(config_path):
        os.remove(config_path)
    
    return {
        'config_name': config_name,
        'success': result.returncode == 0 if 'result' in locals() else False,
        'processing_time': processing_time if 'processing_time' in locals() else None,
        'output_path': config['paths']['output_video']
    }

def find_test_video():
    """Find a suitable test video file"""
    possible_paths = [
        'test_media/person.mp4',
        'test_media/test_video.mp4',
        'test_media/sample.mp4',
        'dataset/test_video.mp4'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📹 Found test video: {path}")
            return path
    
    print("⚠️  No test video found. Please ensure you have a test video file.")
    print("   Suggested paths: test_media/person.mp4")
    return None

def main():
    """Main test function"""
    print("🚀 Phase 1.1: Gradient Masking A/B Testing")
    print("=" * 60)
    
    # Check if we're in the right environment
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"🐍 Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
    else:
        print("⚠️  Conda environment not detected. Please run: conda activate diff2lip")
    
    # Find test video
    test_video_path = find_test_video()
    if not test_video_path:
        print("❌ Cannot proceed without test video")
        return
    
    # Create output directory
    os.makedirs('output_dir', exist_ok=True)
    
    # Create test configurations
    base_config = create_base_config()
    test_configs = create_test_configs()
    
    # Run tests
    results = []
    
    for config_name, test_config in test_configs.items():
        merged_config = merge_configs(base_config, test_config)
        result = run_test_configuration(config_name, merged_config, test_video_path)
        results.append(result)
        
        # Small delay between tests
        time.sleep(2)
    
    # Print summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    successful_tests = 0
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        time_str = f"{result['processing_time']:.1f}s" if result['processing_time'] else "N/A"
        print(f"{result['config_name']:20} | {status:10} | {time_str:8} | {result['output_path']}")
        
        if result['success']:
            successful_tests += 1
    
    print(f"\n🎯 Success rate: {successful_tests}/{len(results)} ({successful_tests/len(results)*100:.1f}%)")
    
    if successful_tests > 0:
        print("\n✅ Gradient masking integration successful!")
        print("💡 Next steps:")
        print("   1. Compare output videos visually")
        print("   2. Look for reduced seam artifacts in gradient masked versions")
        print("   3. Proceed to Phase 1.2 (Quality Presets) and Phase 1.3 (Sharpening)")
    else:
        print("\n❌ Tests failed. Please check:")
        print("   1. Model checkpoint exists at checkpoints/checkpoint.pt")
        print("   2. All dependencies are installed")
        print("   3. CUDA is available if using GPU")

if __name__ == "__main__":
    main()
