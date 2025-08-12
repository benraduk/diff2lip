#!/usr/bin/env python3
"""
Test Script for Face Enhancement Integration
Tests the face enhancement functionality with diff2lip pipeline
"""

import os
import sys
import cv2
import numpy as np
import torch
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_face_enhancer_standalone():
    """Test the face enhancer module standalone"""
    print("🧪 Testing Face Enhancer Standalone")
    print("=" * 50)
    
    try:
        from face_enhancer import create_face_enhancer
        
        # Test configuration
        test_config = {
            'enabled': True,
            'model': 'gfpgan_1.4',
            'blend': 80,
            'weight': 1.0
        }
        
        # Create enhancer
        enhancer = create_face_enhancer(test_config)
        print(f"✅ Face enhancer created successfully")
        
        # Test with dummy image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        start_time = time.time()
        enhanced = enhancer.enhance_face(test_image)
        processing_time = time.time() - start_time
        
        print(f"✅ Face enhancement test completed")
        print(f"📊 Input shape: {test_image.shape}")
        print(f"📊 Output shape: {enhanced.shape}")
        print(f"⏱️  Processing time: {processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration Loading")
    print("=" * 50)
    
    try:
        import yaml
        
        # Test loading the face enhancement config
        with open('test_face_enhancement_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check face enhancement section
        face_config = config.get('face_enhancement', {})
        print(f"✅ Configuration loaded successfully")
        print(f"📊 Face enhancement enabled: {face_config.get('enabled', False)}")
        print(f"📊 Model: {face_config.get('model', 'unknown')}")
        print(f"📊 Blend: {face_config.get('blend', 0)}")
        print(f"📊 Weight: {face_config.get('weight', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        return False

def test_inference_integration():
    """Test inference.py integration (dry run)"""
    print("\n🧪 Testing Inference Integration")
    print("=" * 50)
    
    try:
        # Set environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Import inference module
        from inference import ConfigurableDiff2LipProcessor
        
        # Initialize processor with test config
        processor = ConfigurableDiff2LipProcessor('test_face_enhancement_config.yaml')
        
        print(f"✅ Processor initialized successfully")
        print(f"📊 Face enhancer available: {processor.face_enhancer is not None}")
        
        if processor.face_enhancer:
            print(f"📊 Face enhancer enabled: {processor.face_enhancer.enabled}")
            print(f"📊 Face enhancer model: {processor.face_enhancer.model_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n🧪 Checking Dependencies")
    print("=" * 50)
    
    dependencies = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'pytorch',
        'yaml': 'pyyaml'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing.append(package)
    
    # Check optional dependencies
    optional_deps = {
        'onnxruntime': 'onnxruntime'
    }
    
    for module, package in optional_deps.items():
        try:
            __import__(module)
            print(f"✅ {package}: Available (optional)")
        except ImportError:
            print(f"⚠️  {package}: Missing (optional)")
    
    if missing:
        print(f"\n❌ Missing required dependencies: {', '.join(missing)}")
        return False
    else:
        print(f"\n✅ All required dependencies available")
        return True

def check_test_files():
    """Check if test files exist"""
    print("\n🧪 Checking Test Files")
    print("=" * 50)
    
    test_files = [
        'test_media/person_short.mp4',
        'test_media/speech_short.wav',
        'checkpoints/checkpoint.pt'
    ]
    
    missing_files = []
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Available")
        else:
            print(f"❌ {file_path}: Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing test files: {', '.join(missing_files)}")
        print(f"💡 You can still test with placeholder data")
        return False
    else:
        print(f"\n✅ All test files available")
        return True

def main():
    """Run all tests"""
    print("🚀 Face Enhancement Integration Test Suite")
    print("=" * 70)
    
    # Run tests
    tests = [
        ("Dependencies", check_dependencies),
        ("Test Files", check_test_files),
        ("Face Enhancer Standalone", test_face_enhancer_standalone),
        ("Configuration Loading", test_config_loading),
        ("Inference Integration", test_inference_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("🏁 Test Results Summary")
    print("=" * 70)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for integration testing.")
    elif passed >= total - 1:
        print("⚠️  Almost ready - check failed tests above.")
    else:
        print("❌ Multiple issues found - please fix before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
