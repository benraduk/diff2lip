#!/usr/bin/env python3
"""
Download FaceFusion models for optimized face detection
"""

import os
import urllib.request
from pathlib import Path

# Model URLs (these are examples - you may need to update with actual URLs)
MODELS = {
    'retinaface_10g.onnx': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx',
        'size_mb': 16.9
    },
    'scrfd_2.5g.onnx': {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.onnx', 
        'size_mb': 2.5
    }
}

def download_file(url: str, filepath: str, expected_size_mb: float = None):
    """Download a file with progress"""
    print(f"Downloading {os.path.basename(filepath)}...")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print()  # New line after progress
        
        # Check file size
        actual_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Downloaded: {actual_size_mb:.1f}MB")
        
        if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 1:
            print(f"⚠️  Warning: Expected {expected_size_mb}MB, got {actual_size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filepath}: {e}")
        return False

def main():
    """Download FaceFusion models"""
    print("FaceFusion Model Downloader")
    print("=" * 40)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print(f"Downloading models to: {models_dir.absolute()}")
    print()
    
    success_count = 0
    total_count = len(MODELS)
    
    for model_name, model_info in MODELS.items():
        model_path = models_dir / model_name
        
        # Skip if already exists
        if model_path.exists():
            existing_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_name} already exists ({existing_size_mb:.1f}MB)")
            success_count += 1
            continue
        
        # Download model
        print(f"📥 Downloading {model_name} ({model_info['size_mb']}MB)...")
        
        if download_file(model_info['url'], str(model_path), model_info['size_mb']):
            print(f"✅ {model_name} downloaded successfully")
            success_count += 1
        else:
            print(f"❌ Failed to download {model_name}")
        
        print()
    
    print("=" * 40)
    print(f"Download complete: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print()
        print("You can now use the optimized face detection with:")
        print("  python inference_optimized.py")
    else:
        print("⚠️  Some models failed to download.")
        print("You can still use the original S3FD detector as fallback.")

if __name__ == "__main__":
    main()
