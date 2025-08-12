#!/usr/bin/env python3
"""
Setup script for Diff2Lip dependencies
Installs the guided-diffusion package and checks all dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def main():
    """Setup Diff2Lip environment"""
    print("🚀 Setting up Diff2Lip environment")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("guided-diffusion"):
        print("❌ guided-diffusion directory not found!")
        print("   Make sure you're in the diff2lip root directory")
        sys.exit(1)
    
    # Install main requirements
    print("\n📦 Installing main requirements...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("⚠️  Some requirements may have failed to install")
    
    # Install guided-diffusion package
    print("\n🧠 Installing guided-diffusion package...")
    guided_diffusion_path = Path("guided-diffusion")
    if guided_diffusion_path.exists():
        original_dir = os.getcwd()
        try:
            os.chdir(guided_diffusion_path)
            if run_command("pip install -e .", "Installing guided-diffusion"):
                print("✅ guided-diffusion installed successfully")
            else:
                print("❌ Failed to install guided-diffusion")
                return False
        finally:
            os.chdir(original_dir)
    else:
        print("❌ guided-diffusion directory not found")
        return False
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "import torch",
        "import cv2",
        "import numpy as np",
        "import yaml",
        "from audio import audio",
        "import face_detection",
        "from guided_diffusion import dist_util",
        "from guided_diffusion.script_util import tfg_model_and_diffusion_defaults"
    ]
    
    failed_imports = []
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"✅ {import_stmt}")
        except ImportError as e:
            print(f"❌ {import_stmt} - {e}")
            failed_imports.append(import_stmt)
    
    # Summary
    print("\n" + "=" * 50)
    if failed_imports:
        print("⚠️  Setup completed with some issues:")
        for failed in failed_imports:
            print(f"   - {failed}")
        print("\n💡 Try running the failed imports manually to debug")
    else:
        print("🎉 Setup completed successfully!")
        print("\n🚀 You can now run:")
        print("   python inference.py")
    
    # Check for checkpoint
    if not os.path.exists("checkpoints/checkpoint.pt"):
        print("\n⚠️  Model checkpoint not found!")
        print("   Please download the Diff2Lip checkpoint to:")
        print("   checkpoints/checkpoint.pt")
        print("\n📥 Download from: https://drive.google.com/drive/folders/1UMiHAhVf5M_CKzjVQFC5jkz-IXAAnFo5")
    
    return len(failed_imports) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
