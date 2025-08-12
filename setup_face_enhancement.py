#!/usr/bin/env python3
"""
Setup script for Face Enhancement Integration
Installs dependencies and prepares environment
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a command with error handling"""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        
        print(f"✅ {description} completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr if e.stderr else str(e)}")
        return False

def check_conda_environment():
    """Check if we're in the correct conda environment"""
    print("🔍 Checking conda environment")
    
    # Check if conda is available
    if not shutil.which('conda'):
        print("❌ Conda not found - please install Anaconda/Miniconda")
        return False
    
    # Check current environment
    try:
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True, check=True)
        current_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        
        print(f"📊 Current environment: {current_env}")
        
        if 'diff2lip' in result.stdout:
            print("✅ diff2lip environment found")
            if current_env != 'diff2lip':
                print("⚠️  Please activate diff2lip environment:")
                print("   conda activate diff2lip")
                return False
            return True
        else:
            print("❌ diff2lip environment not found")
            print("💡 Please create the environment first")
            return False
            
    except Exception as e:
        print(f"❌ Error checking conda environment: {e}")
        return False

def install_dependencies():
    """Install face enhancement dependencies"""
    print("\n🔧 Installing face enhancement dependencies")
    
    # Check if we have pip
    if not shutil.which('pip'):
        print("❌ pip not found")
        return False
    
    # Install dependencies
    deps_to_install = [
        'onnxruntime-gpu>=1.16.0',
        'opencv-python>=4.8.0',
        'pillow>=8.0.0'
    ]
    
    for dep in deps_to_install:
        success = run_command(
            ['pip', 'install', dep],
            f"Installing {dep}",
            check=False  # Don't fail if one package fails
        )
        if not success:
            print(f"⚠️  Failed to install {dep} - continuing...")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n🔧 Creating directories")
    
    directories = [
        'models/face_enhancement',
        'output_dir',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def check_test_files():
    """Check and suggest test files"""
    print("\n🔍 Checking test files")
    
    test_files = {
        'test_media/person_short.mp4': 'Short video file for testing (3-5 seconds)',
        'test_media/speech_short.wav': 'Short audio file for testing (3-5 seconds)',
        'checkpoints/checkpoint.pt': 'Diff2Lip model checkpoint'
    }
    
    missing = []
    for file_path, description in test_files.items():
        if os.path.exists(file_path):
            print(f"✅ {file_path}: Available")
        else:
            print(f"❌ {file_path}: Missing - {description}")
            missing.append(file_path)
    
    if missing:
        print(f"\n💡 To create test files:")
        print(f"   1. Copy a short video to test_media/person_short.mp4")
        print(f"   2. Copy a short audio to test_media/speech_short.wav")
        print(f"   3. Ensure your diff2lip checkpoint is at checkpoints/checkpoint.pt")
    
    return len(missing) == 0

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running test suite")
    
    if not os.path.exists('test_face_enhancement.py'):
        print("❌ test_face_enhancement.py not found")
        return False
    
    return run_command(
        [sys.executable, 'test_face_enhancement.py'],
        "Running face enhancement tests",
        check=False
    )

def main():
    """Main setup function"""
    print("🚀 Face Enhancement Setup")
    print("=" * 50)
    
    # Check conda environment
    if not check_conda_environment():
        print("\n❌ Please activate the diff2lip conda environment and try again")
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install - you may need to install manually")
    
    # Check test files
    test_files_ok = check_test_files()
    
    # Run tests
    print("\n" + "=" * 50)
    print("🧪 Running Tests")
    print("=" * 50)
    
    tests_passed = run_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Setup Summary")
    print("=" * 50)
    
    if tests_passed and test_files_ok:
        print("🎉 Setup completed successfully!")
        print("✅ Ready to test face enhancement integration")
        print("\n💡 Next steps:")
        print("   1. Run: python test_face_enhancement.py")
        print("   2. Run: python inference.py --config test_face_enhancement_config.yaml")
    elif tests_passed:
        print("⚠️  Setup mostly completed - missing test files")
        print("💡 Add test files and run: python test_face_enhancement.py")
    else:
        print("❌ Setup had issues - check error messages above")
        print("💡 Try running tests manually: python test_face_enhancement.py")
    
    return tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
