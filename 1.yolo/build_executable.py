#!/usr/bin/env python3
"""
Build script to create standalone executable for crosswalk detection system.
Optimized for deployment on various platforms including embedded systems.
"""

import os
import subprocess
import shutil
import platform
from pathlib import Path

def build_executable():
    """Build standalone executable using PyInstaller."""

    print("Building crosswalk detection executable...")

    # PyInstaller command for creating executable
    cmd = [
        'pyinstaller',
        '--onefile',                    # Create single executable file
        '--noconsole',                  # No console window (for Windows)
        '--name', 'crosswalk_detector', # Output executable name
        '--icon=assets/icon.ico',       # Icon file (if available)
        '--distpath', 'dist',           # Output directory
        '--workpath', 'build',          # Build directory
        '--specpath', '.',              # Spec file location

        # Add data files
        '--add-data', 'best.pt;.' if platform.system() == 'Windows' else 'best.pt:.',

        # Hidden imports for dependencies
        '--hidden-import', 'ultralytics',
        '--hidden-import', 'cv2',
        '--hidden-import', 'pyttsx3',
        '--hidden-import', 'numpy',
        '--hidden-import', 'torch',
        '--hidden-import', 'PIL',

        # Exclude unnecessary modules to reduce size
        '--exclude-module', 'matplotlib',
        '--exclude-module', 'tkinter',
        '--exclude-module', 'PyQt5',
        '--exclude-module', 'PyQt6',

        'main.py'
    ]

    # Run PyInstaller
    try:
        subprocess.run(cmd, check=True)
        print("✓ Executable built successfully!")

        # Get executable path
        exe_name = 'crosswalk_detector.exe' if platform.system() == 'Windows' else 'crosswalk_detector'
        exe_path = f"dist/{exe_name}"

        if os.path.exists(exe_path):
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"✓ Executable created: {exe_path} ({size_mb:.1f} MB)")

    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        return False

    return True

def create_deployment_package():
    """Create a complete deployment package."""

    package_dir = "deployment_package"

    # Create package directory
    if os.path.exists(package_dir):
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)

    # Copy executable
    exe_name = 'crosswalk_detector.exe' if platform.system() == 'Windows' else 'crosswalk_detector'
    src_exe = f"dist/{exe_name}"

    if os.path.exists(src_exe):
        shutil.copy2(src_exe, package_dir)
        print(f"✓ Copied executable to {package_dir}/")

    # Copy model file (if exists)
    model_files = ['best.pt', 'yolov8n.pt']
    for model_file in model_files:
        if os.path.exists(model_file):
            shutil.copy2(model_file, package_dir)
            print(f"✓ Copied {model_file} to package")
            break

    # Create README for deployment
    readme_content = """# Crosswalk Detection System

## Quick Start
1. Connect your camera
2. Run the executable:
   - Windows: double-click crosswalk_detector.exe
   - Linux/Mac: ./crosswalk_detector

## Controls
- Press 'q' to quit
- Audio feedback will guide you for crosswalk alignment

## System Requirements
- Camera (USB webcam or built-in)
- Speakers/headphones for audio feedback
- Windows 10+, Ubuntu 18.04+, or macOS 10.14+

## Troubleshooting
- If camera doesn't work, try changing camera index in settings
- For audio issues, check system volume and audio drivers
"""

    with open(f"{package_dir}/README.txt", "w") as f:
        f.write(readme_content)

    print(f"✓ Deployment package created in {package_dir}/")

if __name__ == "__main__":
    print("Crosswalk Detection Build Tool")
    print("=" * 40)

    # Check if PyInstaller is installed
    try:
        subprocess.run(['pyinstaller', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing PyInstaller...")
        subprocess.run(['pip', 'install', 'pyinstaller'], check=True)

    # Build executable
    if build_executable():
        create_deployment_package()
        print("\n✓ Build complete! Check the 'deployment_package' folder.")
    else:
        print("\n✗ Build failed. Check error messages above.")