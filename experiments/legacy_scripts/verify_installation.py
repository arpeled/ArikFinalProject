#!/usr/bin/env python3
"""
Verification script for auto-improvement pipeline
Checks if all dependencies are installed and configured correctly
"""

import sys
import os

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        print(f"   Install with: pip install {package_name}")
        return False

def check_file(file_path):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {file_path} exists")
        return True
    else:
        print(f"❌ {file_path} NOT found")
        return False

def main():
    print("="*60)
    print("Auto-Improvement Pipeline - Installation Verification")
    print("="*60)
    print()

    all_good = True

    # Check Python version
    print("Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"❌ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        print("   Python 3.8+ required")
        all_good = False
    print()

    # Check required packages
    print("Checking required packages...")
    packages = [
        ("yaml", "pyyaml"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("sklearn", "scikit-learn"),
        ("openai", "openai"),
        ("PIL", "Pillow")
    ]

    for module, package in packages:
        if not check_import(module, package):
            all_good = False
    print()

    # Check OpenAI API key
    print("Checking OpenAI API key...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        print(f"✅ OPENAI_API_KEY is set ({masked_key})")
    else:
        print("⚠️  OPENAI_API_KEY not set (you'll need to provide it when running)")
    print()

    # Check required files
    print("Checking required files...")
    files = [
        "config_baseline.yaml",
        "config_manager.py",
        "config_based_pipeline.py",
        "ai_advisor.py",
        "auto_improvement_loop.py",
        "dataset.py",
        "chest_xray_test_pipeline.py"
    ]

    for file in files:
        if not check_file(file):
            all_good = False
    print()

    # Check data files
    print("Checking data files...")
    data_files = [
        "./ChestX-ray14/train_data.csv",
        "./ChestX-ray14/test_data.csv",
        "./ChestX-ray14/images224"
    ]

    for file in data_files:
        check_file(file)
    print()

    # Final verdict
    print("="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("You're ready to run the auto-improvement pipeline:")
        print("  python auto_improvement_loop.py")
        print("  or")
        print("  ./run_auto_improvement.sh")
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements_auto_improvement.txt")
    print("="*60)

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
