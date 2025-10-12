#!/usr/bin/env python3
"""
Test script for the unified pipeline.

This script provides basic validation and testing functionality for the unified pipeline.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def test_script_executability():
    """Test if the unified pipeline script is executable."""
    script_path = os.path.join(os.path.dirname(__file__), 'unified_pipeline.sh')
    
    if not os.path.exists(script_path):
        print("❌ unified_pipeline.sh not found")
        return False
    
    if not os.access(script_path, os.X_OK):
        print("❌ unified_pipeline.sh is not executable")
        return False
    
    print("✅ unified_pipeline.sh is executable")
    return True

def test_help_functionality():
    """Test if the help functionality works."""
    script_path = os.path.join(os.path.dirname(__file__), 'unified_pipeline.sh')
    
    try:
        result = subprocess.run([script_path, '--help'], capture_output=True, text=True)
        if result.returncode == 0 and 'Usage:' in result.stdout:
            print("✅ Help functionality works")
            return True
        else:
            print("❌ Help functionality failed")
            return False
    except Exception as e:
        print(f"❌ Error testing help functionality: {e}")
        return False

def test_invalid_dataset():
    """Test error handling for invalid dataset."""
    script_path = os.path.join(os.path.dirname(__file__), 'unified_pipeline.sh')
    
    try:
        result = subprocess.run([script_path, 'invalid_dataset', '/tmp'], capture_output=True, text=True)
        if result.returncode != 0:
            print("✅ Invalid dataset handling works")
            return True
        else:
            print("❌ Invalid dataset handling failed")
            return False
    except Exception as e:
        print(f"❌ Error testing invalid dataset: {e}")
        return False

def test_missing_phase_labels():
    """Test error handling for missing phase labels in pars-ct."""
    script_path = os.path.join(os.path.dirname(__file__), 'unified_pipeline.sh')
    
    # Create a temporary directory with pars-ct structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fake pars-ct structure
        case_dir = os.path.join(temp_dir, 'case_001')
        os.makedirs(case_dir)
        os.makedirs(os.path.join(case_dir, 'SCANS'))
        os.makedirs(os.path.join(case_dir, 'ASSESSORS'))
        
        try:
            result = subprocess.run([script_path, 'pars-ct', temp_dir], capture_output=True, text=True)
            if result.returncode != 0:
                print("✅ Missing phase labels handling works")
                return True
            else:
                print("❌ Missing phase labels handling failed")
                return False
        except Exception as e:
            print(f"❌ Error testing missing phase labels: {e}")
            return False

def test_python_dependencies():
    """Test if required Python packages are available."""
    required_packages = [
        'pydicom', 'nibabel', 'pandas', 'numpy', 'SimpleITK'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required Python packages are available")
    return True

def test_totalsegmentator():
    """Test if TotalSegmentator is available."""
    try:
        result = subprocess.run(['TotalSegmentator', '--help'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ TotalSegmentator is available")
            return True
        else:
            print("❌ TotalSegmentator is not working properly")
            return False
    except FileNotFoundError:
        print("❌ TotalSegmentator is not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing TotalSegmentator: {e}")
        return False

def test_supporting_scripts():
    """Test if supporting Python scripts exist and are valid."""
    supporting_scripts = [
        'process_pars_ct_phases.py',
        'apply_segmentation_to_other_series.py',
        'crop_pars_ct_volumes.py',
        'register_pars_ct_volumes.py',
        'unified_configs.py'
    ]
    
    missing_scripts = []
    for script in supporting_scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} is missing")
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ Missing scripts: {', '.join(missing_scripts)}")
        return False
    
    print("✅ All supporting scripts exist")
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("Running unified pipeline tests...\n")
    
    tests = [
        ("Script Executability", test_script_executability),
        ("Help Functionality", test_help_functionality),
        ("Invalid Dataset Handling", test_invalid_dataset),
        ("Missing Phase Labels Handling", test_missing_phase_labels),
        ("Python Dependencies", test_python_dependencies),
        ("TotalSegmentator", test_totalsegmentator),
        ("Supporting Scripts", test_supporting_scripts)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\n--- Test Results ---")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage: python test_unified_pipeline.py")
        print("Run basic validation tests for the unified pipeline.")
        return
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

