#!/usr/bin/env python3
"""
Simple test runner for iteration_012 repeatability validation.
"""

import subprocess
import sys
import os
from datetime import datetime

def main():
    """Run repeatability test for iteration_012"""
    
    print("ITERATION 012 REPEATABILITY TEST")
    print("=" * 40)
    print("Purpose: Verify that iteration_012 model produces consistent results")
    print(f"Started: {datetime.now()}")
    
    # Check if test script exists
    test_script = "iteration_012_test_repeatability.py"
    if not os.path.exists(test_script):
        print(f"❌ Test script not found: {test_script}")
        return
    
    print(f"\nRunning repeatability test...")
    
    try:
        # Run the test
        result = subprocess.run([sys.executable, test_script], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Test completed successfully")
            print("\nOutput:")
            print(result.stdout)
            
            # Check for verdict in output
            if "PERFECTLY REPEATABLE" in result.stdout:
                print("\n🎯 FINAL VERDICT: Model is PERFECTLY REPEATABLE")
            elif "HIGHLY REPEATABLE" in result.stdout:
                print("\n✅ FINAL VERDICT: Model is HIGHLY REPEATABLE")
            elif "NOT REPEATABLE" in result.stdout:
                print("\n❌ FINAL VERDICT: Model is NOT REPEATABLE")
            
        else:
            print("❌ Test failed")
            print("\nError output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
    
    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    main()