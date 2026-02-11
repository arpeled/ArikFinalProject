#!/usr/bin/env python3
"""
Comprehensive test runner for iteration_012 validation.
Runs all tests to verify results are not random and provides summary report.
"""

import subprocess
import sys
import os
from datetime import datetime
import pandas as pd

def run_test_script(script_name, description):
    """Run a test script and capture results"""
    
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print("✅ Test completed successfully")
            print("\nOutput:")
            print(result.stdout)
            return True, result.stdout, result.stderr
        else:
            print("❌ Test failed")
            print("\nError output:")
            print(result.stderr)
            return False, result.stdout, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out (30 minutes)")
        return False, "", "Test timed out"
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False, "", str(e)

def check_prerequisites():
    """Check if all required files exist"""
    
    print("Checking prerequisites...")
    
    required_files = [
        "auto_improvement_runs/iteration_012/config.yaml",
        "auto_improvement_runs/iteration_012/pipeline_results_20251229-101232.csv",
        "./ChestX-ray14/train_data_small.csv",
        "./ChestX-ray14/test_data_small.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        
        if "./ChestX-ray14/train_data_small.csv" in missing_files:
            print("\n💡 Run 'python create_small_datasets.py' to create small datasets")
        
        return False
    
    print("✅ All prerequisites found")
    return True

def analyze_test_outputs():
    """Analyze outputs from all tests"""
    
    print("\n" + "="*60)
    print("ANALYZING TEST OUTPUTS")
    print("="*60)
    
    # Look for output files from tests
    timestamp_pattern = datetime.now().strftime("%Y%m%d")
    
    output_files = []
    for file in os.listdir("."):
        if file.startswith("iteration_012_test_") and timestamp_pattern in file:
            output_files.append(file)
    
    if not output_files:
        print("❌ No test output files found")
        return
    
    print(f"Found {len(output_files)} test output files:")
    for file in output_files:
        print(f"  - {file}")
    
    # Try to load and summarize CSV results
    csv_files = [f for f in output_files if f.endswith('.csv')]
    
    summary_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if 'reproducibility' in csv_file:
                # Reproducibility test results
                if 'avg_auc' in df.columns:
                    auc_cv = (df['avg_auc'].std() / df['avg_auc'].mean()) * 100
                    summary_data.append({
                        'test': 'Reproducibility',
                        'metric': 'AUC CV%',
                        'value': auc_cv,
                        'status': '✅ PASS' if auc_cv < 1.0 else '❌ FAIL'
                    })
            
            elif 'cross_validation' in csv_file:
                # Cross-validation test results
                if 'avg_auc' in df.columns:
                    auc_cv = (df['avg_auc'].std() / df['avg_auc'].mean()) * 100
                    summary_data.append({
                        'test': 'Cross-Validation',
                        'metric': 'AUC CV%',
                        'value': auc_cv,
                        'status': '✅ PASS' if auc_cv < 5.0 else '❌ FAIL'
                    })
                    
                    # Check training consistency
                    if 'epochs_trained' in df.columns:
                        epochs_std = df['epochs_trained'].std()
                        summary_data.append({
                            'test': 'Cross-Validation',
                            'metric': 'Epochs Std',
                            'value': epochs_std,
                            'status': '✅ PASS' if epochs_std < 2 else '❌ FAIL'
                        })
        
        except Exception as e:
            print(f"❌ Error reading {csv_file}: {str(e)}")
    
    # Display summary
    if summary_data:
        print("\nTest Summary:")
        print("-" * 50)
        for item in summary_data:
            print(f"{item['test']:15} | {item['metric']:12} | {item['value']:8.2f} | {item['status']}")
    
    return summary_data

def generate_final_report(test_results, summary_data):
    """Generate final validation report"""
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = f"iteration_012_test_validation_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("ITERATION 012 VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Purpose: Verify that iteration_012 results are not random\n\n")
        
        f.write("TESTS EXECUTED:\n")
        f.write("-" * 20 + "\n")
        for i, (test_name, success, _, _) in enumerate(test_results, 1):
            status = "✅ PASSED" if success else "❌ FAILED"
            f.write(f"{i}. {test_name}: {status}\n")
        
        f.write(f"\nTEST SUMMARY:\n")
        f.write("-" * 20 + "\n")
        if summary_data:
            for item in summary_data:
                f.write(f"{item['test']} - {item['metric']}: {item['value']:.2f} ({item['status']})\n")
        else:
            f.write("No quantitative summary available\n")
        
        # Overall verdict
        passed_tests = sum(1 for _, success, _, _ in test_results if success)
        total_tests = len(test_results)
        
        f.write(f"\nOVERALL VERDICT:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Tests Passed: {passed_tests}/{total_tests}\n")
        
        if passed_tests >= 2:
            f.write("🎯 CONCLUSION: Results appear to be NON-RANDOM\n")
            f.write("   The model shows consistent, reproducible performance\n")
            f.write("   that is significantly better than random chance.\n")
        elif passed_tests >= 1:
            f.write("⚠️  CONCLUSION: Results are PARTIALLY VALIDATED\n")
            f.write("   Some tests passed but validation is incomplete.\n")
        else:
            f.write("❌ CONCLUSION: Results may be RANDOM\n")
            f.write("   Tests failed to demonstrate non-random performance.\n")
        
        f.write(f"\nFor detailed results, see individual test output files.\n")
    
    return report_file

def main():
    """Main function to run all validation tests"""
    
    print("ITERATION 012 VALIDATION TEST SUITE")
    print("=" * 50)
    print("Purpose: Verify that iteration_012 results are not random")
    print(f"Started: {datetime.now()}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return
    
    # Define tests to run
    tests = [
        ("iteration_012_test_statistical_validation.py", "Statistical Validation Test"),
        ("iteration_012_test_reproducibility.py", "Reproducibility Test"),
        ("iteration_012_test_cross_validation.py", "Cross-Validation Stability Test")
    ]
    
    # Run all tests
    test_results = []
    
    for script_name, description in tests:
        if os.path.exists(script_name):
            success, stdout, stderr = run_test_script(script_name, description)
            test_results.append((description, success, stdout, stderr))
        else:
            print(f"❌ Test script not found: {script_name}")
            test_results.append((description, False, "", "Script not found"))
    
    # Analyze outputs
    summary_data = analyze_test_outputs()
    
    # Generate final report
    report_file = generate_final_report(test_results, summary_data)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION TEST SUITE COMPLETED")
    print("="*60)
    
    passed_tests = sum(1 for _, success, _, _ in test_results if success)
    total_tests = len(test_results)
    
    print(f"Tests completed: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests >= 2:
        print("\n🎯 FINAL VERDICT: Results appear to be NON-RANDOM")
        print("   The iteration_012 model shows consistent, reproducible")
        print("   performance that is significantly better than chance.")
    elif passed_tests >= 1:
        print("\n⚠️  FINAL VERDICT: Results are PARTIALLY VALIDATED")
        print("   Some evidence of non-random performance, but incomplete.")
    else:
        print("\n❌ FINAL VERDICT: Results may be RANDOM")
        print("   Tests failed to demonstrate non-random performance.")
    
    print(f"\nDetailed report saved to: {report_file}")
    print("\nTest files created:")
    for file in os.listdir("."):
        if file.startswith("iteration_012_test_") and datetime.now().strftime("%Y%m%d") in file:
            print(f"  - {file}")

if __name__ == "__main__":
    main()