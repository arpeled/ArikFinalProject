"""
Test script to verify that all metrics are included in baseline comparison
"""

import pandas as pd
import numpy as np
from auto_improvement_loop import BASELINE_RESULTS

# Create a mock results CSV with all metrics
def create_test_results():
    """Create test results DataFrame with all metrics"""
    labels = list(BASELINE_RESULTS.keys())

    test_data = []
    for label in labels:
        test_data.append({
            'Label': label,
            'AUC': 0.85,  # Slightly different from baseline
            'Threshold': 0.5,
            'Sensitivity': 0.75,
            'Specificity': 0.80,
            'Accuracy': 0.82,
            'Precision': 0.15,
            'Recall': 0.75,
            'F1_Score': 0.25
        })

    return pd.DataFrame(test_data)

# Test the comparison
def test_comparison():
    """Test that comparison includes all metrics"""
    print("Testing Baseline Comparison with All Metrics\n")
    print("=" * 80)

    # Create test results
    results_df = create_test_results()
    results_file = 'test_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✓ Created test results file: {results_file}")

    # Import the comparison function
    from auto_improvement_loop import AutoImprovementLoop
    import os

    # Create a temporary instance with dummy API key
    loop = AutoImprovementLoop(openai_api_key="dummy_key_for_testing")

    # Run comparison
    comparison_file = 'test_comparison.csv'
    comparison_df = loop.compare_with_baseline(results_file, comparison_file)

    print(f"✓ Created comparison file: {comparison_file}")
    print(f"\nComparison DataFrame columns:")
    print("-" * 80)
    for i, col in enumerate(comparison_df.columns, 1):
        print(f"{i:2d}. {col}")

    # Check that all expected metrics are present
    expected_metrics = ['AUC', 'F1_Score', 'Recall', 'Accuracy', 'Specificity', 'Precision', 'Sensitivity', 'Threshold']

    print(f"\n\nExpected Metrics Check:")
    print("-" * 80)

    all_present = True
    for metric in expected_metrics:
        baseline_col = f'Baseline_{metric}'
        our_col = f'Our_{metric}'
        improvement_col = f'{metric}_Improvement'
        better_col = f'Better_{metric}'

        checks = []
        checks.append(('Baseline', baseline_col in comparison_df.columns))
        checks.append(('Our', our_col in comparison_df.columns))
        checks.append(('Improvement', improvement_col in comparison_df.columns))
        checks.append(('Better', better_col in comparison_df.columns))

        metric_present = all(check[1] for check in checks)
        status = "✓" if metric_present else "✗"

        print(f"{status} {metric:12s}: ", end="")
        print(" | ".join([f"{name}:{' ✓' if present else ' ✗'}" for name, present in checks]))

        if not metric_present:
            all_present = False

    print("\n" + "=" * 80)
    if all_present:
        print("✓ SUCCESS: All metrics are included in the comparison!")
    else:
        print("✗ FAILURE: Some metrics are missing from the comparison!")

    # Display sample comparison data
    print("\n\nSample Comparison Data (first 3 rows):")
    print("-" * 80)
    print(comparison_df.head(3).to_string())

    # Test AI advisor summary
    print("\n\n" + "=" * 80)
    print("Testing AI Advisor Comparison Summary:")
    print("=" * 80)

    from ai_advisor import AIAdvisor
    advisor = AIAdvisor(api_key="dummy_key_for_testing")
    summary = advisor._summarize_comparison(comparison_df)
    print(summary)

    print("\n" + "=" * 80)
    print("Test completed!")

    # Cleanup
    import os
    os.remove(results_file)
    os.remove(comparison_file)
    print("\n✓ Test files cleaned up")

if __name__ == "__main__":
    test_comparison()
