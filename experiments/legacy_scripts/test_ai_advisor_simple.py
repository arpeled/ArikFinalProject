"""
Quick test script for AI Advisor to diagnose empty response issue
"""

import os
import sys
import pandas as pd
from ai_advisor import AIAdvisor

def test_ai_advisor():
    """Test AI advisor with a simple iteration"""

    print("="*80)
    print("AI ADVISOR TEST")
    print("="*80)

    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    claude_key = os.getenv('ANTHROPIC_API_KEY')

    print(f"\n🔑 API Keys:")
    print(f"   OpenAI: {'✅ Set' if openai_key else '❌ Missing'}")
    print(f"   Claude: {'✅ Set' if claude_key else '❌ Missing'}")

    if not openai_key and not claude_key:
        print("\n❌ ERROR: No API keys found!")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Initialize AI Advisor
    print(f"\n📊 Initializing AI Advisor...")
    advisor = AIAdvisor()
    print(f"   Model: {advisor.model}")
    print(f"   Claude available: {advisor.claude_client is not None}")

    # Load sample data from iteration 57 (which worked)
    print(f"\n📁 Loading test data from iteration 057...")

    iteration_dir = "auto_improvement_runs/iteration_057"

    # Load results CSV
    results_csv = f"{iteration_dir}/pipeline_results_20260108-133418.csv"
    if not os.path.exists(results_csv):
        # Try to find any pipeline results file
        import glob
        files = glob.glob(f"{iteration_dir}/pipeline_results_*.csv")
        if files:
            results_csv = files[0]
        else:
            print(f"❌ No results file found in {iteration_dir}")
            return

    results_df = pd.read_csv(results_csv)
    print(f"   Results: {len(results_df)} diseases")

    # Load comparison CSV
    comparison_csv = "baseline_comparison_20260108-133418.csv"
    if not os.path.exists(comparison_csv):
        import glob
        files = glob.glob("baseline_comparison_*.csv")
        if files:
            comparison_csv = sorted(files)[-1]  # Get most recent
        else:
            print(f"❌ No comparison file found")
            return

    comparison_df = pd.read_csv(comparison_csv)
    print(f"   Comparison: {len(comparison_df)} diseases")

    # Simple config (minimal)
    config = {
        'training': {
            'learning_rate': 0.0005,
            'num_epochs': 50
        },
        'loss': {
            'type': 'FocalLoss',
            'gamma': 3.0,
            'alpha': 0.75
        }
    }

    # Call AI advisor
    print(f"\n🤖 Calling AI Advisor...")
    print(f"   (This may take 20-30 seconds)")

    try:
        analysis_text, suggestions = advisor.analyze_results(
            config=config,
            results_df=results_df,
            comparison_df=comparison_df,
            iteration=999,  # Test iteration
            train_loss=0.1,
            val_loss=0.15
        )

        print(f"\n✅ AI Advisor call successful!")
        print(f"   Analysis length: {len(analysis_text)} characters")
        print(f"   Suggestions: {len(suggestions)} keys")

        # Show analysis preview
        print(f"\n📄 Analysis Preview (first 500 chars):")
        print("-" * 80)
        print(analysis_text[:500])
        print("-" * 80)

        # Show suggestions
        print(f"\n💡 Suggestions:")
        for key, value in suggestions.items():
            if key != 'reasoning':
                print(f"   {key}: {value}")

        if 'reasoning' in suggestions:
            print(f"\n📝 Reasoning (first 200 chars):")
            print(f"   {suggestions['reasoning'][:200]}...")

        # Save to test file
        test_output = "ai_advisor_test_output.txt"
        with open(test_output, 'w') as f:
            f.write(analysis_text)
        print(f"\n💾 Full analysis saved to: {test_output}")

    except Exception as e:
        print(f"\n❌ Error calling AI Advisor: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_ai_advisor()
