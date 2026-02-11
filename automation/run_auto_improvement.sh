#!/bin/bash
# Quick start script for auto-improvement pipeline

echo "=========================================="
echo "Chest X-Ray Auto-Improvement Pipeline"
echo "=========================================="
echo ""

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    read -p "Enter your OpenAI API key now (or press Ctrl+C to exit): " api_key
    export OPENAI_API_KEY="$api_key"
    echo ""
fi

# Check if config file exists
if [ ! -f "config_baseline.yaml" ]; then
    echo "❌ ERROR: config_baseline.yaml not found"
    echo "Please ensure you're running this script from the project directory"
    exit 1
fi

echo "✅ Configuration file found"
echo "✅ OpenAI API key set"
echo ""

# Check for existing runs and ask about resume
resume_flag=""
output_dir_default="auto_improvement_runs"

if [ -d "$output_dir_default" ] && [ -n "$(ls -A $output_dir_default/iteration_* 2>/dev/null)" ]; then
    # Count existing iterations
    iteration_count=$(ls -d $output_dir_default/iteration_* 2>/dev/null | wc -l | tr -d ' ')
    last_iteration=$(ls -d $output_dir_default/iteration_* 2>/dev/null | sort | tail -1 | sed 's/.*iteration_0*//')

    echo "🔍 Found $iteration_count previous iteration(s)"
    echo "   Last completed: iteration $last_iteration"
    echo ""
    read -p "Resume from iteration $last_iteration? (y/n): " do_resume

    if [ "$do_resume" = "y" ]; then
        resume_flag="--resume"
        echo "✅ Will resume from last iteration"
    else
        echo "⚠️  Starting fresh (previous runs will not be deleted)"
    fi
    echo ""
fi

# Ask user for number of iterations
read -p "How many iterations to run? (default: 10): " iterations
iterations=${iterations:-10}

# Ask for output directory
read -p "Output directory? (default: $output_dir_default): " output_dir
output_dir=${output_dir:-$output_dir_default}

echo ""
echo "=========================================="
echo "Starting Auto-Improvement Loop"
echo "=========================================="
echo "Iterations: $iterations"
echo "Output directory: $output_dir"
if [ -n "$resume_flag" ]; then
    echo "Mode: RESUME from iteration $last_iteration"
    echo "Will run iterations: $((last_iteration + 1)) to $((last_iteration + iterations))"
else
    echo "Mode: FRESH START"
    echo "Will run iterations: 1 to $iterations"
fi
echo "Expected time: ~$((iterations * 3)) hours (full dataset)"
echo "=========================================="
echo ""

read -p "Continue? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Cancelled"
    exit 0
fi

# Run the auto-improvement loop
python3 auto_improvement_loop.py \
    --config config_baseline.yaml \
    --iterations $iterations \
    --output-dir $output_dir \
    $resume_flag

echo ""
echo "=========================================="
echo "Auto-Improvement Loop Completed"
echo "=========================================="
echo "Check results in: $output_dir/"
echo "Final report: $output_dir/FINAL_REPORT.md"
echo "=========================================="
