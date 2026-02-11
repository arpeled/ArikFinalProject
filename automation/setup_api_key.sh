#!/bin/bash
# Setup OpenAI API Key Helper Script

echo "=========================================="
echo "OpenAI API Key Setup"
echo "=========================================="
echo ""

# Check if already set
if [ -n "$OPENAI_API_KEY" ]; then
    masked_key="${OPENAI_API_KEY:0:7}...${OPENAI_API_KEY: -4}"
    echo "✅ OPENAI_API_KEY is already set: $masked_key"
    echo ""
    read -p "Do you want to update it? (y/n): " update
    if [ "$update" != "y" ]; then
        echo "Keeping existing key. Testing it now..."
        echo ""
        uv run test_openai_key.py
        exit 0
    fi
fi

# Prompt for API key
echo "Please enter your OpenAI API key:"
echo "(Get one at: https://platform.openai.com/api-keys)"
echo ""
read -p "API Key (sk-...): " api_key

# Validate format
if [[ ! $api_key == sk-* ]]; then
    echo ""
    echo "⚠️  Warning: API key should start with 'sk-'"
    echo "Are you sure this is correct?"
    read -p "Continue anyway? (y/n): " continue
    if [ "$continue" != "y" ]; then
        echo "Cancelled."
        exit 1
    fi
fi

# Export to current session
export OPENAI_API_KEY="$api_key"

echo ""
echo "✅ API key set for current session"
echo ""

# Ask if user wants to save permanently
echo "Do you want to save this to your shell configuration?"
echo "(This will add it to your ~/.zshrc or ~/.bashrc)"
echo ""
read -p "Save permanently? (y/n): " save_permanent

if [ "$save_permanent" = "y" ]; then
    # Detect shell
    if [ -f "$HOME/.zshrc" ]; then
        shell_config="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        shell_config="$HOME/.bashrc"
    else
        shell_config="$HOME/.bash_profile"
    fi

    # Check if already exists
    if grep -q "OPENAI_API_KEY" "$shell_config"; then
        echo ""
        echo "⚠️  OPENAI_API_KEY already exists in $shell_config"
        echo "Please update it manually to avoid duplicates:"
        echo "  nano $shell_config"
    else
        echo "" >> "$shell_config"
        echo "# OpenAI API Key" >> "$shell_config"
        echo "export OPENAI_API_KEY=\"$api_key\"" >> "$shell_config"
        echo ""
        echo "✅ Saved to $shell_config"
        echo "To use in new terminals, run:"
        echo "  source $shell_config"
    fi
fi

echo ""
echo "=========================================="
echo "Testing API Key"
echo "=========================================="
echo ""

# Test the API key
uv run test_openai_key.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "You're ready to run the auto-improvement pipeline:"
    echo "  ./run_auto_improvement_uv.sh"
    echo "  or"
    echo "  uv run auto_improvement_loop.py --iterations 10"
else
    echo ""
    echo "=========================================="
    echo "❌ API Key Test Failed"
    echo "=========================================="
    echo ""
    echo "Please check:"
    echo "  1. Your API key is correct"
    echo "  2. You have billing set up: https://platform.openai.com/account/billing"
    echo "  3. Your account has available credits"
    echo ""
    echo "To try again, run:"
    echo "  ./setup_api_key.sh"
fi
