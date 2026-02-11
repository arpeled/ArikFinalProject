#!/bin/bash
# Quick verification script for Anthropic API key

echo "Checking Anthropic API key setup..."
echo "=========================================="

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ ANTHROPIC_API_KEY is NOT set"
    echo ""
    echo "To set it, run:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-your-key-here'"
    echo ""
    echo "Or add to ~/.zshrc for permanent setup:"
    echo "  echo 'export ANTHROPIC_API_KEY=\"sk-ant-your-key-here\"' >> ~/.zshrc"
    echo "  source ~/.zshrc"
    exit 1
else
    echo "✅ ANTHROPIC_API_KEY is set"
    echo ""
    echo "Key prefix: ${ANTHROPIC_API_KEY:0:20}..."
    echo ""
    echo "Testing API connection..."

    # Try to import and test anthropic
    uv run python -c "
import os
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    # Simple test message
    response = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=10,
        messages=[{'role': 'user', 'content': 'Hi'}]
    )
    print('✅ API connection successful!')
    print(f'   Response: {response.content[0].text}')
except ImportError:
    print('❌ anthropic package not installed')
    print('   Install with: uv pip install anthropic')
except Exception as e:
    print(f'❌ API test failed: {e}')
    print('   Check your API key is correct')
"
fi

echo ""
echo "=========================================="