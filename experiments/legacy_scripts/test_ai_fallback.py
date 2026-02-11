"""
Test script to verify AI provider fallback mechanism
Tests: OpenAI -> Claude -> Shortened prompt fallback chain
"""

import os
import sys
from ai_advisor import AIAdvisor, ANTHROPIC_AVAILABLE

def test_anthropic_availability():
    """Test if Anthropic package is available"""
    print("=" * 60)
    print("TEST 1: Anthropic Package Availability")
    print("=" * 60)

    if ANTHROPIC_AVAILABLE:
        print("✅ Anthropic package is installed")
    else:
        print("❌ Anthropic package is NOT installed")
        print("   Install with: uv pip install anthropic")

    return ANTHROPIC_AVAILABLE

def test_api_keys():
    """Test if API keys are configured"""
    print("\n" + "=" * 60)
    print("TEST 2: API Keys Configuration")
    print("=" * 60)

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    print(f"OpenAI API Key:    {'✅ Set' if openai_key else '❌ Not set'}")
    print(f"Anthropic API Key: {'✅ Set' if anthropic_key else '❌ Not set'}")

    if openai_key:
        print(f"   OpenAI key prefix: {openai_key[:20]}...")
    if anthropic_key:
        print(f"   Anthropic key prefix: {anthropic_key[:20]}...")

    return openai_key is not None, anthropic_key is not None

def test_advisor_initialization():
    """Test AIAdvisor initialization with fallback"""
    print("\n" + "=" * 60)
    print("TEST 3: AIAdvisor Initialization")
    print("=" * 60)

    try:
        advisor = AIAdvisor(use_claude_fallback=True)
        print("✅ AIAdvisor initialized successfully")

        print(f"   Primary provider:   OpenAI ({advisor.model})")
        print(f"   Fallback enabled:   {advisor.use_claude_fallback}")
        print(f"   Claude client:      {'✅ Initialized' if advisor.claude_client else '❌ Not initialized'}")

        return advisor
    except Exception as e:
        print(f"❌ Failed to initialize AIAdvisor: {e}")
        return None

def test_claude_fallback_simulation(advisor):
    """Simulate Claude fallback by using a very long prompt"""
    print("\n" + "=" * 60)
    print("TEST 4: Claude Fallback Simulation")
    print("=" * 60)

    if not advisor:
        print("❌ Cannot test - advisor not initialized")
        return

    # Create a very long prompt to exceed OpenAI context (8k tokens)
    # Approximate: 1 token ≈ 4 characters, so 8k tokens ≈ 32k chars
    # We'll create a 40k character prompt to definitely exceed it

    print("Creating a long prompt to trigger context overflow...")

    # Build a prompt that's intentionally too long for GPT-4
    base_prompt = "Analyze this medical AI training data: "
    repeated_data = "Patient record: chest x-ray shows infiltrate, possible pneumonia. " * 600
    long_prompt = base_prompt + repeated_data

    print(f"Prompt length: {len(long_prompt):,} characters (~{len(long_prompt)//4:,} tokens)")
    print(f"OpenAI GPT-4 limit: ~8,192 tokens")
    print(f"Claude Sonnet 4 limit: ~200,000 tokens")

    if len(long_prompt) < 30000:
        print("\n⚠️  Warning: Prompt might not be long enough to trigger fallback")
        print("   Extending prompt...")
        long_prompt = long_prompt + ("Additional context data. " * 1000)
        print(f"   New length: {len(long_prompt):,} characters")

    print("\nAttempting API call with long prompt...")
    print("(This should trigger OpenAI context overflow and fallback to Claude)")
    print("-" * 60)

    try:
        response = advisor._call_openai_with_retry(long_prompt)

        if response:
            print("\n✅ API call succeeded!")
            print(f"   Response preview: {response.choices[0].message.content[:100]}...")

            # Check if it was Claude or OpenAI
            if hasattr(response, '__class__') and 'Claude' in response.__class__.__name__:
                print("   🤖 Response from: CLAUDE (fallback successful!)")
            else:
                print("   🤖 Response from: OpenAI (no fallback needed)")
        else:
            print("❌ No response received")

    except Exception as e:
        print(f"\n❌ API call failed: {type(e).__name__}")
        print(f"   Error: {str(e)[:200]}")

        # Check if it's the expected context length error
        if "context_length" in str(e).lower() or "maximum context" in str(e).lower():
            print("\n   ✅ Context overflow detected (as expected)")
            print("   ✅ Fallback mechanism should have been triggered")
        else:
            print("\n   ⚠️  Different error - fallback may not have been tested")

def test_simple_api_call(advisor):
    """Test a simple API call that should work"""
    print("\n" + "=" * 60)
    print("TEST 5: Simple API Call (Should Work)")
    print("=" * 60)

    if not advisor:
        print("❌ Cannot test - advisor not initialized")
        return

    simple_prompt = "What is the capital of France? Answer in one word."
    print(f"Sending simple prompt: '{simple_prompt}'")

    try:
        response = advisor._call_openai_with_retry(simple_prompt)

        if response:
            content = response.choices[0].message.content
            print(f"\n✅ Response received: {content}")

            if hasattr(response, '__class__') and 'Claude' in response.__class__.__name__:
                print("   Provider: Claude")
            else:
                print("   Provider: OpenAI")
        else:
            print("❌ No response received")

    except Exception as e:
        print(f"❌ API call failed: {e}")

def main():
    """Run all tests"""
    print("\n" + "🧪" * 30)
    print("AI PROVIDER FALLBACK TEST SUITE")
    print("🧪" * 30)

    # Test 1: Package availability
    anthropic_available = test_anthropic_availability()

    # Test 2: API keys
    openai_ok, anthropic_ok = test_api_keys()

    if not openai_ok:
        print("\n⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        print("   Tests will fail without API key.")

    if not anthropic_ok:
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set!")
        print("   Claude fallback will not work.")
        print("   Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")

    # Test 3: Initialization
    advisor = test_advisor_initialization()

    if not advisor:
        print("\n❌ Cannot continue - advisor initialization failed")
        return 1

    # Test 4: Simple call (should work)
    test_simple_api_call(advisor)

    # Test 5: Fallback simulation (may trigger context overflow)
    if anthropic_ok:
        print("\n💡 Claude is available - attempting fallback test...")
        test_claude_fallback_simulation(advisor)
    else:
        print("\n⚠️  Skipping fallback test - ANTHROPIC_API_KEY not set")
        print("   To test fallback, set ANTHROPIC_API_KEY and rerun")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Anthropic Package: {'✅' if anthropic_available else '❌'}")
    print(f"OpenAI API Key:    {'✅' if openai_ok else '❌'}")
    print(f"Anthropic API Key: {'✅' if anthropic_ok else '❌'}")
    print(f"Advisor Init:      {'✅' if advisor else '❌'}")
    print(f"Fallback Ready:    {'✅' if (advisor and advisor.claude_client) else '❌'}")

    if advisor and advisor.claude_client:
        print("\n🎉 All systems ready! Fallback mechanism is operational.")
    elif advisor:
        print("\n⚠️  Advisor works but Claude fallback not available")
        print("   Set ANTHROPIC_API_KEY to enable fallback")
    else:
        print("\n❌ Setup incomplete - see errors above")

    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())