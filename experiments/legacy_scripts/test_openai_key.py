#!/usr/bin/env python3
"""
Test OpenAI API Key
Quick script to validate your OpenAI API key is working
"""

import os
import sys


def test_openai_key():
    """Test if OpenAI API key is set and working"""

    print("=" * 60)
    print("OpenAI API Key Validation Test")
    print("=" * 60)
    print()

    # Step 1: Check if key is set
    print("Step 1: Checking environment variable...")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is NOT set")
        print()
        print("Set it with:")
        print("  export OPENAI_API_KEY='sk-your-api-key-here'")
        print()
        return False

    # Mask the key for display
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
    print(f"✅ OPENAI_API_KEY is set: {masked_key}")
    print()

    # Step 2: Test importing OpenAI
    print("Step 2: Testing OpenAI library import...")
    try:
        from openai import OpenAI
        print("✅ OpenAI library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OpenAI library: {e}")
        print()
        print("Install with:")
        print("  uv sync")
        print("  or")
        print("  pip install openai")
        return False
    print()

    # Step 3: Test API connection
    print("Step 3: Testing API connection...")
    print("(Making a simple API call...)")

    try:
        client = OpenAI(api_key=api_key)

        # Make a minimal API call to test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API test successful' and nothing else."}
            ],
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        print(f"✅ API call successful!")
        print(f"   Response: '{result}'")
        print()

        # Step 4: Show account info
        print("Step 4: Checking API details...")
        print(f"   Model used: {response.model}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        print()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Your OpenAI API key is working correctly.")
        print("You're ready to run the auto-improvement pipeline:")
        print()
        print("  uv run auto_improvement_loop.py --iterations 10")
        print("  or")
        print("  ./run_auto_improvement_uv.sh")
        print()

        return True

    except Exception as e:
        print(f"❌ API call failed: {e}")
        print()

        # Provide helpful error messages
        error_msg = str(e).lower()

        if "authentication" in error_msg or "api key" in error_msg or "401" in error_msg:
            print("This looks like an authentication error.")
            print("Possible issues:")
            print("  1. API key is invalid or expired")
            print("  2. API key doesn't have the correct permissions")
            print()
            print("Get a new API key at: https://platform.openai.com/api-keys")

        elif "quota" in error_msg or "billing" in error_msg or "429" in error_msg:
            print("This looks like a quota or billing error.")
            print("Possible issues:")
            print("  1. No credits remaining on your account")
            print("  2. Rate limit exceeded")
            print()
            print("Check your account at: https://platform.openai.com/account/billing")

        elif "network" in error_msg or "connection" in error_msg:
            print("This looks like a network error.")
            print("Check your internet connection and try again.")

        else:
            print("Unexpected error. Please check:")
            print("  1. Your API key is correct")
            print("  2. You have billing set up at https://platform.openai.com/account/billing")
            print("  3. Your internet connection is working")

        print()
        return False


def main():
    """Main entry point"""
    try:
        success = test_openai_key()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
