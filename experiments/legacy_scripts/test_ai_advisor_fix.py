"""
Test to verify AI advisor f-string fix
"""

from ai_advisor import AIAdvisor

def test_system_prompt():
    """Test that system prompt can be generated without formatting errors"""
    print("Testing AI Advisor system prompt generation...")

    try:
        # Create AI advisor with dummy key
        advisor = AIAdvisor(api_key="dummy_test_key")

        # Try to generate system prompt (this would fail with formatting error before fix)
        system_prompt = advisor._get_system_prompt()

        # Check that the prompt contains the JSON example
        assert "training" in system_prompt, "System prompt missing 'training' section"
        assert "FocalLoss" in system_prompt, "System prompt missing 'FocalLoss' reference"
        assert "reasoning" in system_prompt, "System prompt missing 'reasoning' field"

        # Check that curly braces are properly escaped in the prompt
        # The prompt should contain literal "{" and "}" characters after f-string processing
        assert "{" in system_prompt, "System prompt missing JSON braces"

        print("✅ System prompt generated successfully!")
        print(f"   Prompt length: {len(system_prompt)} characters")
        print(f"   Contains JSON example: Yes")
        print()

        # Show a snippet of the JSON example in the prompt
        lines = system_prompt.split('\n')
        json_start = False
        print("JSON Example in Prompt:")
        print("-" * 60)
        for line in lines:
            if '```json' in line:
                json_start = True
            if json_start:
                print(line)
            if json_start and '```' in line and '```json' not in line:
                break
        print("-" * 60)

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_system_prompt()
    exit(0 if success else 1)
