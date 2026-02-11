"""
Check available OpenAI models
"""
import os
from openai import OpenAI

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not set")
    exit(1)

# Initialize client
client = OpenAI(api_key=api_key)

print("Fetching available OpenAI models...\n")

try:
    # List all models
    models = client.models.list()

    # Filter for GPT models
    gpt_models = []
    all_models = []

    for model in models.data:
        all_models.append(model.id)
        if 'gpt' in model.id.lower():
            gpt_models.append(model.id)

    print("=" * 80)
    print("GPT MODELS AVAILABLE:")
    print("=" * 80)

    # Sort and display GPT models
    gpt_models_sorted = sorted(gpt_models, reverse=True)
    for model in gpt_models_sorted:
        print(f"  - {model}")

    print(f"\nTotal GPT models: {len(gpt_models)}")

    # Check specifically for gpt-5.1
    print("\n" + "=" * 80)
    print("CHECKING FOR GPT-5.1:")
    print("=" * 80)

    gpt5_models = [m for m in gpt_models if 'gpt-5' in m.lower()]
    if gpt5_models:
        print("✅ GPT-5 models found:")
        for model in gpt5_models:
            print(f"  - {model}")
            if 'gpt-5.1' in model.lower() or 'gpt-5-1' in model.lower():
                print(f"    ✅ This matches gpt-5.1!")
    else:
        print("❌ No GPT-5 models found")
        print("\nMost recent GPT models available:")
        latest = [m for m in gpt_models_sorted[:10]]
        for model in latest:
            print(f"  - {model}")

    # Save all models to file
    with open("all_openai_models.txt", "w") as f:
        f.write("ALL OPENAI MODELS\n")
        f.write("=" * 80 + "\n\n")
        for model in sorted(all_models):
            f.write(f"{model}\n")

    print(f"\n📄 Full model list saved to: all_openai_models.txt")
    print(f"   Total models available: {len(all_models)}")

except Exception as e:
    print(f"❌ Error fetching models: {e}")
    import traceback
    traceback.print_exc()
