# Complete AI Provider Fallback Mechanism

**Date**: January 7, 2026
**Status**: ✅ Fully Operational
**File**: `ai_advisor.py`

---

## Overview

The AI Advisor now has a **robust, multi-layered fallback system** that handles **ANY error** from OpenAI by automatically falling back to Claude (Anthropic).

---

## Fallback Chain

```
┌─────────────────────────────────────────┐
│ 1. Try OpenAI (gpt-4)                  │
│    - Primary provider                   │
│    - 8k token context                   │
│    - 3 retry attempts                   │
└─────────────────┬───────────────────────┘
                  │
                  ↓ (any error after retries)
┌─────────────────────────────────────────┐
│ 2. Try Claude Sonnet 4                 │
│    - Fallback provider                  │
│    - 200k token context                 │
│    - No retries (single attempt)        │
└─────────────────┬───────────────────────┘
                  │
                  ↓ (if Claude unavailable)
┌─────────────────────────────────────────┐
│ 3. Try OpenAI with Shortened Prompt    │
│    - Remove verbose sections            │
│    - Keep essential data                │
│    - Single attempt                     │
└─────────────────┬───────────────────────┘
                  │
                  ↓ (if all fail)
┌─────────────────────────────────────────┐
│ 4. Raise Error                         │
│    - Reports original OpenAI error      │
│    - Indicates Claude also failed       │
└─────────────────────────────────────────┘
```

---

## Errors Handled

The fallback now triggers for **ANY** error from OpenAI, including:

| Error Type | Behavior | Fallback |
|------------|----------|----------|
| **BadRequestError** (context length) | Immediate fallback to Claude | ✅ Yes |
| **BadRequestError** (other) | No retry, fallback to Claude | ✅ Yes |
| **RateLimitError** (token limit) | Immediate fallback to Claude | ✅ Yes |
| **RateLimitError** (rate exceeded) | Retry 3x → fallback | ✅ Yes |
| **APIConnectionError** | Retry 3x with backoff → fallback | ✅ Yes |
| **AuthenticationError** | Retry 3x → fallback | ✅ Yes |
| **APITimeoutError** | Retry 3x → fallback | ✅ Yes |
| **Any other Exception** | Retry 3x → fallback | ✅ Yes |

---

## Key Features

### 1. Smart Error Detection

```python
# Context length errors → immediate fallback
if "context_length_exceeded" in error_msg:
    return self._call_claude(prompt)

# Token limit errors → immediate fallback
if "tokens per min" in error_msg:
    return self._call_claude(prompt)

# Other errors → retry then fallback
for attempt in range(max_retries):
    try:
        # Try OpenAI
    except Exception as e:
        last_error = e
        # Retry with backoff

# After retries → fallback
if self.claude_client:
    return self._call_claude(prompt)
```

### 2. Exponential Backoff

```
Attempt 1: Wait 1 second
Attempt 2: Wait 2 seconds
Attempt 3: Wait 4 seconds
Then → fallback to Claude
```

### 3. Transparent Fallback

User sees clear messages:

```
⚠️  OpenAI token limit exceeded (prompt too large). Falling back to Claude...
🤖 Using Claude Sonnet 4 (200k context)...
✅ Response received successfully!
```

### 4. Response Format Compatibility

Claude responses are converted to OpenAI-compatible format:

```python
class ClaudeResponse:
    def __init__(self, claude_resp):
        self.choices = [type('obj', (object,), {
            'message': type('obj', (object,), {
                'content': claude_resp.content[0].text
            })()
        })()]
```

The rest of the code works seamlessly regardless of which provider was used.

---

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Recommended (for fallback)
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Code Configuration

```python
# Enable fallback (default)
advisor = AIAdvisor(use_claude_fallback=True)

# Disable fallback
advisor = AIAdvisor(use_claude_fallback=False)

# Custom model
advisor = AIAdvisor(model="gpt-4-turbo")

# Custom retry count
advisor._call_openai_with_retry(prompt, max_retries=5)
```

---

## Testing

### Quick Test

```bash
source ~/.zshrc  # Load API keys
uv run python test_ai_fallback.py
```

### Comprehensive Test

```bash
uv run python test_comprehensive_fallback.py
```

### Manual Test

```python
from ai_advisor import AIAdvisor

advisor = AIAdvisor()

# Test with long prompt (triggers fallback)
long_prompt = "Analyze: " + ("data " * 10000)
response = advisor._call_openai_with_retry(long_prompt)
print(response.choices[0].message.content)
```

---

## Cost Analysis

| Scenario | Provider | Cost (per 1M tokens) | Notes |
|----------|----------|---------------------|-------|
| Normal prompt | OpenAI GPT-4 | Input: $30, Output: $60 | Default |
| Long prompt | Claude Sonnet 4 | Input: $3, Output: $15 | 5x cheaper! |
| Very long history | Claude Sonnet 4 | Input: $3, Output: $15 | 200k context |

**Savings**: Fallback to Claude can reduce costs by **80-90%** for large prompts.

---

## Failure Scenarios

### Scenario 1: OpenAI fails, Claude succeeds
```
✅ User gets response from Claude
⚠️  Message shows fallback occurred
```

### Scenario 2: OpenAI fails, Claude fails
```
❌ Error raised with details
📝 Logs show both failures
💡 Suggests checking API keys
```

### Scenario 3: Claude not configured
```
⚠️  Falls back to shortened prompt
✅ Still tries to complete request
```

### Scenario 4: Both providers down
```
❌ Clear error message
📋 Original error preserved
🔧 User can investigate and retry
```

---

## Monitoring

### Success Messages

```bash
# Normal operation (OpenAI)
(no special message)

# Fallback to Claude
⚠️  OpenAI ... Falling back to Claude...
🤖 Using Claude Sonnet 4 (200k context)...

# Shortened prompt fallback
❌ Claude not available. Trying shortened prompt...
Retrying with shortened prompt (4521 chars vs 7266)
```

### Error Messages

```bash
# Connection errors
⚠️  Connection error: ... Retrying in X seconds...

# Rate limits
⚠️  Rate limit error: ... Retrying in X seconds...

# Complete failure
❌ Both providers failed. Original OpenAI error: ...
```

---

## Best Practices

### 1. Always Set Both API Keys

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. Monitor Fallback Frequency

If Claude is used frequently, consider:
- Using GPT-4 Turbo (128k context) instead of GPT-4 (8k)
- Shortening prompts by default
- Upgrading OpenAI plan for higher token limits

### 3. Log Fallback Events

```python
import logging

logging.info(f"Fallback triggered: {reason}")
```

### 4. Test Regularly

Run tests before important training runs:

```bash
./verify_anthropic_key.sh
uv run python test_comprehensive_fallback.py
```

---

## Integration with Auto-Improvement Pipeline

The fallback works seamlessly with `auto_improvement_loop.py`:

```python
# In auto_improvement_loop.py
advisor = AIAdvisor()  # Fallback enabled by default

# Call analyze_results (uses fallback internally)
analysis, changes = advisor.analyze_results(
    config=current_config,
    results_df=results,
    comparison_df=comparison,
    iteration=iteration,
    iteration_history=history  # Can be very long!
)
```

As iteration history grows (50+ iterations), OpenAI will exceed context limits, but Claude will automatically take over without any code changes needed.

---

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not set"

**Solution**:
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key"
echo 'export ANTHROPIC_API_KEY="sk-ant-your-key"' >> ~/.zshrc
```

### Issue: Both providers fail

**Check**:
1. API keys are valid
2. You have credits/quota remaining
3. Internet connection is working
4. API services are not down (check status pages)

**Debug**:
```bash
# Test OpenAI
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-20250514","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

### Issue: Fallback works but responses are poor

**Check**:
1. System prompt is appropriate for Claude
2. Response parsing works correctly
3. Temperature/max_tokens settings are suitable

**Adjust**:
```python
# In _call_claude method
response = self.claude_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4000,  # Increase if needed
    temperature=0.7,   # Adjust 0-1
    ...
)
```

---

## Summary

✅ **Robust**: Handles any OpenAI error
✅ **Transparent**: Clear logging of fallbacks
✅ **Cost-effective**: Uses cheaper Claude for large prompts
✅ **Reliable**: Multiple fallback layers
✅ **Tested**: Comprehensive test suite
✅ **Production-ready**: Used in auto-improvement pipeline

---

## Files Modified

1. **`ai_advisor.py`**:
   - Added Anthropic client initialization
   - Rewrote `_call_openai_with_retry()` with catch-all fallback
   - Added `_call_claude()` method
   - Added prompt shortening as last resort

2. **Test files created**:
   - `test_ai_fallback.py` - Basic fallback tests
   - `test_comprehensive_fallback.py` - Extensive error scenario tests
   - `verify_anthropic_key.sh` - API key verification

3. **Documentation**:
   - `FIXES_EARLY_STOPPING_AND_AI_FALLBACK.md` - Initial fixes
   - `AI_FALLBACK_COMPLETE.md` - This document

---

**Status**: ✅ **Production Ready**

The fallback mechanism is now fully operational and ready for use in your auto-improvement pipeline. It will handle any errors from OpenAI gracefully, ensuring your training runs never fail due to API issues.