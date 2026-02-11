# Telegram Notifications Setup Guide

Get real-time updates on your phone when iterations start, complete, or fail!

## 📱 What You'll Get

### Notifications for:
- 🚀 **Iteration Start** - When each iteration begins
- ✅ **Iteration Complete** - Results summary (AUC, F1, Recall, Precision, Time)
- ❌ **Iteration Failed** - Error notifications
- 🎉 **Pipeline Complete** - Final summary with best results

### Example Notifications:

```
🚀 Iteration 3/10 Started

📋 Config: config_iteration_003.yaml
⏰ Started: 2024-12-27 15:30:00

Auto-improvement pipeline is running...
```

```
✅ Iteration 3/10 Completed

📊 Results:
• AUC: 0.7845
• F1: 0.0234
• Recall: 0.0123
• Precision: 0.1234

⏱️ Time: 149.8 minutes
✨ Progress: 30.0%
```

## 🔧 Setup (5 Minutes)

### Step 1: Create a Telegram Bot

1. **Open Telegram** and search for **@BotFather**
2. **Start a chat** with BotFather
3. **Send** `/newbot`
4. **Choose a name** for your bot (e.g., "My Pipeline Notifier")
5. **Choose a username** for your bot (must end in 'bot', e.g., "my_pipeline_bot")
6. **Copy the bot token** - It looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### Step 2: Get Your Chat ID

1. **Search for your bot** in Telegram (the username you just created)
2. **Start a chat** with your bot (click START or send any message)
3. **Visit this URL** in your browser (replace with your bot token):
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   Example:
   ```
   https://api.telegram.org/bot123456789:ABCdefGHIjklMNOpqrsTUVwxyz/getUpdates
   ```
4. **Find your chat ID** in the response - Look for `"chat":{"id":123456789,...}`
5. **Copy the chat ID** (the number after `"id":`)

### Step 3: Set Environment Variables

#### Option A: For Current Session
```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="123456789"
```

#### Option B: Permanent (Recommended)

Add to your `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"' >> ~/.zshrc
echo 'export TELEGRAM_CHAT_ID="123456789"' >> ~/.zshrc
source ~/.zshrc
```

### Step 4: Test the Setup

```bash
uv run telegram_notifier.py
```

Expected output:
```
Testing Telegram notifications...
✅ Test message sent successfully!
```

You should receive a test message in Telegram!

## 🚀 Using Telegram Notifications

### Automatic (Already Integrated!)

Just run your pipeline as normal:

```bash
uv run auto_improvement_loop.py --iterations 10
```

**Notifications are sent automatically** when:
- Each iteration starts
- Each iteration completes
- An iteration fails
- The entire pipeline completes

### Notifications are Optional

If you **don't set** the environment variables:
- Pipeline still works normally
- Just shows a warning: `⚠️ Telegram notifications disabled`
- No errors or interruptions

## 📊 What Each Notification Contains

### Iteration Start
```
🚀 Iteration 1/10 Started

📋 Config: config_baseline.yaml
⏰ Started: 2024-12-27 14:30:00

Auto-improvement pipeline is running...
```

### Iteration Complete
```
✅ Iteration 1/10 Completed

📊 Results:
• AUC: 0.7845
• F1: 0.0234
• Recall: 0.0123
• Precision: 0.1234

⏱️ Time: 149.8 minutes
✨ Progress: 10.0%
```

### Iteration Failed
```
❌ Iteration 1 Failed

⚠️ Error: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'

Check logs for details
```

### Pipeline Complete
```
🎉 Pipeline Completed!

✅ Completed: 10 iterations
⏱️ Total time: 25.3 hours

🏆 Best Result:
• Iteration: 7
• AUC: 0.8523

Check final report for details
```

## 🔍 Verification Checklist

- [ ] Created Telegram bot via @BotFather
- [ ] Got bot token (format: `123456789:ABC...`)
- [ ] Started chat with your bot
- [ ] Got chat ID from getUpdates URL
- [ ] Set TELEGRAM_BOT_TOKEN environment variable
- [ ] Set TELEGRAM_CHAT_ID environment variable
- [ ] Ran test script successfully
- [ ] Received test message in Telegram

## 🐛 Troubleshooting

### "Telegram notifications disabled"
**Problem**: Environment variables not set

**Solution**:
```bash
# Check if variables are set
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# If empty, set them:
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

### "Failed to send Telegram notification"
**Problem**: Invalid token or chat ID

**Solution**:
1. Verify bot token is correct
2. Verify you started a chat with the bot
3. Verify chat ID is correct (use getUpdates URL again)

### No test message received
**Problem**: Wrong chat ID or didn't start chat with bot

**Solution**:
1. Open Telegram and search for your bot
2. Click START or send any message
3. Get chat ID again from getUpdates URL

### Message not formatted correctly
**Problem**: Special characters in messages

**Solution**: This is handled automatically by the library

## 💡 Tips

### For Long Runs
Enable notifications to monitor progress without checking your computer:
- Get notified when iterations complete
- See if performance is improving
- Get alerted if something fails

### For Overnight Runs
Perfect for knowing when to check results:
- Start pipeline before bed
- Wake up to notifications showing progress
- Know exactly when it's done

### For Multi-Day Runs
Track progress across days:
- See each iteration complete
- Monitor performance trends
- Get final summary when done

## 🔐 Security Notes

### Keep Your Tokens Private
- **Don't commit** tokens to git
- **Don't share** tokens publicly
- **Don't hardcode** tokens in scripts

### Revoke if Compromised
If your bot token is exposed:
1. Talk to @BotFather
2. Send `/mybots`
3. Select your bot
4. Choose "API Token" → "Revoke current token"
5. Get new token and update environment variable

## 🎯 Quick Reference

### Environment Variables
```bash
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID="123456789"
```

### Test Command
```bash
uv run telegram_notifier.py
```

### Run with Notifications
```bash
# Just run normally - notifications are automatic
uv run auto_improvement_loop.py --iterations 10
```

### Disable Notifications
```bash
# Unset environment variables
unset TELEGRAM_BOT_TOKEN
unset TELEGRAM_CHAT_ID
```

---

## 📱 Example Setup Session

```bash
# 1. Set variables (use your actual values)
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="987654321"

# 2. Test
uv run telegram_notifier.py
# Output: ✅ Test message sent successfully!

# 3. Run pipeline
uv run auto_improvement_loop.py --iterations 3

# You'll get notifications:
# - When iteration 1 starts
# - When iteration 1 completes (with metrics)
# - When iteration 2 starts
# - When iteration 2 completes (with metrics)
# - When iteration 3 starts
# - When iteration 3 completes (with metrics)
# - When pipeline completes (with best result)
```

---

**Now you can monitor your pipeline from anywhere!** 📱✨

Perfect for long-running experiments - you'll know exactly when each iteration completes and how it's performing!
