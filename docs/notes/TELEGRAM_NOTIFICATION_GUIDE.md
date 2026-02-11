# Telegram Notification Setup & Usage Guide

**Date:** 2025-12-29
**Purpose:** Keep you updated on training progress via Telegram with periodic updates, epoch completions, and iteration summaries

---

## 📱 What You'll Receive

### 1. **Periodic Updates (Every 30 minutes)**
Get the last 5 log entries automatically:
```
⏰ Periodic Update (every 30 min)

📋 Recent Activity (last 5 logs):
`18:30:45 | Epoch 5/20: Train=0.0234, Val=0.0289`
`18:31:12 | Epoch 6/20: Train=0.0221, Val=0.0276`
`18:31:45 | Epoch 7/20: Train=0.0215, Val=0.0270`
`18:32:18 | Testing phase started for iteration 1`
`18:35:22 | Testing completed in 3.1 minutes`

⏱️ Last update: 30.0 minutes ago
🕐 Current time: 2025-12-29 18:35:45
```

### 2. **Epoch Completion (After each epoch)**
```
📚 Epoch 7/20 Complete (Iter 1)

📊 Losses:
• Train: 0.0215
• Val: 0.0270
• Ratio: 1.26x ⚠️ OVERFITTING

⏱️ Time: 42.3s
✨ Progress: 35.0%

_Continuing training..._
```

### 3. **Iteration Completion (After each iteration)**
```
✅ Iteration 1/10 Completed

📊 Test Results:
• AUC: 0.7895
• F1: 0.1234
• Recall: 0.3456
• Precision: 0.0987

📉 Losses:
• Train: 0.0215
• Val: 0.0270 ⚠️ Slight overfitting

⏱️ Time: 45.3 minutes
✨ Progress: 10.0%

📋 Recent Activity:
`AI analysis completed`
`Iteration 1 completed: AUC=0.7895, F1=0.1234`
```

### 4. **Pipeline Completion**
```
🎉 Pipeline Completed!

✅ Completed: 10 iterations
⏱️ Total time: 7.5 hours

🏆 Best Result:
• Iteration: 5
• AUC: 0.8234

_Check final report for details_
```

---

## 🔧 Setup Instructions

### Step 1: Create a Telegram Bot

1. **Open Telegram** and search for `@BotFather`
2. **Send `/newbot`** command
3. **Choose a name** for your bot (e.g., "My Training Monitor")
4. **Choose a username** (must end in 'bot', e.g., "my_training_monitor_bot")
5. **Copy the Bot Token** - it looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

### Step 2: Get Your Chat ID

**Method 1: Using a helper bot (Easiest)**
1. Search for `@userinfobot` in Telegram
2. Start a chat with it
3. It will display your Chat ID (e.g., `987654321`)

**Method 2: Using your bot**
1. Start a chat with your new bot
2. Send any message to it
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Look for `"chat":{"id":987654321}` in the JSON response
5. Copy the number (e.g., `987654321`)

### Step 3: Set Environment Variables

**Option A: Using `.env` file (Recommended)**

Create a file named `.env` in your project root:
```bash
# .env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
```

**Option B: Export in terminal**
```bash
export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
export TELEGRAM_CHAT_ID="987654321"
```

**Option C: Add to `.bashrc` or `.zshrc` (Permanent)**
```bash
echo 'export TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrsTUVwxyz"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="987654321"' >> ~/.bashrc
source ~/.bashrc
```

---

## ✅ Testing Your Setup

### Test 1: Quick Test

Run the test script:
```bash
python telegram_notifier.py
```

**Expected output:**
```
Testing Telegram notifications...
✅ Test message sent successfully!
```

**You should receive:**
```
📢 Test Notification

Telegram notifier is working correctly! ✅
```

### Test 2: Test All Notification Types

Create a test script `test_telegram_full.py`:
```python
from telegram_notifier import TelegramNotifier
import time

# Initialize
telegram = TelegramNotifier()

if not telegram.enabled:
    print("❌ Telegram not configured")
    exit(1)

print("🧪 Testing all notification types...\n")

# Test 1: Iteration start
print("1. Testing iteration start...")
telegram.send_iteration_start(1, 10, "config_baseline.yaml")
time.sleep(2)

# Test 2: Epoch complete
print("2. Testing epoch complete...")
telegram.send_epoch_complete(
    epoch=5,
    total_epochs=20,
    train_loss=0.0234,
    val_loss=0.0289,
    iteration=1,
    time_seconds=42.3
)
time.sleep(2)

# Test 3: Iteration complete
print("3. Testing iteration complete...")
telegram.send_iteration_complete(
    iteration=1,
    total_iterations=10,
    avg_auc=0.7895,
    avg_f1=0.1234,
    avg_recall=0.3456,
    avg_precision=0.0987,
    time_minutes=45.3,
    train_loss=0.0215,
    val_loss=0.0270
)
time.sleep(2)

# Test 4: Add logs and periodic update
print("4. Testing periodic update with logs...")
telegram.add_log("Epoch 5/20: Train=0.0234, Val=0.0289")
telegram.add_log("Epoch 6/20: Train=0.0221, Val=0.0276")
telegram.add_log("Testing phase started")
telegram.add_log("Testing completed in 3.1 minutes")
telegram.add_log("AI analysis started")
telegram.send_periodic_update()
time.sleep(2)

# Test 5: Pipeline complete
print("5. Testing pipeline complete...")
telegram.send_pipeline_complete(
    total_iterations=10,
    total_time_hours=7.5,
    best_auc=0.8234,
    best_iteration=5
)

print("\n✅ All tests completed! Check your Telegram to see the messages.")
```

Run it:
```bash
python test_telegram_full.py
```

---

## 🚀 Using with Auto-Improvement Loop

### Start Training with Telegram Notifications

```bash
# Make sure environment variables are set
echo $TELEGRAM_BOT_TOKEN  # Should show your token
echo $TELEGRAM_CHAT_ID    # Should show your chat ID

# Run the auto-improvement loop
python auto_improvement_loop.py \
    --config config_baseline.yaml \
    --iterations 10 \
    --output-dir auto_improvement_runs
```

### What Happens:

1. **At Start:**
   - Telegram periodic updates start (every 30 minutes)
   - You receive "Iteration Started" message
   - Logging handler captures important log messages

2. **During Training:**
   - After each epoch: Receive epoch completion with train/val loss
   - Every 30 minutes: Receive periodic update with last 5 logs
   - All important log messages are captured automatically

3. **After Iteration:**
   - Receive iteration summary with:
     - Test metrics (AUC, F1, Recall, Precision)
     - Train/Val losses
     - Overfitting/underfitting status
     - Recent activity logs

4. **At End:**
   - Receive pipeline completion summary with best result

---

## 🔍 Troubleshooting

### Issue: "Telegram notifications disabled"

**Solution:**
```bash
# Check if environment variables are set
env | grep TELEGRAM

# If not set, export them
export TELEGRAM_BOT_TOKEN="your_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

### Issue: "Failed to send Telegram notification"

**Possible causes:**
1. **Invalid Bot Token** - Check for typos
2. **Wrong Chat ID** - Make sure it's a number
3. **Bot not started** - Send any message to your bot first
4. **Network issues** - Check internet connection

**Debug:**
```python
from telegram_notifier import TelegramNotifier

telegram = TelegramNotifier()
print(f"Bot Token: {telegram.bot_token[:20]}...")  # Shows first 20 chars
print(f"Chat ID: {telegram.chat_id}")
print(f"Enabled: {telegram.enabled}")

# Try sending a test message
result = telegram.send_custom_message("Test", "Debug test message")
print(f"Send result: {result}")
```

### Issue: Not receiving periodic updates

**Check:**
1. Training must run for at least 30 minutes
2. Telegram notifier must be enabled
3. Periodic timer should be started (check logs for "📱 Telegram periodic updates started")

### Issue: Receiving too many notifications

**Customize the logging handler:**

Edit `telegram_notifier.py` around line 334 to adjust keywords:
```python
self.keywords = [
    'epoch', 'iteration', 'completed',  # Keep these
    # 'batch',  # Comment out to reduce noise
]
```

Or increase the log buffer capture threshold:
```python
# Only capture WARNING and above
telegram_handler = self.telegram.create_logging_handler(level=logging.WARNING)
```

---

## 📊 Notification Frequency

| Notification Type | Frequency | Typical Count (10 iterations, 20 epochs each) |
|------------------|-----------|----------------------------------------------|
| Iteration Start | Once per iteration | 10 |
| Epoch Complete | Once per epoch | 200 (10 × 20) |
| Iteration Complete | Once per iteration | 10 |
| Periodic Update | Every 30 minutes | Depends on duration (~16 for 8 hours) |
| Pipeline Complete | Once at end | 1 |
| **Total** | | **~237 messages** |

**Note:** If this is too many, you can:
- Disable epoch notifications (comment out the `send_epoch_complete` call in `config_based_pipeline.py:497-504`)
- Increase periodic interval to 60 minutes (change `self.periodic_interval = 60 * 60` in `telegram_notifier.py:38`)

---

## 🎨 Customizing Notifications

### Change Periodic Update Interval

Edit `telegram_notifier.py` line 38:
```python
# Change from 30 minutes to 60 minutes
self.periodic_interval = 60 * 60  # seconds
```

### Change Number of Recent Logs

Edit `telegram_notifier.py` when calling `get_recent_logs()`:
```python
# Show last 10 logs instead of 5
recent_logs = self.get_recent_logs(10)
```

### Disable Specific Notifications

**Disable epoch notifications:**
Comment out in `config_based_pipeline.py:494-504`:
```python
# if self.telegram:
#     self.telegram.add_log(...)
#     self.telegram.send_epoch_complete(...)
```

**Disable periodic updates:**
Comment out in `auto_improvement_loop.py:584`:
```python
# self.telegram.start_periodic_updates()
```

---

## 📝 Summary

✅ **Setup Steps:**
1. Create Telegram bot with @BotFather
2. Get bot token and chat ID
3. Set environment variables
4. Test with `python telegram_notifier.py`

✅ **Features:**
- ⏰ Periodic updates every 30 minutes
- 📚 Epoch completion notifications
- ✅ Iteration summaries with metrics
- 🎉 Pipeline completion
- 📋 Automatic log capture

✅ **Customizable:**
- Update frequency
- Number of logs
- Notification types
- Keywords for log capture

---

**Ready to start?**
```bash
# Quick setup
export TELEGRAM_BOT_TOKEN="your_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python telegram_notifier.py  # Test
python auto_improvement_loop.py --config config_baseline.yaml --iterations 1  # Run
```

Stay updated on your training progress from anywhere! 📱🚀
