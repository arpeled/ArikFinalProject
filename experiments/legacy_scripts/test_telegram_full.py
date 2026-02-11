"""
Full Telegram Notification Test Script
Tests all notification types to verify setup is working correctly
"""

from telegram_notifier import TelegramNotifier
import time

def main():
    print("="*60)
    print("🧪 Telegram Notification Full Test")
    print("="*60)
    print()

    # Initialize
    telegram = TelegramNotifier()

    if not telegram.enabled:
        print("❌ Telegram not configured!")
        print()
        print("Please set these environment variables:")
        print("  export TELEGRAM_BOT_TOKEN='your_bot_token'")
        print("  export TELEGRAM_CHAT_ID='your_chat_id'")
        print()
        print("See TELEGRAM_NOTIFICATION_GUIDE.md for setup instructions")
        return 1

    print(f"✅ Telegram enabled")
    print(f"   Bot token: {telegram.bot_token[:20]}...")
    print(f"   Chat ID: {telegram.chat_id}")
    print()

    print("🚀 Starting tests (you should receive 6 Telegram messages)...")
    print()

    # Test 1: Iteration start
    print("1/6 Testing iteration start notification...")
    telegram.send_iteration_start(1, 10, "config_baseline.yaml")
    time.sleep(2)
    print("    ✅ Sent")

    # Test 2: Epoch complete
    print("2/6 Testing epoch complete notification...")
    telegram.send_epoch_complete(
        epoch=5,
        total_epochs=20,
        train_loss=0.0234,
        val_loss=0.0289,
        iteration=1,
        time_seconds=42.3
    )
    time.sleep(2)
    print("    ✅ Sent")

    # Test 3: Another epoch (overfitting)
    print("3/6 Testing epoch with overfitting warning...")
    telegram.send_epoch_complete(
        epoch=10,
        total_epochs=20,
        train_loss=0.0180,
        val_loss=0.0250,  # Ratio > 1.2, should trigger overfitting warning
        iteration=1,
        time_seconds=38.7
    )
    time.sleep(2)
    print("    ✅ Sent")

    # Test 4: Add logs and test log buffer
    print("4/6 Testing log buffer...")
    telegram.add_log("Epoch 5/20: Train=0.0234, Val=0.0289")
    telegram.add_log("Epoch 6/20: Train=0.0221, Val=0.0276")
    telegram.add_log("Testing phase started")
    telegram.add_log("Testing completed in 3.1 minutes")
    telegram.add_log("AI analysis started")
    print("    ✅ Added 5 logs to buffer")

    # Test 5: Periodic update
    print("5/6 Testing periodic update (with logs)...")
    telegram.send_periodic_update()
    time.sleep(2)
    print("    ✅ Sent")

    # Test 6: Iteration complete
    print("6/6 Testing iteration complete notification...")
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
    print("    ✅ Sent")

    # Test 7: Pipeline complete
    print("7/7 Testing pipeline complete notification...")
    telegram.send_pipeline_complete(
        total_iterations=10,
        total_time_hours=7.5,
        best_auc=0.8234,
        best_iteration=5
    )
    time.sleep(2)
    print("    ✅ Sent")

    print()
    print("="*60)
    print("✅ All tests completed successfully!")
    print("="*60)
    print()
    print("Check your Telegram to verify you received 7 messages:")
    print("  1. Iteration Started")
    print("  2. Epoch Complete (balanced)")
    print("  3. Epoch Complete (with overfitting warning)")
    print("  4. Periodic Update (with 5 recent logs)")
    print("  5. Iteration Completed (with metrics & losses)")
    print("  6. Pipeline Completed")
    print()
    print("If you didn't receive them, check:")
    print("  - Bot token is correct")
    print("  - Chat ID is correct")
    print("  - You've started a chat with your bot")
    print("  - Your internet connection is working")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
