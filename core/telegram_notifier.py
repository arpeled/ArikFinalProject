"""
Telegram Notification Module
Sends notifications about pipeline progress to Telegram
"""

import os
import urllib.request
import urllib.parse
import json
import time
import threading
import logging
from typing import Optional, List
from collections import deque
from datetime import datetime


class TelegramNotifier:
    """Send notifications to Telegram"""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram notifier

        Args:
            bot_token: Telegram bot token (or use TELEGRAM_BOT_TOKEN env var)
            chat_id: Telegram chat ID (or use TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)

        # Log buffer to track recent messages
        self.log_buffer = deque(maxlen=100)  # Keep last 100 log entries
        self.lock = threading.Lock()  # Thread-safe log buffer

        # Periodic update timer
        self.periodic_timer = None
        self.periodic_interval = 30 * 60  # 30 minutes in seconds
        self.last_periodic_update = time.time()

        if not self.enabled:
            print("⚠️  Telegram notifications disabled (no bot token or chat ID)")

    def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to Telegram

        Args:
            message: Message text (supports Markdown)
            parse_mode: Message format (Markdown or HTML)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }

            # Encode data
            data_encoded = urllib.parse.urlencode(data).encode('utf-8')

            # Send request
            req = urllib.request.Request(url, data=data_encoded, method='POST')
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('ok', False)

        except urllib.error.URLError as e:
            # Network/DNS errors - not critical, just log
            if not hasattr(self, '_network_error_logged'):
                print(f"⚠️  Telegram network error (notifications disabled): {e}")
                print("    Training will continue without Telegram notifications.")
                print("    Check your internet connection or Telegram configuration.")
                self._network_error_logged = True  # Only show once
            return False
        except Exception as e:
            if not hasattr(self, '_error_logged'):
                print(f"⚠️  Failed to send Telegram notification: {e}")
                self._error_logged = True  # Only show once
            return False

    def add_log(self, log_message: str):
        """
        Add a log message to the buffer

        Args:
            log_message: Log message to add
        """
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_buffer.append(f"{timestamp} | {log_message}")

    def get_recent_logs(self, n: int = 5) -> List[str]:
        """
        Get the most recent n log messages

        Args:
            n: Number of recent logs to retrieve

        Returns:
            List of recent log messages
        """
        with self.lock:
            # Get last n messages from deque
            if len(self.log_buffer) == 0:
                return ["No logs available yet"]
            return list(self.log_buffer)[-n:]

    def start_periodic_updates(self):
        """Start sending periodic updates every 30 minutes"""
        if not self.enabled:
            return

        def send_periodic():
            while True:
                time.sleep(self.periodic_interval)
                self.send_periodic_update()

        self.periodic_timer = threading.Thread(target=send_periodic, daemon=True)
        self.periodic_timer.start()
        print("📱 Telegram periodic updates started (every 30 minutes)")

    def stop_periodic_updates(self):
        """Stop periodic updates"""
        if self.periodic_timer and self.periodic_timer.is_alive():
            # Note: daemon thread will stop when main program exits
            print("📱 Telegram periodic updates will stop when program exits")

    def send_periodic_update(self):
        """Send periodic update with recent logs"""
        if not self.enabled:
            return False

        recent_logs = self.get_recent_logs(5)
        logs_text = "\n".join([f"`{log}`" for log in recent_logs])

        elapsed = (time.time() - self.last_periodic_update) / 60  # minutes
        self.last_periodic_update = time.time()

        message = f"""
⏰ *Periodic Update* (every 30 min)

📋 *Recent Activity* (last 5 logs):
{logs_text}

⏱️ Last update: `{elapsed:.1f}` minutes ago
🕐 Current time: `{self._get_timestamp()}`

_Pipeline is still running..._
"""
        return self.send_message(message.strip())

    def send_epoch_complete(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        iteration: int,
        time_seconds: float
    ):
        """
        Send notification when epoch completes

        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            iteration: Current iteration number
            time_seconds: Time taken for this epoch
        """
        progress = (epoch / total_epochs) * 100
        loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0

        # Determine overfitting status
        if loss_ratio > 1.2:
            status = "⚠️ OVERFITTING"
        elif loss_ratio > 1.1:
            status = "⚠️ Possible overfitting"
        else:
            status = "✅ Balanced"

        message = f"""
📚 *Epoch {epoch}/{total_epochs} Complete* (Iter {iteration})

📊 *Losses:*
• Train: `{train_loss:.4f}`
• Val: `{val_loss:.4f}`
• Ratio: `{loss_ratio:.2f}x` {status}

⏱️ Time: `{time_seconds:.1f}s`
✨ Progress: `{progress:.1f}%`

_Continuing training..._
"""
        return self.send_message(message.strip())

    def send_iteration_start(self, iteration: int, total_iterations: int, config_path: str):
        """Send notification when iteration starts"""
        message = f"""
🚀 *Iteration {iteration}/{total_iterations} Started*

📋 Config: `{config_path}`
⏰ Started: `{self._get_timestamp()}`

_Auto-improvement pipeline is running..._
"""
        # Log this event
        self.add_log(f"Iteration {iteration}/{total_iterations} started")
        return self.send_message(message.strip())

    def send_iteration_complete(
        self,
        iteration: int,
        total_iterations: int,
        avg_auc: float,
        avg_f1: float,
        avg_recall: float,
        avg_precision: float,
        time_minutes: float,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None
    ):
        """Send notification when iteration completes"""
        # Build loss info if available
        loss_info = ""
        if train_loss is not None and val_loss is not None:
            loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0
            if loss_ratio > 1.2:
                status = "⚠️ Overfitting"
            elif loss_ratio > 1.1:
                status = "⚠️ Slight overfitting"
            else:
                status = "✅ Balanced"
            loss_info = f"""
📉 *Losses:*
• Train: `{train_loss:.4f}`
• Val: `{val_loss:.4f}` {status}
"""

        message = f"""
✅ *Iteration {iteration}/{total_iterations} Completed*

📊 *Test Results:*
• AUC: `{avg_auc:.4f}`
• F1: `{avg_f1:.4f}`
• Recall: `{avg_recall:.4f}`
• Precision: `{avg_precision:.4f}`
{loss_info}
⏱️ Time: `{time_minutes:.1f} minutes`
✨ Progress: `{(iteration/total_iterations)*100:.1f}%`

📋 *Recent Activity:*
{chr(10).join([f"`{log}`" for log in self.get_recent_logs(3)])}
"""
        # Log this event
        self.add_log(f"Iteration {iteration} completed: AUC={avg_auc:.4f}, F1={avg_f1:.4f}")
        return self.send_message(message.strip())

    def send_iteration_failed(self, iteration: int, error: str):
        """Send notification when iteration fails"""
        message = f"""
❌ *Iteration {iteration} Failed*

⚠️ Error: `{error[:200]}`

_Check logs for details_
"""
        return self.send_message(message.strip())

    def send_pipeline_complete(
        self,
        total_iterations: int,
        total_time_hours: float,
        best_auc: float,
        best_iteration: int
    ):
        """Send notification when entire pipeline completes"""
        message = f"""
🎉 *Pipeline Completed!*

✅ Completed: `{total_iterations} iterations`
⏱️ Total time: `{total_time_hours:.1f} hours`

🏆 *Best Result:*
• Iteration: `{best_iteration}`
• AUC: `{best_auc:.4f}`

_Check final report for details_
"""
        return self.send_message(message.strip())

    def send_custom_message(self, title: str, details: str):
        """Send a custom notification"""
        message = f"""
📢 *{title}*

{details}
"""
        return self.send_message(message.strip())

    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_logging_handler(self, level=logging.INFO):
        """
        Create a custom logging handler that captures logs to Telegram buffer

        Args:
            level: Logging level (default: INFO)

        Returns:
            TelegramLoggingHandler instance
        """
        return TelegramLoggingHandler(self, level)


class TelegramLoggingHandler(logging.Handler):
    """Custom logging handler that captures logs to Telegram notifier"""

    def __init__(self, telegram_notifier: TelegramNotifier, level=logging.INFO):
        """
        Initialize the handler

        Args:
            telegram_notifier: TelegramNotifier instance
            level: Logging level
        """
        super().__init__(level)
        self.telegram = telegram_notifier

        # Filter out noise - only capture important messages
        self.keywords = [
            'epoch', 'iteration', 'completed', 'started', 'loss', 'auc', 'f1',
            'training', 'testing', 'analysis', 'error', 'warning', 'failed',
            'success', 'finished', 'phase', 'batch'
        ]

    def emit(self, record):
        """
        Emit a log record to Telegram buffer

        Args:
            record: LogRecord instance
        """
        try:
            msg = self.format(record)

            # Only capture important messages (contains keywords or is WARNING+)
            if record.levelno >= logging.WARNING or any(kw in msg.lower() for kw in self.keywords):
                # Remove timestamp prefix if exists (telegram adds its own)
                # Extract just the message part after " - "
                parts = msg.split(" - ", 1)
                if len(parts) > 1:
                    clean_msg = parts[1]
                else:
                    clean_msg = msg

                # Truncate long messages
                if len(clean_msg) > 150:
                    clean_msg = clean_msg[:147] + "..."

                self.telegram.add_log(clean_msg)

        except Exception:
            # Fail silently - don't break logging if Telegram fails
            pass


# Test function
def test_telegram_notifier():
    """Test the Telegram notifier"""
    notifier = TelegramNotifier()

    if not notifier.enabled:
        print("❌ Telegram not configured")
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        return False

    print("Testing Telegram notifications...")

    # Test message
    success = notifier.send_custom_message(
        "Test Notification",
        "Telegram notifier is working correctly! ✅"
    )

    if success:
        print("✅ Test message sent successfully!")
        return True
    else:
        print("❌ Failed to send test message")
        return False


if __name__ == "__main__":
    test_telegram_notifier()
