"""
Notification module for trade alerts and system monitoring.

Supports Telegram, Discord webhooks, and console logging.
"""

import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Severity level for notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"


@dataclass
class Notification:
    """Container for a notification message."""
    level: NotificationLevel
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class NotificationManager:
    """
    Manages sending notifications to configured channels.

    Supports:
    - Telegram bot
    - Discord webhook
    - Console output (always enabled)
    """

    def __init__(
            self,
            telegram_token: Optional[str] = None,
            telegram_chat_id: Optional[str] = None,
            discord_webhook_url: Optional[str] = None,
            min_level: NotificationLevel = NotificationLevel.INFO,
    ):
        """
        Initialize notification manager.

        Args:
            telegram_token: Telegram bot token.
            telegram_chat_id: Telegram chat ID to send message to.
            discord_webhook_url: Discord webhook URL.
            min_level: Minimum level of notifications to send.
        """
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook_url = discord_webhook_url
        self.min_level = min_level

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send(self, notification: Notification) -> None:
        """
        Send notification to all configured channels.

        Args:
            notification: Notification object to send.
        """
        if notification.level.value < self.min_level.value:
            return

        # Always log to console
        self._log_to_console(notification)

        # Send to Telegram and Discord concurrently
        tasks = []
        if self.telegram_token and self.telegram_chat_id:
            tasks.append(self._send_telegram(notification))
        if self.discord_webhook_url:
            tasks.append(self._send_discord(notification))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _log_to_console(self, notification: Notification) -> None:
        """Log notification to console with appropriate level."""
        msg = f"[{notification.level.value.upper()}] {notification.title}: {notification.message}"
        if notification.data:
            msg += f" | Data: {notification.data}"

        if notification.level == NotificationLevel.ERROR:
            logger.error(msg)
        elif notification.level == NotificationLevel.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)

    async def _send_telegram(self, notification: Notification) -> None:
        """Send notification via Telegram bot."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        text = self._format_message(notification, platform="telegram")

        payload = {
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram send failed: {resp.status}")
        except Exception as err:
            logger.error(f"Telegram notification error: {err}")

    async def _send_discord(self, notification: Notification) -> None:
        """Send notification via discord webhook."""
        if not self.discord_webhook_url:
            return

        color_map = {
            NotificationLevel.INFO: 3447003,  # Blue
            NotificationLevel.WARNING: 16776960,  # Yellow
            NotificationLevel.ERROR: 15158332,  # Red
            NotificationLevel.TRADE: 3066993,  # Green
        }

        embed = {
            "title": notification.title,
            "description": notification.message,
            "color": color_map.get(notification.level, 0),
            "timestamp": notification.timestamp.isoformat(),
        }

        if notification.data:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in notification.data.items()
            ]

        payload = {"embeds": [embed]}

        try:
            session = await self._get_session()
            async with session.post(self.discord_webhook_url, json=payload) as resp:
                if resp.status not in (200, 204):
                    logger.error(f"Discord send failed: {resp.status}")
        except Exception as err:
            logger.error(f"Discord notification error: {err}")

    def _format_message(self, notification: Notification, platform: str) -> str:
        """Format notification message for specific platform."""
        if platform == "telegram":
            emoji_map = {
                NotificationLevel.INFO: "ℹ️",
                NotificationLevel.WARNING: "⚠️",
                NotificationLevel.ERROR: "❌",
                NotificationLevel.TRADE: "💰",
            }
            emoji = emoji_map.get(notification.level, "")
            text = f"{emoji} <b>{notification.title}</b>\n{notification.message}"
            if notification.data:
                for k, v in notification.data.items():
                    text += f"\n• {k}: {v}"
            return text
        else:
            return f"{notification.title}\n{notification.message}"

    # Convenience methods for common notifications
    async def send_trade_signal(self, signal_data: Dict[str, Any]) -> None:
        """Send notification about new trading signal."""
        await self.send(Notification(
            level=NotificationLevel.TRADE,
            title="🎯 New Trading Signal",
            message=f"Signal: {signal_data.get('side')} {signal_data.get('market')}",
            data=signal_data,
        ))

    async def send_trade_executed(self, result_data: Dict[str, Any]) -> None:
        """Send notification about executed trade."""
        await self.send(Notification(
            level=NotificationLevel.TRADE,
            title="✅ Trade Executed",
            message=f"Order {result_data.get('order_id')} filled",
            data=result_data,
        ))

    async def send_error(
            self,
            error_msg: str,
            context: Dict[str, Any] = None
    ) -> None:
        """Send error notification."""
        await self.send(Notification(
            level=NotificationLevel.ERROR,
            title="🚨 System Error",
            message=error_msg,
            data=context,
        ))

    async def send_portfolio_update(self, summary: Dict[str, Any]) -> None:
        """Send periodic portfolio update."""
        await self.send(Notification(
            level=NotificationLevel.INFO,
            title="📊 Portfolio Update",
            message=f"Bankroll: ${summary.get('bankroll', 0):.2f} | PnL: {summary.get('total_return_pct', 0):.2f}%",
            data=summary
        ))
