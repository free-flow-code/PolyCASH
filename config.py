"""
Configuration management for the Polymarket trading bot.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class WeatherConfig:
    """Weather API configuration."""
    cache_ttl_seconds: int = 300
    max_concurrent_requests: int = 10
    default_locations: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("New York", 40.7128, -74.060),
        ("London", 51.5074, -0.1278),
        ("Chicago", 41.8781, -87.6298),
        ("Los Angeles", 34.0522, -118.2437),
    ])


@dataclass
class PolymarketConfig:
    """Polymarket API configuration."""
    host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    signature_type: int = 1
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    private_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY", ""))
    funder_address: str = field(default_factory=lambda: os.getenv("POLYMARKET_FUNDER_ADDRESS", ""))


@dataclass
class StrategyConfig:
    """Trading strategy parameters."""
    bankroll_usdc: float = 1000.0
    min_edge_threshold: float = 0.05
    kelly_multiplier: float = 0.25
    max_position_pct: float = 0.03
    max_daily_loss_pct: float = 0.10
    confidence_threshold: float = 0.70
    scan_interval_seconds: int = 300
    paper_trading: bool = True


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_position_age_hours: float = 72.0
    default_stop_loss_pct: float = 0.20
    max_daily_loss_usdc: float = 100.0
    max_portfolio_drawdown_pct: float = 0.25
    check_interval_seconds: int = 60


@dataclass
class NotificationConfig:
    """Notification settings."""
    telegram_token: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID"))
    discord_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL"))
    min_notification_level: str = "INFO"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_file: Optional[str] = "trading_bot.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AppConfig:
    """Master configuration aggregator."""
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Strategy config from env
        if os.getenv("BANKROLL_USDC"):
            config.strategy.bankroll_usdc = float(os.getenv("BANKROLL_USDC"))
        if os.getenv("PAPER_TRADING"):
            config.strategy.paper_trading = os.getenv("PAPER_TRADING").lower() == "true"
        if os.getenv("SCAN_INTERVAL_SECONDS"):
            config.strategy.scan_interval_seconds = int(os.getenv("SCAN_INTERVAL_SECONDS"))
        if os.getenv("MIN_EDGE_THRESHOLD"):
            config.strategy.min_edge_threshold = float(os.getenv("MIN_EDGE_THRESHOLD"))
        if os.getenv("KELLY_MULTIPLIER"):
            config.strategy.kelly_multiplier = float(os.getenv("KELLY_MULTIPLIER"))

        # Risk config from env
        if os.getenv("MAX_DAILY_LOSS_USDC"):
            config.risk.max_daily_loss_usdc = float(os.getenv("MAX_DAILY_LOSS_USDC"))

        # Logging config from env
        if os.getenv("LOG_LEVEL"):
            config.logging.log_level = os.getenv("LOG_LEVEL")

        return config


config = AppConfig.from_env()
