#!/usr/bin/env python3
"""
Polymarket Weather Trading Bot - Main Entry Point.

This bot implements an automated trading strategy that exploits inefficiencies
in Polymarket weather markets using ensemble weather forecasts.

Usage:
    python main.py [--config CONFIG_FILE]

Environment variables can be set in a .env file.
"""

import asyncio
import signal
import sys
from typing import Optional

from config import config, AppConfig
from logger_setup import setup_logging
from weather_client import WeatherAPIManager
from polymarket_client import PolymarketClient
from strategy import WeatherTradingStrategy, TradingEngine
from risk_manager import RiskManager
from notifications import NotificationManager, Notification, NotificationLevel


class TradingBot:
    """
    Main application class that orchestrates all components.
    """

    def __init__(self, config: AppConfig):
        """Initialize trading bot with configuration."""
        self.config = config
        self.running = False

        # Components (initialized later)
        self.notifier: Optional[NotificationManager] = None
        self.weather_manager: Optional[WeatherAPIManager] = None
        self.polymarket_client: Optional[PolymarketClient] = None
        self.strategy: Optional[WeatherTradingStrategy] = None
        self.risk_manager: Optional[RiskManager] = None
        self.trading_engine: Optional[TradingEngine] = None

    async def initialize(self) -> None:
        """Initialize all components asynchronously."""
        # Setup logging first
        setup_logging(
            log_level=self.config.logging.log_level,
            log_file=self.config.logging.log_file,
            log_format=self.config.logging.log_format,
        )

        # Notification manager
        self.notifier = NotificationManager(
            telegram_token=self.config.notification.telegram_token,
            telegram_chat_id=self.config.notification.telegram_chat_id,
            discord_webhook_url=self.config.notification.discord_webhook_url,
        )

        try:
            await self.notifier.send(Notification(
                level=NotificationLevel.INFO,
                title="🚀 Bot Starting",
                message=f"Initializing trading bot (Paper: {self.config.strategy.paper_trading})",
            ))

            # Weather API manager
            self.weather_manager = WeatherAPIManager(
                max_concurrent=self.config.weather.max_concurrent_requests
            )

            # Polymarket client
            if not self.config.polymarket.private_key and not self.config.strategy.paper_trading:
                raise ValueError("POLYMARKET_PRIVATE_KEY required for live trading")

            self.polymarket_client = PolymarketClient(
                private_key=self.config.polymarket.private_key or "",  # Empty string for paper trading
                funder_address=self.config.polymarket.funder_address or "0x0000000000000000000000000000000000000000",
                host=self.config.polymarket.host,
                chain_id=self.config.polymarket.chain_id,
                signature_type=self.config.polymarket.signature_type,
            )
            await self.polymarket_client.initialize()

            # Trading strategy
            self.strategy = WeatherTradingStrategy(
                bankroll_usdc=self.config.strategy.bankroll_usdc,
                min_edge_threshold=self.config.strategy.min_edge_threshold,
                kelly_multiplier=self.config.strategy.kelly_multiplier,
                max_position_pct=self.config.strategy.max_position_pct,
                max_daily_loss_pct=self.config.strategy.max_daily_loss_pct,
                confidence_threshold=self.config.strategy.confidence_threshold,
            )

            # Risk manager
            self.risk_manager = RiskManager(
                polymarket_client=self.polymarket_client,
                max_position_age_hours=self.config.risk.max_position_age_hours,
                default_stop_loss_pct=self.config.risk.default_stop_loss_pct,
                max_daily_loss_usdc=self.config.risk.max_daily_loss_usdc,
                max_portfolio_drawdown_pct=self.config.risk.max_portfolio_drawdown_pct,
                check_interval_seconds=self.config.risk.check_interval_seconds,
            )

            # Trading engine
            self.trading_engine = TradingEngine(
                polymarket_client=self.polymarket_client,
                strategy=self.strategy,
                weather_manager=self.weather_manager,
                locations=self.config.weather.default_locations,
                scan_interval_seconds=self.config.strategy.scan_interval_seconds,
                paper_trading=self.config.strategy.paper_trading,
            )

            await self.notifier.send(Notification(
                level=NotificationLevel.INFO,
                title="✅ Initialization Complete",
                message=f"Bot ready. Monitoring {len(self.config.weather.default_locations)} locations",
            ))

        except Exception as e:
            # Clean up on initialization failure
            await self._cleanup()
            raise

    async def _cleanup(self) -> None:
        """Clean up resources on error."""
        if self.polymarket_client:
            try:
                await self.polymarket_client.close()
            except Exception as e:
                print(f"Error closing Polymarket client: {e}")
        
        if self.weather_manager:
            try:
                await self.weather_manager.close()
            except Exception as e:
                print(f"Error closing weather manager: {e}")
        
        if self.notifier:
            try:
                await self.notifier.close()
            except Exception as e:
                print(f"Error closing notifier: {e}")

    async def run(self) -> None:
        """Run the main trading loop."""
        self.running = True

        # Setup signal handlers for graceful shutdown
        # Note: Windows doesn't support add_signal_handler for SIGINT/SIGTERM
        # We handle KeyboardInterrupt in the main() function instead
        
        # Start risk monitoring in background
        risk_task = asyncio.create_task(self._risk_monitoring_loop())

        # Start portfolio reporting in background
        report_task = asyncio.create_task(self._periodic_reporting())

        try:
            await self.trading_engine.start()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self.notifier.send_error(f"Trading engine error: {e}")
            raise
        finally:
            # Cancel background tasks
            risk_task.cancel()
            report_task.cancel()
            
            # Wait for tasks to finish cancellation
            await asyncio.gather(risk_task, report_task, return_exceptions=True)

    async def _risk_monitoring_loop(self) -> None:
        """Background task for risk monitoring."""
        while self.running:
            try:
                # Update risk manager with current state
                self.risk_manager.update_portfolio_state(
                    positions=self.strategy.positions,
                    bankroll=self.strategy.bankroll,
                    daily_pnl=self.strategy.daily_pnl,
                )
                await self.risk_manager.start_monitoring()
            except Exception as e:
                await self.notifier.send_error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)

    async def _periodic_reporting(self) -> None:
        """Background task for periodic portfolio reporting."""
        while self.running:
            await asyncio.sleep(3600)  # Every hour
            try:
                summary = self.strategy.get_portfolio_summary()
                metrics = self.risk_manager.get_risk_metrics()

                report_data = {
                    **summary,
                    "drawdown_pct": metrics.current_drawdown_pct * 100,
                    "exposure_pct": metrics.max_exposure_pct * 100,
                    "trading_allowed": self.risk_manager.is_trading_allowed(),
                }

                await self.notifier.send_portfolio_update(report_data)
            except Exception as e:
                await self.notifier.send_error(f"Reporting error: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        if not self.running:
            return

        self.running = False
        
        # Stop trading engine first
        if self.trading_engine:
            self.trading_engine.running = False
        
        try:
            await self.notifier.send(Notification(
                level=NotificationLevel.WARNING,
                title="🛑 Shutting Down",
                message="Bot shutdown initiated",
            ))
        except:
            pass

        # Wait a bit for tasks to finish
        await asyncio.sleep(1)
        
        await self._cleanup()
        print("✅ Bot shutdown complete.")


async def main() -> None:
    """Application entry point."""
    bot = TradingBot(config)
    try:
        await bot.initialize()
        
        # Run the bot
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user, shutting down gracefully...")
        await bot.shutdown()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        await bot._cleanup()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
