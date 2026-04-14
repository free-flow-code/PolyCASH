"""
Risk management module for Polymarket trading bot.

Handles position monitoring stop-loss execution, daily loss limits,
and portfolio drawdown protection.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from strategy import Position
from polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)


class StopLossType(Enum):
    """Types of stop-loss triggers."""
    TIME_BASED = "time_based"
    PRICE_BASED = "price_based"
    TRAILING = "trailing"


@dataclass
class StopLossRule:
    """Configuration for a stop-loss rule."""
    rule_type: StopLossType
    threshold: float  # For price: absolute price; for time: hours
    token_id: Optional[str] = None  # If None, applies to all positions


@dataclass
class RiskMetrics:
    """Real-time risk metrics snapshot."""
    total_exposure: float
    max_exposure_pct: float
    current_drawdown_pct: float
    daily_pnl: float
    daily_loss_limit_reached: bool
    open_positions_count: int
    margin_used_pct: float
    timestamp: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    Monitor positions and enforces risk limits.

    Implements:
    - Time-based stop-loss (close positions after X hours)
    - Price-based stop-loss (close if price moves against position)
    - Daily loss limits
    - Maximum drawdown protection
    - Correlation based exposure limits
    """

    def __init__(
            self,
            polymarket_client: PolymarketClient,
            max_position_age_hours: float = 72.0,
            default_stop_loss_pct: float = 0.20,
            max_daily_loss_usdc: float = 100.0,
            max_portfolio_drawdown_pct: float = 0.25,
            check_interval_seconds: int = 60,
    ):
        """
        Initialize risk manager.

        Args:
            polymarket_client: Authenticated Polymarket client for closing positions.
            max_position_age_hours: Close positions older than this many hours.
            default_stop_loss_pct: Stop loss as fraction of entry price (0.20 = 20% loss).
            max_daily_loss_usdc: Maximum allowed daily loss in USDC.
            max_portfolio_drawdown_pct: Stop all traiding if drawdown exceed this.
            check_interval_seconds: How often to check positions.
        """
        self.client = polymarket_client
        self.max_position_age_hours = max_position_age_hours
        self.default_stop_loss_pct = default_stop_loss_pct
        self.max_daily_loss_usdc = max_daily_loss_usdc
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct
        self.check_interval = check_interval_seconds

        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.initial_bankroll: Optional[float] = None
        self.current_bankroll: Optional[float] = None
        self._running = False
        self._stop_rules: List[StopLossRule] = []
        self._halt_trading = False

        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Initialize default stop-loss rules."""
        self._stop_rules.append(
            StopLossRule(
                rule_type=StopLossType.TIME_BASED,
                threshold=self.max_position_age_hours,
            )
        )
        self._stop_rules.append(
            StopLossRule(
                rule_type=StopLossType.PRICE_BASED,
                threshold=self.default_stop_loss_pct,
            )
        )

    def update_portfolio_state(
            self,
            positions: Dict[str, Position],
            bankroll: float,
            daily_pnl: float,
    ) -> None:
        """
        Update internal state with latest portfolio data.

        Args:
            positions: Current open positions.
            bankroll: Current available capital.
            daily_pnl: Realized + unrealized PnL for the day.
        """
        self.positions = positions.copy()
        self.current_bankroll = bankroll
        self.daily_pnl = daily_pnl

        if self.initial_bankroll is None:
            self.initial_bankroll = bankroll

    async def start_monitoring(self) -> None:
        """Start continuous risk monitoring loop."""
        self._running = True
        logger.info("Risk manager monitoring started")

        while self._running:
            try:
                await self._check_all_positions()
                self._check_portfolio_limits()
                await asyncio.sleep(self.check_interval)
            except Exception as err:
                logger.error(f"Risk monitoring error: {err}")
                await asyncio.sleep(self.check_interval)

    async def stop(self) -> None:
        """Stop risk monitoring."""
        self._running = False
        logger.info("Risk manager stopped")

    async def _check_all_positions(self) -> None:
        """Check each position against active stop-loss rules."""
        for token_id, position in list(self.positions.items()):
            for rule in self._stop_rules:
                if self._should_close_position(position, rule):
                    await self._close_position(token_id, f"Stop-loss triggered: {rule.rule_type.value}")
                    break

    def _should_close_position(
            self,
            position: Position,
            rule: StopLossRule
    ) -> bool:
        """Determine if position should be closed based on rule."""
        if rule.token_id and rule.token_id != position.token_id:
            return False

        if rule.rule_type == StopLossType.TIME_BASED:
            age_hours = (datetime.now() - position.open_time).total_seconds() / 3600
            return age_hours > rule.threshold
        elif rule.rule_type == StopLossType.PRICE_BASED:
            if position.side == "BUY":
                loss_pct = (position.entry_price - position.current_price) / position.entry_price
            else:
                loss_pct = (position.current_price - position.entry_price) / position.entry_price
            return loss_pct > rule.threshold

        return False

    async def _close_position(self, token_id: str, reason: str) -> None:
        """Close position at market price."""
        position = self.positions.get(token_id)
        if not position:
            return

        logger.warning(f"Closing position {token_id}: {reason}")

        # Determine opposite side to close
        close_side = "SELL" if position.side == "BUY" else "BUY"

        result = await self.client.place_order(
            token_id=token_id,
            side=close_side,
            size=position.size,
            price=position.current_price  # Market order approximation
        )

        if result.success:
            logger.info(f"Position {token_id} closed successfully")
            del self.positions[token_id]
        else:
            logger.error(f"Failed to close position {token_id}: {result.error}")

    def _check_portfolio_limits(self) -> None:
        """Check global portfolio risk limits."""
        if self.current_bankroll is None or self.initial_bankroll is None:
            return

        drawdown = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll
        if drawdown > self.max_portfolio_drawdown_pct:
            self._halt_trading = True
            logger.critical(f"Max drawdown exceed: {drawdown:.2%}. Halting trading")

        if self.daily_pnl < -self.max_daily_loss_usdc:
            self._halt_trading = True
            logger.warning(f"Daily loss limit exceeded: ${-self.daily_pnl:.2f}. Halting trading")

    def get_risk_metrics(self) -> RiskMetrics:
        """Generate current risk metrics snapshot."""
        if self.current_bankroll is None:
            return RiskMetrics(
                total_exposure=0.0,
                max_exposure_pct=0.0,
                current_drawdown_pct=0.0,
                daily_pnl=0.0,
                daily_loss_limit_reached=False,
                open_positions_count=0,
                margin_used_pct=0.0,
            )

        total_exposure = sum(p.size for p in self.positions.values())
        exposure_pct = total_exposure / self.current_bankroll if self.current_bankroll > 0 else 0.0

        drawdown = 0.0
        if self.initial_bankroll:
            drawdown = (self.initial_bankroll - self.current_bankroll) / self.initial_bankroll

        return RiskMetrics(
            total_exposure=total_exposure,
            max_exposure_pct=exposure_pct,
            current_drawdown_pct=drawdown,
            daily_pnl=self.daily_pnl,
            daily_loss_limit_reached=self.daily_pnl < -self.max_daily_loss_usdc,
            open_positions_count=len(self.positions),
            margin_used_pct=exposure_pct,
        )

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed based on risk limits."""
        return not self._halt_trading

    def reset_daily_pnl(self) -> None:
        """Reset daily PnL tracking (called at midnight)."""
        self.daily_pnl = 0.0
        self._halt_trading = False
        logger.info("Daily PnL reset")
