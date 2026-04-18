import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

from weather_client import WeatherForecast, WeatherAPIManager
from polymarket_client import PolymarketClient, WeatherMarket, OrderResult

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Container for trading signal."""
    market: WeatherMarket
    signal_type: SignalType
    edge: float
    model_prob: float
    market_price: float
    kelly_fraction: float
    position_size_usdc: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Container for open position."""
    token_id: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    open_time: datetime
    order_id: str


class WeatherTradingStrategy:
    """
    Core trading strategy for weather markets on Polymarket.

    Computes edge between ensemble forecast probabilities and market prices,
    generates trading signals, and manages risk with fractional Kelly sizing.
    """

    def __init__(
            self,
            bankroll_usdc: float,
            min_edge_threshold: float = 0.05,
            kelly_multiplier: float = 0.25,
            max_position_pct: float = 0.03,
            max_daily_loss_pct: float = 0.10,
            confidence_threshold: float = 0.70,
    ):
        """
        Initialize trading strategy.

        Args:
            bankroll_usdc: Total available capital in USDC.
            min_edge_threshold: Minimum edge required to trade (0.05 = 5%).
            kelly_multiplier: Fractional Kelly multiplier (0.1-0.5 recommended).
            max_position_pct: Maximum position size as fraction of bankroll.
            max_daily_loss_pct: Maximum daily loss as fraction of bankroll.
            confidence_threshold: Minimum confidence to execute trade.
        """
        self.bankroll = bankroll_usdc
        self.initial_bankroll = bankroll_usdc
        self.min_edge_threshold = min_edge_threshold
        self.kelly_multiplier = kelly_multiplier
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.confidence_threshold = confidence_threshold

        self.positions: Dict[str, Position] = {}
        self.daily_pnl: float = 0.0
        self.last_reset_date: date = datetime.now().date()
        self.trade_history: List[OrderResult] = []

    def compute_signal(
            self,
            market: WeatherMarket,
            forecast: WeatherForecast,
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal by comparing model probability with market price.

        Args:
            market: Polymarket weather market data.
            forecast: Weather forecast with calibrated probabilities.

        Returns:
             TradingSignal if edge exceeds threshold, else None.
        """
        model_prob = self._get_model_probability(market, forecast)
        market_price = market.current_price

        edge = model_prob - market_price

        if abs(edge) < self.min_edge_threshold:
            return None

        signal_type = SignalType.BUY if edge > 0 else SignalType.SELL
        confidence = self._compute_confidence(model_prob, market, forecast)

        if confidence < self.confidence_threshold:
            logger.debug(f"Signal rejected: confidence {confidence:.2f} below threshold")
            return None

        kelly_fraction = self._compute_kelly_fraction(
            model_prob, market_price, signal_type
        )

        position_size = self._compute_position_size(kelly_fraction)

        if not self._check_risk_limits(position_size):
            logger.debug("Signal rejected: risk limits exceeded")
            return None

        return TradingSignal(
            market=market,
            signal_type=signal_type,
            edge=edge,
            model_prob=model_prob,
            market_price=market_price,
            kelly_fraction=kelly_fraction,
            position_size_usdc=position_size,
            confidence=confidence,
        )

    def _get_model_probability(
            self,
            market: WeatherMarket,
            forecast: WeatherForecast
    ) -> float:
        """Extract relevant probability from forecast for given market type."""
        if market.market_type == "high_temp":
            return forecast.high_temp_prob
        elif market.market_type == "low_temp":
            return forecast.low_temp_prob
        elif market.market_type == "precip":
            return forecast.precip_prob
        else:
            return 0.5

    def _compute_confidence(
            self,
            model_prob: float,
            market: WeatherMarket,
            forecast: WeatherForecast
    ) -> float:
        """
        Compute confidence score for the signal.

        Factors: ensemble agreement, market liquidity, time to resolution.
        """
        ensemble_confidence = min(1.0, forecast.ensemble_members / 31.0)

        volume_confidence = min(1.0, market.volume / 10000.0)

        price_confidence = 1.0 - abs(model_prob - market.current_price)

        weights = [0.5, 0.2, 0.3]
        confidence = (
            weights[0] * ensemble_confidence +
            weights[1] * volume_confidence +
            weights[2] * price_confidence
        )

        return confidence

    def _compute_kelly_fraction(
            self,
            prob: float,
            price: float,
            signal_type: SignalType
    ) -> float:
        """
        Compute fractional Kelly criterion for optimal position sizing.

        For binary markets:
            Kelly = (p * b - q) / b
            where: p = probability of winning
                   q = 1 - p
                   b = net odds received ((1/price - 1) for BUY, (1/(1-price) - 1) for SELL)
        """
        if signal_type == SignalType.BUY:
            b = (1.0 / price) - 1.0 if price > 0 else 0
        else:
            b = (1.0 / (1.0 - price)) - 1.0 if price < 1.0 else 0

        if b <= 0:
            return 0.0

        q = 1.0 - prob
        kelly = (prob * b - q) / b

        kelly = max(0.0, min(0.25, kelly))

        return kelly * self.kelly_multiplier

    def _compute_position_size(self, kelly_fraction: float) -> float:
        """Compute position size in USDC based on Kelly fraction."""
        raw_size = self.bankroll * kelly_fraction
        max_allowed = self.bankroll * self.max_position_pct
        return min(raw_size, max_allowed)

    def _check_risk_limits(self, position_size: float) -> bool:
        """Check if trade passes all risk management limits."""
        if position_size <= 0:
            return False

        if self.bankroll < self.initial_bankroll * 0.5:
            logger.warning("Bankroll below 50% of initial, halting trading")
            return False

        self._reset_daily_if_needed()

        if abs(self.daily_pnl) > self.initial_bankroll * self.max_daily_loss_pct:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False

        total_exposure = sum(p.size for p in self.positions.values())
        if total_exposure + position_size > self.bankroll * 0.5:
            logger.debug("Total exposure would exceed 50% of bankroll")
            return False

        return True

    def _reset_daily_if_needed(self) -> None:
        """Reset daily PnL tracking if date has changed."""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = today

    def update_position_pnl(self, token_id: str, current_price: float) -> None:
        """Update unrealized PnL for a position."""
        if token_id in self.positions:
            pos = self.positions[token_id]
            pos.current_price = current_price
            if pos.side == "BUY":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size

    def record_trade_result(self, result: OrderResult) -> None:
        """Record trade result and update bankroll."""
        self.trade_history.append(result)

        if result.side == "BUY":
            self.bankroll -= result.size

        self.positions[result.token_id] = Position(
            token_id=result.token_id,
            side=result.side,
            size=result.size,
            entry_price=result.price,
            current_price=result.price,
            unrealized_pnl=0.0,
            open_time=result.timestamp,
            order_id=result.order_id,
        )

    def get_portfolio_summary(self) -> Dict:
        """Generate current portfolio summary."""
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        return {
            "bankroll": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "total_return_pct": (self.bankroll / self.initial_bankroll - 1) * 100,
            "open_positions": len(self.positions),
            "total_unrealized_pnl": total_unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": len(self.trade_history),
        }


class TradingEngine:
    """
    Main orchestration engine for the weather trading bot.

    Coordinates weather data fetching, market discovery, signal generation,
    and order execution.
    """

    def __init__(
            self,
            polymarket_client: PolymarketClient,
            strategy: WeatherTradingStrategy,
            weather_manager: WeatherAPIManager,
            locations: List[Tuple[str, float, float]],
            scan_interval_seconds: int = 300,
            paper_trading: bool = True,
    ):
        """
        Initialize trading engine.

        Args:
            polymarket_client: Authenticated Polymarket client.
            strategy: Trading strategy instance.
            weather_manager: Weather api manager.
            locations: List of (name, lan, lon) tuples to monitor.
            scan_interval_seconds: Seconds between market scans.
            paper_trading: If True, don't execute real orders.
        """
        self.polymarket = polymarket_client
        self.strategy = strategy
        self.weather_manager = weather_manager
        self.locations = locations
        self.scan_interval = scan_interval_seconds
        self.paper_trading = paper_trading

        self.running = False
        self._markets_cache: List[WeatherMarket] = []
        self._last_market_update: datetime = None

    async def start(self) -> None:
        """Start the main trading loop."""
        self.running = True
        logger.info("Trading engine started")

        while self.running:
            try:
                await self._scan_and_trade()
                await asyncio.sleep(self.scan_interval)
            except Exception as err:
                logger.error(f"Error in trading loop: {err}")
                await asyncio.sleep(60)

    async def stop(self) -> None:
        """Stop the trading engine."""
        self.running = False
        await self.weather_manager.close()
        await self.polymarket.close()
        logger.info("Trading engine stopped")

    async def _scan_and_trade(self) -> None:
        """Perform one full scan and trading cycle."""
        target_date = datetime.now().strftime("%Y-%m-%d")

        forecasts = await self.weather_manager.fetch_all_forecasts(
            self.locations, target_date
        )

        markets = await self._get_weather_markets()

        signals = []
        for market in markets:
            if market.location not in forecasts:
                continue

            forecast = forecasts[market.location]
            signal = self.strategy.compute_signal(market, forecast)

            if signal:
                signals.append(signal)
                logger.info(
                    "Signal: %s %s Edge: %.3f, Size: $%.2f",
                    signal.signal_type.value,
                    signal.market.token_id[:8] + "...",  # Truncate token ID
                    signal.edge,
                    signal.position_size_usdc
                )

        if signals:
            await self._execute_signals(signals)
        else:
            logger.debug("No trading signals generated in this scan")

    async def _get_weather_markets(self) -> List[WeatherMarket]:
        """Get weather markets, using cache if fresh."""
        if (
            self._last_market_update
            and (datetime.now() - self._last_market_update).seconds < 300
        ):
            return self._markets_cache

        self._markets_cache = await self.polymarket.get_weather_markets()
        self._last_market_update = datetime.now()
        return self._markets_cache

    async def _execute_signals(self, signals: List[TradingSignal]) -> None:
        """Execute trading signals."""
        signals.sort(key=lambda s: abs(s.edge), reverse=True)

        for signal in signals[:10]:
            if self.paper_trading:
                logger.info(f"[PAPER] Would execute: {signal}")
                continue

            result = await self.polymarket.place_order(
                token_id=signal.market.token_id,
                side=signal.signal_type.value,
                size=signal.position_size_usdc,
                price=signal.market_price,
            )

            if result.success:
                self.strategy.record_trade_result(result)
            else:
                logger.error(f"Order failed: {result.error}")
