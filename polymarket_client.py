import asyncio
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

logger = logging.getLogger(__name__)


@dataclass
class WeatherMarket:
    """Container for Polymarket weather market data."""
    token_id: str
    question: str
    location: str
    market_type: str  # `high_temp`, `low_temp`, `precip`
    current_price: float
    volume: float
    spread: float
    condition_id: str
    slug: str


@dataclass
class OrderResult:
    """Container for order execution result."""
    success: bool
    order_id: Optional[str]
    token_id: str
    side: str
    size: float
    price: float
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class PolymarketClient:
    """
    Async wrapper for Polymarket CLOB client with market discovery.

    Handles authentication, market data fetching, and order execution.
    """

    GAMMA_API_URL = "https://gamma-api.polymarket.com"

    def __init__(
            self,
            private_key: str,
            funder_address: str,
            host: str = "https://clob.polymarket.com",
            chain_id: int = 137,
            signature_type: int = 1,
    ):
        """
        Initialize Polymarket client.

        Args:
            private_key: Wallet private key for signing orders.
            funder_address: Address that holds the funds.
            host: CLOB API endpoind.
            chain_id: Polygon chain ID (137 for mainnet).
            signature_type: Signature type (1 for email/Magic wallets).
        """
        self.host = host
        self.chain_id = chain_id
        self.private_key = private_key
        self.funder_address = funder_address
        self.signature_type = signature_type

        self._clob_client: Optional[ClobClient] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize CLOB client and derive API credentials."""
        try:
            self._clob_client = ClobClient(
                self.host,
                key=self.private_key,
                chain_id=self.chain_id,
                signature_type=self.signature_type,
                funder=self.funder_address,
            )
            # Try to get existing API credentials or create new ones
            api_creds = self._clob_client.create_or_derive_api_creds()
            if api_creds:
                self._clob_client.set_api_creds(api_creds)
            self._session = aiohttp.ClientSession()
            logger.info("Polymarket client initialized successfully")
        except Exception as err:
            logger.error(f"Failed to initialize Polymarket client: {err}")
            # Don't raise, allow paper trading to continue
            if self._session is None:
                self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_weather_markets(self) -> List[WeatherMarket]:
        """
        Discover active weather markets from Gamma API.

        Returns:
             List of WeatherMarket objects for temperature and precipitation markets.
        """
        if not self._session:
            await self.initialize()

        weather_markets = []

        try:
            url = f"{self.GAMMA_API_URL}/markets"
            params = {
                "closed": "false",
                "limit": 500,
            }

            async with self._session.get(url, params=params) as resp:
                resp.raise_for_status()
                markets = await resp.json()

            for market_data in markets:
                question = market_data.get("question", "").lower()
                if not self._is_weather_market(question):
                    continue

                market_type = self._classify_weather_market(question)
                location = self._extract_location(question)

                token_id = self._get_token_id(market_data)
                if not token_id:
                    continue

                current_price = await self._get_current_price(token_id)

                weather_markets.append(
                    WeatherMarket(
                        token_id=token_id,
                        question=market_data.get("question", ""),
                        location=location,
                        market_type=market_type,
                        current_price=current_price,
                        volume=float(market_data.get("volume", 0)),
                        spread=float(market_data.get("spread", 0.02)),
                        condition_id=market_data.get("conditionId", ""),
                        slug=market_data.get("slug", ""),
                    )
                )

            logger.info(f"Discovered {len(weather_markets)} weather markets")
            return weather_markets

        except Exception as err:
            logger.error(f"Failed to fetch weather markets: {err}")
            return []

    async def place_order(
            self,
            token_id: str,
            side: str,
            size: float,
            price: float,
            order_type: OrderType = OrderType.GTC,
    ) -> OrderResult:
        """
        Place an order on Polymarket CLOB.

        Args:
            token_id: Market token ID.
            side: `BUY` or `SELL`.
            size: Order size in USDC.
            price: Limit price (0.0 to 1.0).
            order_type: Order type (GTC, FOK, IOC).

        Returns:
             OrderResult with execution details.
        """
        if not self._clob_client:
            return OrderResult(
                success=False,
                order_id=None,
                token_id=token_id,
                side=side,
                size=size,
                price=price,
                error="Client not initialized",
            )

        try:
            side_enum = BUY if side.upper() == "BUY" else SELL

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side_enum,
            )

            # Создаём подписанный ордер
            signed_order = await asyncio.to_thread(
                self._clob_client.create_order, order_args
            )

            # order_type передаётся именно сюда, в post_order
            response = await asyncio.to_thread(
                self._clob_client.create_and_post_order, signed_order, order_type
            )

            if response and "orderID" in response:
                logger.info(f"Order placed: {side} {size} @ {price} on {token_id}")
                return OrderResult(
                    success=True,
                    order_id=response["orderID"],
                    token_id=token_id,
                    side=side,
                    size=size,
                    price=price,
                )
            else:
                return OrderResult(
                    success=False,
                    order_id=None,
                    token_id=token_id,
                    side=side,
                    size=size,
                    price=price,
                    error="No order ID in response",
                )

        except Exception as err:
            logger.error(f"Order placement failed: {err}")
            return OrderResult(
                success=False,
                order_id=None,
                token_id=token_id,
                side=side,
                size=size,
                price=price,
                error=str(err),
            )

    async def _get_current_price(self, token_id: str) -> float:
        """Get current midpoint price for a token."""
        try:
            if self._clob_client:
                price = await asyncio.to_thread(
                    self._clob_client.get_midpoint, token_id
                )
                return float(price) if price else 0.5
        except Exception as err:
            logger.debug(f"Failed to get price for {token_id}: {err}")
        return 0.5

    @staticmethod
    def _is_weather_market(question: str) -> bool:
        """Check if market question is weather-related."""
        weather_keywords = [
            "temperature", "precipitation", "rain", "snow",
            "high", "low", "weather", "degrees", "celsius",
            "fahrenheit", "noaa"
        ]
        return any(kw in question.lower() for kw in weather_keywords)

    @staticmethod
    def _classify_weather_market(question: str) -> str:
        """Classify weather market type from question text."""
        q = question.lower()
        if "high" in q or "maximum" in q or "highest" in q:
            return "high_temp"
        if "low" in q or "minimum" in q or "lowest" in q:
            return "low_temp"
        if "precip" in q or "rain" in q or "snow" in q:
            return "precip"
        return "unknown"

    @staticmethod
    def _extract_location(question: str) -> str:
        """Extract location name from market question."""
        import re
        location_patterns = [
            r"(?:in|at|for)\s+([A-Za-z\s]+?)(?:\s+(?:on|this|will|today|tomorrow|the|have|be|temperature))",
            r"temperature\s+(?:in|at)\s+([A-Za-z\s]+)",
            r"([A-Za-z\s]+?)(?:'s|\s+weather|\s+temperature)",
        ]

        for pattern in location_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "unknown"

    @staticmethod
    def _get_token_id(market_data: dict) -> Optional[str]:
        """Extract token ID from market data."""
        if "tokens" in market_data and market_data["tokens"]:
            return market_data["tokens"][0].get("token_id")
        if "tokenId" in market_data:
            return market_data["tokenId"]
        return None
