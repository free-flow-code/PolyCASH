import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import json

import aiohttp
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL
from weather_module.locations import TEMPERATURE_STATIONS

logger = logging.getLogger(__name__)


@dataclass
class WeatherMarket:
    """Container for Polymarket weather market data."""
    token_id: str
    question: str
    location: str
    market_type: str  # `high_temp`, `low_temp`
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
             List of WeatherMarket objects for temperature markets.
        """
        if not self._session:
            await self.initialize()

        weather_markets = []

        offset = 0
        limit = 500  # Maximum page size
            
        url = f"{self.GAMMA_API_URL}/markets"

        while True:
            params = {
                    "closed": "false",
                    "active": "true",
                    "tag_id": 103040,  # Tag id `Daily Temperature`
                    "limit": limit,
                    "offset": offset,
                }

            try:
                async with self._session.get(url, params=params) as resp:
                    resp.raise_for_status()
                    markets_batch = await resp.json()
                    
                    for market in markets_batch:
                        question = market.get("question", "").lower()
                        print(question)
                        if not self._is_weather_market(question):
                            print("NOT WEATHER")
                            continue

                        market_type = self._classify_weather_market(question)
                        print(f"market_type {market_type}")
                        location = self._extract_location(question)
                        print(f"location {location}")

                        token_id = self._get_token_id(market)
                        if not token_id:
                            continue

                        current_price = await self._get_current_price(token_id)

                        weather_markets.append(
                            WeatherMarket(
                                token_id=token_id,
                                question=market.get("question", ""),
                                location=location,
                                market_type=market_type,
                                current_price=current_price,
                                volume=float(market.get("volume", 0)),
                                spread=float(market.get("spread", 0.02)),
                                condition_id=market.get("conditionId", ""),
                                slug=market.get("slug", ""),
                            )
                        )

                        if not markets_batch:
                            # The answer is: the end of the list has been reached.
                            break

                        if len(markets_batch) < limit:
                            break
                        
                        offset += limit
                        await asyncio.sleep(0.5)

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

            # Create a signed order
            signed_order = await asyncio.to_thread(
                self._clob_client.create_order, order_args
            )

            # order_type is passed here, in post_order
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
            "temperature",
            "highest temperature",
            "lowest temperature",
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
        return "unknown"

    @staticmethod
    def _extract_location(question: str) -> Optional[str]:
        """
        Extract location name from market question with improved patterns.
        
        Returns None if no location can be extracted.
        """
        question_lower = question.lower()
        
        # Expanded patterns for location extraction
        patterns = [
            # Temperature patterns
            r"(?:temperature|high|low|temp)\s+(?:in|at|for)\s+([A-Za-z\s\-\.]+?)(?:\s+(?:on|this|will|today|tomorrow|the|have|be|reach|exceed|above|below|between|forecast|weather|degrees|°|celsius|fahrenheit|$))",
            r"([A-Za-z\s\-\.]+?)(?:'s)?\s+(?:temperature|high|low|weather|forecast)",
            r"(?:weather|forecast)\s+(?:in|at|for)\s+([A-Za-z\s\-\.]+)",
            
            # Precipitation patterns
            r"(?:rain|snow|precipitation|rainfall)\s+(?:in|at|for)\s+([A-Za-z\s\-\.]+?)(?:\s+(?:on|this|will|today|tomorrow|the|exceed|above|total|amount|$))",
            r"([A-Za-z\s\-\.]+?)(?:'s)?\s+(?:rain|snow|precipitation)",
            
            # General location patterns
            r"(?:in|at)\s+([A-Za-z\s\-\.]+?)(?:,|\s+(?:on|for|this|will|today|tomorrow|the|have|be))",
            r"([A-Za-z\s\-\.]+?)(?:,)\s+(?:weather|temperature|forecast)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Clean up the location string
                location = re.sub(r'\s+', ' ', location)  # Normalize spaces
                location = location.strip('.,;:-')
                
                # Filter out false positives
                if location.lower() in ['will', 'today', 'tomorrow', 'weather', 'temperature', 
                                        'forecast', 'high', 'low', 'the', 'this', 'have', 'be',
                                        'reach', 'exceed', 'above', 'below']:
                    continue
                    
                if len(location) >= 3:  # Minimum meaningful location length
                    return location.title()
        
        return None
    
    def geocode_location(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Gets latitude/longitude coordinates by location name.
        
        Pre-collected location data from PolyMarket is used..
        """
        if not location or location == "unknown":
            return None
        
        location_data = TEMPERATURE_STATIONS.get(location, "")

        if location_data:
            lat = location_data.get("station_lat", "")
            lon = location_data.get("station_lon", "")
            logger.info(f"Geocoded '{location}' -> ({lat:.4f}, {lon:.4f})")
            return (lat, lon)
        else:
            logger.warning(f"Geocoding failed for '{location}'")
            
        return None

    @staticmethod
    def _get_token_id(market_data: dict) -> Optional[str]:
        """Extract token ID from market data.

        Each market on Polymarket is a question with a binary outcome (yes/no).
        The server returns a list of two tokens.

        - One token represents the "YES" outcome
        (or the first possible option, e.g., "The temperature will be 21°C").

        - The second token represents the "NO" outcome
        (or the second possible option, e.g., "The temperature will not be 21°C").

        The official py-clob-client library and the Polymarket platform
        itself consistently assign the YES token as the first token in the clobTokenIds list.
        """
        clob_token_ids = json.loads(market_data.get("clobTokenIds"))


        if not clob_token_ids:
            return None
        return clob_token_ids[0]
