import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WeatherForecast:
    """Container for weather forecast data."""
    location: str
    latitude: float
    longitude: float
    target_date: str
    high_temp_prob: float
    low_temp_prob: float
    precip_prob: float
    ensemble_members: int
    timestamp: datetime


class OpenMeteoClient:
    """
    Async client for Open-Meteo ensemble weather API.

    Fetches GFS 31-member ensemble forecasts and computes calibrated
    probabilities for temperature thresholds and precipitation events.
    """

    BASE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize Open-Meteo client.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 5 minutes).
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[WeatherForecast, float]] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_ensemble_forecast(
            self,
            latitude: float,
            longitude: float,
            location_name: str,
            target_date: Optional[str] = None,
            models: List[str] = None,
    ) -> WeatherForecast:
        """
        Fetch ensemble weather forecast for a specific location.

        Args:
            latitude: Location latitude.
            longitude: Location longitude.
            location_name: Human-readable location identifier.
            target_date: Target date in YYYY-MM-DD format (default: today).
            models: List of models to include (default: GFS 31-member ensemble).

        Returns:
            WeatherForecast object with calibrated probabilities.
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")

        if models is None:
            models = ["gfs_seamless"]

        cache_key = f"{latitude:.4f}_{longitude:.4f}_{target_date}"
        if cache_key in self._cache:
            forecast, cached_time = self._cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logger.debug(f"Returning cached forecast for {location_name}")
                return forecast

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "UTC",
            "forecast_days": 7,
            "models": models,
        }

        session = await self._get_session()
        try:
            async with session.get(self.BASE_URL, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            forecast = self._parse_ensemble_response(
                data, location_name, latitude, longitude, target_date
            )
            self._cache[cache_key] = (forecast, datetime.now().timestamp())
            return forecast

        except aiohttp.ClientError as err:
            logger.error(f"Failed to fetch weather data: {err}")
            raise
        except Exception as err:
            logger.error(f"Unexpected error parsing weather data: {err}")
            raise

    def _parse_ensemble_response(
            self,
            data: dict,
            location_name: str,
            latitude: float,
            longitude: float,
            target_date: str,
    ) -> WeatherForecast:
        """
        Parse Open-Meteo ensemble response into calibrated probabilities.

        For each ensemble member, checks if the predicted value falls within
        the bucket thresholds defined by Polymarket markets.

        Args:
            data: Raw JSON response from Open-Meteo.
            location_name: Human-readable location identifier.
            latitude: Location latitude.
            longitude: Location longitude.
            target_date: Target date for forecast.

        Returns:
            WeatherForecast with computed probabilities.
        """
        daily_data = data.get("daily", {})
        ensemble_members = len(data.get("hourly", {}).get("temperature_2m", [[]]))

        temp_max_values = daily_data.get("temperature_2m_max", [[]])[0]
        temp_min_values = daily_data.get("temperature_2m_min", [[]])[0]
        precip_values = daily_data.get("precipitation_sum", [[]])[0]

        if not temp_max_values or not temp_min_values:
            raise ValueError(f"No ensemble data available for {location_name}")

        high_temp_prob = self._compute_temperature_probability(temp_max_values)
        low_temp_prob = self._compute_temperature_probability(temp_min_values)
        precip_prob = self._compute_precipitation_probability(precip_values)

        return WeatherForecast(
            location=location_name,
            latitude=latitude,
            longitude=longitude,
            target_date=target_date,
            high_temp_prob=high_temp_prob,
            low_temp_prob=low_temp_prob,
            precip_prob=precip_prob,
            ensemble_members=ensemble_members,
            timestamp=datetime.now(),
        )

    @staticmethod
    def _compute_temperature_probability(
            values: List[float],
            threshold: float = 20.0
    ) -> float:
        """
        Compute probability that temperature exceeds threshold.

        Args:
            values: List of ensemble member temperature predictions.
            threshold: Temperature threshold in Celsius.

        Returns:
             Probability (0.0 to 1.0) thet temperature exceeds threshold.
        """
        if not values:
            return 0.5

        exceed_count = sum(1 for value in values if value > threshold)
        return exceed_count / len(values)

    @staticmethod
    def _compute_precipitation_probability(
            values: List[float],
            threshold: float = 1.0
    ) -> float:
        """
        Compute probability that precipitation exceeds threshold.

        Args:
            values: List of ensemble member precipitation predictions (mm).
            threshold: Precipitation threshold in mm.

        Returns:
            Probability (0.0 to 1.0) that precipitation exceeds threshold.
        """
        if not values:
            return 0.5

        exceed_count = sum(1 for value in values if value > threshold)
        return exceed_count / len(values)


class WeatherAPIManager:
    """
    Manager class for fetching weather data across multiple locations.

    Handles concurrent API requests and rate limiting.
    """

    def __init__(self, max_concurrent: int = 10):
        """
        Initialize weather API manager.

        Args:
            max_concurrent: Maximum number of concorrent API requests.
        """
        self.client = OpenMeteoClient()
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_all_forecasts(
            self,
            locations: List[Tuple[str, float, float]],
            target_date: str,
    ) -> Optional[Dict[str, WeatherForecast]]:
        """
        Fetch forecasts for multiple locations concurrently.

        Args:
            locations: List of (location_name, latitude, longitude) tuples.
            target_date: Target date for all forecasts.

        Returns:
             Dictionary mapping location_name to WeatherForecast.
        """
        async def fetch_one(loc: Tuple[str, float, float]) -> Optional[WeatherForecast]:
            async with self._semaphore:
                try:
                    name, lat, lon = loc
                    return await self.client.get_ensemble_forecast(
                        lat, lon, name, target_date
                    )
                except Exception as err:
                    logger.error(f"Failed to fetch forecast for {loc[0]: {err}}")
                    return None

        tasks = [fetch_one(loc) for loc in locations]
        results = await asyncio.gather(*tasks)

        forecasts = {}
        for loc, forecast in zip(locations, results):
            if forecast is not None:
                forecasts[loc[0]] = forecast

        return forecasts

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.close()
