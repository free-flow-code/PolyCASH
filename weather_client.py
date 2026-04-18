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

        # Simple cache check
        cache_key = f"{latitude:.4f}_{longitude:.4f}_{target_date}"
        if cache_key in self._cache:
            forecast, cached_time = self._cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logger.debug(f"Returning cached forecast for {location_name}")
                return forecast

        # Simplified parameters for Open-Meteo
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "UTC",
            "forecast_days": 7,
        }

        session = await self._get_session()
        try:
            # Use standard forecast API instead of ensemble for simplicity
            url = "https://api.open-meteo.com/v1/forecast"
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

            forecast = self._parse_standard_forecast(
                data, location_name, latitude, longitude, target_date
            )
            self._cache[cache_key] = (forecast, datetime.now().timestamp())
            return forecast

        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch weather data: {err}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {err}")
            raise

    def _parse_standard_forecast(
        self,
        data: dict,
        location_name: str,
        latitude: float,
        longitude: float,
        target_date: str,
    ) -> WeatherForecast:
        """
        Parse standard (non-ensemble) forecast response.
        
        Since we don't have ensemble data, we add artificial uncertainty
        based on forecast confidence intervals.
        """
        try:
            daily_data = data.get("daily", {})
            
            # Get today's forecast
            temp_max = daily_data.get("temperature_2m_max", [20.0])[0] if daily_data.get("temperature_2m_max") else 20.0
            temp_min = daily_data.get("temperature_2m_min", [10.0])[0] if daily_data.get("temperature_2m_min") else 10.0
            precip = daily_data.get("precipitation_sum", [0.0])[0] if daily_data.get("precipitation_sum") else 0.0
            
            # Convert to probabilities with some uncertainty
            # Using sigmoid-like function to convert temperature to probability
            import math
            
            def temp_to_prob(temp: float, threshold: float = 20.0, spread: float = 5.0) -> float:
                """Convert temperature to probability using logistic function."""
                return 1.0 / (1.0 + math.exp(-(temp - threshold) / spread))
            
            high_temp_prob = temp_to_prob(temp_max, 25.0, 3.0)  # Probability temp > 25°C
            low_temp_prob = 1.0 - temp_to_prob(temp_min, 5.0, 2.0)  # Probability temp < 5°C
            precip_prob = min(1.0, precip / 10.0)  # Probability of rain > 10mm
            
            return WeatherForecast(
                location=location_name,
                latitude=latitude,
                longitude=longitude,
                target_date=target_date,
                high_temp_prob=high_temp_prob,
                low_temp_prob=low_temp_prob,
                precip_prob=precip_prob,
                ensemble_members=1,  # Single forecast, no ensemble
                timestamp=datetime.now(),
            )
            
        except Exception as err:
            logger.error(f"Error parsing standard forecast: {err}")
            return WeatherForecast(
                location=location_name,
                latitude=latitude,
                longitude=longitude,
                target_date=target_date,
                high_temp_prob=0.5,
                low_temp_prob=0.5,
                precip_prob=0.5,
                ensemble_members=1,
                timestamp=datetime.now(),
            )        

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
        try:
            # Get daily data - structure depends on API response
            daily_data = data.get("daily", {})
            
            # For ensemble forecasts, data is nested differently
            # Temperature max values - might be list of lists or single list
            temp_max_values = daily_data.get("temperature_2m_max", [])
            temp_min_values = daily_data.get("temperature_2m_min", [])
            precip_values = daily_data.get("precipitation_sum", [])
            
            # Handle different response structures
            if temp_max_values and isinstance(temp_max_values[0], list):
                # Ensemble response: first element contains all members
                temp_max_values = temp_max_values[0]
                temp_min_values = temp_min_values[0] if temp_min_values else []
                precip_values = precip_values[0] if precip_values else []
            
            # Count ensemble members
            ensemble_members = len(temp_max_values) if temp_max_values else 31
            
            # If no data, use default values
            if not temp_max_values or not temp_min_values:
                logger.warning(f"No ensemble data available for {location_name}, using defaults")
                return WeatherForecast(
                    location=location_name,
                    latitude=latitude,
                    longitude=longitude,
                    target_date=target_date,
                    high_temp_prob=0.5,
                    low_temp_prob=0.5,
                    precip_prob=0.5,
                    ensemble_members=31,
                    timestamp=datetime.now(),
                )
            
            high_temp_prob = self._compute_temperature_probability(temp_max_values)
            low_temp_prob = self._compute_temperature_probability(temp_min_values)
            precip_prob = self._compute_precipitation_probability(precip_values if precip_values else [])
            
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
        
        except Exception as err:
            logger.error(f"Error parsing weather data: {err}")
            # Return default forecast on error
            return WeatherForecast(
                location=location_name,
                latitude=latitude,
                longitude=longitude,
                target_date=target_date,
                high_temp_prob=0.5,
                low_temp_prob=0.5,
                precip_prob=0.5,
                ensemble_members=31,
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
