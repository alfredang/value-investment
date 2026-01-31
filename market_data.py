"""
Market Data Provider Module

Provides unified interface for market data from multiple sources:
- Morningstar API (comprehensive fundamentals + market data)
- Twelve Data (real-time prices, historical data)
- NewsAPI (financial news)
- Yahoo Finance (free alternative)
"""
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import httpx


class DataProvider(Enum):
    MORNINGSTAR = "morningstar"
    TWELVE_DATA = "twelve_data"
    NEWS_API = "news_api"
    YAHOO_FINANCE = "yahoo_finance"


@dataclass
class StockQuote:
    """Real-time stock quote."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    source: str
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    previous_close: Optional[float] = None
    market_cap: Optional[float] = None


@dataclass
class HistoricalPrice:
    """Historical price data point."""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


@dataclass
class CompanyFundamentals:
    """Company fundamental data."""
    symbol: str
    name: str
    sector: str
    industry: str
    description: str
    market_cap: float
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None
    source: str = ""


@dataclass
class NewsArticle:
    """News article."""
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    sentiment: Optional[str] = None  # positive, negative, neutral


@dataclass
class AnalystEstimate:
    """Analyst estimates and ratings."""
    symbol: str
    target_price: Optional[float] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    num_analysts: int = 0
    buy_ratings: int = 0
    hold_ratings: int = 0
    sell_ratings: int = 0
    consensus: str = ""  # buy, hold, sell
    eps_estimate_current_year: Optional[float] = None
    eps_estimate_next_year: Optional[float] = None
    revenue_estimate_current_year: Optional[float] = None
    source: str = ""


# ============================================================================
# Abstract Base Provider
# ============================================================================

class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote for a symbol."""
        pass

    @abstractmethod
    def get_historical(self, symbol: str, period: str = "1Y") -> List[HistoricalPrice]:
        """Get historical price data. Period: 1D, 1W, 1M, 3M, 6M, 1Y, 5Y"""
        pass

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Optional[CompanyFundamentals]:
        """Get company fundamentals."""
        pass

    @abstractmethod
    def get_news(self, symbol: str = None, query: str = None, limit: int = 10) -> List[NewsArticle]:
        """Get news articles."""
        pass

    def get_analyst_estimates(self, symbol: str) -> Optional[AnalystEstimate]:
        """Get analyst estimates (optional implementation)."""
        return None


# ============================================================================
# Morningstar API Provider
# ============================================================================

class MorningstarProvider(MarketDataProvider):
    """
    Morningstar API provider.

    Morningstar provides comprehensive financial data including:
    - Real-time and historical quotes
    - Detailed fundamentals
    - Analyst estimates
    - Fair value estimates
    - Economic moat ratings

    API Documentation: https://developer.morningstar.com/
    """

    BASE_URL = "https://api.morningstar.com/v2"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MORNINGSTAR_API_KEY", "")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request."""
        if not self.api_key:
            return None

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(
                    f"{self.BASE_URL}/{endpoint}",
                    headers=self.headers,
                    params=params or {}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Morningstar API error: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote from Morningstar."""
        data = self._request(f"stocks/{symbol}/quote")
        if not data:
            return None

        try:
            quote_data = data.get("quote", data)
            return StockQuote(
                symbol=symbol,
                price=float(quote_data.get("lastPrice", 0)),
                change=float(quote_data.get("netChange", 0)),
                change_percent=float(quote_data.get("percentChange", 0)),
                volume=int(quote_data.get("volume", 0)),
                timestamp=datetime.now(),
                source="Morningstar",
                high=quote_data.get("dayHigh"),
                low=quote_data.get("dayLow"),
                open=quote_data.get("openPrice"),
                previous_close=quote_data.get("previousClose"),
                market_cap=quote_data.get("marketCap")
            )
        except Exception as e:
            print(f"Error parsing Morningstar quote: {e}")
            return None

    def get_historical(self, symbol: str, period: str = "1Y") -> List[HistoricalPrice]:
        """Get historical price data from Morningstar."""
        period_map = {
            "1D": 1, "1W": 7, "1M": 30, "3M": 90,
            "6M": 180, "1Y": 365, "5Y": 1825
        }
        days = period_map.get(period, 365)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        data = self._request(f"stocks/{symbol}/priceHistory", {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d")
        })

        if not data:
            return []

        prices = []
        for item in data.get("priceHistory", []):
            try:
                prices.append(HistoricalPrice(
                    date=datetime.strptime(item["date"], "%Y-%m-%d"),
                    open=float(item.get("open", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    close=float(item.get("close", 0)),
                    volume=int(item.get("volume", 0)),
                    adjusted_close=item.get("adjustedClose")
                ))
            except Exception:
                continue

        return prices

    def get_fundamentals(self, symbol: str) -> Optional[CompanyFundamentals]:
        """Get company fundamentals from Morningstar."""
        data = self._request(f"stocks/{symbol}/fundamentals")
        if not data:
            return None

        try:
            fund = data.get("fundamentals", data)
            profile = data.get("profile", {})

            return CompanyFundamentals(
                symbol=symbol,
                name=profile.get("name", symbol),
                sector=profile.get("sector", ""),
                industry=profile.get("industry", ""),
                description=profile.get("description", ""),
                market_cap=float(fund.get("marketCap", 0)),
                pe_ratio=fund.get("peRatio"),
                pb_ratio=fund.get("pbRatio"),
                ps_ratio=fund.get("psRatio"),
                dividend_yield=fund.get("dividendYield"),
                eps=fund.get("eps"),
                revenue=fund.get("revenue"),
                net_income=fund.get("netIncome"),
                gross_margin=fund.get("grossMargin"),
                operating_margin=fund.get("operatingMargin"),
                net_margin=fund.get("netMargin"),
                roe=fund.get("returnOnEquity"),
                roa=fund.get("returnOnAssets"),
                debt_to_equity=fund.get("debtToEquity"),
                current_ratio=fund.get("currentRatio"),
                quick_ratio=fund.get("quickRatio"),
                free_cash_flow=fund.get("freeCashFlow"),
                source="Morningstar"
            )
        except Exception as e:
            print(f"Error parsing Morningstar fundamentals: {e}")
            return None

    def get_news(self, symbol: str = None, query: str = None, limit: int = 10) -> List[NewsArticle]:
        """Get news from Morningstar."""
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        if query:
            params["query"] = query

        data = self._request("news", params)
        if not data:
            return []

        articles = []
        for item in data.get("articles", []):
            try:
                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    url=item.get("url", ""),
                    source="Morningstar",
                    published_at=datetime.strptime(
                        item.get("publishedAt", ""),
                        "%Y-%m-%dT%H:%M:%SZ"
                    ) if item.get("publishedAt") else datetime.now(),
                    sentiment=item.get("sentiment")
                ))
            except Exception:
                continue

        return articles[:limit]

    def get_analyst_estimates(self, symbol: str) -> Optional[AnalystEstimate]:
        """Get analyst estimates from Morningstar."""
        data = self._request(f"stocks/{symbol}/analystEstimates")
        if not data:
            return None

        try:
            est = data.get("estimates", data)
            return AnalystEstimate(
                symbol=symbol,
                target_price=est.get("targetPrice"),
                target_high=est.get("targetHigh"),
                target_low=est.get("targetLow"),
                num_analysts=int(est.get("numAnalysts", 0)),
                buy_ratings=int(est.get("buyRatings", 0)),
                hold_ratings=int(est.get("holdRatings", 0)),
                sell_ratings=int(est.get("sellRatings", 0)),
                consensus=est.get("consensus", ""),
                eps_estimate_current_year=est.get("epsCurrentYear"),
                eps_estimate_next_year=est.get("epsNextYear"),
                revenue_estimate_current_year=est.get("revenueCurrentYear"),
                source="Morningstar"
            )
        except Exception as e:
            print(f"Error parsing Morningstar estimates: {e}")
            return None

    def get_fair_value(self, symbol: str) -> Optional[Dict]:
        """Get Morningstar fair value estimate (unique to Morningstar)."""
        data = self._request(f"stocks/{symbol}/valuation")
        if not data:
            return None

        return {
            "fair_value": data.get("fairValue"),
            "star_rating": data.get("starRating"),
            "economic_moat": data.get("economicMoat"),
            "moat_trend": data.get("moatTrend"),
            "uncertainty_rating": data.get("uncertaintyRating"),
            "price_to_fair_value": data.get("priceToFairValue")
        }


# ============================================================================
# Twelve Data Provider
# ============================================================================

class TwelveDataProvider(MarketDataProvider):
    """Twelve Data API provider for real-time and historical data."""

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY", "")

    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request."""
        if not self.api_key:
            return None

        params = params or {}
        params["apikey"] = self.api_key

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(f"{self.BASE_URL}/{endpoint}", params=params)
                response.raise_for_status()
                data = response.json()
                if "status" in data and data["status"] == "error":
                    print(f"Twelve Data error: {data.get('message')}")
                    return None
                return data
        except Exception as e:
            print(f"Twelve Data API error: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time quote."""
        data = self._request("quote", {"symbol": symbol})
        if not data:
            return None

        try:
            return StockQuote(
                symbol=symbol,
                price=float(data.get("close", 0)),
                change=float(data.get("change", 0)),
                change_percent=float(data.get("percent_change", 0)),
                volume=int(data.get("volume", 0)),
                timestamp=datetime.now(),
                source="Twelve Data",
                high=float(data.get("high", 0)) if data.get("high") else None,
                low=float(data.get("low", 0)) if data.get("low") else None,
                open=float(data.get("open", 0)) if data.get("open") else None,
                previous_close=float(data.get("previous_close", 0)) if data.get("previous_close") else None
            )
        except Exception as e:
            print(f"Error parsing Twelve Data quote: {e}")
            return None

    def get_historical(self, symbol: str, period: str = "1Y") -> List[HistoricalPrice]:
        """Get historical price data."""
        period_map = {
            "1D": ("1day", 1),
            "1W": ("1day", 7),
            "1M": ("1day", 30),
            "3M": ("1day", 90),
            "6M": ("1day", 180),
            "1Y": ("1day", 365),
            "5Y": ("1week", 260)
        }
        interval, output_size = period_map.get(period, ("1day", 365))

        data = self._request("time_series", {
            "symbol": symbol,
            "interval": interval,
            "outputsize": output_size
        })

        if not data or "values" not in data:
            return []

        prices = []
        for item in data["values"]:
            try:
                prices.append(HistoricalPrice(
                    date=datetime.strptime(item["datetime"], "%Y-%m-%d"),
                    open=float(item.get("open", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    close=float(item.get("close", 0)),
                    volume=int(item.get("volume", 0))
                ))
            except Exception:
                continue

        return prices

    def get_fundamentals(self, symbol: str) -> Optional[CompanyFundamentals]:
        """Get company fundamentals (limited in Twelve Data)."""
        # Twelve Data has limited fundamental data
        profile = self._request("profile", {"symbol": symbol})
        stats = self._request("statistics", {"symbol": symbol})

        if not profile:
            return None

        try:
            return CompanyFundamentals(
                symbol=symbol,
                name=profile.get("name", symbol),
                sector=profile.get("sector", ""),
                industry=profile.get("industry", ""),
                description=profile.get("description", ""),
                market_cap=float(stats.get("market_capitalization", 0)) if stats else 0,
                pe_ratio=float(stats.get("pe_ratio", 0)) if stats and stats.get("pe_ratio") else None,
                eps=float(stats.get("eps", 0)) if stats and stats.get("eps") else None,
                dividend_yield=float(stats.get("dividend_yield", 0)) if stats and stats.get("dividend_yield") else None,
                source="Twelve Data"
            )
        except Exception as e:
            print(f"Error parsing Twelve Data fundamentals: {e}")
            return None

    def get_news(self, symbol: str = None, query: str = None, limit: int = 10) -> List[NewsArticle]:
        """Twelve Data doesn't have news endpoint - return empty."""
        return []


# ============================================================================
# NewsAPI Provider
# ============================================================================

class NewsAPIProvider(MarketDataProvider):
    """NewsAPI provider for financial news."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "")

    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request."""
        if not self.api_key:
            return None

        params = params or {}
        params["apiKey"] = self.api_key

        try:
            with httpx.Client(timeout=30) as client:
                response = client.get(f"{self.BASE_URL}/{endpoint}", params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"NewsAPI error: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """NewsAPI doesn't provide quotes."""
        return None

    def get_historical(self, symbol: str, period: str = "1Y") -> List[HistoricalPrice]:
        """NewsAPI doesn't provide historical data."""
        return []

    def get_fundamentals(self, symbol: str) -> Optional[CompanyFundamentals]:
        """NewsAPI doesn't provide fundamentals."""
        return None

    def get_news(self, symbol: str = None, query: str = None, limit: int = 10) -> List[NewsArticle]:
        """Get news from NewsAPI."""
        search_query = query or symbol or "stock market"

        data = self._request("everything", {
            "q": search_query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit
        })

        if not data or "articles" not in data:
            return []

        articles = []
        for item in data["articles"]:
            try:
                articles.append(NewsArticle(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    published_at=datetime.strptime(
                        item.get("publishedAt", "")[:19],
                        "%Y-%m-%dT%H:%M:%S"
                    ) if item.get("publishedAt") else datetime.now()
                ))
            except Exception:
                continue

        return articles[:limit]


# ============================================================================
# Unified Market Data Service
# ============================================================================

class MarketDataService:
    """
    Unified market data service that can use multiple providers.

    Priority order for each data type can be configured.
    Falls back to alternative providers if primary fails.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize with configuration.

        Config example:
        {
            "quote_provider": "morningstar",  # or "twelve_data"
            "fundamentals_provider": "morningstar",
            "news_provider": "morningstar",  # or "news_api"
            "historical_provider": "twelve_data",  # or "morningstar"
        }
        """
        self.config = config or {}

        # Initialize providers
        self.providers = {
            DataProvider.MORNINGSTAR: MorningstarProvider(),
            DataProvider.TWELVE_DATA: TwelveDataProvider(),
            DataProvider.NEWS_API: NewsAPIProvider(),
        }

    def _get_provider(self, data_type: str) -> MarketDataProvider:
        """Get provider for specific data type."""
        provider_name = self.config.get(f"{data_type}_provider", "morningstar")

        provider_map = {
            "morningstar": DataProvider.MORNINGSTAR,
            "twelve_data": DataProvider.TWELVE_DATA,
            "news_api": DataProvider.NEWS_API,
        }

        provider_enum = provider_map.get(provider_name, DataProvider.MORNINGSTAR)
        return self.providers.get(provider_enum)

    def _get_fallback_providers(self, exclude: DataProvider) -> List[MarketDataProvider]:
        """Get fallback providers excluding the specified one."""
        return [p for k, p in self.providers.items() if k != exclude]

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get quote with fallback."""
        primary = self._get_provider("quote")
        quote = primary.get_quote(symbol)

        if quote:
            return quote

        # Try fallbacks
        for provider in self._get_fallback_providers(DataProvider.MORNINGSTAR):
            quote = provider.get_quote(symbol)
            if quote:
                return quote

        return None

    def get_historical(self, symbol: str, period: str = "1Y") -> List[HistoricalPrice]:
        """Get historical data with fallback."""
        primary = self._get_provider("historical")
        data = primary.get_historical(symbol, period)

        if data:
            return data

        # Try fallbacks
        for provider in self._get_fallback_providers(DataProvider.TWELVE_DATA):
            data = provider.get_historical(symbol, period)
            if data:
                return data

        return []

    def get_fundamentals(self, symbol: str) -> Optional[CompanyFundamentals]:
        """Get fundamentals with fallback."""
        primary = self._get_provider("fundamentals")
        fundamentals = primary.get_fundamentals(symbol)

        if fundamentals:
            return fundamentals

        # Try fallbacks
        for provider in self._get_fallback_providers(DataProvider.MORNINGSTAR):
            fundamentals = provider.get_fundamentals(symbol)
            if fundamentals:
                return fundamentals

        return None

    def get_news(self, symbol: str = None, query: str = None, limit: int = 10) -> List[NewsArticle]:
        """Get news with fallback."""
        primary = self._get_provider("news")
        news = primary.get_news(symbol, query, limit)

        if news:
            return news

        # Try fallbacks
        for provider in self._get_fallback_providers(DataProvider.NEWS_API):
            news = provider.get_news(symbol, query, limit)
            if news:
                return news

        return []

    def get_analyst_estimates(self, symbol: str) -> Optional[AnalystEstimate]:
        """Get analyst estimates (Morningstar only)."""
        morningstar = self.providers.get(DataProvider.MORNINGSTAR)
        if morningstar:
            return morningstar.get_analyst_estimates(symbol)
        return None

    def get_fair_value(self, symbol: str) -> Optional[Dict]:
        """Get Morningstar fair value estimate."""
        morningstar = self.providers.get(DataProvider.MORNINGSTAR)
        if isinstance(morningstar, MorningstarProvider):
            return morningstar.get_fair_value(symbol)
        return None

    def get_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """Get all available data for a symbol."""
        return {
            "quote": self.get_quote(symbol),
            "fundamentals": self.get_fundamentals(symbol),
            "historical": self.get_historical(symbol, "1Y"),
            "news": self.get_news(symbol, limit=5),
            "analyst_estimates": self.get_analyst_estimates(symbol),
            "fair_value": self.get_fair_value(symbol)
        }
