"""
Financial Modeling Prep (FMP) API integration for fetching real financial data.

Provides methods to fetch financial metrics, ratios, statements, and valuation data
from FMP API without requiring local files.
"""
import requests
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime
import time


class FMPClient:
    """Client for Financial Modeling Prep API."""

    def __init__(self, api_key: str):
        """
        Initialize FMP API client.

        Args:
            api_key: FMP API key from financialmodelingprep.com
        """
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # Delay between requests to avoid rate limiting

    def _get(self, endpoint: str, params: dict = None) -> Dict[str, Any]:
        """
        Make GET request to FMP API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        time.sleep(self.rate_limit_delay)
        
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=10)
            response.raise_for_status()
            return response.json() if response.text else {}
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return {}

    def get_financial_ratios(self, symbol: str, period: str = "annual", limit: int = 10) -> pd.DataFrame:
        """
        Get financial ratios for a company.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods to retrieve

        Returns:
            DataFrame with financial ratios
        """
        data = self._get(f"/financial-ratios-ttm/{symbol}")
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        return pd.DataFrame(data)

    def get_profile(self, symbol: str) -> Dict[str, Any]:
        """
        Get company profile/information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        data = self._get(f"/profile/{symbol}")
        return data[0] if isinstance(data, list) and data else {}

    def get_ratios(self, symbol: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
        """
        Get detailed financial ratios.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            DataFrame with ratios
        """
        endpoint = f"/ratios/{symbol}"
        params = {"limit": limit}
        data = self._get(endpoint, params)
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df

    def get_income_statement(self, symbol: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            DataFrame with income statement
        """
        endpoint = f"/income-statement/{symbol}"
        params = {"period": period, "limit": limit}
        data = self._get(endpoint, params)
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df

    def get_balance_sheet(self, symbol: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
        """
        Get balance sheet data.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            DataFrame with balance sheet
        """
        endpoint = f"/balance-sheet-statement/{symbol}"
        params = {"period": period, "limit": limit}
        data = self._get(endpoint, params)
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df

    def get_cash_flow(self, symbol: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
        """
        Get cash flow statement data.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            DataFrame with cash flow statement
        """
        endpoint = f"/cash-flow-statement/{symbol}"
        params = {"period": period, "limit": limit}
        data = self._get(endpoint, params)
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time stock quote.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with quote data
        """
        data = self._get(f"/quote/{symbol}")
        return data[0] if isinstance(data, list) and data else {}

    def get_key_metrics(self, symbol: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
        """
        Get key financial metrics.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            DataFrame with key metrics
        """
        endpoint = f"/key-metrics/{symbol}"
        params = {"period": period, "limit": limit}
        data = self._get(endpoint, params)
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df

    def get_enterprise_value(self, symbol: str, period: str = "annual", limit: int = 20) -> pd.DataFrame:
        """
        Get enterprise value metrics.

        Args:
            symbol: Stock ticker symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            DataFrame with enterprise value data
        """
        endpoint = f"/enterprise-values/{symbol}"
        params = {"period": period, "limit": limit}
        data = self._get(endpoint, params)
        
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        return df

    def get_financial_statements(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive financial statements for analysis.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing all financial data
        """
        return {
            'profile': self.get_profile(symbol),
            'quote': self.get_quote(symbol),
            'ratios': self.get_ratios(symbol),
            'key_metrics': self.get_key_metrics(symbol),
            'income_statement': self.get_income_statement(symbol),
            'balance_sheet': self.get_balance_sheet(symbol),
            'cash_flow': self.get_cash_flow(symbol),
            'enterprise_value': self.get_enterprise_value(symbol)
        }

    def build_screener_row(self, symbol: str) -> Dict[str, Any]:
        """
        Build a screening row with key metrics for a company.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with screening metrics
        """
        statements = self.get_financial_statements(symbol)
        profile = statements['profile']
        quote = statements['quote']
        ratios = statements['ratios']
        key_metrics = statements['key_metrics']
        
        # Get latest values from ratios and metrics
        latest_ratios = ratios.iloc[0].to_dict() if not ratios.empty else {}
        latest_metrics = key_metrics.iloc[0].to_dict() if not key_metrics.empty else {}
        
        row = {
            'Symbol': symbol,
            'Company': profile.get('companyName', 'N/A'),
            'Sector': profile.get('sector', 'N/A'),
            'Industry': profile.get('industry', 'N/A'),
            'Market Cap ($M)': latest_metrics.get('marketCapitalization', quote.get('marketCap', 0)) / 1_000_000 if latest_metrics.get('marketCapitalization') else 0,
            'Current Price': quote.get('price', 0),
            'ROE %': latest_ratios.get('returnOnEquity', 0) * 100 if latest_ratios.get('returnOnEquity') else 0,
            'ROA %': latest_ratios.get('returnOnAssets', 0) * 100 if latest_ratios.get('returnOnAssets') else 0,
            'Gross Margin %': latest_ratios.get('grossMargin', 0) * 100 if latest_ratios.get('grossMargin') else 0,
            'Net Margin %': latest_ratios.get('netMargin', 0) * 100 if latest_ratios.get('netMargin') else 0,
            'Operating Margin %': latest_metrics.get('operatingMargin', 0) * 100 if latest_metrics.get('operatingMargin') else 0,
            'Debt-to-Equity': latest_ratios.get('debtRatio', 0),
            'Current Ratio': latest_ratios.get('currentRatio', 0),
            'Quick Ratio': latest_ratios.get('quickRatio', 0),
            'ROIC %': latest_metrics.get('roic', 0) * 100 if latest_metrics.get('roic') else 0,
            'ROIC-WACC': latest_metrics.get('roic', 0) - profile.get('wacc', 0) if latest_metrics.get('roic') and profile.get('wacc') else 0,
            'PE Ratio': latest_metrics.get('peRatio', 0),
            'PB Ratio': latest_metrics.get('pbRatio', 0),
            'FCF Margin %': 0,  # Will need to calculate from statements
            '5-Year Revenue Growth Rate (Per Share)': latest_metrics.get('revenuePerShare', 0),
            '5-Year EPS without NRI Growth Rate': latest_metrics.get('eps', 0),
        }
        
        return row

    def get_symbol_list(self, exchange: str = "NASDAQ") -> List[str]:
        """
        Get list of available symbols for an exchange.

        Args:
            exchange: Exchange name (NASDAQ, NYSE, etc.)

        Returns:
            List of stock symbols
        """
        data = self._get(f"/available-traded/list")
        
        if not data or not isinstance(data, list):
            return []
        
        df = pd.DataFrame(data)
        if 'symbol' in df.columns:
            return df[df['exchangeShortName'] == exchange]['symbol'].tolist()
        
        return []


def fetch_stock_data(api_key: str, symbols: List[str]) -> pd.DataFrame:
    """
    Convenience function to fetch stock screening data for multiple symbols.

    Args:
        api_key: FMP API key
        symbols: List of stock symbols

    Returns:
        DataFrame with screening data for all symbols
    """
    client = FMPClient(api_key)
    rows = []
    
    for symbol in symbols:
        try:
            row = client.build_screener_row(symbol)
            rows.append(row)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            continue
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    api_key = "fqrzJCICPFC6oxRCEc1812YewLXb5Oh4"
    client = FMPClient(api_key)
    
    print("Fetching MSFT data...")
    profile = client.get_profile("MSFT")
    print(f"Company: {profile.get('companyName')}")
    
    print("\nFetching MSFT ratios...")
    ratios = client.get_ratios("MSFT")
    print(ratios.head())
    
    print("\nBuilding screening row...")
    row = client.build_screener_row("MSFT")
    print(row)
