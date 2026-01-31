"""
Valuation module for classifying stocks based on EPV vs Market Cap.
"""
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValuationStatus(Enum):
    """Stock valuation classification."""
    UNDERVALUED = "Undervalued"
    FAIR_VALUE = "Fair Value"
    OVERVALUED = "Overvalued"
    NA = "N/A"
    NEGATIVE_EPV = "N/A (Negative EPV)"


@dataclass
class ValuationResult:
    """Result of a valuation analysis."""
    symbol: str
    company: str
    epv: float
    market_cap: float
    epv_mc_ratio: float
    status: ValuationStatus
    margin_of_safety: float  # Percentage above/below fair value


class Valuator:
    """
    Performs valuation analysis using EPV vs Market Cap comparison.

    Classification thresholds:
    - Undervalued: EPV/MC > 1.3 (EPV is 30%+ above Market Cap)
    - Fair Value: 0.7 <= EPV/MC <= 1.3
    - Overvalued: EPV/MC < 0.7 (EPV is 30%+ below Market Cap)
    """

    def __init__(
        self,
        undervalued_threshold: float = 1.3,
        overvalued_threshold: float = 0.7
    ):
        """
        Initialize the valuator.

        Args:
            undervalued_threshold: EPV/MC ratio above which stock is undervalued
            overvalued_threshold: EPV/MC ratio below which stock is overvalued
        """
        self.undervalued_threshold = undervalued_threshold
        self.overvalued_threshold = overvalued_threshold

    def classify(self, epv: float, market_cap: float) -> Tuple[ValuationStatus, float, float]:
        """
        Classify a single stock's valuation.

        Args:
            epv: Earnings Power Value
            market_cap: Market Capitalization

        Returns:
            Tuple of (status, epv_mc_ratio, margin_of_safety)
        """
        # Handle edge cases
        if pd.isna(epv) or pd.isna(market_cap):
            return ValuationStatus.NA, None, None

        if market_cap <= 0:
            return ValuationStatus.NA, None, None

        if epv <= 0:
            return ValuationStatus.NEGATIVE_EPV, None, None

        # Calculate ratio
        ratio = epv / market_cap

        # Calculate margin of safety (positive = undervalued, negative = overvalued)
        margin_of_safety = (ratio - 1.0) * 100  # as percentage

        # Classify
        if ratio > self.undervalued_threshold:
            return ValuationStatus.UNDERVALUED, ratio, margin_of_safety
        elif ratio >= self.overvalued_threshold:
            return ValuationStatus.FAIR_VALUE, ratio, margin_of_safety
        else:
            return ValuationStatus.OVERVALUED, ratio, margin_of_safety

    def analyze_stock(
        self,
        symbol: str,
        company: str,
        epv: float,
        market_cap: float
    ) -> ValuationResult:
        """
        Perform full valuation analysis on a single stock.

        Args:
            symbol: Stock ticker symbol
            company: Company name
            epv: Earnings Power Value
            market_cap: Market Capitalization

        Returns:
            ValuationResult with full analysis
        """
        status, ratio, margin = self.classify(epv, market_cap)

        return ValuationResult(
            symbol=symbol,
            company=company,
            epv=epv,
            market_cap=market_cap,
            epv_mc_ratio=ratio,
            status=status,
            margin_of_safety=margin
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add valuation columns to a DataFrame.

        Expected columns: 'Earnings Power Value (EPV)', 'Market Cap ($M)'

        Args:
            df: DataFrame with EPV and Market Cap columns

        Returns:
            DataFrame with added valuation columns
        """
        df = df.copy()

        epv_col = 'Earnings Power Value (EPV)'
        mc_col = 'Market Cap ($M)'

        if epv_col not in df.columns or mc_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{epv_col}' and '{mc_col}' columns")

        results = []
        for _, row in df.iterrows():
            status, ratio, margin = self.classify(row[epv_col], row[mc_col])
            results.append({
                'EPV/MC Ratio': ratio,
                'Valuation': status.value,
                'Margin of Safety %': margin
            })

        result_df = pd.DataFrame(results)
        for col in result_df.columns:
            df[col] = result_df[col].values

        return df

    def get_undervalued(self, df: pd.DataFrame, min_margin: float = 0) -> pd.DataFrame:
        """
        Filter DataFrame to only undervalued stocks.

        Args:
            df: DataFrame with valuation columns
            min_margin: Minimum margin of safety percentage (default 0)

        Returns:
            Filtered DataFrame
        """
        if 'Valuation' not in df.columns:
            df = self.analyze_dataframe(df)

        mask = (df['Valuation'] == ValuationStatus.UNDERVALUED.value)

        if min_margin > 0 and 'Margin of Safety %' in df.columns:
            mask = mask & (df['Margin of Safety %'] >= min_margin)

        return df[mask]

    def get_valuation_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of valuation distribution.

        Args:
            df: DataFrame with valuation columns

        Returns:
            Dictionary with counts and percentages
        """
        if 'Valuation' not in df.columns:
            df = self.analyze_dataframe(df)

        total = len(df)
        counts = df['Valuation'].value_counts().to_dict()

        summary = {
            'total': total,
            'counts': counts,
            'percentages': {k: round(v / total * 100, 1) for k, v in counts.items()}
        }

        # Add average margin of safety for undervalued stocks
        undervalued = df[df['Valuation'] == ValuationStatus.UNDERVALUED.value]
        if len(undervalued) > 0 and 'Margin of Safety %' in df.columns:
            summary['avg_margin_undervalued'] = undervalued['Margin of Safety %'].mean()

        return summary

    def rank_by_value(self, df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
        """
        Rank stocks by EPV/MC ratio (value potential).

        Args:
            df: DataFrame with valuation columns
            ascending: If True, rank from most overvalued to most undervalued

        Returns:
            Sorted DataFrame
        """
        if 'EPV/MC Ratio' not in df.columns:
            df = self.analyze_dataframe(df)

        # Filter out N/A values for sorting
        valid_df = df[df['EPV/MC Ratio'].notna()].copy()
        valid_df = valid_df.sort_values('EPV/MC Ratio', ascending=ascending)

        return valid_df


if __name__ == "__main__":
    # Test the valuator
    valuator = Valuator()

    # Test individual classification
    test_cases = [
        ("Strong Undervalued", 150, 100),  # EPV/MC = 1.5
        ("Moderate Undervalued", 135, 100),  # EPV/MC = 1.35
        ("Fair Value High", 125, 100),  # EPV/MC = 1.25
        ("Fair Value Mid", 100, 100),  # EPV/MC = 1.0
        ("Fair Value Low", 75, 100),  # EPV/MC = 0.75
        ("Overvalued", 60, 100),  # EPV/MC = 0.6
        ("Negative EPV", -50, 100),  # Negative
    ]

    print("Individual valuation tests:")
    print("-" * 60)
    for name, epv, mc in test_cases:
        status, ratio, margin = valuator.classify(epv, mc)
        ratio_str = f"{ratio:.2f}" if ratio else "N/A"
        margin_str = f"{margin:+.1f}%" if margin else "N/A"
        print(f"{name:25} EPV={epv:6} MC={mc:6} -> {status.value:20} (Ratio: {ratio_str}, Margin: {margin_str})")

    # Test with sample DataFrame
    print("\n\nDataFrame analysis test:")
    print("-" * 60)

    sample_data = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOG', 'META', 'TSLA'],
        'Company': ['Apple', 'Microsoft', 'Alphabet', 'Meta', 'Tesla'],
        'Earnings Power Value (EPV)': [150, 120, 80, 200, -30],
        'Market Cap ($M)': [100, 100, 100, 100, 100]
    })

    result = valuator.analyze_dataframe(sample_data)
    print(result[['Symbol', 'Earnings Power Value (EPV)', 'Market Cap ($M)',
                  'EPV/MC Ratio', 'Valuation', 'Margin of Safety %']].to_string(index=False))

    print("\nValuation Summary:")
    summary = valuator.get_valuation_summary(result)
    print(f"  Total stocks: {summary['total']}")
    print(f"  Distribution: {summary['counts']}")
    print(f"  Percentages: {summary['percentages']}")
