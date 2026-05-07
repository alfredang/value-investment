"""
Stock screener module for filtering stocks based on fundamental criteria.
"""
import pandas as pd
from typing import Dict, Optional, List, Any
from data_loader import DataLoader


class StockScreener:
    """Filter stocks based on fundamental analysis criteria."""

    # Mapping of user-friendly names to DataFrame column names
    CRITERIA_COLUMNS = {
        'gross_margin': 'Gross Margin %',
        'net_margin': 'Net Margin %',
        'roa': 'ROA %',
        'roe': 'ROE %',
        'revenue_growth_5y': '5-Year Revenue Growth Rate (Per Share)',
        'eps_growth_5y': '5-Year EPS without NRI Growth Rate',
        'debt_to_equity': 'Debt-to-Equity',
        'fcf_margin': 'FCF Margin %',
        'roic_wacc': 'ROIC-WACC',
        'rote_wacc': 'ROTE-WACC'
    }

    # Criteria where lower is better (use <= instead of >=)
    LOWER_IS_BETTER = {'debt_to_equity'}

    def __init__(self, data_loader: DataLoader = None):
        """
        Initialize the screener.

        Args:
            data_loader: DataLoader instance. Creates new one if not provided.
        """
        self.data_loader = data_loader or DataLoader()
        self._data_cache = {}

    def screen(
        self,
        market: str = "US",
        criteria: Dict[str, float] = None,
        include_valuation: bool = True
    ) -> pd.DataFrame:
        """
        Screen stocks based on specified criteria.

        Args:
            market: "US" or "SG" market
            criteria: Dictionary of criteria name to threshold value
                     e.g., {'gross_margin': 20, 'roe': 15, 'debt_to_equity': 1.5}
            include_valuation: Whether to include valuation classification

        Returns:
            DataFrame of stocks meeting all criteria
        """
        # Load data (use cache if available)
        cache_key = market.upper()
        if cache_key not in self._data_cache:
            self._data_cache[cache_key] = self.data_loader.load_screener_data(market)

        df = self._data_cache[cache_key].copy()

        # Apply criteria filters
        if criteria:
            for criterion_name, threshold in criteria.items():
                if threshold is None:
                    continue

                if criterion_name not in self.CRITERIA_COLUMNS:
                    print(f"Warning: Unknown criterion '{criterion_name}', skipping")
                    continue

                column = self.CRITERIA_COLUMNS[criterion_name]
                if column not in df.columns:
                    print(f"Warning: Column '{column}' not found in data, skipping")
                    continue

                # Apply filter based on criterion type
                if criterion_name in self.LOWER_IS_BETTER:
                    # For debt-to-equity, lower is better
                    mask = (df[column] <= threshold) & (df[column] >= 0)
                else:
                    # For most metrics, higher is better
                    mask = df[column] >= threshold

                df = df[mask]

        # Add valuation classification using the in-house formula
        if include_valuation:
            df = self._add_valuation(df)

        return df

    def _add_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add valuation columns using the client's in-house two-stage DCF formula.

        Adds:
        - Low IV       (conservative intrinsic value per share)
        - High IV      (aggressive intrinsic value per share)
        - Valuation    (Undervalued / Fair Value / Overvalued / N/A)

        Verdict logic:
        - Undervalued: Current Price < Low IV  (cheap by both scenarios)
        - Fair Value : Low IV ≤ Current Price ≤ High IV
        - Overvalued : Current Price > High IV (expensive by both scenarios)
        """
        from valuation import (
            compute_in_house_valuation,
            derive_eps_inputs_from_csv_row,
            classify_verdict,
            ValuationStatus,
        )

        df = df.copy()
        low_ivs, high_ivs, verdicts = [], [], []

        for _, row in df.iterrows():
            inputs = derive_eps_inputs_from_csv_row(row)
            if inputs is None:
                low_ivs.append(None)
                high_ivs.append(None)
                verdicts.append(ValuationStatus.NA.value)
                continue

            present_eps, past_eps, years_back = inputs
            iv = compute_in_house_valuation(present_eps, past_eps, years_back=years_back)
            if iv is None:
                low_ivs.append(None)
                high_ivs.append(None)
                verdicts.append(ValuationStatus.NA.value)
                continue

            low_ivs.append(iv["low_iv"])
            high_ivs.append(iv["high_iv"])

            try:
                cp = float(row.get("Current Price")) if row.get("Current Price") is not None else None
            except (TypeError, ValueError):
                cp = None
            verdicts.append(classify_verdict(cp, iv["low_iv"], iv["high_iv"]).value)

        df["Low IV"] = low_ivs
        df["High IV"] = high_ivs
        df["Valuation"] = verdicts
        return df

    def get_available_criteria(self) -> Dict[str, str]:
        """
        Get list of available screening criteria.

        Returns:
            Dictionary mapping criterion name to description
        """
        return {
            'gross_margin': 'Gross Margin % (>= threshold)',
            'net_margin': 'Net Margin % (>= threshold)',
            'roa': 'Return on Assets % (>= threshold)',
            'roe': 'Return on Equity % (>= threshold)',
            'revenue_growth_5y': '5-Year Revenue Growth Rate (>= threshold)',
            'eps_growth_5y': '5-Year EPS Growth Rate (>= threshold)',
            'debt_to_equity': 'Debt-to-Equity Ratio (<= threshold)',
            'fcf_margin': 'Free Cash Flow Margin % (>= threshold)',
            'roic_wacc': 'ROIC minus WACC (>= threshold)',
            'rote_wacc': 'ROTE minus WACC (>= threshold)'
        }

    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for screened results.

        Args:
            df: Screened DataFrame

        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_stocks': len(df),
            'by_sector': df['Sector'].value_counts().to_dict() if 'Sector' in df.columns else {},
            'by_valuation': df['Valuation'].value_counts().to_dict() if 'Valuation' in df.columns else {}
        }

        # Add average metrics
        for criterion, column in self.CRITERIA_COLUMNS.items():
            if column in df.columns:
                stats[f'avg_{criterion}'] = df[column].mean()

        return stats

    def format_results(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        sort_by: str = 'Valuation'
    ) -> pd.DataFrame:
        """
        Format results for display.

        Args:
            df: Screened DataFrame
            columns: List of columns to include. None for default set.
            sort_by: Column to sort results by

        Returns:
            Formatted DataFrame
        """
        default_columns = [
            'Symbol', 'Company', 'Sector',
            'Gross Margin %', 'Net Margin %', 'ROE %', 'ROA %',
            'Debt-to-Equity', 'FCF Margin %',
            'ROIC-WACC', 'ROTE-WACC',
            'Market Cap ($M)', 'EPV/MC Ratio', 'Valuation'
        ]

        columns = columns or default_columns
        available_columns = [c for c in columns if c in df.columns]

        result = df[available_columns].copy()

        # Sort by valuation priority if sorting by Valuation
        if sort_by == 'Valuation' and 'Valuation' in result.columns:
            valuation_order = {'Undervalued': 0, 'Fair Value': 1, 'Overvalued': 2, 'N/A': 3, 'N/A (Negative EPV)': 4}
            result['_sort_key'] = result['Valuation'].map(valuation_order)
            result = result.sort_values('_sort_key').drop('_sort_key', axis=1)
        elif sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=False)

        return result


if __name__ == "__main__":
    # Test the screener
    screener = StockScreener()

    print("Available criteria:")
    for name, desc in screener.get_available_criteria().items():
        print(f"  {name}: {desc}")

    # Example screening
    criteria = {
        'gross_margin': 30,
        'roe': 15,
        'debt_to_equity': 1.0
    }

    print(f"\nScreening US stocks with criteria: {criteria}")

    try:
        results = screener.screen(market="US", criteria=criteria)
        print(f"Found {len(results)} stocks matching criteria")

        if len(results) > 0:
            formatted = screener.format_results(results)
            print("\nTop 10 results:")
            print(formatted.head(10).to_string(index=False))

            print("\nSummary:")
            stats = screener.get_summary_stats(results)
            print(f"  By valuation: {stats.get('by_valuation', {})}")
    except Exception as e:
        print(f"Error: {e}")
