"""
Data loader module for loading stock data from CSV and XLS files.
"""
import os
import glob
import pandas as pd
import xlrd
from typing import Optional, Dict, List, Any


class DataLoader:
    """Handles loading of stock screening data and financial statements."""

    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing data files. Defaults to current directory.
        """
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))

    def load_screener_data(self, market: str = "US") -> pd.DataFrame:
        """
        Load stock screener data for a specific market.

        Args:
            market: "US" or "SG" for United States or Singapore markets.

        Returns:
            DataFrame with stock screening data.
        """
        market = market.upper()
        pattern = os.path.join(self.data_dir, f"{market} All In One Screeners*.csv")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"No screener file found for market '{market}' in {self.data_dir}")

        # Use the most recent file if multiple exist
        file_path = sorted(files)[-1]

        df = pd.read_csv(file_path, encoding='utf-8-sig')

        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()

        # Convert numeric columns
        numeric_cols = [
            'Gross Margin %', 'Net Margin %', 'ROA %', 'ROE %',
            'Debt-to-Equity', 'Quick Ratio', 'ROIC %', 'WACC %',
            'ROIC-WACC', 'ROTE', 'ROTE-WACC', 'FCF Margin %',
            'Earnings Power Value (EPV)', 'Market Cap ($M)', 'Current Price',
            '5-Year EPS without NRI Growth Rate', '5-Year Revenue Growth Rate (Per Share)'
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def load_anomaly_data(self, symbol: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load financial data for anomaly detection from XLS file.

        Args:
            symbol: Optional stock symbol to load. If None, loads all available.

        Returns:
            Dictionary mapping symbol to DataFrame with 30-year financials.
        """
        pattern = os.path.join(self.data_dir, "Companies with anomalies*.xls")
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"No anomaly data file found in {self.data_dir}")

        file_path = files[0]

        try:
            book = xlrd.open_workbook(file_path, ignore_workbook_corruption=True)
        except Exception as e:
            raise RuntimeError(f"Error reading XLS file: {e}")

        result = {}

        for sheet_name in book.sheet_names():
            # Extract symbol from sheet name (format: "NAS_DDI" -> "DDI")
            parts = sheet_name.split('_')
            if len(parts) >= 2:
                sheet_symbol = parts[1]
            else:
                sheet_symbol = sheet_name

            # Filter by symbol if specified
            if symbol and sheet_symbol.upper() != symbol.upper():
                continue

            sheet = book.sheet_by_name(sheet_name)
            data = self._parse_financial_sheet(sheet)
            result[sheet_symbol] = data

        return result

    def _parse_financial_sheet(self, sheet) -> Dict[str, Any]:
        """
        Parse a financial data sheet into structured data.

        Args:
            sheet: xlrd sheet object

        Returns:
            Dictionary with parsed financial data.
        """
        data = {
            'key_stats': {},
            'growth_rates': {},
            'per_share_data': {},
            'ratios': {},
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'valuation_ratios': {},
            'quality_metrics': {},
            'fiscal_periods': []
        }

        # Find fiscal period row to get column headers
        fiscal_row = None
        for row_idx in range(min(25, sheet.nrows)):
            cell_val = sheet.cell_value(row_idx, 0)
            if 'Fiscal Period' in str(cell_val):
                fiscal_row = row_idx
                break

        if fiscal_row:
            # Extract fiscal periods (column headers)
            for col_idx in range(1, sheet.ncols):
                val = sheet.cell_value(fiscal_row, col_idx)
                if val and str(val).startswith('Dec'):
                    data['fiscal_periods'].append(str(val))

        # Define row mappings for each section
        row_mappings = {
            'key_stats': {
                'Price ($)': 'price',
                'Market Cap ($ Million)': 'market_cap',
                'PE Ratio': 'pe_ratio',
                'PB Ratio': 'pb_ratio',
                'PS Ratio': 'ps_ratio'
            },
            'per_share_data': {
                'Revenue per Share': 'revenue_per_share',
                'EBITDA per Share': 'ebitda_per_share',
                'Earnings per Share (Diluted)': 'eps_diluted',
                'EPS without NRI': 'eps_without_nri',
                'Free Cash Flow per Share': 'fcf_per_share',
                'Operating Cash Flow per Share': 'ocf_per_share',
                'Book Value per Share': 'book_value_per_share'
            },
            'ratios': {
                'ROE %': 'roe',
                'ROA %': 'roa',
                'ROIC %': 'roic',
                'WACC %': 'wacc',
                'Gross Margin %': 'gross_margin',
                'Net Margin %': 'net_margin',
                'Operating Margin %': 'operating_margin',
                'FCF Margin %': 'fcf_margin',
                'Debt-to-Equity': 'debt_to_equity',
                'Return-on-Tangible-Equity': 'rote'
            },
            'income_statement': {
                'Revenue': 'revenue',
                'Gross Profit': 'gross_profit',
                'Operating Income': 'operating_income',
                'Net Income': 'net_income',
                'EBITDA': 'ebitda'
            },
            'balance_sheet': {
                'Total Assets': 'total_assets',
                'Total Liabilities': 'total_liabilities',
                'Total Stockholders Equity': 'total_equity',
                'Total Receivables': 'receivables',
                'Total Inventories': 'inventories',
                'Goodwill': 'goodwill'
            },
            'cash_flow': {
                'Cash Flow from Operations': 'operating_cash_flow',
                'Cash Flow from Investing': 'investing_cash_flow',
                'Cash Flow from Financing': 'financing_cash_flow',
                'Free Cash Flow': 'free_cash_flow',
                'Capital Expenditure': 'capex'
            },
            'quality_metrics': {
                'Altman Z-Score': 'z_score',
                'Piotroski F-Score': 'f_score',
                'Beneish M-Score': 'm_score',
                'Sloan Ratio %': 'sloan_ratio',
                'Current Ratio': 'current_ratio',
                'Quick Ratio': 'quick_ratio'
            }
        }

        # Parse each row
        for row_idx in range(sheet.nrows):
            row_label = str(sheet.cell_value(row_idx, 0)).strip()

            # Find which section and key this row belongs to
            for section, mappings in row_mappings.items():
                if row_label in mappings:
                    key = mappings[row_label]
                    values = []

                    # Extract values for each fiscal period
                    for col_idx in range(1, min(sheet.ncols, len(data['fiscal_periods']) + 1)):
                        val = sheet.cell_value(row_idx, col_idx)
                        if val == '-' or val == '':
                            values.append(None)
                        elif isinstance(val, (int, float)):
                            values.append(val)
                        else:
                            try:
                                values.append(float(val))
                            except (ValueError, TypeError):
                                values.append(None)

                    data[section][key] = values
                    break

        return data

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in the anomaly data file.

        Returns:
            List of stock symbols.
        """
        pattern = os.path.join(self.data_dir, "Companies with anomalies*.xls")
        files = glob.glob(pattern)

        if not files:
            return []

        try:
            book = xlrd.open_workbook(files[0], ignore_workbook_corruption=True)
            symbols = []
            for sheet_name in book.sheet_names():
                parts = sheet_name.split('_')
                if len(parts) >= 2:
                    symbols.append(parts[1])
            return symbols
        except Exception:
            return []


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()

    print("Testing US screener data...")
    try:
        us_data = loader.load_screener_data("US")
        print(f"Loaded {len(us_data)} US stocks")
        print(f"Columns: {list(us_data.columns)[:10]}...")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting SG screener data...")
    try:
        sg_data = loader.load_screener_data("SG")
        print(f"Loaded {len(sg_data)} SG stocks")
    except Exception as e:
        print(f"Error: {e}")

    print("\nAvailable anomaly symbols:", loader.get_available_symbols())
