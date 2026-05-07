"""
In-house valuation formula (per client Excel: "In-house valuation.xlsx").

Two-stage DCF model that produces a Low and High intrinsic value per stock:

    historical_growth   = (Present_EPS / Past_EPS) ^ (1/years_back) - 1
    Low_IV_growth       = MIN(historical_growth / 8, 3%)    # conservative
    High_IV_growth      = MIN(historical_growth / 2, 12%)   # aggressive

    For each scenario, IV = PV(growth phase) + PV(terminal phase):

        Stage 1 = F * (1+g) * (1 - ((1+g)/(1+I))^H) / (I - g)
        Stage 2 = F * (1+g)^H * (1+J) * (1 - ((1+J)/(1+I))^K) / (I - J) / (1+I)^H
        IV      = (Stage 1 + Stage 2) * (1 - L)

    Where (defaults from the client's Excel):
        F = Present EPS
        H = 10  (growth phase years)
        I = 10% (discount rate)
        J = 2%  (inflation / terminal growth rate)
        K = 10  (terminal phase years)
        L = 0   (tax/discount adjustment)

Verdict mapping:
    Undervalued  → current_price < Low IV   (cheap by both scenarios)
    Fair Value   → Low IV ≤ current_price ≤ High IV
    Overvalued   → current_price > High IV  (expensive by both scenarios)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import pandas as pd


# ============================================================================
# Defaults — match the client's Excel
# ============================================================================
DEFAULT_DISCOUNT_RATE = 0.10           # I
DEFAULT_INFLATION_RATE = 0.02          # J (terminal growth)
DEFAULT_GROWTH_YEARS = 10              # H
DEFAULT_TERMINAL_YEARS = 10            # K
DEFAULT_MARGIN_OF_SAFETY = 0.30        # L (30% — per client's Excel)

LOW_IV_GROWTH_DIVISOR = 8              # conservative dampening
LOW_IV_GROWTH_CAP = 0.03               # 3%
HIGH_IV_GROWTH_DIVISOR = 2             # aggressive dampening
HIGH_IV_GROWTH_CAP = 0.12              # 12%


# ============================================================================
# Verdict status
# ============================================================================
class ValuationStatus(Enum):
    UNDERVALUED = "Undervalued"
    FAIR_VALUE = "Fair Value"
    OVERVALUED = "Overvalued"
    NA = "N/A"
    NEGATIVE_EPS = "N/A (Negative EPS)"


@dataclass
class ValuationResult:
    """In-house valuation output for a single stock."""
    symbol: str
    company: str
    present_eps: Optional[float]
    past_eps: Optional[float]
    historical_growth: Optional[float]
    low_iv: Optional[float]
    high_iv: Optional[float]
    current_price: Optional[float]
    status: ValuationStatus


# ============================================================================
# Core formula
# ============================================================================

def _two_stage_dcf(
    present_eps: float,
    growth_rate: float,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    growth_years: int = DEFAULT_GROWTH_YEARS,
    terminal_years: int = DEFAULT_TERMINAL_YEARS,
    margin_of_safety: float = DEFAULT_MARGIN_OF_SAFETY,
) -> float:
    """Two-stage DCF — exact transcription of the client Excel formula."""
    F = present_eps
    g = growth_rate
    I = discount_rate
    J = inflation_rate
    H = growth_years
    K = terminal_years
    L = margin_of_safety

    # Stage 1: PV of EPS growing at g for H years, discounted at I
    if abs(I - g) < 1e-9:
        stage1 = F * (1 + g) * H / (1 + I)  # L'Hopital limit
    else:
        stage1 = (F * (1 + g) * (1 - ((1 + g) / (1 + I)) ** H)) / (I - g)

    # Stage 2: terminal phase — EPS at end of growth phase, then growing at J for K years
    eps_at_end = F * (1 + g) ** H
    if abs(I - J) < 1e-9:
        stage2 = eps_at_end * (1 + J) * K / (1 + I) ** (H + 1)
    else:
        stage2 = (
            eps_at_end * (1 + J)
            * (1 - ((1 + J) / (1 + I)) ** K)
            / (I - J)
            / (1 + I) ** H
        )

    return (stage1 + stage2) * (1 - L)


def compute_in_house_valuation(
    present_eps: float,
    past_eps: float,
    years_back: int = 5,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    inflation_rate: float = DEFAULT_INFLATION_RATE,
    growth_years: int = DEFAULT_GROWTH_YEARS,
    terminal_years: int = DEFAULT_TERMINAL_YEARS,
    margin_of_safety: float = DEFAULT_MARGIN_OF_SAFETY,
) -> Optional[Dict[str, float]]:
    """
    Compute Low IV and High IV for a stock from EPS data.

    Args:
        present_eps: current EPS
        past_eps: EPS `years_back` years ago
        years_back: lookback window for historical growth (default 5)

    Returns:
        Dict with keys:
            historical_growth (decimal, e.g. 0.12 = 12%)
            low_iv_growth (the dampened growth used for Low IV)
            high_iv_growth (the dampened growth used for High IV)
            low_iv (conservative intrinsic value per share)
            high_iv (aggressive intrinsic value per share)
        Or None if inputs are invalid (negative/zero EPS).
    """
    if present_eps is None or past_eps is None:
        return None
    if present_eps <= 0 or past_eps <= 0 or years_back <= 0:
        return None

    historical_growth = (present_eps / past_eps) ** (1.0 / years_back) - 1.0

    low_g = min(historical_growth / LOW_IV_GROWTH_DIVISOR, LOW_IV_GROWTH_CAP)
    high_g = min(historical_growth / HIGH_IV_GROWTH_DIVISOR, HIGH_IV_GROWTH_CAP)

    low_iv = _two_stage_dcf(
        present_eps, low_g,
        discount_rate=discount_rate,
        inflation_rate=inflation_rate,
        growth_years=growth_years,
        terminal_years=terminal_years,
        margin_of_safety=margin_of_safety,
    )
    high_iv = _two_stage_dcf(
        present_eps, high_g,
        discount_rate=discount_rate,
        inflation_rate=inflation_rate,
        growth_years=growth_years,
        terminal_years=terminal_years,
        margin_of_safety=margin_of_safety,
    )

    return {
        "historical_growth": historical_growth,
        "low_iv_growth": low_g,
        "high_iv_growth": high_g,
        "low_iv": round(low_iv, 2),
        "high_iv": round(high_iv, 2),
    }


# ============================================================================
# CSV-row helpers — derive EPS inputs from screener metrics
# ============================================================================

def derive_eps_inputs_from_csv_row(row) -> Optional[Tuple[float, float, int]]:
    """
    Derive (present_eps, past_eps, years_back) from a screener CSV row.

    Two-path strategy (some CSV exports don't include PE Ratio):
      Path A — PE-based:  Present EPS = Current Price / PE Ratio (TTM)
      Path B — EPV-based: Present EPS = EPV × (WACC% / 100)
                          (EPV is a no-growth perpetuity of EPS at WACC, so
                          EPS = EPV × WACC by definition)

      Past EPS  = Present EPS / (1 + 5Y EPS Growth Rate)^5
      years_back = 5

    Returns None if neither path produces valid positive inputs.
    """
    def _num(v):
        try:
            f = float(v)
            return f if f == f else None  # filter NaN
        except (TypeError, ValueError):
            return None

    current_price = _num(row.get('Current Price'))
    eps_growth_pct = _num(row.get('5-Year EPS without NRI Growth Rate'))
    if current_price is None or eps_growth_pct is None:
        return None

    present_eps: Optional[float] = None

    # Path A: PE-based
    pe_ratio = _num(row.get('PE Ratio (TTM)'))
    if pe_ratio is not None and pe_ratio > 0:
        present_eps = current_price / pe_ratio

    # Path B: EPV-based (fallback when PE column is absent or invalid)
    if present_eps is None or present_eps <= 0:
        epv = _num(row.get('Earnings Power Value (EPV)'))
        wacc_pct = _num(row.get('WACC %'))
        if epv is not None and epv > 0 and wacc_pct is not None and wacc_pct > 0:
            present_eps = epv * (wacc_pct / 100.0)

    if present_eps is None or present_eps <= 0:
        return None

    growth_decimal = eps_growth_pct / 100.0
    past_eps = present_eps / ((1 + growth_decimal) ** 5)
    if past_eps <= 0:
        return None

    return present_eps, past_eps, 5


# ============================================================================
# Verdict + DataFrame integration
# ============================================================================

def classify_verdict(
    current_price: Optional[float],
    low_iv: Optional[float],
    high_iv: Optional[float],
) -> ValuationStatus:
    """Map current price to a verdict against the IV range."""
    if current_price is None or low_iv is None or high_iv is None:
        return ValuationStatus.NA
    if low_iv <= 0 or high_iv <= 0:
        return ValuationStatus.NEGATIVE_EPS
    if current_price < low_iv:
        return ValuationStatus.UNDERVALUED
    if current_price <= high_iv:
        return ValuationStatus.FAIR_VALUE
    return ValuationStatus.OVERVALUED


# ============================================================================
# Backward-compatibility: Valuator class (for existing app.py imports)
# ============================================================================

class Valuator:
    """
    Backward-compatible wrapper. The old EPV-based logic has been replaced
    by the in-house two-stage DCF formula from the client's Excel.
    """

    def __init__(self, undervalued_threshold: float = None, overvalued_threshold: float = None):
        # Thresholds are now defined by Low IV / High IV bounds (per stock),
        # not a global ratio. These args are kept for signature compatibility.
        self.undervalued_threshold = undervalued_threshold
        self.overvalued_threshold = overvalued_threshold

    def classify(self, epv, market_cap):
        """Legacy EPV/MC classifier — retained for any old call sites."""
        if pd.isna(epv) or pd.isna(market_cap) or market_cap <= 0:
            return ValuationStatus.NA, None, None
        if epv <= 0:
            return ValuationStatus.NEGATIVE_EPS, None, None
        ratio = epv / market_cap
        margin = (ratio - 1.0) * 100
        if ratio > 1.3:
            return ValuationStatus.UNDERVALUED, ratio, margin
        if ratio >= 0.7:
            return ValuationStatus.FAIR_VALUE, ratio, margin
        return ValuationStatus.OVERVALUED, ratio, margin

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'Low IV', 'High IV', 'Valuation' columns using the in-house formula."""
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
            low_ivs.append(iv['low_iv'])
            high_ivs.append(iv['high_iv'])
            current_price = row.get('Current Price')
            try:
                cp = float(current_price) if current_price is not None else None
            except (TypeError, ValueError):
                cp = None
            verdicts.append(classify_verdict(cp, iv['low_iv'], iv['high_iv']).value)

        df['Low IV'] = low_ivs
        df['High IV'] = high_ivs
        df['Valuation'] = verdicts
        return df


if __name__ == "__main__":
    # Sanity check — replicate the 3 examples from the client's Excel
    test_cases = [
        ("AAA", 0.68, 2.12, 10, 14.28, 19.12),
        ("BBB", 2.14, 11.05, 9, 79.34, 129.66),
        ("CCC", 1.50, 10.39, 5, 77.03, 139.27),
    ]
    print(f"{'Ticker':6} {'PastEPS':>8} {'PresentEPS':>10} {'Years':>6} | "
          f"{'ExpLow':>8} {'GotLow':>8} {'ExpHigh':>8} {'GotHigh':>8}")
    print("-" * 80)
    for ticker, past, present, years, exp_low, exp_high in test_cases:
        iv = compute_in_house_valuation(present, past, years)
        if iv:
            print(f"{ticker:6} {past:>8.2f} {present:>10.2f} {years:>6} | "
                  f"{exp_low:>8.2f} {iv['low_iv']:>8.2f} {exp_high:>8.2f} {iv['high_iv']:>8.2f}")
