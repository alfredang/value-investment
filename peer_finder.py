"""
Peer-finding helpers — identifies competitors of a target company from
the loaded screening universe based on Sector + Industry match.

Used by:
- Step 3 Streamlit competitor comparison view
- DOCX report Section E (competitor bar charts)
"""
from typing import Optional
import pandas as pd


PEER_METRIC_COLUMNS = [
    'ROE %',
    'Net Margin %',
    'Gross Margin %',
    'FCF Margin %',
    '5-Year Revenue Growth Rate (Per Share)',
    'Debt-to-Equity',
    'ROIC-WACC',
]


def find_peers(
    target_symbol: str,
    universe: pd.DataFrame,
    limit: int = 5,
    same_industry_first: bool = True,
) -> pd.DataFrame:
    """
    Find peer companies for the target from the screening universe.

    Match priority (narrow → wide), so peers are in the same actual business:
      1. Same Subindustry (e.g. "Marine Shipping" — most specific)
      2. Same Industry    (e.g. "Transportation")
      3. Same Sector      (e.g. "Industrials")

    The Subindustry classification comes from the user's screener export
    (Gurufocus All-In-One uses GICS-style sub-industry tags). This is real
    industry-classification data, not heuristics.

    Args:
        target_symbol: ticker of the target company
        universe: DataFrame of all screened/loaded stocks
        limit: max number of peers to return
        same_industry_first: kept for API compatibility (ignored — match
            priority is now Subindustry > Industry > Sector unconditionally)

    Returns:
        DataFrame of peer rows, sorted by Market Cap descending. May be empty
        if no peers exist at any tier.
    """
    if universe is None or universe.empty:
        return pd.DataFrame()

    target_rows = universe[universe['Symbol'] == target_symbol]
    if target_rows.empty:
        return pd.DataFrame()

    target = target_rows.iloc[0]
    target_sector = target.get('Sector')
    target_industry = target.get('Industry')
    target_subindustry = target.get('Subindustry')

    others = universe[universe['Symbol'] != target_symbol].copy()

    def _valid(s):
        return s is not None and pd.notna(s) and str(s).strip() not in ("", "N/A")

    # Tier 1: Same Subindustry — actual same-business peers
    tier1 = pd.DataFrame()
    if 'Subindustry' in others.columns and _valid(target_subindustry):
        tier1 = others[others['Subindustry'] == target_subindustry]

    candidates = tier1

    # Tier 2: widen to same Industry if Tier 1 didn't yield enough
    if len(candidates) < limit and _valid(target_industry):
        tier2 = others[others['Industry'] == target_industry]
        if not tier2.empty:
            existing_syms = set(candidates['Symbol']) if not candidates.empty else set()
            extras = tier2[~tier2['Symbol'].isin(existing_syms)]
            candidates = pd.concat([candidates, extras], ignore_index=True)

    # Tier 3: widen to same Sector if still too few
    if len(candidates) < limit and _valid(target_sector):
        tier3 = others[others['Sector'] == target_sector]
        if not tier3.empty:
            existing_syms = set(candidates['Symbol']) if not candidates.empty else set()
            extras = tier3[~tier3['Symbol'].isin(existing_syms)]
            candidates = pd.concat([candidates, extras], ignore_index=True)

    if candidates.empty:
        return pd.DataFrame()

    # Sort by Market Cap descending and take top N
    if 'Market Cap ($M)' in candidates.columns:
        candidates = candidates.sort_values('Market Cap ($M)', ascending=False, na_position='last')

    return candidates.head(limit)


def build_peer_metrics_frame(target_row: pd.Series, peers: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy DataFrame for charting: rows = companies (target + peers),
    columns = the metrics we care about.

    Returns:
        DataFrame with index = ticker symbols, columns = metric values.
        Always has the target as the first row.
    """
    metric_cols = [c for c in PEER_METRIC_COLUMNS if c in peers.columns or c in target_row.index]
    rows = []

    target_record = {'Symbol': target_row.get('Symbol', 'TARGET')}
    for c in metric_cols:
        target_record[c] = target_row.get(c)
    rows.append(target_record)

    for _, peer in peers.iterrows():
        rec = {'Symbol': peer.get('Symbol', '')}
        for c in metric_cols:
            rec[c] = peer.get(c)
        rows.append(rec)

    df = pd.DataFrame(rows).set_index('Symbol')
    # Coerce to numeric; ignore non-numeric junk
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def peer_search_summary(
    target_symbol: str,
    peers: pd.DataFrame,
    target_industry: Optional[str],
    target_subindustry: Optional[str] = None,
) -> str:
    """One-line description of what was found, for UI display.

    Prefers the most specific classification we actually matched against —
    Subindustry if available, otherwise Industry.
    """
    n = len(peers)
    if n == 0:
        return f"No peers found for {target_symbol} in the screening universe."

    def _valid(s):
        return s is not None and pd.notna(s) and str(s).strip() not in ("", "N/A")

    # Show the most specific classification we actually have
    if _valid(target_subindustry):
        bucket = f"{target_subindustry} sub-industry"
    elif _valid(target_industry):
        bucket = f"{target_industry} industry"
    else:
        bucket = "screening universe"

    return f"{n} peer{'s' if n != 1 else ''} found for {target_symbol} in the {bucket}."
