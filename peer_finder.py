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

    Strategy:
    1. Same Sector AND Industry → top N by Market Cap (preferred)
    2. If fewer than `limit` peers found, widen to same Sector only
    3. Always exclude the target itself

    Args:
        target_symbol: ticker of the target company
        universe: DataFrame of all screened/loaded stocks
        limit: max number of peers to return
        same_industry_first: if True, narrow Sector+Industry first

    Returns:
        DataFrame of peer rows, sorted by Market Cap descending. May be empty
        if no peers exist.
    """
    if universe is None or universe.empty:
        return pd.DataFrame()

    target_rows = universe[universe['Symbol'] == target_symbol]
    if target_rows.empty:
        return pd.DataFrame()

    target = target_rows.iloc[0]
    target_sector = target.get('Sector')
    target_industry = target.get('Industry')

    others = universe[universe['Symbol'] != target_symbol].copy()

    # Pass 1: Sector + Industry match
    candidates = others
    if same_industry_first and target_sector and target_industry:
        narrow = others[
            (others['Sector'] == target_sector) &
            (others['Industry'] == target_industry)
        ]
        if not narrow.empty:
            candidates = narrow

    # Pass 2: widen to Sector only if narrow yielded too few
    if len(candidates) < limit and target_sector:
        widened = others[others['Sector'] == target_sector]
        # merge: prefer narrow matches, fill with widened
        if not widened.empty:
            existing_syms = set(candidates['Symbol']) if not candidates.empty else set()
            extras = widened[~widened['Symbol'].isin(existing_syms)]
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


def peer_search_summary(target_symbol: str, peers: pd.DataFrame, target_industry: Optional[str]) -> str:
    """One-line description of what was found, for UI display."""
    n = len(peers)
    if n == 0:
        return f"No peers found for {target_symbol} in the screening universe."
    industry_str = f" in the {target_industry} industry" if target_industry else ""
    return f"{n} peer{'s' if n != 1 else ''} found for {target_symbol}{industry_str}."
