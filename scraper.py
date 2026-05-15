"""
Firecrawl-based scraper for real 10-year financial history.

Used by Step 2 (AI Anomaly Analysis) so the LLM analyzes ACTUAL filings
data instead of hallucinating from training memory.

Sources are configurable from the UI. The default set (per client spec) is:
  Bloomberg, Reuters, Morningstar, Gurufocus.

URL templates accept these placeholders:
  {symbol}        — original case from CSV
  {symbol_upper}  — uppercase ticker
  {symbol_lower}  — lowercase ticker
  {market}        — 'us' or 'sg'

Each call returns either:
  {"ok": True, "markdown": "...", "source_url": "...", "source": "..."}
  {"ok": False, "error": "..."}
"""
import os
from typing import Dict, Optional, List, Tuple

try:
    from firecrawl import Firecrawl
    _FIRECRAWL_AVAILABLE = True
except ImportError:
    _FIRECRAWL_AVAILABLE = False


# Default source set — the client's preferred sites.
# Each entry: (label, homepage_url, ticker_url_template).
#   - homepage_url is what we DISPLAY in the UI (clean bare domain, per client spec).
#   - ticker_url_template is what Firecrawl actually hits (the per-ticker subpath
#     where the real financial data lives — the homepage itself has none).
DEFAULT_SOURCES: List[Tuple[str, str, str]] = [
    # Gurufocus first — most universal across exchanges and rarely paywalled.
    # For SGX tickers the canonical Gurufocus path is SGX:{ticker}.
    ("Gurufocus",
     "https://www.gurufocus.com/",
     "https://www.gurufocus.com/stock/{gurufocus_path}/financials"),
    # Stock Analysis — universal, scraper-friendly. For SGX use ".si"-suffix.
    ("Stock Analysis",
     "https://stockanalysis.com/",
     "https://stockanalysis.com/{stockanalysis_path}/financials/"),
    # Bloomberg — ":US" for US listings, ":SP" for Singapore.
    ("Bloomberg",
     "https://www.bloomberg.com/",
     "https://www.bloomberg.com/quote/{symbol_upper}:{bloomberg_suffix}"),
    # Reuters — bare ticker for US, {ticker}.SI for SGX.
    ("Reuters",
     "https://www.reuters.com/",
     "https://www.reuters.com/markets/companies/{reuters_path}"),
    # Morningstar — use the search URL so we don't have to guess exchange code.
    # For SGX, prepend xses/ to land on the canonical SGX page.
    ("Morningstar",
     "https://www.morningstar.com/",
     "https://www.morningstar.com/stocks/{morningstar_path}"),
]


def render_url(template: str, symbol: str, market: str = "US") -> str:
    """Substitute placeholders in a URL template.

    Market-aware: when market='SG' the URLs are rewritten to use the
    SGX-specific paths (`:SP` Bloomberg suffix, `xses` Morningstar code,
    `.SI` Reuters/Stock Analysis suffix, `SGX:` Gurufocus prefix).
    """
    sym_upper = symbol.upper()
    sym_lower = symbol.lower()
    is_sg = market.upper() == "SG"

    # Per-source path / suffix overrides for SG tickers
    if is_sg:
        gurufocus_path = f"SGX:{sym_upper}"
        stockanalysis_path = f"quote/sgx/{sym_upper}"
        bloomberg_suffix = "SP"
        reuters_path = f"{sym_upper}.SI"
        morningstar_path = f"xses/{sym_lower}"
    else:
        gurufocus_path = sym_upper
        stockanalysis_path = f"stocks/{sym_lower}"
        bloomberg_suffix = "US"
        reuters_path = sym_upper
        morningstar_path = sym_lower

    return (
        template
        .replace("{symbol_upper}", sym_upper)
        .replace("{symbol_lower}", sym_lower)
        .replace("{market}", market.lower())
        .replace("{symbol}", symbol)
        .replace("{gurufocus_path}", gurufocus_path)
        .replace("{stockanalysis_path}", stockanalysis_path)
        .replace("{bloomberg_suffix}", bloomberg_suffix)
        .replace("{reuters_path}", reuters_path)
        .replace("{morningstar_path}", morningstar_path)
    )


def _client() -> Optional["Firecrawl"]:
    if not _FIRECRAWL_AVAILABLE:
        return None
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        return None
    return Firecrawl(api_key=api_key)


def _markdown_from_result(result) -> str:
    """Pull markdown text out of a Firecrawl result regardless of return shape."""
    if result is None:
        return ""
    # v4 returns a Document object with .markdown
    if hasattr(result, "markdown"):
        return (result.markdown or "")
    # Older SDKs may return a dict
    if isinstance(result, dict):
        return result.get("markdown") or result.get("data", {}).get("markdown", "") or ""
    return str(result) if result else ""


def _scrape_url(client, url: str, timeout_ok: int = 60,
                max_retries: int = 1) -> Tuple[Optional[str], str]:
    """Scrape a single URL with retry on transient errors.

    Returns:
        (markdown_or_None, status_string).
        status_string is one of:
            "ok:<N> chars"       — usable markdown returned
            "empty:<N> chars"    — got something <200 chars (likely blocked/paywall/nav)
            "error:<class>:<msg>" — request itself failed
    """
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            result = client.scrape(url, formats=["markdown"])
            md = _markdown_from_result(result)
            n = len(md) if md else 0
            if md and n > 200:
                return md, f"ok:{n} chars"
            # Got something but too thin to be useful — don't retry, it's likely
            # a blocked page / paywall, not a transient failure
            return None, f"empty:{n} chars"
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt < max_retries:
                continue
            return None, f"error:{last_error}"
    return None, "error:unknown"


def scrape_financial_data(
    symbol: str,
    company_name: str = "",
    market: str = "US",
    custom_sources: Optional[List[Tuple[str, str]]] = None,
) -> Dict:
    """
    Fetch real 10-year financial history for a company.

    Args:
        symbol: ticker (e.g. AFYA, D05, 1AZ)
        company_name: optional, used for fallback search query
        market: 'US' or 'SG' — passed to URL templates via {market}
        custom_sources: optional list of (label, url_template) pairs. If None,
            uses DEFAULT_SOURCES (Bloomberg, Reuters, Morningstar, Gurufocus).
            Templates support {symbol}, {symbol_upper}, {symbol_lower}, {market}.

    Returns dict with:
        ok (bool), markdown (str), source_url (str), source (str), error (str)
    """
    client = _client()
    if client is None:
        return {
            "ok": False,
            "error": "FIRECRAWL_API_KEY not configured. Add it to your .env file.",
        }

    # `custom_sources` accepts either:
    #   - (label, template)            — backwards-compatible, template is what we hit
    #   - (label, homepage, template)  — homepage is UI-only, template is what we hit
    if custom_sources:
        sources = custom_sources
    else:
        sources = DEFAULT_SOURCES

    candidates: List[Tuple[str, str]] = []
    for entry in sources:
        if not entry:
            continue
        if len(entry) >= 3:
            label, _homepage, template = entry[0], entry[1], entry[2]
        elif len(entry) == 2:
            label, template = entry[0], entry[1]
        else:
            continue
        if label and template:
            candidates.append((label, render_url(template, symbol, market)))

    combined_md_parts: List[str] = []
    sources_used: List[str] = []
    last_url = ""
    # source_status: per-source outcome so the UI can show exactly what happened.
    # List of dicts: {label, url, status, bytes}
    source_status: List[Dict] = []

    for source_name, url in candidates:
        md, status = _scrape_url(client, url, max_retries=1)
        n = len(md) if md else 0
        source_status.append({
            "label": source_name, "url": url, "status": status, "bytes": n
        })
        if md:
            combined_md_parts.append(f"### Source: {source_name}\n### URL: {url}\n\n{md}")
            sources_used.append(source_name)
            last_url = url

    # If structured pages failed, fall back to Firecrawl search
    search_fallback_status = None
    if not combined_md_parts:
        try:
            query = (f"{company_name or symbol} {symbol} annual revenue "
                     f"net income free cash flow 10 year history")
            search_result = client.search(query=query, limit=3)
            web_results = []
            if hasattr(search_result, "web"):
                web_results = search_result.web or []
            elif isinstance(search_result, dict):
                web_results = search_result.get("web", []) or search_result.get("data", [])

            search_fallback_status = {
                "label": "search-fallback", "url": f"search:{query[:60]}...",
                "status": f"ok:{len(web_results)} results", "bytes": 0
            }
            source_status.append(search_fallback_status)

            for item in web_results[:3]:
                if hasattr(item, "url"):
                    item_url = item.url
                elif isinstance(item, dict):
                    item_url = item.get("url")
                else:
                    continue
                md, status = _scrape_url(client, item_url, max_retries=1)
                n = len(md) if md else 0
                source_status.append({
                    "label": "search-result", "url": item_url,
                    "status": status, "bytes": n
                })
                if md:
                    combined_md_parts.append(
                        f"### Source: search-result\n### URL: {item_url}\n\n{md}"
                    )
                    sources_used.append("search-fallback")
                    last_url = item_url
        except Exception as e:
            source_status.append({
                "label": "search-fallback",
                "url": "(Firecrawl search API)",
                "status": f"error:{type(e).__name__}: {str(e)[:120]}",
                "bytes": 0,
            })

    if not combined_md_parts:
        return {
            "ok": False,
            "error": (f"All configured sources returned empty or errored for {symbol}. "
                      f"See source_status for per-URL outcome."),
            "source_status": source_status,
        }

    combined = "\n\n---\n\n".join(combined_md_parts)
    # Cap at ~30k chars to keep prompts manageable for Claude
    if len(combined) > 30000:
        combined = combined[:30000] + "\n\n[... truncated for prompt size ...]"

    return {
        "ok": True,
        "markdown": combined,
        "source_url": last_url,
        "source": ", ".join(sources_used),
        "source_status": source_status,
    }


def is_firecrawl_available() -> bool:
    return _FIRECRAWL_AVAILABLE and bool(os.getenv("FIRECRAWL_API_KEY", "").strip())


if __name__ == "__main__":
    # Smoke test
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv()

    for sym, name, mkt in [("AFYA", "Afya Limited", "US"), ("INMD", "InMode", "US")]:
        print(f"\n=== {sym} ({name}) — {mkt} ===")
        out = scrape_financial_data(sym, name, mkt)
        if out["ok"]:
            print(f"OK — sources: {out['source']}, {len(out['markdown'])} chars")
        else:
            print(f"FAIL — {out['error']}")
