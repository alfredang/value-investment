"""
Firecrawl-based scraper for real 10-year financial history.

Used by Step 2 (AI Anomaly Analysis) so the LLM analyzes ACTUAL filings
data instead of hallucinating from training memory.

Sources tried in order:
  1. stockanalysis.com — Income statement page (works for most US tickers)
  2. stockanalysis.com — SGX quote page (works for many SG tickers)
  3. Firecrawl search — fallback for obscure tickers

Each call returns either:
  {"ok": True, "markdown": "...", "source_url": "...", "source": "..."}
  {"ok": False, "error": "..."}
"""
import os
from typing import Dict, Optional, List

try:
    from firecrawl import Firecrawl
    _FIRECRAWL_AVAILABLE = True
except ImportError:
    _FIRECRAWL_AVAILABLE = False


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


def _scrape_url(client, url: str, timeout_ok: int = 60) -> Optional[str]:
    """Scrape a single URL, return markdown or None on failure."""
    try:
        result = client.scrape(url, formats=["markdown"])
        md = _markdown_from_result(result)
        if md and len(md) > 200:  # filter out empty / blocked pages
            return md
    except Exception:
        return None
    return None


def scrape_financial_data(
    symbol: str,
    company_name: str = "",
    market: str = "US",
) -> Dict:
    """
    Fetch real 10-year financial history for a company.

    Args:
        symbol: ticker (e.g. AFYA, D05, 1AZ)
        company_name: optional, used for fallback search query
        market: 'US' or 'SG' — determines which URL patterns to try first

    Returns dict with:
        ok (bool), markdown (str), source_url (str), source (str), error (str)
    """
    client = _client()
    if client is None:
        return {
            "ok": False,
            "error": "FIRECRAWL_API_KEY not configured. Add it to your .env file.",
        }

    sym = symbol.strip().lower()
    candidates: List[tuple] = []

    if market.upper() == "US":
        candidates.extend([
            ("stockanalysis-us-income", f"https://stockanalysis.com/stocks/{sym}/financials/"),
            ("stockanalysis-us-cashflow", f"https://stockanalysis.com/stocks/{sym}/financials/cash-flow-statement/"),
        ])
    else:
        candidates.extend([
            ("stockanalysis-sgx-income", f"https://stockanalysis.com/quote/sgx/{symbol.upper()}/financials/"),
        ])

    combined_md_parts: List[str] = []
    sources_used: List[str] = []
    last_url = ""

    for source_name, url in candidates:
        md = _scrape_url(client, url)
        if md:
            combined_md_parts.append(f"### Source: {source_name}\n### URL: {url}\n\n{md}")
            sources_used.append(source_name)
            last_url = url

    # If structured pages failed, fall back to Firecrawl search
    if not combined_md_parts:
        try:
            query = f"{company_name or symbol} {symbol} annual revenue net income free cash flow 10 year history"
            search_result = client.search(query=query, limit=3)
            # search returns a SearchResult object with .web (list of results) in v4
            web_results = []
            if hasattr(search_result, "web"):
                web_results = search_result.web or []
            elif isinstance(search_result, dict):
                web_results = search_result.get("web", []) or search_result.get("data", [])

            for item in web_results[:3]:
                # item could be a dict or a Document-like object
                if hasattr(item, "url"):
                    item_url = item.url
                elif isinstance(item, dict):
                    item_url = item.get("url")
                else:
                    continue
                md = _scrape_url(client, item_url)
                if md:
                    combined_md_parts.append(f"### Source: search-result\n### URL: {item_url}\n\n{md}")
                    sources_used.append("search-fallback")
                    last_url = item_url
        except Exception as e:
            return {
                "ok": False,
                "error": f"Direct URLs failed and search fallback errored: {e}",
            }

    if not combined_md_parts:
        return {
            "ok": False,
            "error": f"No usable pages scraped for {symbol}. Try a more well-known ticker.",
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
