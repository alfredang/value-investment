"""
AI-powered analysis module using Claude (via claude-agent-sdk).

Provides intelligent analysis, summaries, and recommendations for stocks.
All LLM calls route through llm.claude_complete which uses the local
Claude Code CLI subscription auth.
"""
import json
from typing import Dict, List, Any
import pandas as pd

from llm import claude_complete, DEFAULT_MODEL


class AIAnalyzer:
    """AI-powered stock analysis using Claude via claude-agent-sdk."""

    def __init__(self, api_key: str = None, model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: Ignored. Auth is handled by Claude Code CLI login.
                     Kept for backward-compat with callers that still pass it.
            model: Claude model ID. Defaults to claude-sonnet-4-6.
        """
        self.model = model

    def _complete(self, system: str, user: str) -> str:
        """Single-turn Claude completion."""
        return claude_complete(user=user, system=system, model=self.model)

    def analyze_screened_stocks(
        self,
        stocks_df: pd.DataFrame,
        criteria: Dict[str, float],
        top_n: int = 10
    ) -> str:
        """Provide AI analysis of screened stocks."""
        if len(stocks_df) == 0:
            return "No stocks matched the criteria. Consider relaxing your filters."

        top_stocks = stocks_df.head(top_n).to_dict('records')

        summary = {
            'total_matches': len(stocks_df),
            'sectors': stocks_df['Sector'].value_counts().head(5).to_dict() if 'Sector' in stocks_df.columns else {},
            'valuation_distribution': stocks_df['Valuation'].value_counts().to_dict() if 'Valuation' in stocks_df.columns else {},
            'avg_roe': stocks_df['ROE %'].mean() if 'ROE %' in stocks_df.columns else None,
            'avg_gross_margin': stocks_df['Gross Margin %'].mean() if 'Gross Margin %' in stocks_df.columns else None,
        }

        prompt = f"""You are a value investing expert analyst. Analyze the following stock screening results and provide actionable insights.

SCREENING CRITERIA USED:
{json.dumps(criteria, indent=2)}

SUMMARY STATISTICS:
- Total stocks matching criteria: {summary['total_matches']}
- Top sectors: {summary['sectors']}
- Valuation distribution: {summary['valuation_distribution']}
- Average ROE: {summary['avg_roe']:.1f}% (if available)
- Average Gross Margin: {summary['avg_gross_margin']:.1f}% (if available)

TOP {top_n} STOCKS (sample for detailed analysis):
{json.dumps(top_stocks, indent=2, default=str)}

Please provide:
1. **Executive Summary** (2-3 sentences on overall quality of matches)
2. **Top Picks** (3-5 stocks with brief reasoning based on the data)
3. **Sector Insights** (which sectors are well-represented and why)
4. **Risk Factors** (potential concerns based on the financial metrics)
5. **Recommendations** (what additional due diligence to perform)

Keep the analysis concise but insightful. Focus on value investing principles."""

        return self._complete(
            system="You are a professional value investing analyst providing data-driven stock analysis.",
            user=prompt,
        )

    def analyze_anomaly_report(
        self,
        symbol: str,
        company_name: str,
        anomalies: List[Dict],
        quality_scores: Dict[str, float],
        risk_level: str
    ) -> str:
        """Provide AI analysis of anomaly detection results."""
        prompt = f"""You are a forensic accounting expert analyzing financial anomalies. Review the following anomaly report and provide interpretation.

COMPANY: {symbol} - {company_name}
OVERALL RISK LEVEL: {risk_level}

QUALITY SCORES:
{json.dumps(quality_scores, indent=2)}

Reference thresholds:
- M-Score > -1.78 suggests potential earnings manipulation
- Z-Score < 1.8 indicates financial distress zone
- F-Score < 3 suggests weak financials
- Sloan Ratio > 10% suggests earnings quality issues

DETECTED ANOMALIES ({len(anomalies)} total):
{json.dumps(anomalies[:20], indent=2, default=str)}

Please provide:
1. **Risk Assessment** (interpret what the scores and anomalies mean)
2. **Key Concerns** (most significant red flags to investigate)
3. **Potential Explanations** (legitimate business reasons vs. concerning patterns)
4. **Investigation Priorities** (what to look at in SEC filings, earnings calls)
5. **Investment Implications** (whether to avoid, proceed with caution, or investigate further)

Be balanced - distinguish between normal business volatility and genuinely concerning patterns."""

        return self._complete(
            system="You are a forensic accounting expert specializing in detecting financial fraud and anomalies.",
            user=prompt,
        )

    def generate_investment_thesis(
        self,
        stock_data: Dict,
        valuation_status: str,
        epv_mc_ratio: float
    ) -> str:
        """Generate an investment thesis for a specific stock."""
        prompt = f"""You are a value investing analyst. Generate a brief investment thesis for the following stock.

STOCK DATA:
{json.dumps(stock_data, indent=2, default=str)}

VALUATION:
- Status: {valuation_status}
- EPV/MC Ratio: {epv_mc_ratio:.2f}
- Interpretation: EPV is {(epv_mc_ratio - 1) * 100:.0f}% {"above" if epv_mc_ratio > 1 else "below"} market cap

Generate a concise investment thesis covering:
1. **Bull Case** (2-3 points why this could be a good investment)
2. **Bear Case** (2-3 risks or concerns)
3. **Key Metrics to Monitor** (what would change the thesis)
4. **Verdict** (buy/hold/avoid with brief reasoning)

Keep it practical and actionable for a value investor."""

        return self._complete(
            system="You are a value investing analyst creating investment theses.",
            user=prompt,
        )

    def compare_stocks(
        self,
        stocks: List[Dict],
        criteria: str = "value"
    ) -> str:
        """Compare multiple stocks and rank them."""
        prompt = f"""You are a value investing analyst. Compare the following stocks and provide a ranking.

COMPARISON FOCUS: {criteria}

STOCKS TO COMPARE:
{json.dumps(stocks, indent=2, default=str)}

Provide:
1. **Ranking** (best to worst with scores out of 10)
2. **Comparison Table** (key metrics side by side)
3. **Top Pick** (which stock and why)
4. **Avoid** (which stock to avoid and why)

Focus on value investing principles: margin of safety, earnings power, balance sheet strength."""

        return self._complete(
            system="You are a value investing analyst comparing investment opportunities.",
            user=prompt,
        )

    def summarize_for_executive(
        self,
        analysis_text: str,
        context: str = "stock screening"
    ) -> str:
        """Create an executive summary of analysis."""
        prompt = f"""Summarize the following {context} analysis into an executive brief.

ANALYSIS:
{analysis_text}

Create a 3-5 bullet point executive summary that a busy investor can scan in 30 seconds.
Focus on: key findings, top recommendations, and critical action items."""

        return self._complete(
            system="You create concise executive summaries for busy investors.",
            user=prompt,
        )


def check_anthropic_available() -> bool:
    """Check if Claude Agent SDK is available (kept for backward compat with caller name)."""
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except ImportError:
        return False
