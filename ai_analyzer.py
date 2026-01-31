"""
AI-powered analysis module using OpenAI.

Provides intelligent analysis, summaries, and recommendations for stocks.
"""
import json
from typing import Dict, List, Optional, Any
import pandas as pd

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIAnalyzer:
    """AI-powered stock analysis using OpenAI."""

    def __init__(self, api_key: str = None):
        """
        Initialize the AI analyzer.

        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # Use GPT-4o for best analysis

    def analyze_screened_stocks(
        self,
        stocks_df: pd.DataFrame,
        criteria: Dict[str, float],
        top_n: int = 10
    ) -> str:
        """
        Provide AI analysis of screened stocks.

        Args:
            stocks_df: DataFrame of screened stocks
            criteria: Screening criteria used
            top_n: Number of top stocks to analyze in detail

        Returns:
            AI-generated analysis and recommendations
        """
        # Prepare stock data for analysis
        if len(stocks_df) == 0:
            return "No stocks matched the criteria. Consider relaxing your filters."

        # Get top stocks for detailed analysis
        top_stocks = stocks_df.head(top_n).to_dict('records')

        # Summary statistics
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional value investing analyst providing data-driven stock analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        return response.choices[0].message.content

    def analyze_anomaly_report(
        self,
        symbol: str,
        company_name: str,
        anomalies: List[Dict],
        quality_scores: Dict[str, float],
        risk_level: str
    ) -> str:
        """
        Provide AI analysis of anomaly detection results.

        Args:
            symbol: Stock ticker
            company_name: Company name
            anomalies: List of detected anomalies
            quality_scores: M-Score, Z-Score, F-Score, etc.
            risk_level: Overall risk assessment

        Returns:
            AI-generated analysis and interpretation
        """
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a forensic accounting expert specializing in detecting financial fraud and anomalies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        return response.choices[0].message.content

    def generate_investment_thesis(
        self,
        stock_data: Dict,
        valuation_status: str,
        epv_mc_ratio: float
    ) -> str:
        """
        Generate an investment thesis for a specific stock.

        Args:
            stock_data: Dictionary with stock fundamentals
            valuation_status: Undervalued/Fair Value/Overvalued
            epv_mc_ratio: EPV to Market Cap ratio

        Returns:
            AI-generated investment thesis
        """
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a value investing analyst creating investment theses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )

        return response.choices[0].message.content

    def compare_stocks(
        self,
        stocks: List[Dict],
        criteria: str = "value"
    ) -> str:
        """
        Compare multiple stocks and rank them.

        Args:
            stocks: List of stock dictionaries with fundamentals
            criteria: Comparison focus (value, quality, growth)

        Returns:
            AI-generated comparison and ranking
        """
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a value investing analyst comparing investment opportunities."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content

    def summarize_for_executive(
        self,
        analysis_text: str,
        context: str = "stock screening"
    ) -> str:
        """
        Create an executive summary of analysis.

        Args:
            analysis_text: Full analysis text to summarize
            context: Type of analysis (screening, anomaly, comparison)

        Returns:
            Brief executive summary (3-5 bullet points)
        """
        prompt = f"""Summarize the following {context} analysis into an executive brief.

ANALYSIS:
{analysis_text}

Create a 3-5 bullet point executive summary that a busy investor can scan in 30 seconds.
Focus on: key findings, top recommendations, and critical action items."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You create concise executive summaries for busy investors."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )

        return response.choices[0].message.content


def check_openai_available() -> bool:
    """Check if OpenAI package is available."""
    return OPENAI_AVAILABLE
