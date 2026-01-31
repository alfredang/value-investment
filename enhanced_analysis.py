"""
Enhanced Analysis Module

Provides advanced analysis features:
- Charts and visualizations
- CEO commitment tracking and validation
- Deep dive anomaly validation with context
- Rigorous company selection criteria
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Chart imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# AI imports for CEO tracking
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CEOCommitment:
    """Represents a CEO commitment and its fulfillment status."""
    year: int
    commitment: str
    category: str  # revenue, margin, product, expansion, etc.
    target_metric: Optional[str] = None
    target_value: Optional[float] = None
    actual_value: Optional[float] = None
    fulfilled: Optional[bool] = None
    evidence: str = ""
    source: str = ""


@dataclass
class AnomalyValidation:
    """Deep dive validation of an anomaly."""
    anomaly_type: str
    original_flag: str
    severity: str

    # Validation results
    is_valid_concern: bool = True
    explanation: str = ""
    mitigating_factors: List[str] = field(default_factory=list)
    aggravating_factors: List[str] = field(default_factory=list)

    # Context
    industry_context: str = ""
    one_time_event: bool = False
    management_explanation: str = ""

    # Final assessment
    adjusted_severity: str = ""  # May be upgraded or downgraded
    recommendation: str = ""


@dataclass
class CompanyScore:
    """Comprehensive scoring for company selection."""
    symbol: str
    company_name: str

    # Individual scores (0-100)
    valuation_score: float = 0
    quality_score: float = 0
    growth_score: float = 0
    safety_score: float = 0
    management_score: float = 0

    # Weighted total
    total_score: float = 0

    # Selection status
    passed_screening: bool = False
    disqualification_reasons: List[str] = field(default_factory=list)

    # Ranking
    rank: int = 0
    percentile: float = 0


# ============================================================================
# Chart Generation
# ============================================================================

class ChartGenerator:
    """Generate interactive charts for investment analysis."""

    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for chart generation")

    def create_valuation_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create valuation comparison chart (EPV vs Market Cap)."""
        # Prepare data
        chart_df = df[['Symbol', 'Company', 'Earnings Power Value (EPV)',
                       'Market Cap ($M)', 'Valuation']].copy()
        chart_df = chart_df.dropna()

        # Color mapping
        color_map = {
            'Undervalued': '#2ecc71',
            'Fair Value': '#f39c12',
            'Overvalued': '#e74c3c',
            'N/A': '#95a5a6'
        }

        fig = go.Figure()

        for valuation, color in color_map.items():
            mask = chart_df['Valuation'] == valuation
            subset = chart_df[mask]
            if len(subset) > 0:
                fig.add_trace(go.Scatter(
                    x=subset['Market Cap ($M)'],
                    y=subset['Earnings Power Value (EPV)'],
                    mode='markers+text',
                    name=valuation,
                    text=subset['Symbol'],
                    textposition='top center',
                    marker=dict(size=12, color=color, opacity=0.7),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Market Cap: $%{x:.1f}M<br>' +
                                  'EPV: $%{y:.1f}M<br>' +
                                  '<extra></extra>'
                ))

        # Add fair value line (EPV = MC)
        max_val = max(chart_df['Market Cap ($M)'].max(),
                      chart_df['Earnings Power Value (EPV)'].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Fair Value Line',
            line=dict(dash='dash', color='gray', width=1)
        ))

        fig.update_layout(
            title='Valuation: EPV vs Market Cap',
            xaxis_title='Market Cap ($M)',
            yaxis_title='Earnings Power Value ($M)',
            showlegend=True,
            height=500
        )

        return fig

    def create_fundamentals_radar(self, stock_data: Dict) -> go.Figure:
        """Create radar chart for fundamental metrics."""
        categories = ['Gross Margin', 'Net Margin', 'ROE', 'ROA',
                      'FCF Margin', 'ROIC-WACC']

        # Normalize values to 0-100 scale
        values = []
        for cat in categories:
            col_map = {
                'Gross Margin': 'Gross Margin %',
                'Net Margin': 'Net Margin %',
                'ROE': 'ROE %',
                'ROA': 'ROA %',
                'FCF Margin': 'FCF Margin %',
                'ROIC-WACC': 'ROIC-WACC'
            }
            val = stock_data.get(col_map[cat], 0)
            if pd.isna(val):
                val = 0
            # Normalize (assuming reasonable ranges)
            normalized = min(max(float(val), 0), 100)
            values.append(normalized)

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=stock_data.get('Symbol', 'Stock'),
            line_color='#3498db'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title=f"Fundamental Profile: {stock_data.get('Symbol', '')}",
            showlegend=True,
            height=400
        )

        return fig

    def create_risk_heatmap(self, companies: List[Dict]) -> go.Figure:
        """Create risk heatmap for multiple companies."""
        symbols = [c.get('symbol', '') for c in companies]

        risk_metrics = ['M-Score Risk', 'Z-Score Risk', 'F-Score Risk',
                        'Sloan Risk', 'Debt Risk']

        # Build risk matrix
        z_data = []
        for company in companies:
            row = []
            # M-Score (higher is worse)
            m = company.get('m_score')
            row.append(80 if m and m > -1.78 else 20 if m else 50)

            # Z-Score (lower is worse)
            z = company.get('z_score')
            row.append(80 if z and z < 1.8 else 20 if z and z > 3 else 50)

            # F-Score (lower is worse)
            f = company.get('f_score')
            row.append(80 if f and f < 3 else 20 if f and f >= 7 else 50)

            # Sloan (higher absolute is worse)
            s = company.get('sloan_ratio')
            row.append(80 if s and abs(s) > 10 else 20 if s and abs(s) < 5 else 50)

            # Debt (higher is worse)
            d = company.get('debt_to_equity')
            row.append(80 if d and d > 2 else 20 if d and d < 0.5 else 50)

            z_data.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=risk_metrics,
            y=symbols,
            colorscale=[[0, '#2ecc71'], [0.5, '#f1c40f'], [1, '#e74c3c']],
            showscale=True,
            colorbar=dict(title='Risk Level')
        ))

        fig.update_layout(
            title='Risk Heatmap',
            xaxis_title='Risk Metric',
            yaxis_title='Company',
            height=max(300, len(symbols) * 40)
        )

        return fig

    def create_financial_trends(self, historical_data: pd.DataFrame,
                                 symbol: str) -> go.Figure:
        """Create financial trends chart from historical data."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trend', 'Net Income Trend',
                           'Margins', 'ROE & ROA')
        )

        years = historical_data.get('Year', range(len(historical_data)))

        # Revenue
        if 'Revenue' in historical_data.columns:
            fig.add_trace(
                go.Scatter(x=years, y=historical_data['Revenue'],
                          mode='lines+markers', name='Revenue'),
                row=1, col=1
            )

        # Net Income
        if 'Net Income' in historical_data.columns:
            fig.add_trace(
                go.Scatter(x=years, y=historical_data['Net Income'],
                          mode='lines+markers', name='Net Income'),
                row=1, col=2
            )

        # Margins
        if 'Gross Margin' in historical_data.columns:
            fig.add_trace(
                go.Scatter(x=years, y=historical_data['Gross Margin'],
                          mode='lines+markers', name='Gross Margin'),
                row=2, col=1
            )
        if 'Net Margin' in historical_data.columns:
            fig.add_trace(
                go.Scatter(x=years, y=historical_data['Net Margin'],
                          mode='lines+markers', name='Net Margin'),
                row=2, col=1
            )

        # ROE/ROA
        if 'ROE' in historical_data.columns:
            fig.add_trace(
                go.Scatter(x=years, y=historical_data['ROE'],
                          mode='lines+markers', name='ROE'),
                row=2, col=2
            )
        if 'ROA' in historical_data.columns:
            fig.add_trace(
                go.Scatter(x=years, y=historical_data['ROA'],
                          mode='lines+markers', name='ROA'),
                row=2, col=2
            )

        fig.update_layout(
            title=f'Financial Trends: {symbol}',
            height=600,
            showlegend=True
        )

        return fig

    def create_peer_comparison(self, companies_df: pd.DataFrame,
                                metric: str = 'ROE %') -> go.Figure:
        """Create peer comparison bar chart."""
        df = companies_df.sort_values(metric, ascending=False).head(20)

        colors = ['#2ecc71' if v > df[metric].median() else '#e74c3c'
                  for v in df[metric]]

        fig = go.Figure(data=[
            go.Bar(
                x=df['Symbol'],
                y=df[metric],
                marker_color=colors,
                text=df[metric].round(1),
                textposition='outside'
            )
        ])

        fig.add_hline(y=df[metric].median(), line_dash="dash",
                      annotation_text=f"Median: {df[metric].median():.1f}")

        fig.update_layout(
            title=f'Peer Comparison: {metric}',
            xaxis_title='Company',
            yaxis_title=metric,
            height=400
        )

        return fig

    def create_selection_funnel(self, stages: List[Tuple[str, int]]) -> go.Figure:
        """Create funnel chart showing company selection process."""
        stage_names = [s[0] for s in stages]
        stage_values = [s[1] for s in stages]

        fig = go.Figure(go.Funnel(
            y=stage_names,
            x=stage_values,
            textposition="inside",
            textinfo="value+percent initial",
            opacity=0.85,
            marker=dict(
                color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6'][:len(stages)]
            )
        ))

        fig.update_layout(
            title='Company Selection Funnel',
            height=400
        )

        return fig


# ============================================================================
# CEO Commitment Tracker
# ============================================================================

class CEOCommitmentTracker:
    """Track and validate CEO commitments from earnings calls and reports."""

    def __init__(self):
        self.tavily_client = None
        if TAVILY_AVAILABLE and os.getenv('TAVILY_API_KEY'):
            self.tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

    def search_ceo_commitments(self, symbol: str, company_name: str,
                                year: int = None) -> List[CEOCommitment]:
        """Search for CEO commitments from earnings calls and investor presentations."""
        if not self.tavily_client:
            return []

        if year is None:
            year = datetime.now().year - 1

        commitments = []

        # Search for earnings call transcripts
        query = f"{company_name} {symbol} CEO earnings call guidance {year}"

        try:
            results = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5
            )

            # Parse commitments from search results
            for result in results.get('results', []):
                commitment = self._parse_commitment(result, year)
                if commitment:
                    commitments.append(commitment)

        except Exception as e:
            print(f"Error searching CEO commitments: {e}")

        return commitments

    def _parse_commitment(self, search_result: Dict, year: int) -> Optional[CEOCommitment]:
        """Parse a commitment from search result."""
        content = search_result.get('content', '')

        # Look for common commitment patterns
        commitment_keywords = [
            'expect', 'target', 'guidance', 'forecast', 'project',
            'anticipate', 'commit', 'goal', 'aim'
        ]

        # Check if content contains commitment language
        has_commitment = any(kw in content.lower() for kw in commitment_keywords)

        if has_commitment:
            # Determine category
            category = 'general'
            if any(w in content.lower() for w in ['revenue', 'sales', 'growth']):
                category = 'revenue'
            elif any(w in content.lower() for w in ['margin', 'profit', 'earnings']):
                category = 'margin'
            elif any(w in content.lower() for w in ['product', 'launch', 'innovation']):
                category = 'product'
            elif any(w in content.lower() for w in ['expand', 'market', 'international']):
                category = 'expansion'

            return CEOCommitment(
                year=year,
                commitment=content[:500],  # Truncate
                category=category,
                source=search_result.get('url', '')
            )

        return None

    def validate_commitment(self, commitment: CEOCommitment,
                           actual_data: Dict) -> CEOCommitment:
        """Validate if a commitment was fulfilled based on actual data."""
        # Check against actual metrics if available
        if commitment.category == 'revenue':
            actual = actual_data.get('revenue_growth')
            if actual and commitment.target_value:
                commitment.actual_value = actual
                commitment.fulfilled = actual >= commitment.target_value

        elif commitment.category == 'margin':
            actual = actual_data.get('net_margin')
            if actual and commitment.target_value:
                commitment.actual_value = actual
                commitment.fulfilled = actual >= commitment.target_value

        return commitment

    def generate_commitment_report(self, symbol: str, company_name: str,
                                    commitments: List[CEOCommitment]) -> str:
        """Generate a report on CEO commitment fulfillment."""
        if not commitments:
            return f"No CEO commitments found for {symbol} ({company_name})"

        report = [f"## CEO Commitment Analysis: {symbol}"]
        report.append("")

        fulfilled = sum(1 for c in commitments if c.fulfilled is True)
        unfulfilled = sum(1 for c in commitments if c.fulfilled is False)
        unknown = sum(1 for c in commitments if c.fulfilled is None)

        report.append(f"**Summary**: {fulfilled} fulfilled, {unfulfilled} not fulfilled, {unknown} unknown")
        report.append("")

        # Group by category
        categories = {}
        for c in commitments:
            if c.category not in categories:
                categories[c.category] = []
            categories[c.category].append(c)

        for cat, items in categories.items():
            report.append(f"### {cat.title()} Commitments")
            for item in items:
                status = "✅" if item.fulfilled else "❌" if item.fulfilled is False else "❓"
                report.append(f"- {status} ({item.year}): {item.commitment[:200]}...")
                if item.source:
                    report.append(f"  Source: {item.source}")
            report.append("")

        return "\n".join(report)


# ============================================================================
# Anomaly Deep Dive Validator
# ============================================================================

class AnomalyValidator:
    """Deep dive validation of detected anomalies."""

    def __init__(self):
        self.tavily_client = None
        if TAVILY_AVAILABLE and os.getenv('TAVILY_API_KEY'):
            self.tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

    def validate_anomaly(self, symbol: str, company_name: str,
                         anomaly: Dict, industry: str = None) -> AnomalyValidation:
        """Perform deep dive validation on an anomaly."""
        validation = AnomalyValidation(
            anomaly_type=anomaly.get('category', 'Unknown'),
            original_flag=anomaly.get('description', ''),
            severity=anomaly.get('severity', 'MEDIUM')
        )

        # Check for common excusable patterns
        validation = self._check_industry_context(validation, industry)
        validation = self._check_one_time_events(validation, symbol, company_name, anomaly)
        validation = self._search_management_explanation(validation, symbol, company_name, anomaly)

        # Determine final assessment
        validation = self._assess_final_severity(validation)

        return validation

    def _check_industry_context(self, validation: AnomalyValidation,
                                 industry: str) -> AnomalyValidation:
        """Check if anomaly is normal for the industry."""
        industry_norms = {
            'Technology': {
                'high_rnd': True,  # High R&D spending is normal
                'negative_earnings': True,  # Growth companies may have losses
                'high_goodwill': True  # Tech M&A is common
            },
            'Financial Services': {
                'high_leverage': True,  # Banks normally have high leverage
                'low_tangible_assets': True
            },
            'Real Estate': {
                'high_debt': True,  # REITs use leverage
                'low_cash_flow': True  # Depreciation-heavy
            },
            'Biotech': {
                'negative_earnings': True,
                'high_rnd': True,
                'revenue_volatility': True
            },
            'Retail': {
                'low_margins': True,
                'high_inventory': True,
                'seasonal_variance': True
            }
        }

        if industry in industry_norms:
            norms = industry_norms[industry]
            anomaly_type = validation.anomaly_type.lower()

            for norm, is_normal in norms.items():
                if norm in anomaly_type and is_normal:
                    validation.mitigating_factors.append(
                        f"This is typical for the {industry} industry"
                    )
                    validation.industry_context = f"Normal for {industry} sector"

        return validation

    def _check_one_time_events(self, validation: AnomalyValidation,
                                symbol: str, company_name: str,
                                anomaly: Dict) -> AnomalyValidation:
        """Check if anomaly is due to one-time events."""
        one_time_keywords = [
            'restructuring', 'acquisition', 'merger', 'divestiture',
            'impairment', 'write-off', 'legal settlement', 'pandemic',
            'natural disaster', 'strike', 'recall'
        ]

        year = anomaly.get('year')
        description = anomaly.get('description', '').lower()

        # Check description for one-time event indicators
        for keyword in one_time_keywords:
            if keyword in description:
                validation.one_time_event = True
                validation.mitigating_factors.append(
                    f"Likely due to one-time event: {keyword}"
                )
                break

        # Search for news about one-time events if we have Tavily
        if self.tavily_client and year:
            try:
                query = f"{company_name} {symbol} {year} restructuring impairment one-time charge"
                results = self.tavily_client.search(query=query, max_results=3)

                for result in results.get('results', []):
                    content = result.get('content', '').lower()
                    for keyword in one_time_keywords:
                        if keyword in content:
                            validation.one_time_event = True
                            validation.mitigating_factors.append(
                                f"News confirms one-time event: {keyword}"
                            )
                            break
            except Exception:
                pass

        return validation

    def _search_management_explanation(self, validation: AnomalyValidation,
                                        symbol: str, company_name: str,
                                        anomaly: Dict) -> AnomalyValidation:
        """Search for management's explanation of the anomaly."""
        if not self.tavily_client:
            return validation

        year = anomaly.get('year', datetime.now().year)
        category = anomaly.get('category', '')

        try:
            query = f"{company_name} CEO CFO explains {category} {year} earnings call"
            results = self.tavily_client.search(query=query, max_results=2)

            for result in results.get('results', []):
                content = result.get('content', '')
                # Look for explanation patterns
                if any(p in content.lower() for p in ['explained', 'due to', 'because', 'result of']):
                    validation.management_explanation = content[:500]
                    break

        except Exception:
            pass

        return validation

    def _assess_final_severity(self, validation: AnomalyValidation) -> AnomalyValidation:
        """Assess final severity based on all factors."""
        original_severity = validation.severity

        # Count factors
        mitigating = len(validation.mitigating_factors)
        aggravating = len(validation.aggravating_factors)

        # Adjust severity
        if validation.one_time_event or mitigating >= 2:
            # Downgrade severity
            if original_severity == 'HIGH':
                validation.adjusted_severity = 'MEDIUM'
            elif original_severity == 'MEDIUM':
                validation.adjusted_severity = 'LOW'
            else:
                validation.adjusted_severity = 'LOW'
            validation.is_valid_concern = False
            validation.recommendation = "Monitor but likely not a critical concern"
        elif aggravating > mitigating:
            # Upgrade severity
            if original_severity == 'LOW':
                validation.adjusted_severity = 'MEDIUM'
            elif original_severity == 'MEDIUM':
                validation.adjusted_severity = 'HIGH'
            else:
                validation.adjusted_severity = 'HIGH'
            validation.is_valid_concern = True
            validation.recommendation = "Requires deeper investigation"
        else:
            validation.adjusted_severity = original_severity
            validation.is_valid_concern = original_severity in ['HIGH', 'MEDIUM']
            validation.recommendation = "Standard monitoring recommended"

        return validation

    def generate_validation_report(self, symbol: str,
                                    validations: List[AnomalyValidation]) -> str:
        """Generate a comprehensive anomaly validation report."""
        report = [f"## Anomaly Deep Dive: {symbol}"]
        report.append("")

        # Summary
        valid_concerns = sum(1 for v in validations if v.is_valid_concern)
        excusable = len(validations) - valid_concerns

        report.append(f"**Summary**: {valid_concerns} valid concerns, {excusable} excusable anomalies")
        report.append("")

        # Valid concerns first
        report.append("### Valid Concerns (Require Attention)")
        concerns = [v for v in validations if v.is_valid_concern]
        if concerns:
            for v in concerns:
                report.append(f"**{v.anomaly_type}** - Severity: {v.adjusted_severity}")
                report.append(f"- Original: {v.original_flag}")
                if v.aggravating_factors:
                    report.append(f"- Aggravating: {', '.join(v.aggravating_factors)}")
                report.append(f"- Recommendation: {v.recommendation}")
                report.append("")
        else:
            report.append("*No critical concerns requiring immediate attention*")
            report.append("")

        # Excusable anomalies
        report.append("### Excusable Anomalies")
        excusables = [v for v in validations if not v.is_valid_concern]
        if excusables:
            for v in excusables:
                report.append(f"**{v.anomaly_type}** - Downgraded from {v.severity} to {v.adjusted_severity}")
                report.append(f"- Original: {v.original_flag}")
                if v.mitigating_factors:
                    report.append(f"- Mitigating: {', '.join(v.mitigating_factors)}")
                if v.one_time_event:
                    report.append("- Likely one-time event")
                if v.industry_context:
                    report.append(f"- Industry context: {v.industry_context}")
                if v.management_explanation:
                    report.append(f"- Management explanation: {v.management_explanation[:200]}...")
                report.append("")
        else:
            report.append("*All anomalies are valid concerns*")

        return "\n".join(report)


# ============================================================================
# Enhanced Company Selection
# ============================================================================

class CompanySelector:
    """Rigorous company selection with multiple criteria."""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        return {
            # Minimum thresholds
            'min_market_cap': 100,  # $100M
            'min_roe': 10,
            'max_debt_equity': 2.0,
            'min_gross_margin': 20,
            'min_net_margin': 0,  # Must be profitable

            # Scoring weights
            'valuation_weight': 0.25,
            'quality_weight': 0.25,
            'growth_weight': 0.20,
            'safety_weight': 0.20,
            'management_weight': 0.10,

            # Selection criteria
            'min_total_score': 60,
            'max_companies': 10
        }

    def score_company(self, stock_data: Dict,
                      anomaly_report: Dict = None) -> CompanyScore:
        """Score a company on multiple dimensions."""
        symbol = stock_data.get('Symbol', '')
        company_name = stock_data.get('Company', '')

        score = CompanyScore(symbol=symbol, company_name=company_name)

        # Calculate individual scores
        score.valuation_score = self._score_valuation(stock_data)
        score.quality_score = self._score_quality(stock_data)
        score.growth_score = self._score_growth(stock_data)
        score.safety_score = self._score_safety(stock_data, anomaly_report)
        score.management_score = self._score_management(stock_data, anomaly_report)

        # Calculate weighted total
        score.total_score = (
            score.valuation_score * self.config['valuation_weight'] +
            score.quality_score * self.config['quality_weight'] +
            score.growth_score * self.config['growth_weight'] +
            score.safety_score * self.config['safety_weight'] +
            score.management_score * self.config['management_weight']
        )

        # Check if passes screening
        score.passed_screening, score.disqualification_reasons = \
            self._check_disqualifications(stock_data, anomaly_report, score)

        return score

    def _score_valuation(self, data: Dict) -> float:
        """Score valuation (0-100)."""
        epv = self._get_float(data, 'Earnings Power Value (EPV)')
        mc = self._get_float(data, 'Market Cap ($M)')

        if not epv or not mc or mc <= 0:
            return 50  # Neutral

        ratio = epv / mc

        if ratio > 2.0:
            return 100  # Significantly undervalued
        elif ratio > 1.5:
            return 90
        elif ratio > 1.3:
            return 80  # Undervalued
        elif ratio > 1.0:
            return 70
        elif ratio > 0.7:
            return 50  # Fair value
        elif ratio > 0.5:
            return 30
        else:
            return 10  # Significantly overvalued

    def _score_quality(self, data: Dict) -> float:
        """Score quality (0-100)."""
        scores = []

        # Gross margin
        gm = self._get_float(data, 'Gross Margin %')
        if gm is not None:
            scores.append(min(gm / 50 * 100, 100))  # 50% gross margin = 100

        # Net margin
        nm = self._get_float(data, 'Net Margin %')
        if nm is not None:
            scores.append(min(max(nm, 0) / 20 * 100, 100))  # 20% net margin = 100

        # ROE
        roe = self._get_float(data, 'ROE %')
        if roe is not None:
            scores.append(min(max(roe, 0) / 25 * 100, 100))  # 25% ROE = 100

        # ROIC-WACC
        roic_wacc = self._get_float(data, 'ROIC-WACC')
        if roic_wacc is not None:
            scores.append(min(max(roic_wacc + 10, 0) / 20 * 100, 100))  # 10% spread = 100

        return np.mean(scores) if scores else 50

    def _score_growth(self, data: Dict) -> float:
        """Score growth (0-100)."""
        scores = []

        # Revenue growth
        rev_growth = self._get_float(data, '5-Year Revenue Growth Rate (Per Share)')
        if rev_growth is not None:
            scores.append(min(max(rev_growth + 10, 0) / 30 * 100, 100))

        # EPS growth
        eps_growth = self._get_float(data, '5-Year EPS without NRI Growth Rate')
        if eps_growth is not None:
            scores.append(min(max(eps_growth + 10, 0) / 30 * 100, 100))

        return np.mean(scores) if scores else 50

    def _score_safety(self, data: Dict, anomaly_report: Dict = None) -> float:
        """Score safety (0-100)."""
        scores = []

        # Debt/Equity (lower is better)
        de = self._get_float(data, 'Debt-to-Equity')
        if de is not None:
            de_score = max(100 - de * 40, 0)  # 0 D/E = 100, 2.5 D/E = 0
            scores.append(de_score)

        # FCF Margin (positive is better)
        fcf = self._get_float(data, 'FCF Margin %')
        if fcf is not None:
            scores.append(min(max(fcf + 10, 0) / 30 * 100, 100))

        # Check anomaly scores if available
        if anomaly_report:
            # Z-Score
            z = anomaly_report.get('z_score')
            if z is not None:
                if z > 3:
                    scores.append(100)  # Safe
                elif z > 1.8:
                    scores.append(60)  # Gray zone
                else:
                    scores.append(20)  # Distress

            # M-Score
            m = anomaly_report.get('m_score')
            if m is not None:
                if m < -2.22:
                    scores.append(100)  # Very safe
                elif m < -1.78:
                    scores.append(70)  # Safe
                else:
                    scores.append(30)  # Manipulation risk

        return np.mean(scores) if scores else 50

    def _score_management(self, data: Dict, anomaly_report: Dict = None) -> float:
        """Score management quality (0-100)."""
        scores = []

        # F-Score as proxy for management quality
        if anomaly_report:
            f = anomaly_report.get('f_score')
            if f is not None:
                scores.append(f / 9 * 100)  # 9 = perfect score

        # Consistent margins (lower volatility = better management)
        # This would need historical data in practice

        return np.mean(scores) if scores else 50

    def _check_disqualifications(self, data: Dict, anomaly_report: Dict,
                                  score: CompanyScore) -> Tuple[bool, List[str]]:
        """Check for hard disqualification criteria."""
        reasons = []

        # Market cap minimum
        mc = self._get_float(data, 'Market Cap ($M)')
        if mc and mc < self.config['min_market_cap']:
            reasons.append(f"Market cap ${mc:.0f}M < ${self.config['min_market_cap']}M minimum")

        # Profitability
        nm = self._get_float(data, 'Net Margin %')
        if nm is not None and nm < self.config['min_net_margin']:
            reasons.append(f"Net margin {nm:.1f}% below minimum {self.config['min_net_margin']}%")

        # Debt level
        de = self._get_float(data, 'Debt-to-Equity')
        if de and de > self.config['max_debt_equity']:
            reasons.append(f"Debt/Equity {de:.1f} exceeds maximum {self.config['max_debt_equity']}")

        # Anomaly checks
        if anomaly_report:
            z = anomaly_report.get('z_score')
            if z and z < 1.0:
                reasons.append(f"Z-Score {z:.1f} indicates high bankruptcy risk")

            m = anomaly_report.get('m_score')
            if m and m > -1.5:
                reasons.append(f"M-Score {m:.2f} indicates possible earnings manipulation")

        # Score minimum
        if score.total_score < self.config['min_total_score']:
            reasons.append(f"Total score {score.total_score:.0f} below minimum {self.config['min_total_score']}")

        passed = len(reasons) == 0
        return passed, reasons

    def _get_float(self, data: Dict, key: str) -> Optional[float]:
        """Safely get float value from dict."""
        val = data.get(key)
        if val is None or pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def select_companies(self, stocks_df: pd.DataFrame,
                         anomaly_reports: Dict[str, Dict] = None) -> List[CompanyScore]:
        """Select top companies from DataFrame."""
        if anomaly_reports is None:
            anomaly_reports = {}

        scores = []
        for _, row in stocks_df.iterrows():
            stock_data = row.to_dict()
            symbol = stock_data.get('Symbol', '')
            anomaly_report = anomaly_reports.get(symbol)

            score = self.score_company(stock_data, anomaly_report)
            scores.append(score)

        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)

        # Assign ranks
        for i, s in enumerate(scores):
            s.rank = i + 1
            s.percentile = (len(scores) - i) / len(scores) * 100

        # Filter to only passed and top N
        selected = [s for s in scores if s.passed_screening][:self.config['max_companies']]

        return selected

    def generate_selection_report(self, selected: List[CompanyScore],
                                   all_scores: List[CompanyScore]) -> str:
        """Generate a selection report."""
        report = ["## Company Selection Report"]
        report.append("")

        total = len(all_scores)
        passed = sum(1 for s in all_scores if s.passed_screening)
        selected_count = len(selected)

        report.append(f"**Selection Funnel**:")
        report.append(f"- Total companies analyzed: {total}")
        report.append(f"- Passed all criteria: {passed}")
        report.append(f"- Final selection (top {self.config['max_companies']}): {selected_count}")
        report.append("")

        # Selected companies table
        report.append("### Selected Companies")
        report.append("")
        report.append("| Rank | Symbol | Company | Score | Valuation | Quality | Growth | Safety |")
        report.append("|------|--------|---------|-------|-----------|---------|--------|--------|")

        for s in selected:
            report.append(
                f"| {s.rank} | {s.symbol} | {s.company_name[:20]} | "
                f"{s.total_score:.0f} | {s.valuation_score:.0f} | "
                f"{s.quality_score:.0f} | {s.growth_score:.0f} | {s.safety_score:.0f} |"
            )

        report.append("")

        # Disqualified companies summary
        disqualified = [s for s in all_scores if not s.passed_screening]
        if disqualified:
            report.append("### Disqualified Companies (Top 10)")
            report.append("")
            for s in disqualified[:10]:
                report.append(f"**{s.symbol}** ({s.company_name}):")
                for reason in s.disqualification_reasons:
                    report.append(f"  - {reason}")
            report.append("")

        return "\n".join(report)
