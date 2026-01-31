"""
Anomaly detector module for identifying financial distortions in company data.

Detects:
1. Earnings manipulation (Beneish M-Score analysis)
2. One-off events affecting fundamentals
3. Quality issues (Z-Score, F-Score, Sloan Ratio)
4. Custom anomaly flags
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd
from data_loader import DataLoader


class Severity(Enum):
    """Anomaly severity levels."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    category: str
    description: str
    severity: Severity
    metric_name: str
    current_value: Any
    threshold: Any = None
    year: str = None
    details: str = None


@dataclass
class AnomalyReport:
    """Complete anomaly report for a company."""
    symbol: str
    company_name: str
    anomalies: List[Anomaly] = field(default_factory=list)
    m_score: float = None
    z_score: float = None
    f_score: float = None
    sloan_ratio: float = None
    risk_level: str = "Unknown"

    def add_anomaly(self, anomaly: Anomaly):
        self.anomalies.append(anomaly)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for a in self.anomalies if a.severity == Severity.HIGH)

    @property
    def total_anomalies(self) -> int:
        return len(self.anomalies)


class AnomalyDetector:
    """
    Detects financial anomalies and distortions in company data.

    Analysis includes:
    - Beneish M-Score components
    - Quality metrics (Z-Score, F-Score, Sloan Ratio)
    - Year-over-year changes for one-off detection
    - Cash flow vs earnings consistency
    """

    # Thresholds for anomaly detection
    M_SCORE_THRESHOLD = -1.78  # Above this suggests manipulation
    Z_SCORE_DISTRESS = 1.8     # Below this = distress zone
    F_SCORE_WEAK = 3           # Below this = weak financials
    SLOAN_HIGH_ACCRUALS = 10   # Above this % = earnings quality issues

    # YoY change thresholds for one-off detection
    REVENUE_SPIKE_THRESHOLD = 50   # % change
    MARGIN_VARIANCE_THRESHOLD = 10  # Percentage points
    EPS_NRI_VARIANCE_THRESHOLD = 50  # % difference between EPS and EPS w/o NRI

    def __init__(self, data_loader: DataLoader = None):
        """
        Initialize the anomaly detector.

        Args:
            data_loader: DataLoader instance for loading financial data.
        """
        self.data_loader = data_loader or DataLoader()

    def analyze(self, symbol: str) -> AnomalyReport:
        """
        Perform full anomaly analysis on a company.

        Args:
            symbol: Stock ticker symbol (e.g., "DDI", "TEX", "CHCI")

        Returns:
            AnomalyReport with all detected anomalies
        """
        # Load financial data
        data = self.data_loader.load_anomaly_data(symbol)

        if symbol.upper() not in data:
            raise ValueError(f"No data found for symbol '{symbol}'")

        fin_data = data[symbol.upper()]
        company_name = self._extract_company_name(symbol)

        report = AnomalyReport(
            symbol=symbol.upper(),
            company_name=company_name
        )

        # Run all detection methods
        self._check_quality_metrics(fin_data, report)
        self._check_one_off_events(fin_data, report)
        self._check_earnings_quality(fin_data, report)
        self._check_cash_flow_consistency(fin_data, report)
        self._check_balance_sheet_anomalies(fin_data, report)

        # Determine overall risk level
        report.risk_level = self._calculate_risk_level(report)

        return report

    def _extract_company_name(self, symbol: str) -> str:
        """Extract company name from available data."""
        symbol_names = {
            'DDI': 'DoubleDown Interactive Co Ltd',
            'TEX': 'Terex Corp',
            'CHCI': 'Comstock Holding Companies Inc'
        }
        return symbol_names.get(symbol.upper(), symbol)

    def _check_quality_metrics(self, data: Dict, report: AnomalyReport):
        """Check M-Score, Z-Score, F-Score, and Sloan Ratio."""
        quality = data.get('quality_metrics', {})

        # Get most recent values (last non-null value in series)
        def get_latest(series_key):
            if series_key in quality:
                vals = quality[series_key]
                for v in reversed(vals):
                    if v is not None:
                        return v
            return None

        # M-Score Analysis
        m_score = get_latest('m_score')
        if m_score is not None:
            report.m_score = m_score
            if m_score > self.M_SCORE_THRESHOLD:
                report.add_anomaly(Anomaly(
                    category="Earnings Manipulation",
                    description=f"Beneish M-Score ({m_score:.2f}) exceeds threshold ({self.M_SCORE_THRESHOLD})",
                    severity=Severity.HIGH,
                    metric_name="M-Score",
                    current_value=m_score,
                    threshold=self.M_SCORE_THRESHOLD,
                    details="M-Score > -1.78 suggests higher probability of earnings manipulation"
                ))

        # Z-Score Analysis
        z_score = get_latest('z_score')
        if z_score is not None:
            report.z_score = z_score
            if z_score < self.Z_SCORE_DISTRESS:
                severity = Severity.HIGH if z_score < 1.1 else Severity.MEDIUM
                report.add_anomaly(Anomaly(
                    category="Financial Distress",
                    description=f"Altman Z-Score ({z_score:.2f}) indicates distress zone",
                    severity=severity,
                    metric_name="Z-Score",
                    current_value=z_score,
                    threshold=self.Z_SCORE_DISTRESS,
                    details="Z-Score < 1.8 = distress zone, < 1.1 = high bankruptcy probability"
                ))

        # F-Score Analysis
        f_score = get_latest('f_score')
        if f_score is not None:
            report.f_score = f_score
            if f_score < self.F_SCORE_WEAK:
                report.add_anomaly(Anomaly(
                    category="Weak Financials",
                    description=f"Piotroski F-Score ({f_score:.0f}) indicates weak financial health",
                    severity=Severity.MEDIUM,
                    metric_name="F-Score",
                    current_value=f_score,
                    threshold=self.F_SCORE_WEAK,
                    details="F-Score < 3 suggests weak financial position"
                ))

        # Sloan Ratio Analysis
        sloan = get_latest('sloan_ratio')
        if sloan is not None:
            report.sloan_ratio = sloan
            if abs(sloan) > self.SLOAN_HIGH_ACCRUALS:
                report.add_anomaly(Anomaly(
                    category="Earnings Quality",
                    description=f"High Sloan Ratio ({sloan:.1f}%) suggests accrual-based earnings",
                    severity=Severity.MEDIUM,
                    metric_name="Sloan Ratio",
                    current_value=sloan,
                    threshold=self.SLOAN_HIGH_ACCRUALS,
                    details="High accruals relative to cash flow may indicate earnings quality issues"
                ))

    def _check_one_off_events(self, data: Dict, report: AnomalyReport):
        """Detect unusual year-over-year changes indicating one-off events."""
        periods = data.get('fiscal_periods', [])
        per_share = data.get('per_share_data', {})
        ratios = data.get('ratios', {})

        if len(periods) < 2:
            return

        # Check EPS vs EPS without NRI discrepancy
        eps = per_share.get('eps_diluted', [])
        eps_nri = per_share.get('eps_without_nri', [])

        for i in range(len(periods)):
            if i < len(eps) and i < len(eps_nri):
                e, e_nri = eps[i], eps_nri[i]
                if e is not None and e_nri is not None and e_nri != 0:
                    # Calculate percentage difference
                    pct_diff = abs((e - e_nri) / abs(e_nri)) * 100
                    if pct_diff > self.EPS_NRI_VARIANCE_THRESHOLD:
                        report.add_anomaly(Anomaly(
                            category="Non-Recurring Items",
                            description=f"Large difference between EPS ({e:.2f}) and EPS w/o NRI ({e_nri:.2f})",
                            severity=Severity.MEDIUM,
                            metric_name="EPS vs EPS w/o NRI",
                            current_value=pct_diff,
                            threshold=self.EPS_NRI_VARIANCE_THRESHOLD,
                            year=periods[i] if i < len(periods) else None,
                            details=f"Suggests significant non-recurring items affecting {pct_diff:.0f}% of earnings"
                        ))

        # Check revenue spikes/drops
        revenue = per_share.get('revenue_per_share', [])
        for i in range(1, len(revenue)):
            if revenue[i] is not None and revenue[i-1] is not None and revenue[i-1] != 0:
                pct_change = ((revenue[i] - revenue[i-1]) / abs(revenue[i-1])) * 100
                if abs(pct_change) > self.REVENUE_SPIKE_THRESHOLD:
                    direction = "spike" if pct_change > 0 else "drop"
                    report.add_anomaly(Anomaly(
                        category="Revenue Volatility",
                        description=f"Revenue {direction} of {pct_change:.0f}%",
                        severity=Severity.MEDIUM if abs(pct_change) < 100 else Severity.HIGH,
                        metric_name="YoY Revenue Change",
                        current_value=pct_change,
                        threshold=self.REVENUE_SPIKE_THRESHOLD,
                        year=periods[i] if i < len(periods) else None,
                        details=f"Large {direction} may indicate M&A, divestiture, or one-off events"
                    ))

        # Check margin volatility
        net_margin = ratios.get('net_margin', [])
        for i in range(1, len(net_margin)):
            if net_margin[i] is not None and net_margin[i-1] is not None:
                margin_change = abs(net_margin[i] - net_margin[i-1])
                if margin_change > self.MARGIN_VARIANCE_THRESHOLD:
                    report.add_anomaly(Anomaly(
                        category="Margin Volatility",
                        description=f"Net margin changed by {margin_change:.1f} percentage points",
                        severity=Severity.MEDIUM,
                        metric_name="Net Margin Change",
                        current_value=margin_change,
                        threshold=self.MARGIN_VARIANCE_THRESHOLD,
                        year=periods[i] if i < len(periods) else None,
                        details="Large margin swings may indicate restructuring or one-off charges"
                    ))

    def _check_earnings_quality(self, data: Dict, report: AnomalyReport):
        """Check for discrepancies suggesting earnings quality issues."""
        per_share = data.get('per_share_data', {})

        eps = per_share.get('eps_diluted', [])
        ocf = per_share.get('ocf_per_share', [])

        # Check if net income consistently exceeds operating cash flow
        for i in range(len(eps)):
            if i < len(ocf) and eps[i] is not None and ocf[i] is not None:
                if eps[i] > 0 and ocf[i] < 0:
                    periods = data.get('fiscal_periods', [])
                    report.add_anomaly(Anomaly(
                        category="Cash Flow Mismatch",
                        description=f"Positive EPS ({eps[i]:.2f}) but negative operating cash flow ({ocf[i]:.2f})",
                        severity=Severity.HIGH,
                        metric_name="EPS vs OCF",
                        current_value=f"EPS: {eps[i]:.2f}, OCF: {ocf[i]:.2f}",
                        year=periods[i] if i < len(periods) else None,
                        details="Earnings not backed by cash flow may indicate low quality or manipulation"
                    ))

    def _check_cash_flow_consistency(self, data: Dict, report: AnomalyReport):
        """Analyze cash flow patterns for inconsistencies."""
        cash_flow = data.get('cash_flow', {})
        periods = data.get('fiscal_periods', [])

        ocf = cash_flow.get('operating_cash_flow', [])
        fcf = cash_flow.get('free_cash_flow', [])
        icf = cash_flow.get('investing_cash_flow', [])

        # Count consecutive years of negative FCF
        neg_fcf_years = 0
        for i, f in enumerate(fcf):
            if f is not None and f < 0:
                neg_fcf_years += 1
            else:
                neg_fcf_years = 0

            if neg_fcf_years >= 3:
                report.add_anomaly(Anomaly(
                    category="Cash Flow Pattern",
                    description=f"Negative free cash flow for {neg_fcf_years}+ consecutive years",
                    severity=Severity.MEDIUM,
                    metric_name="Consecutive Negative FCF",
                    current_value=neg_fcf_years,
                    year=periods[i] if i < len(periods) else None,
                    details="Persistent negative FCF may indicate structural business issues"
                ))
                break  # Only report once

    def _check_balance_sheet_anomalies(self, data: Dict, report: AnomalyReport):
        """Check for balance sheet red flags."""
        balance = data.get('balance_sheet', {})
        per_share = data.get('per_share_data', {})
        periods = data.get('fiscal_periods', [])

        receivables = balance.get('receivables', [])
        inventory = balance.get('inventories', [])
        revenue = per_share.get('revenue_per_share', [])

        # Check if receivables growth >> revenue growth
        for i in range(1, min(len(receivables), len(revenue))):
            if (receivables[i] is not None and receivables[i-1] is not None and
                revenue[i] is not None and revenue[i-1] is not None and
                receivables[i-1] != 0 and revenue[i-1] != 0):

                recv_growth = ((receivables[i] - receivables[i-1]) / abs(receivables[i-1])) * 100
                rev_growth = ((revenue[i] - revenue[i-1]) / abs(revenue[i-1])) * 100

                if recv_growth > rev_growth + 20 and recv_growth > 15:
                    report.add_anomaly(Anomaly(
                        category="Receivables Growth",
                        description=f"Receivables growing ({recv_growth:.0f}%) faster than revenue ({rev_growth:.0f}%)",
                        severity=Severity.MEDIUM,
                        metric_name="Receivables vs Revenue Growth",
                        current_value=f"Recv: {recv_growth:.0f}%, Rev: {rev_growth:.0f}%",
                        year=periods[i] if i < len(periods) else None,
                        details="May indicate aggressive revenue recognition or collection issues"
                    ))

        # Check for inventory buildup vs revenue
        for i in range(1, min(len(inventory), len(revenue))):
            if (inventory[i] is not None and inventory[i-1] is not None and
                revenue[i] is not None and revenue[i-1] is not None and
                inventory[i-1] != 0 and revenue[i-1] != 0):

                inv_growth = ((inventory[i] - inventory[i-1]) / abs(inventory[i-1])) * 100
                rev_growth = ((revenue[i] - revenue[i-1]) / abs(revenue[i-1])) * 100

                if inv_growth > rev_growth + 30 and inv_growth > 20:
                    report.add_anomaly(Anomaly(
                        category="Inventory Buildup",
                        description=f"Inventory growing ({inv_growth:.0f}%) faster than revenue ({rev_growth:.0f}%)",
                        severity=Severity.MEDIUM,
                        metric_name="Inventory vs Revenue Growth",
                        current_value=f"Inv: {inv_growth:.0f}%, Rev: {rev_growth:.0f}%",
                        year=periods[i] if i < len(periods) else None,
                        details="May indicate demand slowdown or obsolescence risk"
                    ))

    def _calculate_risk_level(self, report: AnomalyReport) -> str:
        """Calculate overall risk level based on detected anomalies."""
        high_count = report.high_severity_count
        total = report.total_anomalies

        # Consider quality scores
        m_score_risk = report.m_score is not None and report.m_score > self.M_SCORE_THRESHOLD
        z_score_risk = report.z_score is not None and report.z_score < self.Z_SCORE_DISTRESS

        if high_count >= 3 or (high_count >= 2 and m_score_risk):
            return "HIGH RISK"
        elif high_count >= 1 or total >= 5 or m_score_risk or z_score_risk:
            return "ELEVATED RISK"
        elif total >= 2:
            return "MODERATE RISK"
        elif total >= 1:
            return "LOW RISK"
        else:
            return "MINIMAL RISK"

    def format_report(self, report: AnomalyReport) -> str:
        """Format anomaly report as readable text."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"ANOMALY REPORT: {report.symbol} - {report.company_name}")
        lines.append("=" * 70)
        lines.append("")

        # Quality Scores Summary
        lines.append("QUALITY SCORES:")
        lines.append("-" * 40)
        if report.m_score is not None:
            status = "ALERT" if report.m_score > self.M_SCORE_THRESHOLD else "OK"
            lines.append(f"  Beneish M-Score: {report.m_score:>8.2f}  [{status}]")
        if report.z_score is not None:
            status = "ALERT" if report.z_score < self.Z_SCORE_DISTRESS else "OK"
            lines.append(f"  Altman Z-Score:  {report.z_score:>8.2f}  [{status}]")
        if report.f_score is not None:
            status = "ALERT" if report.f_score < self.F_SCORE_WEAK else "OK"
            lines.append(f"  Piotroski F-Score: {report.f_score:>6.0f}  [{status}]")
        if report.sloan_ratio is not None:
            status = "ALERT" if abs(report.sloan_ratio) > self.SLOAN_HIGH_ACCRUALS else "OK"
            lines.append(f"  Sloan Ratio:     {report.sloan_ratio:>8.1f}% [{status}]")
        lines.append("")

        # Risk Level
        lines.append(f"OVERALL RISK LEVEL: {report.risk_level}")
        lines.append(f"Total Anomalies: {report.total_anomalies} (High: {report.high_severity_count})")
        lines.append("")

        # Detailed Anomalies
        if report.anomalies:
            lines.append("DETECTED ANOMALIES:")
            lines.append("-" * 40)

            # Group by severity
            for severity in [Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
                anomalies = [a for a in report.anomalies if a.severity == severity]
                if anomalies:
                    lines.append(f"\n[{severity.value}]")
                    for i, a in enumerate(anomalies, 1):
                        year_info = f" ({a.year})" if a.year else ""
                        lines.append(f"  {i}. [{a.category}]{year_info}")
                        lines.append(f"     {a.description}")
                        if a.details:
                            lines.append(f"     -> {a.details}")
        else:
            lines.append("No significant anomalies detected.")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


if __name__ == "__main__":
    # Test the anomaly detector
    detector = AnomalyDetector()

    print("Available symbols for analysis:", detector.data_loader.get_available_symbols())

    for symbol in ['DDI', 'TEX', 'CHCI']:
        try:
            print(f"\nAnalyzing {symbol}...")
            report = detector.analyze(symbol)
            print(detector.format_report(report))
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
