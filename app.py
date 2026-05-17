"""
Value Investment Tool - Streamlit Web App

A stock screening and analysis tool for value investors with AI-powered agents.
"""
import os
import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from screener import StockScreener
from valuation import Valuator
from anomaly_detector import AnomalyDetector, Severity
from data_loader import DataLoader

# Try to import AI agents
AI_IMPORT_ERROR = None
try:
    from ai_agents import ValueInvestmentAgent, ScreeningAgent, AnomalyAgent, ResearchAgent
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = str(e)
except Exception as e:
    AI_AVAILABLE = False
    AI_IMPORT_ERROR = str(e)

# Try to import report generator
try:
    from report_generator import ReportGenerator, CompanyAnalysis, create_company_analysis_from_data
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

# Import admin module
try:
    from admin import (
        show_admin_login, show_admin_panel, get_api_keys, apply_api_keys_to_env,
        load_config, APIKeys, LLMConfig, AnalysisConfig
    )
    ADMIN_AVAILABLE = True
except ImportError:
    ADMIN_AVAILABLE = False

# Import enhanced analysis module
try:
    from enhanced_analysis import (
        ChartGenerator, CEOCommitmentTracker, AnomalyValidator,
        CompanySelector, CompanyScore
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Import FMP API client
try:
    from fmp_api import FMPClient
    FMP_AVAILABLE = True
except ImportError:
    FMP_AVAILABLE = False


def get_fmp_client():
    """Initialize FMP client if API key is configured."""
    fmp_key = os.getenv('FMP_API_KEY', '')
    if fmp_key and FMP_AVAILABLE:
        if 'fmp_client' not in st.session_state:
            st.session_state.fmp_client = FMPClient(fmp_key)
        return st.session_state.fmp_client
    return None

# Page configuration
st.set_page_config(
    page_title="VIA Financial Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Dark Theme
st.markdown("""
<style>
    /* Reduce Streamlit's default top padding so the title sits closer to the header */
    .block-container { padding-top: 1.5rem !important; }
    header[data-testid="stHeader"] { height: 2rem; }

    :root {
        --primary: #1a1a2e;
        --secondary: #16213e;
        --accent: #0f3460;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --border: #334155;
    }
    
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Metrics cards */
    .stMetric {
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8)) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stMetric label {
        color: #cbd5e1 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    /* Stock cards */
    .stock-card {
        padding: 20px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9)) !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 12px !important;
    }
    
    /* Stock symbol header */
    .stock-symbol {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    
    .stock-company {
        font-size: 12px !important;
        color: #cbd5e1 !important;
        margin-top: 4px !important;
    }
    
    /* Positive/Negative indicators */
    .metric-positive {
        color: #10b981 !important;
    }
    
    .metric-negative {
        color: #ef4444 !important;
    }
    
    .metric-neutral {
        color: #f59e0b !important;
    }
    
    /* AI Analysis box */
    .ai-analysis {
        padding: 20px !important;
        border-radius: 12px !important;
        border-left: 4px solid #0f3460 !important;
        background: linear-gradient(135deg, rgba(15, 52, 96, 0.1), rgba(15, 23, 42, 0.8)) !important;
        background-color: rgba(0, 102, 204, 0.08) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 12px 12px 0 0 !important;
        padding: 16px 24px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(51, 65, 85, 0.3) !important;
        color: #cbd5e1 !important;
        background: rgba(30, 41, 59, 0.5) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f8fafc !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(15, 23, 42, 0.9)) !important;
        border-color: rgba(16, 185, 129, 0.3) !important;
        color: #10b981 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8)) !important;
        border: 1px solid rgba(51, 65, 85, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9)) !important;
    }
    
    /* Buttons */
    .stButton > button {
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        background: linear-gradient(135deg, #0f3460, #0a1e35) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e5a96, #0f3460) !important;
        box-shadow: 0 8px 16px rgba(15, 212, 243, 0.2) !important;
    }
    
    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #10b981, #059669) !important;
    }
    
    [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #34d399, #10b981) !important;
        box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Section headings */
    h1, h2, h3 {
        color: #f8fafc !important;
    }
    
    h2 {
        border-bottom: 2px solid rgba(16, 185, 129, 0.3) !important;
        padding-bottom: 12px !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(51, 65, 85, 0.3) !important;
    }
    
    /* Data table */
    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.8) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


def _fmt(val, suffix='%', decimals=1):
    """Format a numeric value for display."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return 'N/A'


def _metric_color(val, good_thresh, ok_thresh=None, lower_better=False):
    """Return hex color based on value vs thresholds."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '#cbd5e1'
    try:
        v = float(val)
    except (ValueError, TypeError):
        return '#cbd5e1'
    if lower_better:
        if v < good_thresh:
            return '#10b981'
        if ok_thresh is not None and v < ok_thresh:
            return '#f59e0b'
        return '#ef4444'
    else:
        if v > good_thresh:
            return '#10b981'
        if ok_thresh is not None and v > ok_thresh:
            return '#f59e0b'
        return '#ef4444'


def _progress_bar_html(value, label, min_val=-20, max_val=50):
    """Render an HTML progress bar for ROIC-WACC / ROTE-WACC."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"<div style='font-size:11px;color:#cbd5e1;margin-bottom:4px;'>{label}: N/A</div>"
    try:
        v = float(value)
    except (ValueError, TypeError):
        return f"<div style='font-size:11px;color:#cbd5e1;margin-bottom:4px;'>{label}: N/A</div>"
    pct = max(0, min(100, (v - min_val) / (max_val - min_val) * 100))
    bar_color = '#10b981' if v > 0 else '#ef4444'
    val_color = '#10b981' if v > 0 else '#ef4444'
    sign = '+' if v > 0 else ''
    return f"""<div style='margin-bottom:6px;'>
        <div style='font-size:11px;color:#94a3b8;margin-bottom:2px;'>{label}</div>
        <div style='display:flex;align-items:center;gap:8px;'>
            <div style='flex:1;height:8px;background:rgba(51,65,85,0.5);border-radius:4px;overflow:hidden;'>
                <div style='width:{pct:.0f}%;height:100%;background:{bar_color};border-radius:4px;'></div>
            </div>
            <span style='color:{val_color};font-weight:600;font-size:12px;min-width:55px;text-align:right;'>{sign}{v:.1f}%</span>
        </div>
    </div>"""


def _metric_cell(label, value, suffix='%', good=None, ok=None, lower_better=False):
    """Build HTML for a single metric cell in a card grid."""
    color = _metric_color(value, good, ok, lower_better) if good is not None else '#cbd5e1'
    display = _fmt(value, suffix)
    return f"""<div>
        <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:0.3px;'>{label}</div>
        <div style='font-size:13px;font-weight:600;color:{color};'>{display}</div>
    </div>"""


def render_stock_card(symbol: str, company: str, data: dict, card_index: int = 0):
    """Render an enhanced stock card showing all 10 screening metrics."""
    valuation = data.get('Valuation', 'N/A')
    current_price = data.get('Current Price')
    epv = data.get('Earnings Power Value (EPV)')

    # All 10 metrics
    gross_margin = data.get('Gross Margin %')
    net_margin = data.get('Net Margin %')
    roa = data.get('ROA %')
    roe = data.get('ROE %')
    rev_growth = data.get('5-Year Revenue Growth Rate (Per Share)')
    eps_growth = data.get('5-Year EPS without NRI Growth Rate')
    debt_equity = data.get('Debt-to-Equity')
    fcf_margin = data.get('FCF Margin %')
    roic_wacc = data.get('ROIC-WACC')
    rote_wacc = data.get('ROTE-WACC')

    # Valuation badge
    val_map = {
        'Undervalued': ('#10b981', '#052e16'),
        'Fair': ('#f59e0b', '#451a03'),
        'Fair Value': ('#f59e0b', '#451a03'),
        'Overvalued': ('#ef4444', '#450a0a'),
    }
    val_fg, val_bg = val_map.get(valuation, ('#cbd5e1', '#1e293b'))

    # Buy-price RANGE from in-house two-stage DCF (per client's Excel formula).
    # Low IV / High IV are per-share — same units as Current Price — so the
    # comparison "is the price inside the buy range?" is meaningful.
    low_iv = data.get('Low IV')
    high_iv = data.get('High IV')

    def _fnum(v):
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    low_iv_n = _fnum(low_iv)
    high_iv_n = _fnum(high_iv)

    price_str = f"${float(current_price):.2f}" if current_price else "N/A"
    if low_iv_n is not None and high_iv_n is not None:
        if low_iv_n == 0.0 and high_iv_n == 0.0:
            # Screener's EPV is non-positive — no positive intrinsic value
            # could be derived. NOT a statement about current profitability.
            buy_str = "EPV ≤ $0"
        elif abs(high_iv_n - low_iv_n) < 0.005:
            buy_str = f"${low_iv_n:.2f}"
        else:
            buy_str = f"${low_iv_n:.2f} – ${high_iv_n:.2f}"
    else:
        buy_str = "N/A"

    company_display = (company[:40] + '...') if company and len(company) > 40 else (company or 'N/A')

    # Build metric cells
    m1 = _metric_cell('Gross Margin', gross_margin, good=30, ok=15)
    m2 = _metric_cell('Net Margin', net_margin, good=10, ok=5)
    m3 = _metric_cell('ROA', roa, good=10, ok=5)
    m4 = _metric_cell('ROE', roe, good=15, ok=10)
    m5 = _metric_cell('5Y Rev Growth', rev_growth, good=10, ok=0)
    m6 = _metric_cell('5Y EPS Growth', eps_growth, good=10, ok=0)
    m7 = _metric_cell('Debt/Equity', debt_equity, suffix='x', good=1.0, ok=1.5, lower_better=True)
    m8 = _metric_cell('FCF Margin', fcf_margin, good=10, ok=5)

    roic_bar = _progress_bar_html(roic_wacc, 'ROIC-WACC')
    rote_bar = _progress_bar_html(rote_wacc, 'ROTE-WACC', min_val=-50, max_val=100)

    card_html = f"""
    <div style='padding:16px 20px;border-radius:10px;border:1px solid rgba(51,65,85,0.4);
                background:linear-gradient(135deg,rgba(30,41,59,0.95),rgba(15,23,42,0.95));
                margin-bottom:10px;box-shadow:0 4px 12px rgba(0,0,0,0.3);'>
        <!-- Row 1: Ticker + Company + Valuation badge -->
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;'>
            <div style='display:flex;align-items:baseline;gap:12px;'>
                <span style='font-size:20px;font-weight:800;color:#f8fafc;letter-spacing:0.5px;'>{symbol}</span>
                <span style='font-size:12px;color:#94a3b8;'>{company_display}</span>
                <span style='font-size:16px;font-weight:700;color:#f8fafc;'>{price_str}</span>
            </div>
            <div style='display:flex;align-items:center;gap:12px;'>
                <span style='font-size:11px;color:#94a3b8;' title='Intrinsic value range (Low IV – High IV) from the in-house two-stage DCF formula. Undervalued when price < Low IV, Overvalued when price > High IV.'>IV Range: {buy_str}</span>
                <span style='background:{val_bg};color:{val_fg};padding:4px 12px;border-radius:4px;
                             font-size:11px;font-weight:700;text-transform:uppercase;border:1px solid {val_fg};'>
                    {valuation}
                </span>
            </div>
        </div>
        <!-- Row 2: All 10 metrics in a grid -->
        <div style='display:grid;grid-template-columns:repeat(5, 1fr);gap:8px 12px;'>
            {m1}{m2}{m3}{m4}{m5}
            {m6}{m7}{m8}
            <div>{roic_bar}</div>
            <div>{rote_bar}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # View Trends button
    if st.button("View Trends", key=f"view_{symbol}_{card_index}", help=f"View detailed analysis for {symbol}"):
        st.session_state[f'show_detail_{symbol}'] = True
        st.rerun()


def show_company_detail_popup(symbol: str, data: dict):
    """Display detailed company information in an expander. Uses FMP API for enrichment."""
    with st.expander(f"📊 {symbol} - {data.get('Company', 'N/A')} | Detailed Analysis", expanded=True):
        # Section 1: Core metrics from CSV
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sector", data.get('Sector', 'N/A'))
        with col2:
            st.metric("Industry", data.get('Industry', 'N/A'))
        with col3:
            mc = data.get('Market Cap ($M)')
            st.metric("Market Cap", f"${mc:,.0f}M" if mc else "N/A")
        with col4:
            epv = data.get('Earnings Power Value (EPV)')
            st.metric("EPV", f"${float(epv):,.0f}M" if epv else "N/A")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cp = data.get('Current Price')
            st.metric("Price", f"${float(cp):.2f}" if cp else "N/A")
        with col2:
            val = data.get('Valuation', 'N/A')
            emoji = {'Undervalued': '🟢', 'Fair': '🟡', 'Fair Value': '🟡', 'Overvalued': '🔴'}.get(val, '⚪')
            st.metric("Valuation", f"{emoji} {val}")
        with col3:
            hi = data.get('High IV')
            st.metric("High IV (per share)", f"${float(hi):.2f}" if hi is not None and pd.notna(hi) else "N/A",
                      help="In-house two-stage DCF — aggressive scenario")
        with col4:
            lo = data.get('Low IV')
            st.metric("Low IV (per share)", f"${float(lo):.2f}" if lo is not None and pd.notna(lo) else "N/A",
                      help="In-house two-stage DCF — conservative scenario (with 30% margin of safety)")

        st.markdown("---")

        # Section 2: FMP API enrichment
        fmp = get_fmp_client()
        if fmp:
            # Cache to avoid repeat API calls
            if 'fmp_cache' not in st.session_state:
                st.session_state.fmp_cache = {}

            cache_key = f"detail_{symbol}"
            if cache_key not in st.session_state.fmp_cache:
                with st.spinner(f"Fetching live data for {symbol}..."):
                    try:
                        profile = fmp.get_profile(symbol)
                        income_stmt = fmp.get_income_statement(symbol, limit=10)
                        cash_flow_data = fmp.get_cash_flow(symbol, limit=10)
                        st.session_state.fmp_cache[cache_key] = {
                            'profile': profile,
                            'income_stmt': income_stmt,
                            'cash_flow': cash_flow_data,
                            'error': None
                        }
                    except Exception as e:
                        st.session_state.fmp_cache[cache_key] = {'error': str(e)}

            cached = st.session_state.fmp_cache[cache_key]

            if cached.get('error'):
                st.warning(f"Could not fetch FMP data: {cached['error']}")
            else:
                profile = cached.get('profile', {})
                income_stmt = cached.get('income_stmt', pd.DataFrame())
                cash_flow_data = cached.get('cash_flow', pd.DataFrame())

                # Business description
                if profile.get('description'):
                    st.markdown("#### Business Intelligence")
                    desc = profile['description']
                    st.write(desc[:600] + '...' if len(desc) > 600 else desc)

                # CEO + Location
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**CEO:** {profile.get('ceo', 'N/A')}")
                with col2:
                    city = profile.get('city', '')
                    state = profile.get('state', '')
                    loc = f"{city}, {state}" if city else "N/A"
                    st.markdown(f"**Location:** {loc}")
                with col3:
                    st.markdown(f"**Country:** {profile.get('country', 'N/A')}")

                st.markdown("---")

                # Charts
                try:
                    import plotly.graph_objects as go

                    chart_col1, chart_col2 = st.columns(2)

                    # Revenue vs Net Income (10Y)
                    with chart_col1:
                        if not income_stmt.empty and 'revenue' in income_stmt.columns:
                            stmt_sorted = income_stmt.sort_values('date') if 'date' in income_stmt.columns else income_stmt
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=stmt_sorted['date'], y=stmt_sorted['revenue'] / 1e6,
                                name='Revenue ($M)', marker_color='#3b82f6', opacity=0.7
                            ))
                            fig.add_trace(go.Scatter(
                                x=stmt_sorted['date'], y=stmt_sorted['netIncome'] / 1e6,
                                name='Net Income ($M)', mode='lines+markers',
                                marker_color='#10b981', line=dict(width=3)
                            ))
                            fig.update_layout(
                                title=f"Revenue vs Net Income (10Y)",
                                height=350, margin=dict(t=40, b=30, l=40, r=20),
                                legend=dict(orientation="h", y=1.12),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#cbd5e1'),
                                xaxis=dict(gridcolor='rgba(51,65,85,0.3)'),
                                yaxis=dict(gridcolor='rgba(51,65,85,0.3)', title='$M'),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Free Cash Flow Trend (10Y)
                    with chart_col2:
                        if not cash_flow_data.empty and 'freeCashFlow' in cash_flow_data.columns:
                            cf_sorted = cash_flow_data.sort_values('date') if 'date' in cash_flow_data.columns else cash_flow_data
                            fcf_vals = cf_sorted['freeCashFlow'] / 1e6
                            colors = ['#10b981' if v >= 0 else '#ef4444' for v in fcf_vals]
                            fig2 = go.Figure()
                            fig2.add_trace(go.Bar(
                                x=cf_sorted['date'], y=fcf_vals,
                                name='FCF ($M)', marker_color=colors
                            ))
                            fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                            fig2.update_layout(
                                title=f"Free Cash Flow Trend (10Y)",
                                height=350, margin=dict(t=40, b=30, l=40, r=20),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#cbd5e1'),
                                xaxis=dict(gridcolor='rgba(51,65,85,0.3)'),
                                yaxis=dict(gridcolor='rgba(51,65,85,0.3)', title='$M'),
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                except ImportError:
                    st.info("Install plotly for interactive charts: `pip install plotly`")

        else:
            st.info("💡 Configure `FMP_API_KEY` in .env to unlock: business description, CEO, location, and 10-year financial charts.")

        if st.button("Close", key=f"close_{symbol}"):
            st.session_state[f'show_detail_{symbol}'] = False
            st.rerun()


def _bridge_streamlit_secrets_to_env():
    """
    Copy Streamlit Cloud secrets into os.environ so the rest of the app
    (which uses os.getenv) finds them. Locally this is a no-op because
    .env loads into env vars directly via python-dotenv.

    Streamlit Cloud users add their keys in the app's Secrets dashboard
    (Settings → Secrets) in TOML format, e.g.:

        FIRECRAWL_API_KEY = "fc-..."
        CLAUDE_CODE_OAUTH_TOKEN = "sk-ant-oat01-..."
        FMP_API_KEY = "..."

    Without this bridge, st.secrets is the only way to read them — but
    helper modules (scraper.py, llm.py) all use os.getenv.
    """
    keys_to_bridge = [
        "FIRECRAWL_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "FMP_API_KEY",
        "TAVILY_API_KEY",
        "NEWS_API_KEY",
        "TWELVE_DATA_API_KEY",
    ]
    try:
        secrets = getattr(st, "secrets", None)
        if not secrets:
            return
        for key in keys_to_bridge:
            try:
                if key in secrets and not os.environ.get(key):
                    val = secrets[key]
                    if val:
                        os.environ[key] = str(val)
            except Exception:
                continue
    except Exception:
        pass


def init_config():
    """Initialize configuration from admin settings.

    Auth: uses Claude Code CLI subscription via claude-agent-sdk. A user-supplied
    subscription token in session_state['anthropic_credential'] is pushed to
    CLAUDE_CODE_OAUTH_TOKEN env var on each rerun. Run `claude setup-token` in
    a terminal to generate a token.
    """
    # Bridge Streamlit Cloud secrets → env vars (no-op locally)
    _bridge_streamlit_secrets_to_env()

    if ADMIN_AVAILABLE:
        api_keys = get_api_keys()
        apply_api_keys_to_env(api_keys)

        # Load LLM config into session state
        config = load_config()
        if 'llm_model' not in st.session_state:
            st.session_state['llm_model'] = config.llm_config.model
        if 'llm_temperature' not in st.session_state:
            st.session_state['llm_temperature'] = config.llm_config.temperature
        if 'analysis_config' not in st.session_state:
            st.session_state['analysis_config'] = config.analysis_config

    # Re-apply the user-supplied Claude Code subscription token on each rerun.
    cred = st.session_state.get('anthropic_credential', '').strip()
    if cred:
        os.environ['CLAUDE_CODE_OAUTH_TOKEN'] = cred


def get_ai_agent(agent_type: str = "general"):
    """Get AI agent instance. Auth is via Claude Code CLI subscription."""
    if AI_AVAILABLE:
        try:
            if agent_type == "screening":
                return ScreeningAgent()
            elif agent_type == "anomaly":
                return AnomalyAgent()
            else:
                # Use ResearchAgent as default - it has all capabilities without handoff issues
                return ResearchAgent()
        except Exception as e:
            st.error(f"Error initializing AI agent: {e}")
    return None


def show_persistent_chat():
    """Show persistent AI chat at the bottom of every page."""
    st.markdown("---")
    st.markdown("### 💬 AI Assistant")

    # Initialize chat history
    if 'persistent_chat_messages' not in st.session_state:
        st.session_state.persistent_chat_messages = []

    # Display chat messages in a container with fixed height
    if st.session_state.persistent_chat_messages:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.persistent_chat_messages[-5:]:  # Show last 5 messages
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")

    # Claude auth is handled by Claude Code CLI subscription — no key check needed
    if AI_AVAILABLE:
        with st.form("persistent_chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask anything about value investing...",
                placeholder="e.g., What makes a stock undervalued?",
                key="persistent_chat_input",
                label_visibility="collapsed"
            )
            col1, col2 = st.columns([5, 1])
            with col2:
                submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

        if submitted and user_input and user_input.strip():
            # Add user message
            st.session_state.persistent_chat_messages.append({
                "role": "user",
                "content": user_input
            })

            # Get AI response via claude-agent-sdk (uses Claude Code CLI subscription auth)
            with st.spinner("Thinking..."):
                try:
                    from llm import claude_complete

                    # Build context from session
                    system_prompt = """You are a helpful value investing assistant. You help users:
- Understand value investing principles (margin of safety, intrinsic value, EPV)
- Interpret financial metrics (M-Score, Z-Score, F-Score, ROE, etc.)
- Analyze stocks for potential red flags
- Make informed investment decisions

Be concise but thorough. Use specific numbers when available."""

                    # Add context from current session
                    context_info = ""
                    if 'workflow_data' in st.session_state:
                        if st.session_state.workflow_data.get('filtered_df') is not None:
                            df = st.session_state.workflow_data['filtered_df']
                            context_info += f"\nUser has screened {len(df)} stocks."
                        if st.session_state.workflow_data.get('selected'):
                            context_info += f"\nSelected for analysis: {', '.join(st.session_state.workflow_data['selected'])}"
                        if st.session_state.workflow_data.get('final'):
                            context_info += f"\nFinal candidates: {', '.join(st.session_state.workflow_data['final'])}"

                    if context_info:
                        system_prompt += f"\n\nCurrent session context:{context_info}"

                    # claude-agent-sdk's query() is one-shot per call. Fold recent
                    # conversation history into the user prompt so context survives.
                    recent = st.session_state.persistent_chat_messages[-6:]
                    if len(recent) > 1:
                        history_lines = []
                        for msg in recent[:-1]:
                            role = "User" if msg["role"] == "user" else "Assistant"
                            history_lines.append(f"{role}: {msg['content']}")
                        history_lines.append(f"User: {recent[-1]['content']}")
                        user_prompt = "Recent conversation:\n" + "\n".join(history_lines)
                    else:
                        user_prompt = recent[-1]["content"]

                    model = st.session_state.get('llm_model', 'claude-sonnet-4-6')

                    assistant_message = claude_complete(
                        user=user_prompt,
                        system=system_prompt,
                        model=model,
                    )

                    st.session_state.persistent_chat_messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                except Exception as e:
                    st.session_state.persistent_chat_messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
            st.rerun()

        # Clear chat button
        if st.session_state.persistent_chat_messages:
            if st.button("Clear Chat", key="clear_persistent_chat"):
                st.session_state.persistent_chat_messages = []
                st.rerun()
    else:
        # Show chat input disabled
        st.text_input("Ask anything...", disabled=True, placeholder="claude-agent-sdk not available")
        st.caption("Install claude-agent-sdk and run `claude /login` to enable AI Assistant")


def main():
    # Initialize configuration from admin settings (includes API key setup)
    init_config()

    # Check if settings page should be shown
    if st.session_state.get('show_settings', False):
        # Minimal sidebar for settings page
        if st.sidebar.button("← Back to App", use_container_width=True):
            st.session_state['show_settings'] = False
            st.rerun()
        show_admin_page()
        return

    # Main page - Header with settings button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("📈 VIA Financial Analysis Platform")
    with col2:
        if ADMIN_AVAILABLE:
            if st.button("⚙️", help="Settings"):
                st.session_state['show_settings'] = True
                st.rerun()

    st.markdown("*AI-powered stock screening, anomaly detection, and investment analysis*")

    # ----------------------------------------------------------------
    # Auth: Claude Code subscription OAuth token only.
    # To get a token, run `claude setup-token` in your terminal.
    # ----------------------------------------------------------------
    _saved_credential = st.session_state.get('anthropic_credential', '').strip()

    if _saved_credential:
        _auth_label = "🎫 Subscription token configured — using Claude Code subscription"
    else:
        _auth_label = "🔓 Enter your Claude Code Token"

    with st.expander(_auth_label, expanded=False):
        st.caption(
            "Paste your Claude Code **subscription OAuth token**. "
            "To get a token, run `claude setup-token` in your terminal. "
            "Leave blank to use the local `claude /login` session."
        )
        col_key, col_btn = st.columns([4, 1])
        with col_key:
            new_credential = st.text_input(
                "Subscription Token",
                value=_saved_credential,
                type="password",
                placeholder="Paste your Claude Code subscription token",
                help="Run `claude setup-token` in your terminal to generate this token.",
                label_visibility="collapsed",
                key="credential_input",
            )
        with col_btn:
            if st.button("Save", use_container_width=True, key="save_credential_btn"):
                cleaned = (new_credential or '').strip()
                os.environ.pop('CLAUDE_CODE_OAUTH_TOKEN', None)
                if cleaned:
                    os.environ['CLAUDE_CODE_OAUTH_TOKEN'] = cleaned
                    st.session_state['anthropic_credential'] = cleaned
                else:
                    st.session_state['anthropic_credential'] = ''
                st.rerun()
        if _saved_credential:
            if st.button("Clear token (revert to local CLI login)",
                         key="clear_credential_btn"):
                st.session_state['anthropic_credential'] = ''
                os.environ.pop('CLAUDE_CODE_OAUTH_TOKEN', None)
                st.rerun()

    # Show the tab-based workflow
    show_tabbed_workflow()

    # Show persistent chat at the bottom
    show_persistent_chat()


def show_tabbed_workflow():
    """Show the 3-step workflow with step navigation."""

    # Initialize session state
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = {}
    if 'agent_results' not in st.session_state:
        st.session_state.agent_results = {}
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1

    # Progress indicator
    step1_done = st.session_state.agent_results.get('screened', False) and st.session_state.workflow_data.get('selected')
    step2_done = st.session_state.workflow_data.get('final', [])
    current = st.session_state.current_step

    # Custom CSS for clickable step tabs
    st.markdown("""
    <style>
    div[data-testid="column"] > div > div > div > div > button {
        border-radius: 12px 12px 0 0 !important;
        padding: 20px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="column"] > div > div > div > div > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Clickable step tabs
    col1, col2, col3 = st.columns(3)

    # Step 1 button
    with col1:
        step1_icon = '✅' if step1_done and current != 1 else '📊'
        step1_label = f"{step1_icon} STEP 1: Companies Screening"
        if current == 1:
            st.button(step1_label, key="nav1", use_container_width=True, type="primary")
        elif step1_done:
            if st.button(step1_label, key="nav1", use_container_width=True, type="secondary"):
                st.session_state.current_step = 1
                st.rerun()
        else:
            if st.button(step1_label, key="nav1", use_container_width=True, type="secondary"):
                st.session_state.current_step = 1
                st.rerun()

    # Step 2 button
    with col2:
        step2_icon = '✅' if step2_done and current != 2 else '🔍'
        step2_label = f"{step2_icon} STEP 2: Anomaly Analysis"
        if current == 2:
            st.button(step2_label, key="nav2", use_container_width=True, type="primary")
        elif step2_done:
            if st.button(step2_label, key="nav2", use_container_width=True, type="secondary"):
                st.session_state.current_step = 2
                st.rerun()
        else:
            if st.button(step2_label, key="nav2", use_container_width=True, type="secondary"):
                st.session_state.current_step = 2
                st.rerun()

    # Step 3 button
    with col3:
        step3_icon = '📝'
        step3_label = f"{step3_icon} STEP 3: Summary Report"
        if current == 3:
            st.button(step3_label, key="nav3", use_container_width=True, type="primary")
        else:
            if st.button(step3_label, key="nav3", use_container_width=True, type="secondary"):
                st.session_state.current_step = 3
                st.rerun()

    st.markdown("---")

    # ===========================================
    # STEP 1: SCREEN STOCKS
    # ===========================================
    if current == 1:
        st.markdown("### 📊 Companies Screening: Upload data and set screening criteria")

        # Valuation methodology — in-house two-stage DCF (per client's Excel)
        with st.expander("⚙️ Valuation Formula (In-House Two-Stage DCF)", expanded=False):
            st.markdown("""
#### Valuation: In-House Two-Stage DCF

Every screened stock is valued using the client's in-house formula (transcribed
exactly from `In-house valuation.xlsx`). Two intrinsic-value scenarios are
computed per share, giving a **range** instead of a single point estimate:

```
historical_growth = (Present_EPS / Past_EPS) ^ (1/years_back) − 1
Low_IV_growth     = MIN(historical_growth / 8, 3%)    ← conservative
High_IV_growth    = MIN(historical_growth / 2, 12%)   ← aggressive

For each scenario:
  Stage 1 = F·(1+g)·(1 − ((1+g)/(1+I))^H) / (I − g)        ← growth phase
  Stage 2 = F·(1+g)^H · (1+J)·(1 − ((1+J)/(1+I))^K) / (I − J) / (1+I)^H
  IV      = (Stage 1 + Stage 2) · (1 − L)
```

**Constants** (fixed, per client's Excel):

| Symbol | Meaning | Value |
|---|---|---|
| F | Present EPS | from CSV (Current Price ÷ PE, or EPV × WACC) |
| H | Growth-phase years | **10** |
| I | Discount rate | **10 %** |
| J | Terminal growth / inflation | **2 %** |
| K | Terminal-phase years | **10** |
| L | Margin of safety | **30 %** (built in) |

**Verdict mapping** (no user-tunable thresholds — the 30% margin of safety in
the formula already encodes the buy discipline):

| Status | Condition |
|---|---|
| 🟢 **Undervalued** | Current Price < **Low IV** (cheap by both scenarios) |
| 🟡 **Fair Value** | **Low IV** ≤ Current Price ≤ **High IV** |
| 🔴 **Overvalued** | Current Price > **High IV** (expensive by both scenarios) |

The card on each stock shows `IV Range: $low – $high` — the buy range is per
share and directly comparable to the live price next to it.
            """)

        # File upload section
        col1, col2 = st.columns(2)
        with col1:
            screener_files = st.file_uploader("Screener CSV (US/SG)", type=['csv'], key="step1_screener", accept_multiple_files=True)
            if screener_files:
                try:
                    dfs = []
                    for f in screener_files:
                        df = pd.read_csv(f, encoding='utf-8-sig')
                        df.columns = df.columns.str.strip()
                        filename_upper = f.name.upper()
                        if 'US' in filename_upper:
                            df['Market'] = 'US'
                            # Standardize exchange names
                            if 'Exchange' in df.columns:
                                exchange_map = {'NAS': 'NASDAQ', 'NYSE': 'NYSE', 'AMEX': 'AMEX', 'NASDAQ': 'NASDAQ'}
                                df['Exchange'] = df['Exchange'].map(exchange_map).fillna(df['Exchange'])
                        elif 'SG' in filename_upper:
                            df['Market'] = 'SG'
                            if 'Exchange' not in df.columns or df['Exchange'].isna().all():
                                df['Exchange'] = 'SGX'
                        else:
                            df['Market'] = 'Other'
                        dfs.append(df)
                    combined_df = pd.concat(dfs, ignore_index=True)
                    # Dedupe on Symbol + Market so same-ticker-different-market companies
                    # (e.g. US BAC = Bank of America vs SG BAC = Camsing Healthcare) are both kept.
                    dedupe_keys = ['Symbol', 'Market'] if 'Market' in combined_df.columns else ['Symbol']
                    combined_df = combined_df.drop_duplicates(subset=dedupe_keys, keep='first')
                    st.session_state.workflow_data['screener_df'] = combined_df

                    # Upload summary
                    us_count = len(combined_df[combined_df['Market'] == 'US'])
                    sg_count = len(combined_df[combined_df['Market'] == 'SG'])
                    st.success(f"✅ Loaded {len(combined_df)} unique stocks from {len(screener_files)} file(s)")
                    mcol1, mcol2, mcol3 = st.columns(3)
                    with mcol1:
                        st.metric("US Stocks", us_count)
                    with mcol2:
                        st.metric("SG Stocks", sg_count)
                    with mcol3:
                        exchanges = sorted(combined_df['Exchange'].dropna().unique()) if 'Exchange' in combined_df.columns else []
                        st.caption(f"Exchanges: {', '.join(exchanges)}" if exchanges else "")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            fin_file = st.file_uploader("Financials XLS (optional)", type=['xls', 'xlsx'], key="step1_fin")
            if fin_file:
                try:
                    import xlrd
                    with open("/tmp/Companies with anomalies.xls", "wb") as f:
                        f.write(fin_file.getvalue())
                    book = xlrd.open_workbook("/tmp/Companies with anomalies.xls", ignore_workbook_corruption=True)
                    symbols = [s.split('_')[1] for s in book.sheet_names() if '_' in s]
                    st.session_state.workflow_data['available_symbols'] = symbols
                    st.success(f"✅ {len(symbols)} companies with financials data")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Screening criteria panel
        if 'screener_df' in st.session_state.workflow_data:
            st.markdown("---")
            src_df = st.session_state.workflow_data['screener_df']

            # --- Synced slider+input helper ---
            def _on_slider_change(key):
                """Slider changed -> push value to number input key."""
                st.session_state[f'{key}_i'] = st.session_state[f'{key}_s']

            def _on_input_change(key):
                """Number input changed -> push value to slider key."""
                st.session_state[f'{key}_s'] = st.session_state[f'{key}_i']

            def synced_criterion(label, key, min_val, max_val, default, step=1, is_float=False):
                """Render a synced slider + number input pair. Returns the current value."""
                # Initialize both widget keys to default on first render only
                if f'{key}_s' not in st.session_state:
                    st.session_state[f'{key}_s'] = float(default) if is_float else int(default)
                if f'{key}_i' not in st.session_state:
                    st.session_state[f'{key}_i'] = float(default) if is_float else int(default)

                st.markdown(f"**{label}**")
                c1, c2 = st.columns([3, 1])
                with c1:
                    if is_float:
                        st.slider("s", min_value=float(min_val), max_value=float(max_val),
                                  step=float(step), key=f"{key}_s",
                                  label_visibility="collapsed",
                                  on_change=_on_slider_change, args=(key,))
                    else:
                        st.slider("s", min_value=int(min_val), max_value=int(max_val),
                                  step=int(step), key=f"{key}_s",
                                  label_visibility="collapsed",
                                  on_change=_on_slider_change, args=(key,))
                with c2:
                    if is_float:
                        st.number_input("n", min_value=float(min_val), max_value=float(max_val),
                                        step=float(step), key=f"{key}_i",
                                        label_visibility="collapsed",
                                        on_change=_on_input_change, args=(key,))
                    else:
                        st.number_input("n", min_value=int(min_val), max_value=int(max_val),
                                        step=int(step), key=f"{key}_i",
                                        label_visibility="collapsed",
                                        on_change=_on_input_change, args=(key,))
                return st.session_state[f'{key}_s']

            # Exchange filter
            st.markdown("#### 🌍 Market Filter")
            available_exchanges = sorted(src_df['Exchange'].dropna().unique().tolist()) if 'Exchange' in src_df.columns else []
            selected_exchanges = st.multiselect(
                "Filter by Exchange",
                options=available_exchanges,
                default=available_exchanges,
                help="Select one or more exchanges to include"
            )

            st.markdown("---")

            # Profitability criteria
            st.markdown("#### 📊 Profitability")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                gm = synced_criterion("Min Gross Margin %", "gm", 0, 100, 20)
                nm = synced_criterion("Min Net Margin %", "nm", -50, 100, 5)
                roa = synced_criterion("Min ROA %", "roa", -20, 50, 5)
            with p_col2:
                roe = synced_criterion("Min ROE %", "roe", -20, 100, 10)
                fcf = synced_criterion("Min FCF Margin %", "fcf", -50, 100, 0)

            st.markdown("---")

            # Growth criteria
            st.markdown("#### 📈 Growth")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                rev_growth = synced_criterion("Min 5Y Revenue Growth %", "rev", -50, 100, 0)
            with g_col2:
                eps_growth = synced_criterion("Min 5Y EPS Growth %", "eps", -50, 100, 0)

            st.markdown("---")

            # Efficiency criteria
            st.markdown("#### ⚡ Efficiency")
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                roic = synced_criterion("Min ROIC-WACC", "roic", -20, 50, 0)
            with e_col2:
                rote = synced_criterion("Min ROTE-WACC", "rote", -50, 100, 0)

            st.markdown("---")

            # Balance Sheet criteria
            st.markdown("#### 💰 Balance Sheet")
            de = synced_criterion("Max Debt-to-Equity", "de", 0.0, 5.0, 1.5, step=0.1, is_float=True)

            st.markdown("---")

            # Screen button + reset button
            screen_c1, screen_c2 = st.columns([3, 1])
            with screen_c1:
                run_screen = st.button("🔍 Execute Screen", type="primary", use_container_width=True)
            with screen_c2:
                if st.button("↺ Reset filters", use_container_width=True,
                             help="Reset all sliders to their default values"):
                    # Clear all slider/input keys so they pick up defaults on next render
                    for k in ['gm', 'nm', 'roa', 'roe', 'fcf', 'rev', 'eps',
                              'roic', 'rote', 'de']:
                        st.session_state.pop(f'{k}_s', None)
                        st.session_state.pop(f'{k}_i', None)
                    st.rerun()

            if run_screen:
                df = src_df.copy()
                filter_trace = [("Uploaded", "", "", len(df), len(df))]

                # Exchange filter
                if selected_exchanges and 'Exchange' in df.columns:
                    before = len(df)
                    df = df[df['Exchange'].isin(selected_exchanges)]
                    filter_trace.append(("Exchange", "in", ",".join(selected_exchanges), before, len(df)))

                # Apply all criteria
                filter_config = {
                    'Gross Margin %': ('>=', gm),
                    'Net Margin %': ('>=', nm),
                    'ROA %': ('>=', roa),
                    'ROE %': ('>=', roe),
                    'FCF Margin %': ('>=', fcf),
                    '5-Year Revenue Growth Rate (Per Share)': ('>=', rev_growth),
                    '5-Year EPS without NRI Growth Rate': ('>=', eps_growth),
                    'ROIC-WACC': ('>=', roic),
                    'ROTE-WACC': ('>=', rote),
                    'Debt-to-Equity': ('<=', de),
                }

                for col, (op, threshold) in filter_config.items():
                    if col in df.columns:
                        before = len(df)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if op == '>=':
                            df = df[df[col] >= threshold]
                        else:
                            df = df[(df[col] <= threshold) & (df[col] >= 0)]
                        filter_trace.append((col, op, threshold, before, len(df)))

                st.session_state.workflow_data['filter_trace'] = filter_trace

                # In-house two-stage DCF valuation (per client's Excel formula).
                # Produces per-share Low IV / High IV columns and classifies the
                # current price against that RANGE — not a single fixed number.
                from valuation import Valuator
                df = Valuator().analyze_dataframe(df)

                # Keep Live Share Price for downstream display.
                df['Live Share Price'] = pd.to_numeric(df.get('Current Price'), errors='coerce')

                # Map "Fair Value" -> "Fair" so existing UI logic keeps working.
                if 'Valuation' in df.columns:
                    df['Valuation'] = df['Valuation'].replace({'Fair Value': 'Fair'})

                # Back-compat aliases so downstream code that reads the old column
                # names still works without further churn.
                df['Highest Intrinsic Value'] = df.get('High IV')
                df['Lowest Intrinsic Value'] = df.get('Low IV')

                # Store criteria for report
                st.session_state.workflow_data['criteria'] = {
                    'exchanges': selected_exchanges,
                    'gross_margin': gm, 'net_margin': nm, 'roa': roa, 'roe': roe,
                    'fcf_margin': fcf, 'revenue_growth_5y': rev_growth,
                    'eps_growth_5y': eps_growth, 'roic_wacc': roic,
                    'rote_wacc': rote, 'debt_to_equity': de,
                }
                st.session_state.workflow_data['filtered_df'] = df
                st.session_state.agent_results['screened'] = True

        if st.session_state.agent_results.get('screened'):
            df = st.session_state.workflow_data['filtered_df']

            # Filter cascade diagnostic — shows which criterion eliminated which rows.
            # Always visible when result is empty, collapsed expander otherwise.
            trace = st.session_state.workflow_data.get('filter_trace', [])
            if trace:
                def _fmt_threshold(op, thr):
                    if op == "" or op == "in":
                        return ""
                    if isinstance(thr, float):
                        return f" {op} {thr:.2f}"
                    return f" {op} {thr}"

                def _render_trace_table():
                    lines = ["| Criterion | Threshold | Before | After | Dropped |",
                             "|---|---|---|---|---|"]
                    for name, op, thr, before, after in trace:
                        dropped = before - after
                        thr_str = _fmt_threshold(op, thr) if op else "—"
                        marker = " 🔴" if dropped > 0 and after == 0 else ""
                        lines.append(f"| {name}{marker} | {thr_str} | {before} | {after} | {dropped} |")
                    st.markdown("\n".join(lines))

                if len(df) == 0:
                    # Identify the criterion that killed the result
                    killer = None
                    for name, op, thr, before, after in trace:
                        if before > 0 and after == 0:
                            killer = (name, op, thr)
                            break
                    st.error(
                        f"⚠️ **0 stocks passed the filter.** "
                        + (f"The bottleneck was **{killer[0]}{_fmt_threshold(killer[1], killer[2])}** — "
                           f"loosen this slider to recover results."
                           if killer else "Loosen one or more sliders and re-run.")
                    )
                    with st.expander("📉 Filter elimination cascade — see how many stocks each criterion dropped",
                                     expanded=True):
                        _render_trace_table()
                else:
                    with st.expander("📉 Filter elimination cascade", expanded=False):
                        _render_trace_table()

            # Results header with count + export
            hdr_col1, hdr_col2 = st.columns([4, 1])
            with hdr_col1:
                st.markdown(f"### 📊 {len(df)} Stocks Found")
            with hdr_col2:
                export_cols = ['Symbol', 'Company', 'Exchange', 'Sector', 'Gross Margin %', 'Net Margin %',
                               'ROE %', 'ROA %', 'Debt-to-Equity', 'FCF Margin %', 'ROIC-WACC', 'ROTE-WACC',
                               'Market Cap ($M)', 'Low IV', 'High IV',
                               'Live Share Price', 'Valuation']
                avail_export = [c for c in export_cols if c in df.columns]
                csv_data = df[avail_export].to_csv(index=False)
                st.download_button("📥 Export CSV", csv_data, "screened_stocks.csv", "text/csv",
                                   use_container_width=True)

            # Valuation summary metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                undervalued = len(df[df['Valuation'] == 'Undervalued']) if 'Valuation' in df.columns else 0
                st.metric("Undervalued", undervalued)
            with m_col2:
                fair = len(df[df['Valuation'] == 'Fair']) if 'Valuation' in df.columns else 0
                st.metric("Fair Value", fair)
            with m_col3:
                overvalued = len(df[df['Valuation'] == 'Overvalued']) if 'Valuation' in df.columns else 0
                st.metric("Overvalued", overvalued)
            with m_col4:
                avg_roe = df['ROE %'].mean() if 'ROE %' in df.columns else 0
                st.metric("Avg ROE", f"{avg_roe:.1f}%")

            st.markdown("---")

            # Sort results so both markets surface: Undervalued first, then Fair,
            # then Overvalued / N/A — within each rating, biggest market cap first.
            # Previously df was in upload order (US then SG, or vice versa) and
            # display was capped at head(30), which hid the second file's results.
            df_display = df.copy()
            if 'Valuation' in df_display.columns:
                val_order = {'Undervalued': 0, 'Fair': 1, 'Overvalued': 2, 'N/A': 3}
                df_display['_v_rank'] = df_display['Valuation'].map(
                    lambda v: val_order.get(v, 4)
                )
                sort_cols = ['_v_rank']
                ascending = [True]
                if 'Market Cap ($M)' in df_display.columns:
                    df_display['Market Cap ($M)'] = pd.to_numeric(
                        df_display['Market Cap ($M)'], errors='coerce'
                    )
                    sort_cols.append('Market Cap ($M)')
                    ascending.append(False)
                df_display = df_display.sort_values(sort_cols, ascending=ascending)
                df_display = df_display.drop(columns=['_v_rank'])

            # Display all screened cards (no 30-row cap — was hiding stocks from
            # the second uploaded file when there were >30 survivors).
            for card_idx, (idx, row) in enumerate(df_display.iterrows()):
                row_data = row.to_dict()
                sym = row.get('Symbol', 'N/A')
                render_stock_card(
                    symbol=sym,
                    company=row.get('Company', 'N/A'),
                    data=row_data,
                    card_index=card_idx
                )
                # Show company detail if toggled
                if st.session_state.get(f'show_detail_{sym}', False):
                    show_company_detail_popup(sym, row_data)

            all_symbols = df['Symbol'].tolist()
            available = st.session_state.workflow_data.get('available_symbols', [])

            if all_symbols:
                st.markdown("---")
                with_data = [s for s in all_symbols if s in available]
                without_data = [s for s in all_symbols if s not in available]

                if without_data and not available:
                    st.info("💡 Upload Financials XLS (optional) for detailed anomaly scores")
                elif without_data:
                    st.caption(f"ℹ️ {len(with_data)} have financials data, {len(without_data)} will have limited analysis")

                default_selection = (with_data[:5] if with_data else all_symbols[:5])
                selected = st.multiselect("Select companies for Anomaly Analysis:", all_symbols, default_selection)
                st.session_state.workflow_data['selected'] = selected

                if selected:
                    st.success(f"✅ {len(selected)} companies selected")
                    st.markdown("---")
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button("➡️ Proceed to Anomaly Analysis", type="primary", use_container_width=True):
                            st.session_state.current_step = 2
                            st.rerun()

    # ===========================================
    # STEP 2: AI ANOMALY ANALYSIS
    # ===========================================
    elif current == 2:
        selected = st.session_state.workflow_data.get('selected', [])
        available = st.session_state.workflow_data.get('available_symbols', [])

        if not selected:
            st.warning("⚠️ No companies selected. Go back to Companies Screening.")
            if st.button("← Back to Companies Screening"):
                st.session_state.current_step = 1
                st.rerun()
        else:
            st.markdown("### 🔍 AI Anomaly Analysis: Detect One-Off Financial Distortions")
            st.markdown(f"**{len(selected)} companies:** {', '.join(selected)}")

            st.info("AI analyzes 10-year financial history to detect: **business model changes**, "
                     "**one-off significant expenses/income**, **unusual margin shifts**, "
                     "**revenue/earnings spikes or drops**, and other distortions that may not reflect "
                     "the company's normal earning power.")

            # ────────────────────────────────────────────────────────────────
            # Configurable scraping sources
            # ────────────────────────────────────────────────────────────────
            from scraper import DEFAULT_SOURCES as _DEFAULT_SOURCES, render_url as _render_url

            # Bumped whenever DEFAULT_SOURCES URLs change so stale sessions
            # refresh their cached source list automatically.
            _SOURCES_VERSION = 3

            # Seed session state from defaults on first render, or refresh
            # them when DEFAULT_SOURCES version was bumped. User-added custom
            # sources are preserved across the refresh.
            need_refresh = (
                'scraping_sources' not in st.session_state
                or st.session_state.get('scraping_sources_version', 0) < _SOURCES_VERSION
            )
            if need_refresh:
                existing_custom = [
                    s for s in st.session_state.get('scraping_sources', [])
                    if not s.get("is_default")
                ]
                st.session_state.scraping_sources = [
                    {"label": lbl, "homepage": home, "url_template": tpl,
                     "enabled": True, "is_default": True}
                    for lbl, home, tpl in _DEFAULT_SOURCES
                ] + existing_custom
                st.session_state.scraping_sources_version = _SOURCES_VERSION
                # Clear any cached checkbox/text-input widget keys so the new
                # default rows render with their fresh URLs.
                for k in list(st.session_state.keys()):
                    if str(k).startswith(("src_en_", "src_lbl_", "src_url_")):
                        del st.session_state[k]

            with st.expander("🌐 Scraping Sources — control which sites Firecrawl pulls real data from",
                              expanded=False):
                st.markdown(
                    "**Default sources (client spec):** Bloomberg, Reuters, Morningstar, Gurufocus.\n\n"
                    "Each enabled source is scraped per company by Firecrawl. The data Claude "
                    "analyzes is **only what comes back from these sites** — no hallucinated numbers."
                )

                # Header
                hc1, hc2, hc3, hc4 = st.columns([0.7, 2, 5, 0.6])
                with hc1: st.markdown("**On**")
                with hc2: st.markdown("**Source**")
                with hc3: st.markdown("**Website**")
                with hc4: st.markdown("**Del**")

                # Rows
                to_delete = []
                for idx, src in enumerate(st.session_state.scraping_sources):
                    c1, c2, c3, c4 = st.columns([0.7, 2, 5, 0.6])
                    with c1:
                        src["enabled"] = st.checkbox(
                            "enabled", value=src.get("enabled", True),
                            key=f"src_en_{idx}", label_visibility="collapsed"
                        )
                    is_default = src.get("is_default", False)
                    with c2:
                        if is_default:
                            # Defaults: label is fixed (client-spec sites)
                            st.markdown(f"**{src.get('label', '')}**")
                        else:
                            src["label"] = st.text_input(
                                "label", value=src.get("label", ""),
                                key=f"src_lbl_{idx}", label_visibility="collapsed",
                                placeholder="e.g. SimplyWall.St"
                            )
                    with c3:
                        if is_default:
                            # Defaults: show clean homepage URL (read-only display)
                            home = src.get("homepage", "")
                            st.markdown(f"`{home}`")
                        else:
                            # Custom sites: user provides URL with placeholder
                            src["url_template"] = st.text_input(
                                "url", value=src.get("url_template", ""),
                                key=f"src_url_{idx}", label_visibility="collapsed",
                                placeholder="https://site.com/{symbol_upper}/financials"
                            )
                            src["homepage"] = src["url_template"]  # for display parity
                    with c4:
                        if st.button("🗑️", key=f"src_del_{idx}",
                                     help=("Delete this source (default site)"
                                           if is_default else "Delete this source")):
                            to_delete.append(idx)

                if to_delete:
                    for i in reversed(to_delete):
                        st.session_state.scraping_sources.pop(i)
                    st.rerun()

                # Add / reset buttons
                btn_c1, btn_c2, _ = st.columns([1.8, 1.8, 5])
                with btn_c1:
                    if st.button("➕ Add custom source", use_container_width=True,
                                  help="Add a site beyond the 4 client defaults"):
                        st.session_state.scraping_sources.append(
                            {"label": "", "homepage": "", "url_template": "",
                             "enabled": True, "is_default": False}
                        )
                        st.rerun()
                with btn_c2:
                    if st.button("↺ Reset to defaults", use_container_width=True,
                                  help="Restore Bloomberg/Reuters/Morningstar/Gurufocus"):
                        st.session_state.scraping_sources = [
                            {"label": lbl, "homepage": home, "url_template": tpl,
                             "enabled": True, "is_default": True}
                            for lbl, home, tpl in _DEFAULT_SOURCES
                        ]
                        for k in list(st.session_state.keys()):
                            if str(k).startswith(("src_en_", "src_lbl_", "src_url_")):
                                del st.session_state[k]
                        st.rerun()

                # Custom-source URL placeholder help
                if any(not s.get("is_default") for s in st.session_state.scraping_sources):
                    st.caption(
                        "💡 For custom sources, use placeholders in the URL: "
                        "`{symbol_upper}`, `{symbol_lower}`, `{symbol}`, `{market}` "
                        "(us/sg). Example: `https://example.com/quote/{symbol_upper}`"
                    )

                st.markdown("---")

                # Live preview of resolved URLs for the first selected ticker
                if selected:
                    sample_sym = selected[0]
                    sample_mkt = 'US'
                    fdf = st.session_state.workflow_data.get('filtered_df')
                    if fdf is not None and sample_sym in fdf['Symbol'].values:
                        ex = (fdf[fdf['Symbol'] == sample_sym].iloc[0].get('Exchange') or '').upper()
                        if 'SGX' in ex:
                            sample_mkt = 'SG'
                    st.markdown(
                        f"**🔍 Preview** — exact URLs Firecrawl will fetch for "
                        f"**{sample_sym}** ({sample_mkt}):"
                    )
                    enabled_srcs = [s for s in st.session_state.scraping_sources
                                    if s.get("enabled")
                                    and s.get("url_template", "").strip()
                                    and s.get("label", "").strip()]
                    if not enabled_srcs:
                        st.warning("⚠️ No sources enabled — scraping will skip and analysis "
                                   "will fall back to limited CSV data only.")
                    else:
                        for s in enabled_srcs:
                            resolved = _render_url(s["url_template"], sample_sym, sample_mkt)
                            st.code(f"{s['label']}: {resolved}", language=None)

            # Strict real-data mode is now ALWAYS ON per client spec: never feed
            # Claude a CSV-snapshot fallback. If Firecrawl + XLS + FMP all fail
            # for a ticker, the row reports "no real data available" instead of
            # running a single-period snapshot analysis.
            strict_real_data = True

            # Auth handled by Claude Code CLI subscription (no API key needed)
            if not AI_AVAILABLE:
                st.error("claude-agent-sdk not installed. Run `pip install -r requirements.txt`.")
            else:
                if st.button("🤖 Run AI Anomaly Detection", type="primary", use_container_width=True):
                    results = {}
                    progress = st.progress(0)
                    status_text = st.empty()

                    for i, sym in enumerate(selected):
                        progress.progress((i + 1) / len(selected))

                        # --- Gather 10-year financial data ---
                        fin_data = None

                        # Resolve company name + market once (used for Firecrawl + prompt)
                        filt_df_pre = st.session_state.workflow_data.get('filtered_df')
                        company_name_pre = ''
                        market_pre = 'US'
                        if filt_df_pre is not None and sym in filt_df_pre['Symbol'].values:
                            row_pre = filt_df_pre[filt_df_pre['Symbol'] == sym].iloc[0]
                            company_name_pre = row_pre.get('Company', '') or ''
                            ex = (row_pre.get('Exchange') or '').upper()
                            if 'SGX' in ex:
                                market_pre = 'SG'

                        # Source 0: Firecrawl — scrape REAL 10-year filings data (preferred)
                        scrape_diag: dict = {"source_status": [], "error": None}
                        try:
                            from scraper import scrape_financial_data, is_firecrawl_available
                            if not is_firecrawl_available():
                                scrape_diag["error"] = (
                                    "Firecrawl unavailable — FIRECRAWL_API_KEY not set "
                                    "in environment / .env"
                                )
                            else:
                                # Build the source list from session state (user-configurable).
                                # Pass the 3-tuple form so the scraper can hit the per-ticker
                                # subpath while we still display the bare-domain in UI.
                                user_sources = [
                                    (s["label"].strip(),
                                     s.get("homepage", "").strip(),
                                     s["url_template"].strip())
                                    for s in st.session_state.get('scraping_sources', [])
                                    if s.get("enabled")
                                       and s.get("url_template", "").strip()
                                       and s.get("label", "").strip()
                                ]
                                src_summary = (", ".join(lbl for lbl, _, _ in user_sources)
                                               if user_sources else "Bloomberg, Reuters, Morningstar, Gurufocus (defaults)")
                                status_text.text(
                                    f"Scraping real 10-year data for {sym} "
                                    f"({i+1}/{len(selected)}) — Firecrawl ({src_summary})..."
                                )
                                scraped = scrape_financial_data(
                                    sym, company_name_pre, market_pre,
                                    custom_sources=user_sources if user_sources else None,
                                )
                                scrape_diag["source_status"] = scraped.get('source_status', [])
                                if scraped.get('ok'):
                                    fin_data = {
                                        'source': f"Firecrawl ({scraped.get('source', 'web')})",
                                        'source_url': scraped.get('source_url', ''),
                                        'scraped_markdown': scraped.get('markdown', ''),
                                    }
                                else:
                                    scrape_diag["error"] = scraped.get(
                                        'error', 'Unknown Firecrawl failure'
                                    )
                        except Exception as e:
                            scrape_diag["error"] = (
                                f"Firecrawl call raised {type(e).__name__}: {str(e)[:200]}"
                            )

                        status_text.text(f"Analyzing {sym} ({i+1}/{len(selected)})...")

                        # Source 0.5: yfinance fallback (no API key, ~4y of real
                        # income statement + cash flow from Yahoo Finance).
                        # Runs whenever Firecrawl returned no usable scraped
                        # markdown — ensures we still have REAL multi-year data
                        # to feed Claude before declaring NO_DATA in strict mode.
                        if fin_data is None:
                            # yfinance uses {ticker}.SI for SGX-listed stocks. Compute
                            # this OUTSIDE the try so the diagnostic line below in the
                            # except branch can still reference yf_symbol if the
                            # yfinance import itself fails.
                            yf_symbol = f"{sym}.SI" if market_pre == "SG" else sym
                            try:
                                import yfinance as _yf
                                t = _yf.Ticker(yf_symbol)
                                inc = t.income_stmt
                                cf = t.cashflow
                                if inc is not None and not inc.empty:
                                    # yfinance columns are timestamps, most-recent first.
                                    # Reverse so they go oldest → newest like the other sources.
                                    cols = list(inc.columns)[::-1]
                                    periods = [c.strftime("%Y-%m-%d") if hasattr(c, "strftime") else str(c)
                                               for c in cols]
                                    def _row(df, key):
                                        if df is None or df.empty or key not in df.index:
                                            return []
                                        vals = df.loc[key, cols].tolist()
                                        return [None if (v is None or (isinstance(v, float) and v != v)) else float(v)
                                                for v in vals]
                                    fin_data = {
                                        'source': 'yfinance (Yahoo Finance — real filings data, no API key)',
                                        'periods': periods,
                                        'revenue': _row(inc, 'Total Revenue'),
                                        'net_income': _row(inc, 'Net Income'),
                                        'gross_profit': _row(inc, 'Gross Profit'),
                                        'operating_income': _row(inc, 'Operating Income'),
                                        'eps': _row(inc, 'Diluted EPS'),
                                        'operating_cf': _row(cf, 'Operating Cash Flow'),
                                        'fcf': _row(cf, 'Free Cash Flow'),
                                    }
                                    scrape_diag.setdefault("source_status", []).append({
                                        "label": "yfinance",
                                        "url": f"yfinance.Ticker('{yf_symbol}')",
                                        "status": f"ok:{len(periods)} periods",
                                        "bytes": 0,
                                    })
                            except Exception as e:
                                scrape_diag.setdefault("source_status", []).append({
                                    "label": "yfinance",
                                    "url": f"yfinance.Ticker('{yf_symbol}')",
                                    "status": f"error:{type(e).__name__}: {str(e)[:120]}",
                                    "bytes": 0,
                                })

                        # Source 1: XLS file (if available)
                        if fin_data is None and sym in available:
                            try:
                                loader = DataLoader("/tmp")
                                parsed = loader.load_anomaly_data(sym)
                                if sym in parsed:
                                    d = parsed[sym]
                                    periods = d.get('fiscal_periods', [])[:10]
                                    fin_data = {
                                        'source': 'XLS',
                                        'periods': periods,
                                        'revenue': d.get('income_statement', {}).get('revenue', [])[:10],
                                        'net_income': d.get('income_statement', {}).get('net_income', [])[:10],
                                        'gross_profit': d.get('income_statement', {}).get('gross_profit', [])[:10],
                                        'operating_income': d.get('income_statement', {}).get('operating_income', [])[:10],
                                        'fcf': d.get('cash_flow', {}).get('free_cash_flow', [])[:10],
                                        'operating_cf': d.get('cash_flow', {}).get('operating_cash_flow', [])[:10],
                                        'eps': d.get('per_share_data', {}).get('eps_diluted', [])[:10],
                                        'eps_nri': d.get('per_share_data', {}).get('eps_without_nri', [])[:10],
                                        'gross_margin': d.get('ratios', {}).get('gross_margin', [])[:10],
                                        'net_margin': d.get('ratios', {}).get('net_margin', [])[:10],
                                        'roe': d.get('ratios', {}).get('roe', [])[:10],
                                        'debt_to_equity': d.get('ratios', {}).get('debt_to_equity', [])[:10],
                                    }
                            except Exception:
                                pass

                        # Source 2: FMP API (fallback)
                        if fin_data is None:
                            fmp = get_fmp_client()
                            if fmp:
                                try:
                                    inc = fmp.get_income_statement(sym, limit=10)
                                    cf = fmp.get_cash_flow(sym, limit=10)
                                    if not inc.empty:
                                        inc_sorted = inc.sort_values('date') if 'date' in inc.columns else inc
                                        cf_sorted = cf.sort_values('date') if not cf.empty and 'date' in cf.columns else cf
                                        fin_data = {
                                            'source': 'FMP',
                                            'periods': inc_sorted['date'].tolist() if 'date' in inc_sorted.columns else [],
                                            'revenue': inc_sorted['revenue'].tolist() if 'revenue' in inc_sorted.columns else [],
                                            'net_income': inc_sorted['netIncome'].tolist() if 'netIncome' in inc_sorted.columns else [],
                                            'gross_profit': inc_sorted['grossProfit'].tolist() if 'grossProfit' in inc_sorted.columns else [],
                                            'operating_income': inc_sorted['operatingIncome'].tolist() if 'operatingIncome' in inc_sorted.columns else [],
                                            'eps': inc_sorted['eps'].tolist() if 'eps' in inc_sorted.columns else [],
                                            'fcf': cf_sorted['freeCashFlow'].tolist() if not cf_sorted.empty and 'freeCashFlow' in cf_sorted.columns else [],
                                            'operating_cf': cf_sorted['operatingCashFlow'].tolist() if not cf_sorted.empty and 'operatingCashFlow' in cf_sorted.columns else [],
                                        }
                                except Exception:
                                    pass

                        # Source 3: CSV screening data only (minimal).
                        # SKIPPED when strict_real_data is on — we refuse to run
                        # Claude on a single-period snapshot, per client spec
                        # that input must be 100% real multi-year data.
                        if fin_data is None and not strict_real_data:
                            filt_df = st.session_state.workflow_data.get('filtered_df')
                            if filt_df is not None and sym in filt_df['Symbol'].values:
                                row = filt_df[filt_df['Symbol'] == sym].iloc[0]
                                fin_data = {
                                    'source': 'CSV (limited)',
                                    'screening_metrics': {
                                        k: row.get(k) for k in [
                                            'Gross Margin %', 'Net Margin %', 'ROE %', 'ROA %',
                                            'Debt-to-Equity', 'FCF Margin %', 'ROIC-WACC', 'ROTE-WACC',
                                            'Market Cap ($M)', 'Earnings Power Value (EPV)', 'Current Price'
                                        ] if row.get(k) is not None
                                    }
                                }

                        # Strict mode + no real data anywhere → bail without
                        # calling Claude. Record a "no data" result with the
                        # per-source diagnostic so the user sees WHY.
                        if fin_data is None and strict_real_data:
                            results[sym] = {
                                'analysis': (
                                    f"**Analysis skipped — no real multi-year data available for {sym}.**\n\n"
                                    f"Strict real-data mode is ON. The configured scraping sources "
                                    f"(Bloomberg / Reuters / Morningstar / Gurufocus or your custom list) "
                                    f"did not return usable financial tables for this ticker, and no "
                                    f"XLS / FMP fallback produced multi-year data either. "
                                    f"No CSV-snapshot analysis was generated because that would not "
                                    f"satisfy the client's '100% real data' requirement.\n\n"
                                    f"**What to do:**\n"
                                    f"1. Check the source diagnostic below to see which URLs returned empty.\n"
                                    f"2. Edit the URL pattern for that source in the 'Scraping Sources' "
                                    f"expander, or add a custom source that has data for this ticker.\n"
                                    f"3. Or untick 'Strict real-data mode' to allow CSV-snapshot analysis."
                                ),
                                'rating': 'NO_DATA',
                                'data_source': 'none (strict mode)',
                                'has_data': False,
                                'scrape_diag': scrape_diag,
                            }
                            continue

                        # --- Call Claude via claude-agent-sdk for AI analysis ---
                        try:
                            from llm import claude_complete
                            model = st.session_state.get('llm_model', 'claude-sonnet-4-6')

                            # Get company name from screening data
                            filt_df = st.session_state.workflow_data.get('filtered_df')
                            company_name = ''
                            if filt_df is not None and sym in filt_df['Symbol'].values:
                                company_name = filt_df[filt_df['Symbol'] == sym].iloc[0].get('Company', '')

                            # Build the data block — prefer raw scraped markdown
                            # (preserves real financial tables) over JSON dump.
                            grounding_rule = (
                                "GROUNDING RULE — STRICT, NO EXCEPTIONS:\n"
                                "• Every number, year, and ratio in your response MUST appear "
                                "  somewhere in the FINANCIAL DATA block above. No exceptions.\n"
                                "• Do NOT use any prior knowledge about this company. Treat it as "
                                "  if you've never heard of it — your only source is the block above.\n"
                                "• If a year, segment, or figure is not in the data, you MAY NOT "
                                "  mention it. Say 'cannot assess — not in data' instead.\n"
                                "• If you reference a number, it must be a number that physically "
                                "  appears in the text above (rounding for readability is fine, but "
                                "  the underlying value must be present).\n"
                                "• If the data is empty or insufficient, your verdict must reflect "
                                "  that — do not invent a forensic finding from nothing."
                            )

                            if fin_data and fin_data.get('scraped_markdown'):
                                data_summary = (
                                    f"DATA SOURCE: {fin_data.get('source', 'web')}\n"
                                    f"SOURCE URL: {fin_data.get('source_url', '')}\n\n"
                                    f"{fin_data['scraped_markdown']}"
                                )
                                data_provenance_note = (
                                    "The data above is REAL, scraped live from the configured "
                                    "sources (Bloomberg / Reuters / Morningstar / Gurufocus or "
                                    "whatever the user configured). Base your analysis ONLY on "
                                    "what is in this block. " + grounding_rule
                                )
                            elif fin_data:
                                data_summary = json.dumps(fin_data, default=str, indent=2)
                                data_provenance_note = (
                                    "The data above is structured from the user's uploaded "
                                    "screener / XLS / FMP API — all real, no AI invention. "
                                    + grounding_rule
                                )
                            else:
                                data_summary = "No financial history available."
                                data_provenance_note = (
                                    "No financial data was scraped or loaded. State explicitly "
                                    "that you cannot perform a forensic analysis without filings "
                                    "data, and rate the verdict accordingly. " + grounding_rule
                                )

                            prompt = f"""Analyze the financial history of {sym} ({company_name}) for anomalies and one-off distortions.

FINANCIAL DATA (up to 10 years):
{data_summary}

{data_provenance_note}

ANALYSIS REQUIRED:
1. **One-Off Income/Expenses**: Identify any years with unusual spikes or drops in revenue, net income, or operating income that appear to be one-off events (e.g., asset sales, write-downs, restructuring charges, legal settlements, pandemic impact).

2. **Business Model Changes**: Detect any signs of fundamental shifts in the business (e.g., sudden margin profile changes, revenue composition shifts, pivot from product to services).

3. **Earnings Quality Flags**: Flag years where EPS differs significantly from EPS without non-recurring items, or where operating cash flow diverges from net income.

4. **Margin Distortions**: Identify unusual year-over-year changes in gross margin or net margin (>5 percentage points) that may indicate temporary factors.

5. **Overall Assessment**: Rate the financial consistency as CLEAN (no significant distortions), MINOR (small one-offs that don't materially affect valuation), or MATERIAL (significant distortions that require adjustment to EPV).

FORMAT YOUR RESPONSE AS:
**Overall: [CLEAN/MINOR/MATERIAL]**

**Findings:**
- [Year]: [Description of anomaly and likely cause]
- [Year]: [Description]

**Impact on Valuation:**
[1-2 sentences on whether EPV needs adjustment and why]"""

                            ai_text = claude_complete(
                                user=prompt,
                                system=(
                                    "You are a forensic financial analyst detecting one-off "
                                    "distortions in company financials. You are working under a "
                                    "STRICT NO-HALLUCINATION rule: your only source of truth is "
                                    "the FINANCIAL DATA block in the user message. You may NOT "
                                    "use any prior knowledge you have about the company — pretend "
                                    "you have never heard of it. If a fact (year, segment, figure, "
                                    "event) is not in the data block, you may NOT cite it; say "
                                    "'cannot assess — not in data' instead. Be specific about "
                                    "years and magnitudes, but only those that appear in the data. "
                                    "ALWAYS begin your response with one of these exact phrases: "
                                    "'Overall: CLEAN', 'Overall: MINOR', or 'Overall: MATERIAL'."
                                ),
                                model=model,
                            )

                            # Parse overall rating - prefer explicit "Overall: X" pattern,
                            # fall back to first occurrence anywhere in response
                            import re as _re
                            rating = 'UNKNOWN'
                            ai_upper = ai_text.upper()
                            m = _re.search(r"OVERALL\s*[:.\-]?\s*\*{0,2}\s*(CLEAN|MINOR|MATERIAL)", ai_upper)
                            if m:
                                rating = m.group(1)
                            else:
                                for r in ['CLEAN', 'MINOR', 'MATERIAL']:
                                    if r in ai_upper:
                                        rating = r
                                        break

                            results[sym] = {
                                'analysis': ai_text,
                                'rating': rating,
                                'data_source': fin_data.get('source', 'None') if fin_data else 'None',
                                'has_data': True,
                                'scrape_diag': scrape_diag,
                            }
                        except Exception as e:
                            results[sym] = {
                                'analysis': f"Error during AI analysis: {str(e)}",
                                'rating': 'ERROR',
                                'data_source': fin_data.get('source', 'None') if fin_data else 'None',
                                'has_data': False,
                                'scrape_diag': scrape_diag,
                            }

                    progress.empty()
                    status_text.empty()
                    st.session_state.agent_results['anomalies'] = results

            # Display results
            if 'anomalies' in st.session_state.agent_results:
                st.markdown("### AI Analysis Results")
                passed = []

                for sym, d in st.session_state.agent_results['anomalies'].items():
                    rating = d.get('rating', 'UNKNOWN')
                    rating_map = {
                        'CLEAN':    ('✅', '#10b981', 'No significant distortions'),
                        'MINOR':    ('🟡', '#f59e0b', 'Minor one-offs detected'),
                        'MATERIAL': ('🔴', '#ef4444', 'Material distortions found'),
                        'NO_DATA':  ('🚫', '#94a3b8', 'No real data — strict mode'),
                        'ERROR':    ('⚠️', '#94a3b8', 'Analysis error'),
                        'UNKNOWN':  ('❓', '#94a3b8', 'Could not determine'),
                    }
                    icon, color, desc = rating_map.get(rating, rating_map['UNKNOWN'])

                    expand_default = (rating in ('MATERIAL', 'NO_DATA'))
                    with st.expander(
                        f"{icon} **{sym}** — {rating} ({desc}) | Data: "
                        f"{d.get('data_source', 'N/A')}",
                        expanded=expand_default,
                    ):
                        # Escape `$` so Streamlit's markdown renderer doesn't
                        # interpret $46.02M ... $27.77M as a LaTeX math block
                        # and italicize/concatenate everything between them.
                        raw = d.get('analysis', 'No analysis available.')
                        safe = raw.replace('$', '\\$')
                        st.markdown(safe)

                    if rating in ('CLEAN', 'MINOR'):
                        passed.append(sym)

                st.markdown("---")
                final_candidates = list(st.session_state.agent_results['anomalies'].keys())
                final_selection = st.multiselect(
                    "Select for final report:",
                    final_candidates,
                    passed if passed else final_candidates[:3]
                )
                st.session_state.workflow_data['final'] = final_selection

                if final_selection:
                    st.success(f"✅ {len(final_selection)} companies ready for report")
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col2:
                        if st.button("← Back", use_container_width=True):
                            st.session_state.current_step = 1
                            st.rerun()
                    with col3:
                        if st.button("➡️ Proceed to Summary Report", type="primary", use_container_width=True):
                            st.session_state.current_step = 3
                            st.rerun()

    # ===========================================
    # STEP 3: GENERATE SUMMARY REPORT
    # ===========================================
    elif current == 3:
        final = st.session_state.workflow_data.get('final', [])
        if not final:
            st.warning("⚠️ No companies selected. Go back to Anomaly Analysis.")
            if st.button("← Back to Anomaly Analysis"):
                st.session_state.current_step = 2
                st.rerun()
        else:
            # Build detailed report data
            report_data = []
            for sym in final:
                df = st.session_state.workflow_data.get('filtered_df')
                anom = st.session_state.agent_results.get('anomalies', {}).get(sym, {})
                stock = df[df['Symbol'] == sym].iloc[0].to_dict() if df is not None and sym in df['Symbol'].values else {}
                report_data.append({
                    'symbol': sym,
                    'company': stock.get('Company', 'N/A'),
                    'sector': stock.get('Sector', 'N/A'),
                    'industry': stock.get('Industry', 'N/A'),
                    'subindustry': stock.get('Subindustry', 'N/A'),
                    'exchange': stock.get('Exchange', 'N/A'),
                    'currency': stock.get('Currency', 'USD'),
                    'valuation': stock.get('Valuation', 'N/A'),
                    'ai_rating': anom.get('rating', 'N/A'),
                    'ai_analysis': anom.get('analysis', ''),
                    'data_source': anom.get('data_source', 'N/A'),
                    'roe': stock.get('ROE %'),
                    'roa': stock.get('ROA %'),
                    'gross_margin': stock.get('Gross Margin %'),
                    'net_margin': stock.get('Net Margin %'),
                    'debt_equity': stock.get('Debt-to-Equity'),
                    'fcf_margin': stock.get('FCF Margin %'),
                    'roic_wacc': stock.get('ROIC-WACC'),
                    'rote_wacc': stock.get('ROTE-WACC'),
                    'rev_growth': stock.get('5-Year Revenue Growth Rate (Per Share)'),
                    'eps_growth': stock.get('5-Year EPS without NRI Growth Rate'),
                    'epv': stock.get('Earnings Power Value (EPV)'),
                    'market_cap': stock.get('Market Cap ($M)'),
                    'current_price': stock.get('Current Price'),
                    'pe_ratio': stock.get('PE Ratio (TTM)'),
                    'fcf_growth': stock.get('5-Year FCF Growth Rate (Per Share)'),
                    'low_iv': stock.get('Low IV'),
                    'high_iv': stock.get('High IV'),
                })

            # Render professional summary report dashboard
            from summary_report import render_summary_report
            render_summary_report(report_data)

            st.markdown("---")

            # ============================================================
            # COMPETITOR COMPARISON (financials of company vs peers)
            # ============================================================
            st.markdown("### 📊 Competitor Comparison")
            st.caption("Compare each final candidate against peers in the same Sector + Industry from your screening universe.")

            # Use the full uploaded universe (screener_df) for peer search so
            # every company finds real sector peers, not just the screened few.
            full_universe = st.session_state.workflow_data.get(
                'screener_df',
                st.session_state.workflow_data.get('filtered_df'),
            )
            if full_universe is None or full_universe.empty:
                st.info("No screening universe available — re-run Step 1 to enable competitor analysis.")
            else:
                from peer_finder import find_peers, build_peer_metrics_frame, peer_search_summary
                from chart_engine import make_competitor_bar_chart_plotly

                comparable_symbols = [d['symbol'] for d in report_data]
                if not comparable_symbols:
                    st.info("No companies selected for comparison.")
                else:
                    selected_for_compare = st.radio(
                        "Choose a company to compare against peers:",
                        options=comparable_symbols,
                        format_func=lambda s: f"{s} — {next((d['company'] for d in report_data if d['symbol'] == s), s)}",
                        horizontal=True,
                        key="competitor_compare_select",
                    )

                    target_rows = full_universe[full_universe['Symbol'] == selected_for_compare]
                    if target_rows.empty:
                        st.warning(f"Could not find {selected_for_compare} in the screening data.")
                    else:
                        target_row = target_rows.iloc[0]
                        peers = find_peers(selected_for_compare, full_universe, limit=5)
                        st.markdown(
                            f"**{peer_search_summary(selected_for_compare, peers, target_row.get('Industry'), target_row.get('Subindustry'))}**"
                        )

                        if peers.empty:
                            st.info("No peers found in the screening universe — try widening your screening criteria to include more companies in the same industry.")
                        else:
                            # Side-by-side metrics table
                            metrics_df = build_peer_metrics_frame(target_row, peers)
                            display_df = metrics_df.copy()
                            display_df.insert(0, 'Company', [
                                full_universe[full_universe['Symbol'] == s].iloc[0].get('Company', '')[:30]
                                if s in full_universe['Symbol'].values else ''
                                for s in display_df.index
                            ])

                            st.markdown("**Metrics side-by-side**")
                            st.dataframe(display_df.round(2), use_container_width=True)

                            # Bar charts — 5 key metrics
                            chart_specs = [
                                ('ROE %', 'Return on Equity', True),
                                ('Net Margin %', 'Net Margin', True),
                                ('FCF Margin %', 'FCF Margin', True),
                                ('5-Year Revenue Growth Rate (Per Share)', '5-Year Revenue Growth', True),
                                ('Debt-to-Equity', 'Debt-to-Equity', False),
                            ]
                            chart_cols = st.columns(2)
                            for i, (col_name, label, is_pct) in enumerate(chart_specs):
                                fig = make_competitor_bar_chart_plotly(
                                    metric_label=label,
                                    metrics_df=metrics_df,
                                    metric_column=col_name,
                                    target_symbol=selected_for_compare,
                                    is_percentage=is_pct,
                                )
                                if fig is not None:
                                    with chart_cols[i % 2]:
                                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Generate Reports + Navigation
            col1, col2, col3 = st.columns([1.5, 2, 2])
            with col1:
                if st.button("← Back to Anomaly Analysis"):
                    st.session_state.current_step = 2
                    st.rerun()
            with col2:
                if st.button("📄 Generate DOCX Report", type="secondary", use_container_width=True):
                    if not AI_AVAILABLE:
                        st.error("claude-agent-sdk not installed. Run `pip install -r requirements.txt`.")
                    else:
                        try:
                            from enhanced_report import generate_professional_report
                            from datetime import datetime

                            criteria = st.session_state.workflow_data.get('criteria', {})

                            progress_container = st.empty()
                            progress_bar = st.progress(0)

                            def update_progress(message):
                                progress_container.text(message)

                            # Peer comparison searches the FULL uploaded universe
                            # (screener_df, ~1500 stocks) not the ~19 that passed
                            # screening — otherwise most companies have no sector
                            # peers and Section E comes out empty.
                            peer_universe = st.session_state.workflow_data.get(
                                'screener_df',
                                st.session_state.workflow_data.get('filtered_df'),
                            )

                            with st.spinner("Generating AI-enhanced professional report... This may take a few minutes for deep analysis."):
                                buffer = generate_professional_report(
                                    report_data=report_data,
                                    criteria=criteria,
                                    api_key=None,
                                    progress_callback=update_progress,
                                    universe_df=peer_universe,
                                )

                            progress_bar.progress(100)
                            progress_container.empty()

                            # Cache so the download button survives reruns
                            st.session_state['docx_report_bytes'] = buffer.getvalue() if hasattr(buffer, 'getvalue') else buffer
                            st.session_state['docx_report_filename'] = f"Value_Investment_AI_Report_{datetime.now().strftime('%Y%m%d')}.docx"
                        except Exception as e:
                            st.error(f"Error generating DOCX report: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            with col3:
                if st.button("📊 Generate Excel Report (VIA Style)", type="primary", use_container_width=True):
                    try:
                        from excel_report import generate_excel_report
                        from datetime import datetime

                        criteria = st.session_state.workflow_data.get('criteria', {})

                        progress_container = st.empty()
                        progress_bar = st.progress(0)

                        def update_progress_xlsx(message):
                            progress_container.text(message)

                        # Peer comparison searches the FULL uploaded universe
                        # (screener_df) not just the screened subset, so every
                        # company finds real sector peers for Section E.
                        peer_universe = st.session_state.workflow_data.get(
                            'screener_df',
                            st.session_state.workflow_data.get('filtered_df'),
                        )

                        with st.spinner("Preparing your investment report — this may take a few minutes while we gather the latest financial data for each company."):
                            xlsx_buffer = generate_excel_report(
                                report_data=report_data,
                                criteria=criteria,
                                progress_callback=update_progress_xlsx,
                                universe_df=peer_universe,
                            )

                        progress_bar.progress(100)
                        progress_container.empty()

                        # Cache so the download button survives reruns
                        st.session_state['xlsx_report_bytes'] = xlsx_buffer.getvalue() if hasattr(xlsx_buffer, 'getvalue') else xlsx_buffer
                        st.session_state['xlsx_report_filename'] = f"Value_Investment_AI_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
                    except Exception as e:
                        st.error(f"Error generating Excel report: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            # Persistent download buttons (rendered outside click blocks so they survive reruns)
            if st.session_state.get('docx_report_bytes') or st.session_state.get('xlsx_report_bytes'):
                st.markdown("---")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    if st.session_state.get('docx_report_bytes'):
                        st.download_button(
                            label="📥 Download DOCX Report",
                            data=st.session_state['docx_report_bytes'],
                            file_name=st.session_state['docx_report_filename'],
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                            key="dl_docx",
                        )
                        st.caption("✅ DOCX report ready.")
                with dl_col2:
                    if st.session_state.get('xlsx_report_bytes'):
                        st.download_button(
                            label="📥 Download Excel Report (VIA Style)",
                            data=st.session_state['xlsx_report_bytes'],
                            file_name=st.session_state['xlsx_report_filename'],
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="dl_xlsx",
                        )
                        st.caption("✅ Excel report ready.")


def show_screener_page():
    st.header("🔍 Stock Screener")

    # File upload section
    st.subheader("1. Upload Data")
    col1, col2 = st.columns(2)

    with col1:
        us_file = st.file_uploader(
            "Upload US Screener CSV",
            type=['csv'],
            key="us_upload",
            help="Upload 'US All In One Screeners' CSV file"
        )

    with col2:
        sg_file = st.file_uploader(
            "Upload SG Screener CSV",
            type=['csv'],
            key="sg_upload",
            help="Upload 'SG All In One Screeners' CSV file"
        )

    # Market selection
    market = st.radio("Select Market", ["US", "SG"], horizontal=True)

    # Check if appropriate file is uploaded
    selected_file = us_file if market == "US" else sg_file
    if selected_file is None:
        st.info(f"Please upload the {market} screener CSV file to continue.")
        return

    # Load data
    try:
        df = pd.read_csv(selected_file, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        st.success(f"Loaded {len(df)} {market} stocks")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return

    # Screening criteria
    st.subheader("2. Set Screening Criteria")
    st.markdown("*Adjust the sliders to set your filtering thresholds*")

    # Profitability Metrics
    st.markdown("#### 📊 Profitability Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gross_margin = st.slider("Min Gross Margin %", 0, 100, 20, help="Minimum gross margin percentage")
        net_margin = st.slider("Min Net Margin %", -50, 100, 5, help="Minimum net margin percentage")

    with col2:
        roa = st.slider("Min ROA %", -20, 50, 5, help="Minimum return on assets")
        roe = st.slider("Min ROE %", -20, 100, 10, help="Minimum return on equity")

    with col3:
        roic = st.slider("Min ROIC %", -20, 50, 8, help="Minimum return on invested capital")
        operating_margin = st.slider("Min Operating Margin %", -50, 100, 5, help="Minimum operating margin")

    with col4:
        fcf_margin = st.slider("Min FCF Margin %", -50, 100, 0, help="Minimum free cash flow margin")
        roic_wacc = st.slider("Min ROIC-WACC", -20, 50, 0, help="Minimum ROIC minus WACC")

    # Growth Metrics
    st.markdown("#### 📈 Growth Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        revenue_growth = st.slider("Min 5Y Revenue Growth %", -50, 100, 0, help="Minimum 5-year revenue growth rate")

    with col2:
        eps_growth = st.slider("Min 5Y EPS Growth %", -50, 100, 0, help="Minimum 5-year EPS growth rate")

    with col3:
        fcf_growth = st.slider("Min 5Y FCF Growth %", -100, 200, 0, help="Minimum 5-year FCF growth rate")

    with col4:
        rote_wacc = st.slider("Min ROTE-WACC", -50, 100, 0, help="Minimum ROTE minus WACC")

    # Balance Sheet Metrics
    st.markdown("#### 💰 Balance Sheet & Valuation Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        debt_equity = st.slider("Max Debt-to-Equity", 0.0, 5.0, 1.5, 0.1, help="Maximum debt-to-equity ratio")

    with col2:
        current_ratio = st.slider("Min Current Ratio", 0.0, 5.0, 1.0, 0.1, help="Minimum current ratio (liquidity)")

    with col3:
        pe_ratio = st.slider("Max P/E Ratio", 0, 100, 50, help="Maximum price-to-earnings ratio (0 = no filter)")

    with col4:
        pb_ratio = st.slider("Max P/B Ratio", 0.0, 20.0, 5.0, 0.5, help="Maximum price-to-book ratio (0 = no filter)")

    # Build criteria dict for AI report
    criteria_dict = {
        'gross_margin': gross_margin,
        'net_margin': net_margin,
        'operating_margin': operating_margin,
        'roa': roa,
        'roe': roe,
        'roic': roic,
        'revenue_growth_5y': revenue_growth,
        'eps_growth_5y': eps_growth,
        'fcf_growth_5y': fcf_growth,
        'debt_to_equity': debt_equity,
        'current_ratio': current_ratio,
        'pe_ratio': pe_ratio if pe_ratio > 0 else None,
        'pb_ratio': pb_ratio if pb_ratio > 0 else None,
        'fcf_margin': fcf_margin,
        'roic_wacc': roic_wacc,
        'rote_wacc': rote_wacc
    }

    # Column mapping for filtering (column_name: (operator, threshold, apply_filter))
    criteria_mapping = {
        'Gross Margin %': ('>=', gross_margin, True),
        'Net Margin %': ('>=', net_margin, True),
        'Operating Margin %': ('>=', operating_margin, True),
        'ROA %': ('>=', roa, True),
        'ROE %': ('>=', roe, True),
        'ROIC %': ('>=', roic, True),
        '5-Year Revenue Growth Rate (Per Share)': ('>=', revenue_growth, True),
        '5-Year EPS without NRI Growth Rate': ('>=', eps_growth, True),
        '5-Year FCF Growth Rate': ('>=', fcf_growth, True),
        'Debt-to-Equity': ('<=', debt_equity, True),
        'Current Ratio': ('>=', current_ratio, True),
        'PE Ratio': ('<=', pe_ratio, pe_ratio > 0),  # Only apply if not 0
        'PB Ratio': ('<=', pb_ratio, pb_ratio > 0),  # Only apply if not 0
        'FCF Margin %': ('>=', fcf_margin, True),
        'ROIC-WACC': ('>=', roic_wacc, True),
        'ROTE-WACC': ('>=', rote_wacc, True)
    }

    # Apply filters
    if st.button("🔍 Screen Stocks", type="primary"):
        filtered_df = df.copy()

        for col, (op, threshold, apply_filter) in criteria_mapping.items():
            if apply_filter and col in filtered_df.columns:
                filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                if op == '>=':
                    filtered_df = filtered_df[filtered_df[col] >= threshold]
                else:  # <=
                    filtered_df = filtered_df[(filtered_df[col] <= threshold) & (filtered_df[col] >= 0)]

        # Add valuation
        if 'Earnings Power Value (EPV)' in filtered_df.columns and 'Market Cap ($M)' in filtered_df.columns:
            filtered_df['Earnings Power Value (EPV)'] = pd.to_numeric(filtered_df['Earnings Power Value (EPV)'], errors='coerce')
            filtered_df['Market Cap ($M)'] = pd.to_numeric(filtered_df['Market Cap ($M)'], errors='coerce')

            epv_mc = filtered_df['Earnings Power Value (EPV)'] / filtered_df['Market Cap ($M)']
            epv_mc = epv_mc.replace([float('inf'), float('-inf')], pd.NA)
            filtered_df['EPV/MC Ratio'] = epv_mc

            def classify_valuation(ratio):
                if pd.isna(ratio) or ratio is None:
                    return 'N/A'
                if ratio <= 0:
                    return 'N/A (Negative EPV)'
                if ratio > 1.3:
                    return 'Undervalued'
                if ratio >= 0.7:
                    return 'Fair Value'
                return 'Overvalued'

            filtered_df['Valuation'] = epv_mc.apply(classify_valuation)

        # Store in session state for AI analysis
        st.session_state['screened_stocks'] = filtered_df
        st.session_state['screening_criteria'] = criteria_dict

        # Display results
        st.subheader("3. Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(filtered_df))
        with col2:
            if 'Valuation' in filtered_df.columns:
                undervalued = len(filtered_df[filtered_df['Valuation'] == 'Undervalued'])
            else:
                undervalued = 0
            st.metric("Undervalued", undervalued)
        with col3:
            if 'Valuation' in filtered_df.columns:
                fair = len(filtered_df[filtered_df['Valuation'] == 'Fair Value'])
            else:
                fair = 0
            st.metric("Fair Value", fair)
        with col4:
            if 'Valuation' in filtered_df.columns:
                overvalued = len(filtered_df[filtered_df['Valuation'] == 'Overvalued'])
            else:
                overvalued = 0
            st.metric("Overvalued", overvalued)

        if len(filtered_df) > 0:
            # Display columns
            display_cols = ['Symbol', 'Company', 'Sector', 'Gross Margin %', 'Net Margin %',
                          'ROE %', 'ROA %', 'Debt-to-Equity', 'FCF Margin %',
                          'ROIC-WACC', 'ROTE-WACC', 'Market Cap ($M)', 'EPV/MC Ratio', 'Valuation']
            available_cols = [c for c in display_cols if c in filtered_df.columns]

            # Sort by valuation
            if 'Valuation' in filtered_df.columns:
                val_order = {'Undervalued': 0, 'Fair Value': 1, 'Overvalued': 2, 'N/A': 3, 'N/A (Negative EPV)': 4}
                filtered_df['_sort'] = filtered_df['Valuation'].map(val_order)
                filtered_df = filtered_df.sort_values('_sort').drop('_sort', axis=1)

            st.dataframe(
                filtered_df[available_cols],
                use_container_width=True,
                height=400
            )

            # Download button
            csv = filtered_df[available_cols].to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "screened_stocks.csv",
                "text/csv"
            )
        else:
            st.warning("No stocks match the current criteria. Try relaxing the filters.")

    # AI Agent Analysis Section
    if 'screened_stocks' in st.session_state and len(st.session_state['screened_stocks']) > 0:
        st.markdown("---")
        st.subheader("🤖 AI Agent Analysis")

        agent = get_ai_agent("screening")

        if agent:
            # Chat interface using form for proper state handling
            with st.form("screening_ai_form", clear_on_submit=False):
                user_question = st.text_input(
                    "Ask the AI agent about your screening results:",
                    placeholder="e.g., Which stocks look most promising? What are the key risks?",
                    key="screening_question"
                )
                submitted = st.form_submit_button("✨ Ask AI Agent", type="secondary")

            if submitted:
                if user_question and user_question.strip():
                    with st.spinner("AI agent is analyzing..."):
                        try:
                            # Prepare context
                            top_stocks = st.session_state['screened_stocks'].head(10).to_dict('records')
                            context = f"""Based on screening results with {len(st.session_state['screened_stocks'])} matches.
Top stocks: {json.dumps(top_stocks[:5], default=str)}
Criteria: {st.session_state['screening_criteria']}

User question: {user_question}"""

                            response = agent.chat(context)

                            st.markdown("### 📊 AI Agent Response")
                            st.markdown(f'<div class="ai-analysis">{response.content}</div>', unsafe_allow_html=True)

                            # Show tool calls if any
                            if response.tool_calls:
                                with st.expander("🔧 Tools Used"):
                                    for tc in response.tool_calls:
                                        st.json(tc)

                        except Exception as e:
                            st.error(f"AI agent error: {e}")
                else:
                    st.warning("Please enter a question for the AI agent.")
        else:
            st.info("💡 Install claude-agent-sdk and run `claude /login` to enable AI agent features.")


def show_anomaly_page():
    st.header("🔎 Anomaly Detector")
    st.markdown("*Detect financial distortions and red flags in company data*")

    # File upload
    anomaly_file = st.file_uploader(
        "Upload Financial Data (XLS)",
        type=['xls', 'xlsx'],
        help="Upload 'Companies with anomalies' XLS file with 30-year financials"
    )

    if anomaly_file is None:
        st.info("Please upload the financial data XLS file to analyze companies for anomalies.")

        # Show what the tool detects
        st.subheader("What This Tool Detects")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Quality Metrics**
            - 🔴 Beneish M-Score (earnings manipulation)
            - 🟡 Altman Z-Score (bankruptcy risk)
            - 🟢 Piotroski F-Score (financial strength)
            - 📊 Sloan Ratio (accrual quality)
            """)

        with col2:
            st.markdown("""
            **Anomaly Types**
            - Revenue spikes/drops (>50%)
            - Margin volatility (>10pp change)
            - EPS vs EPS w/o NRI discrepancy
            - Cash flow mismatches
            - Receivables/Inventory buildup
            """)
        return

    # Try to read the file
    try:
        import xlrd
        # Save uploaded file with correct name that DataLoader expects
        temp_file_path = "/tmp/Companies with anomalies.xls"
        with open(temp_file_path, "wb") as f:
            f.write(anomaly_file.getvalue())

        book = xlrd.open_workbook(temp_file_path, ignore_workbook_corruption=True)
        available_symbols = []
        for sheet_name in book.sheet_names():
            parts = sheet_name.split('_')
            if len(parts) >= 2:
                available_symbols.append(parts[1])

        st.success(f"Found {len(available_symbols)} companies: {', '.join(available_symbols)}")

        # Symbol selection
        selected_symbol = st.selectbox("Select Company to Analyze", available_symbols)

        if st.button("🔍 Analyze for Anomalies", type="primary"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                # Create custom data loader that uses the uploaded file
                loader = DataLoader("/tmp")
                detector = AnomalyDetector(loader)
                report = detector.analyze(selected_symbol)

            # Store for AI analysis
            st.session_state['anomaly_report'] = report
            st.session_state['anomaly_detector'] = detector

            # Display results
            st.subheader(f"Anomaly Report: {report.symbol}")

            # Risk level with color
            risk_colors = {
                "HIGH RISK": "🔴",
                "ELEVATED RISK": "🟠",
                "MODERATE RISK": "🟡",
                "LOW RISK": "🟢",
                "MINIMAL RISK": "✅"
            }
            risk_icon = risk_colors.get(report.risk_level, "⚪")
            st.markdown(f"### {risk_icon} Overall Risk: **{report.risk_level}**")

            # Quality scores
            st.subheader("Quality Scores")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if report.m_score is not None:
                    status = "🔴 ALERT" if report.m_score > -1.78 else "🟢 OK"
                    st.metric("M-Score", f"{report.m_score:.2f}", status)
                else:
                    st.metric("M-Score", "N/A")

            with col2:
                if report.z_score is not None:
                    status = "🔴 ALERT" if report.z_score < 1.8 else "🟢 OK"
                    st.metric("Z-Score", f"{report.z_score:.2f}", status)
                else:
                    st.metric("Z-Score", "N/A")

            with col3:
                if report.f_score is not None:
                    status = "🔴 ALERT" if report.f_score < 3 else "🟢 OK"
                    st.metric("F-Score", f"{report.f_score:.0f}", status)
                else:
                    st.metric("F-Score", "N/A")

            with col4:
                if report.sloan_ratio is not None:
                    status = "🔴 ALERT" if abs(report.sloan_ratio) > 10 else "🟢 OK"
                    st.metric("Sloan Ratio", f"{report.sloan_ratio:.1f}%", status)
                else:
                    st.metric("Sloan Ratio", "N/A")

            # Anomalies by severity
            st.subheader(f"Detected Anomalies ({report.total_anomalies} total)")

            if report.anomalies:
                # Group by severity
                for severity in [Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
                    anomalies = [a for a in report.anomalies if a.severity == severity]
                    if anomalies:
                        severity_icons = {Severity.HIGH: "🔴", Severity.MEDIUM: "🟡", Severity.LOW: "🟢"}
                        with st.expander(f"{severity_icons[severity]} {severity.value} Severity ({len(anomalies)})", expanded=(severity == Severity.HIGH)):
                            for a in anomalies:
                                year_info = f" ({a.year})" if a.year else ""
                                st.markdown(f"**[{a.category}]{year_info}**")
                                st.markdown(f"  {a.description}")
                                if a.details:
                                    st.caption(f"  → {a.details}")
                                st.divider()
            else:
                st.success("No significant anomalies detected!")

            # Download report
            report_text = detector.format_report(report)
            st.download_button(
                "📥 Download Full Report",
                report_text,
                f"{selected_symbol}_anomaly_report.txt",
                "text/plain"
            )

        # AI Agent Analysis for Anomalies
        if 'anomaly_report' in st.session_state:
            report = st.session_state['anomaly_report']

            st.markdown("---")
            st.subheader("🤖 AI Agent Interpretation")

            agent = get_ai_agent("anomaly")

            if agent:
                # Chat interface using form for proper state handling
                with st.form("anomaly_ai_form", clear_on_submit=False):
                    user_question = st.text_input(
                        "Ask the AI agent about these anomalies:",
                        placeholder="e.g., Should I be concerned? What should I investigate further?",
                        key="anomaly_question"
                    )
                    submitted = st.form_submit_button("✨ Ask AI Agent", type="secondary")

                if submitted:
                    question = user_question.strip() if user_question else f"Analyze {report.symbol} for financial anomalies and explain the key risks."

                    with st.spinner("AI agent is analyzing..."):
                        try:
                            response = agent.chat(question)

                            st.markdown("### 🔬 AI Agent Analysis")
                            st.markdown(f'<div class="ai-analysis">{response.content}</div>', unsafe_allow_html=True)

                            # Show tool calls if any
                            if response.tool_calls:
                                with st.expander("🔧 Tools Used"):
                                    for tc in response.tool_calls:
                                        st.json(tc)

                        except Exception as e:
                            st.error(f"AI agent error: {e}")
            else:
                st.info("💡 Install claude-agent-sdk and run `claude /login` to enable AI agent features.")

    except Exception as e:
        st.error(f"Error reading file: {e}")


def show_chatbot_page():
    st.header("💬 AI Chatbot")
    st.markdown("*Chat with AI agents to analyze stocks, detect anomalies, and get investment insights*")

    # Check if AI is available
    agent = get_ai_agent("general")
    if not agent:
        st.warning("⚠️ AI features require **claude-agent-sdk** + Claude Code CLI authenticated.")
        st.info("""
        **How to enable AI (one-time setup):**
        1. Install Node.js
        2. Install Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
        3. Run `claude /login` and complete OAuth (uses your Claude subscription)
        4. Install Python deps: `pip install -r requirements.txt`
        """)
        return

    # Agent selection
    col1, col2 = st.columns([1, 3])
    with col1:
        agent_type = st.selectbox(
            "Select Agent",
            ["general", "screening", "anomaly"],
            format_func=lambda x: {
                "general": "🤖 General Agent",
                "screening": "📊 Screening Agent",
                "anomaly": "🔍 Anomaly Agent"
            }.get(x, x),
            help="Choose which specialized agent to chat with"
        )

    with col2:
        st.markdown("""
        | Agent | Specialization |
        |-------|----------------|
        | General | All-purpose value investing assistant |
        | Screening | Find value stocks, analyze criteria |
        | Anomaly | Detect financial red flags, interpret M-Score/Z-Score |
        """)

    # Initialize chat history in session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    if 'chat_agent_type' not in st.session_state:
        st.session_state.chat_agent_type = agent_type

    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = None

    # Reset chat if agent type changes
    if st.session_state.chat_agent_type != agent_type:
        st.session_state.chat_agent_type = agent_type
        st.session_state.chat_messages = []
        st.session_state.chat_agent = None

    # Get or create agent (maintains conversation history)
    if st.session_state.chat_agent is None:
        st.session_state.chat_agent = get_ai_agent(agent_type)

    agent = st.session_state.chat_agent

    # Clear chat button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", help="Clear conversation history"):
            st.session_state.chat_messages = []
            st.session_state.chat_agent = get_ai_agent(agent_type)
            st.rerun()

    st.divider()

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show tool calls if present
            if message.get("tool_calls"):
                with st.expander("🔧 Tools Used"):
                    for tc in message["tool_calls"]:
                        st.json(tc)

    # Chat input
    if prompt := st.chat_input("Ask about stocks, valuations, or anomalies..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Include context from screened stocks if available
                    context_prompt = prompt
                    if 'screened_stocks' in st.session_state and len(st.session_state['screened_stocks']) > 0:
                        stocks_summary = st.session_state['screened_stocks'].head(5).to_dict('records')
                        criteria = st.session_state.get('screening_criteria', {})
                        context_prompt = f"""Context: User has screened stocks with these results:
- {len(st.session_state['screened_stocks'])} stocks matched
- Top stocks: {json.dumps(stocks_summary, default=str)}
- Criteria used: {criteria}

User question: {prompt}"""

                    # Include anomaly report context if available
                    if 'anomaly_report' in st.session_state:
                        report = st.session_state['anomaly_report']
                        context_prompt = f"""Context: User is analyzing {report.symbol}
- Risk Level: {report.risk_level}
- M-Score: {report.m_score}
- Z-Score: {report.z_score}
- F-Score: {report.f_score}
- Total Anomalies: {report.total_anomalies}

User question: {prompt}"""

                    response = agent.chat(context_prompt)

                    # Display response
                    st.markdown(response.content)

                    # Show tool calls if any
                    tool_calls_data = []
                    if response.tool_calls:
                        with st.expander("🔧 Tools Used"):
                            for tc in response.tool_calls:
                                st.json(tc)
                                tool_calls_data.append(tc)

                    # Add assistant message to chat history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": tool_calls_data
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Suggested prompts
    if not st.session_state.chat_messages:
        st.markdown("### 💡 Suggested Questions")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Screening:**
            - "Find undervalued US stocks with ROE > 15%"
            - "What stocks have low debt and high margins?"
            - "Compare AAPL and MSFT fundamentals"
            """)

        with col2:
            st.markdown("""
            **Anomaly Analysis:**
            - "Analyze DDI for financial red flags"
            - "What does an M-Score of -1.5 mean?"
            - "Explain the Altman Z-Score calculation"
            """)


def show_about_page():
    st.header("ℹ️ About")

    st.markdown("""
    ## Value Investment Tool

    A comprehensive tool for value investors to screen stocks and detect financial anomalies, now with **AI-powered analysis**.

    ### Features

    **1. Stock Screener**
    - Filter by 10 fundamental criteria
    - Support for US and Singapore markets
    - Automatic valuation using EPV vs Market Cap
    - 🤖 **AI Analysis**: Get intelligent insights and recommendations

    **2. Anomaly Detector**
    - Beneish M-Score analysis (earnings manipulation)
    - Altman Z-Score (bankruptcy risk)
    - Piotroski F-Score (financial strength)
    - One-off event detection
    - Cash flow consistency checks
    - 🤖 **AI Interpretation**: Understand what anomalies mean

    ### AI Agent Features (requires Claude Code CLI authenticated)

    - **Screening Agent**: Specialized AI that helps find value stocks and explains picks
    - **Anomaly Agent**: Forensic AI that interprets financial red flags
    - **Research Agent**: AI that generates investment theses and recommendations
    - **Tool Use**: Agents can call tools to screen stocks, detect anomalies, compare companies

    ### Valuation Classification (In-House Two-Stage DCF)

    Each stock gets a per-share **IV range** [Low IV, High IV] from the client's
    in-house DCF formula (10% discount, 2% terminal growth, 30% margin of safety):

    | Classification | Condition |
    |----------------|-----------|
    | 🟢 Undervalued | Current Price < **Low IV** |
    | 🟡 Fair Value | **Low IV** ≤ Current Price ≤ **High IV** |
    | 🔴 Overvalued | Current Price > **High IV** |

    ### How to Enable AI

    1. Install Node.js, then `npm install -g @anthropic-ai/claude-code`
    2. Run `claude /login` once to authenticate with your Claude subscription
    3. AI features are enabled automatically — no API key needed

    ---
    Built with Streamlit & Claude | [GitHub Repository](https://github.com/alfredang/value-investment)
    """)


def show_admin_page():
    """Show the admin configuration page."""
    if not ADMIN_AVAILABLE:
        st.error("Admin module not available. Please check the installation.")
        return

    # Check authentication
    if not st.session_state.get('admin_authenticated', False):
        show_admin_login()
    else:
        show_admin_panel()


if __name__ == "__main__":
    main()
