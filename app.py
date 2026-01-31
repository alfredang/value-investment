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
try:
    from ai_agents import ValueInvestmentAgent, ScreeningAgent, AnomalyAgent, ResearchAgent
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

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

# Page configuration
st.set_page_config(
    page_title="Value Investment Academy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - theme compatible
st.markdown("""
<style>
    .stMetric {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stMetric label {
        color: inherit !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: inherit !important;
    }
    .ai-analysis {
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
        background-color: rgba(0, 102, 204, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def init_config():
    """Initialize configuration from admin settings."""
    # First, check for API key in environment (from .env file)
    env_api_key = os.getenv('OPENAI_API_KEY', '')
    if env_api_key:
        st.session_state['openai_api_key'] = env_api_key

    if ADMIN_AVAILABLE:
        api_keys = get_api_keys()
        apply_api_keys_to_env(api_keys)

        # Also set in session state if available from config
        if api_keys.openai_api_key:
            st.session_state['openai_api_key'] = api_keys.openai_api_key

        # Load LLM config into session state
        config = load_config()
        if 'llm_model' not in st.session_state:
            st.session_state['llm_model'] = config.llm_config.model
        if 'llm_temperature' not in st.session_state:
            st.session_state['llm_temperature'] = config.llm_config.temperature
        if 'analysis_config' not in st.session_state:
            st.session_state['analysis_config'] = config.analysis_config


def get_ai_agent(agent_type: str = "general"):
    """Get AI agent instance if API key is available."""
    # Check for API key in session state or environment
    api_key = st.session_state.get('openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')

    if api_key and AI_AVAILABLE:
        try:
            # Set the API key in environment for agents
            os.environ['OPENAI_API_KEY'] = api_key

            if agent_type == "screening":
                return ScreeningAgent()
            elif agent_type == "anomaly":
                return AnomalyAgent()
            else:
                return ValueInvestmentAgent()
        except Exception as e:
            st.error(f"Error initializing AI agent: {e}")
    return None


def show_persistent_chat():
    """Show persistent AI chat at the bottom of every page."""
    st.markdown("---")

    # Expandable chat section
    with st.expander("💬 AI Assistant", expanded=False):
        # Initialize chat history
        if 'persistent_chat_messages' not in st.session_state:
            st.session_state.persistent_chat_messages = []

        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.persistent_chat_messages[-5:]:  # Show last 5 messages
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")

        # Chat input
        agent = get_ai_agent("general")
        if agent:
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

                # Get AI response
                with st.spinner("Thinking..."):
                    try:
                        # Add context from current session
                        context = user_input
                        if 'screened_stocks' in st.session_state and len(st.session_state['screened_stocks']) > 0:
                            context = f"Context: User has {len(st.session_state['screened_stocks'])} screened stocks. Question: {user_input}"

                        response = agent.chat(context)
                        st.session_state.persistent_chat_messages.append({
                            "role": "assistant",
                            "content": response.content
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
            # Show appropriate message based on what's missing
            api_key = st.session_state.get('openai_api_key', '') or os.getenv('OPENAI_API_KEY', '')
            if not AI_AVAILABLE:
                st.warning("AI agents module not available. Check openai-agents package installation.")
            elif not api_key:
                st.info("Enter OpenAI API key in Settings to enable AI Assistant")
            else:
                st.warning("AI initialization failed. Check API key validity.")


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
        st.title("📈 Value Investment Academy")
    with col2:
        if ADMIN_AVAILABLE:
            if st.button("⚙️", help="Settings"):
                st.session_state['show_settings'] = True
                st.rerun()

    st.markdown("*AI-powered stock screening, anomaly detection, and investment analysis*")

    # Show the tab-based workflow
    show_tabbed_workflow()

    # Show persistent chat at the bottom
    show_persistent_chat()


def show_tabbed_workflow():
    """Show the 3-step workflow with tabs - each tab runs an agent."""

    # Initialize session state
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = {}
    if 'agent_results' not in st.session_state:
        st.session_state.agent_results = {}

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Step 1: Screen Stocks",
        "🔍 Step 2: Anomaly Review",
        "📝 Step 3: Report"
    ])

    # ===========================================
    # TAB 1: SCREEN STOCKS
    # ===========================================
    with tab1:
        st.markdown("### Upload data and set screening criteria")

        col1, col2 = st.columns(2)
        with col1:
            screener_file = st.file_uploader("Screener CSV", type=['csv'], key="tab_screener")
            if screener_file:
                try:
                    df = pd.read_csv(screener_file, encoding='utf-8-sig')
                    df.columns = df.columns.str.strip()
                    st.session_state.workflow_data['screener_df'] = df
                    st.success(f"✓ Loaded {len(df)} stocks")
                    market = st.radio("Market", ["US", "SG"], horizontal=True)
                    st.session_state.workflow_data['market'] = market
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            fin_file = st.file_uploader("Financials XLS (for anomalies)", type=['xls', 'xlsx'], key="tab_fin")
            if fin_file:
                try:
                    import xlrd
                    with open("/tmp/Companies with anomalies.xls", "wb") as f:
                        f.write(fin_file.getvalue())
                    book = xlrd.open_workbook("/tmp/Companies with anomalies.xls", ignore_workbook_corruption=True)
                    symbols = [s.split('_')[1] for s in book.sheet_names() if '_' in s]
                    st.session_state.workflow_data['available_symbols'] = symbols
                    st.success(f"✓ {len(symbols)} companies available")
                except Exception as e:
                    st.error(f"Error: {e}")

        if 'screener_df' in st.session_state.workflow_data:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                gm = st.slider("Min Gross Margin %", 0, 100, 20)
                roe = st.slider("Min ROE %", -20, 100, 10)
                de = st.slider("Max Debt/Equity", 0.0, 5.0, 1.5, 0.1)
            with col2:
                fcf = st.slider("Min FCF Margin %", -50, 100, 0)
                roic = st.slider("Min ROIC-WACC", -20, 50, 0)

            if st.button("🔍 Screen Stocks", type="primary"):
                df = st.session_state.workflow_data['screener_df'].copy()
                # Apply filters
                for col, op, val in [('Gross Margin %', '>=', gm), ('ROE %', '>=', roe),
                                      ('Debt-to-Equity', '<=', de), ('FCF Margin %', '>=', fcf),
                                      ('ROIC-WACC', '>=', roic)]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df = df[df[col] >= val] if op == '>=' else df[(df[col] <= val) & (df[col] >= 0)]

                # Add valuation
                if 'Earnings Power Value (EPV)' in df.columns and 'Market Cap ($M)' in df.columns:
                    df['EPV/MC'] = pd.to_numeric(df['Earnings Power Value (EPV)'], errors='coerce') / pd.to_numeric(df['Market Cap ($M)'], errors='coerce')
                    df['Valuation'] = df['EPV/MC'].apply(lambda x: 'Undervalued' if x and x > 1.3 else 'Fair' if x and x >= 0.7 else 'Overvalued' if x else 'N/A')

                st.session_state.workflow_data['filtered_df'] = df
                st.session_state.agent_results['screened'] = True

        if st.session_state.agent_results.get('screened'):
            df = st.session_state.workflow_data['filtered_df']
            st.metric("Matches", len(df))
            cols = ['Symbol', 'Company', 'Valuation', 'ROE %', 'Debt-to-Equity']
            st.dataframe(df[[c for c in cols if c in df.columns]].head(30), use_container_width=True)

            available = st.session_state.workflow_data.get('available_symbols', [])
            analyzable = [s for s in df['Symbol'].tolist() if s in available]
            if analyzable:
                selected = st.multiselect("Select for anomaly analysis:", analyzable, analyzable[:5])
                st.session_state.workflow_data['selected'] = selected

    # ===========================================
    # TAB 2: ANOMALY REVIEW
    # ===========================================
    with tab2:
        selected = st.session_state.workflow_data.get('selected', [])
        if not selected:
            st.info("Complete Step 1 first")
        else:
            if st.button("🔍 Run Anomaly Detection", type="primary"):
                results = {}
                for sym in selected:
                    try:
                        loader = DataLoader("/tmp")
                        detector = AnomalyDetector(loader)
                        r = detector.analyze(sym)
                        results[sym] = {'m': r.m_score, 'z': r.z_score, 'f': r.f_score, 'risk': r.risk_level,
                                       'high': len([a for a in r.anomalies if a.severity == Severity.HIGH])}
                    except:
                        pass
                st.session_state.agent_results['anomalies'] = results

            if 'anomalies' in st.session_state.agent_results:
                passed = []
                for sym, d in st.session_state.agent_results['anomalies'].items():
                    ok = (d.get('m') or -99) < -1.78 and (d.get('z') or 0) > 1.8 and d.get('high', 99) == 0
                    icon = "✅" if ok else "❌"
                    st.markdown(f"{icon} **{sym}** - M:{d.get('m','N/A'):.1f}, Z:{d.get('z','N/A'):.1f}, Risk:{d.get('risk','N/A')}")
                    if ok:
                        passed.append(sym)
                st.session_state.workflow_data['final'] = passed or list(st.session_state.agent_results['anomalies'].keys())

    # ===========================================
    # TAB 3: REPORT
    # ===========================================
    with tab3:
        final = st.session_state.workflow_data.get('final', [])
        if not final:
            st.info("Complete Step 2 first")
        else:
            st.markdown(f"### Report for {len(final)} companies")
            for sym in final:
                df = st.session_state.workflow_data.get('filtered_df')
                anom = st.session_state.agent_results.get('anomalies', {}).get(sym, {})
                stock = df[df['Symbol'] == sym].iloc[0].to_dict() if df is not None and sym in df['Symbol'].values else {}
                with st.expander(f"📊 {sym} - {stock.get('Company', '')}"):
                    st.markdown(f"- Valuation: {stock.get('Valuation', 'N/A')}")
                    st.markdown(f"- Risk: {anom.get('risk', 'N/A')}")
                    st.markdown(f"- M-Score: {anom.get('m', 'N/A')}")
                    agent = get_ai_agent()
                    if agent and st.button(f"Get AI Analysis for {sym}", key=f"ai_{sym}"):
                        with st.spinner("Analyzing..."):
                            resp = agent.chat(f"Brief investment thesis for {sym}: valuation {stock.get('Valuation')}, risk {anom.get('risk')}")
                            st.markdown(resp.content)


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

    col1, col2, col3 = st.columns(3)

    with col1:
        gross_margin = st.slider("Min Gross Margin %", 0, 100, 20, help="Minimum gross margin percentage")
        net_margin = st.slider("Min Net Margin %", -50, 100, 5, help="Minimum net margin percentage")
        roa = st.slider("Min ROA %", -20, 50, 5, help="Minimum return on assets")
        roe = st.slider("Min ROE %", -20, 100, 10, help="Minimum return on equity")

    with col2:
        revenue_growth = st.slider("Min 5Y Revenue Growth %", -50, 100, 0, help="Minimum 5-year revenue growth rate")
        eps_growth = st.slider("Min 5Y EPS Growth %", -50, 100, 0, help="Minimum 5-year EPS growth rate")
        debt_equity = st.slider("Max Debt-to-Equity", 0.0, 5.0, 1.5, 0.1, help="Maximum debt-to-equity ratio")

    with col3:
        fcf_margin = st.slider("Min FCF Margin %", -50, 100, 0, help="Minimum free cash flow margin")
        roic_wacc = st.slider("Min ROIC-WACC", -20, 50, 0, help="Minimum ROIC minus WACC")
        rote_wacc = st.slider("Min ROTE-WACC", -50, 100, 0, help="Minimum ROTE minus WACC")

    # Build criteria dict for AI
    criteria_dict = {
        'gross_margin': gross_margin,
        'net_margin': net_margin,
        'roa': roa,
        'roe': roe,
        'revenue_growth_5y': revenue_growth,
        'eps_growth_5y': eps_growth,
        'debt_to_equity': debt_equity,
        'fcf_margin': fcf_margin,
        'roic_wacc': roic_wacc,
        'rote_wacc': rote_wacc
    }

    # Column mapping for filtering
    criteria_mapping = {
        'Gross Margin %': ('>=', gross_margin),
        'Net Margin %': ('>=', net_margin),
        'ROA %': ('>=', roa),
        'ROE %': ('>=', roe),
        '5-Year Revenue Growth Rate (Per Share)': ('>=', revenue_growth),
        '5-Year EPS without NRI Growth Rate': ('>=', eps_growth),
        'Debt-to-Equity': ('<=', debt_equity),
        'FCF Margin %': ('>=', fcf_margin),
        'ROIC-WACC': ('>=', roic_wacc),
        'ROTE-WACC': ('>=', rote_wacc)
    }

    # Apply filters
    if st.button("🔍 Screen Stocks", type="primary"):
        filtered_df = df.copy()

        for col, (op, threshold) in criteria_mapping.items():
            if col in filtered_df.columns:
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
            st.info("💡 Add your OpenAI API key to .env or enter in sidebar to enable AI agent features.")


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
                st.info("💡 Add your OpenAI API key to .env or enter in sidebar to enable AI agent features.")

    except Exception as e:
        st.error(f"Error reading file: {e}")


def show_chatbot_page():
    st.header("💬 AI Chatbot")
    st.markdown("*Chat with AI agents to analyze stocks, detect anomalies, and get investment insights*")

    # Check if AI is available
    agent = get_ai_agent("general")
    if not agent:
        st.warning("⚠️ AI features require an OpenAI API key. Please add your API key in the sidebar or create a .env file.")
        st.info("""
        **How to enable AI:**
        1. Get an API key from [OpenAI](https://platform.openai.com)
        2. Either enter the key in the sidebar, or
        3. Create a `.env` file with: `OPENAI_API_KEY=your-key-here`
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

    ### AI Agent Features (requires OpenAI API key)

    - **Screening Agent**: Specialized AI that helps find value stocks and explains picks
    - **Anomaly Agent**: Forensic AI that interprets financial red flags
    - **Research Agent**: AI that generates investment theses and recommendations
    - **Tool Use**: Agents can call tools to screen stocks, detect anomalies, compare companies

    ### Valuation Classification

    | Classification | EPV/MC Ratio | Meaning |
    |----------------|--------------|---------|
    | Undervalued | > 1.3 | EPV 30%+ above Market Cap |
    | Fair Value | 0.7 - 1.3 | Within reasonable range |
    | Overvalued | < 0.7 | EPV 30%+ below Market Cap |

    ### How to Enable AI

    1. Get an API key from [OpenAI](https://platform.openai.com)
    2. Enter the key in the sidebar
    3. Click "Generate AI Analysis" on any results page

    ---
    Built with Streamlit & OpenAI | [GitHub Repository](https://github.com/alfredang/value-investment)
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
