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
            st.info("Enter OpenAI API key in Settings to enable AI Assistant")


def main():
    # Initialize configuration from admin settings
    init_config()

    st.title("📈 Value Investment Academy")
    st.markdown("*Screen stocks, analyze valuations, and detect financial anomalies with AI*")

    # Sidebar
    st.sidebar.title("Value Investment Academy")

    # Navigation - Main pages (removed AI Chatbot - now persistent at bottom)
    pages = ["Stock Screener", "Anomaly Detector", "Summary Report"]

    page = st.sidebar.radio(
        "Features",
        pages,
        label_visibility="collapsed"
    )

    # AI Status indicator
    st.sidebar.markdown("---")
    env_api_key = os.getenv('OPENAI_API_KEY', '')

    if AI_AVAILABLE:
        if env_api_key:
            st.sidebar.success("✓ AI Enabled")
            st.session_state['openai_api_key'] = env_api_key
            if 'llm_model' in st.session_state:
                st.sidebar.caption(f"Model: {st.session_state['llm_model']}")
        else:
            api_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                help="Enter your OpenAI API key or configure in Settings"
            )
            if api_key:
                st.sidebar.success("✓ AI Enabled")
            else:
                st.sidebar.info("Enter API key or configure in Settings")
    else:
        st.sidebar.warning("Install openai package for AI features")

    # Settings button at the bottom
    st.sidebar.markdown("---")
    if ADMIN_AVAILABLE:
        if st.sidebar.button("⚙️ Settings", use_container_width=True):
            st.session_state['show_settings'] = True

    # Check if settings page should be shown
    if st.session_state.get('show_settings', False):
        show_admin_page()
        if st.sidebar.button("← Back to App", use_container_width=True):
            st.session_state['show_settings'] = False
            st.rerun()
    else:
        # Show the selected page
        if page == "Stock Screener":
            show_screener_page()
        elif page == "Anomaly Detector":
            show_anomaly_page()
        elif page == "Summary Report":
            show_full_analysis_page()

        # Always show persistent chat at the bottom (except settings page)
        show_persistent_chat()


def show_full_analysis_page():
    st.header("📊 Full Investment Analysis")
    st.markdown("*Complete workflow: Upload data → Screen stocks → Analyze anomalies → Generate report*")

    # Initialize session state for workflow
    if 'workflow_step' not in st.session_state:
        st.session_state.workflow_step = 1
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = {}

    # Progress indicator
    steps = ["Upload Data", "Screen Stocks", "Select Companies", "Analyze & Report"]
    current_step = st.session_state.workflow_step

    # Show progress
    cols = st.columns(4)
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current_step:
                st.success(f"✓ {step_name}")
            elif i == current_step:
                st.info(f"→ {step_name}")
            else:
                st.text(f"○ {step_name}")

    st.divider()

    # Step 1: Upload Data
    if current_step == 1:
        st.subheader("Step 1: Upload Your Data Files")

        st.markdown("""
        Upload your data files to begin the analysis:
        - **Screener CSV**: Contains stock fundamentals (US or SG market)
        - **Financials XLS** (optional): 30-year financial data for anomaly detection
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Stock Screener Data**")
            screener_file = st.file_uploader(
                "Upload Screener CSV",
                type=['csv'],
                key="workflow_screener",
                help="US or SG All In One Screeners CSV file"
            )

            if screener_file:
                try:
                    df = pd.read_csv(screener_file, encoding='utf-8-sig')
                    df.columns = df.columns.str.strip()
                    st.session_state.workflow_data['screener_df'] = df
                    st.success(f"✓ Loaded {len(df)} stocks")

                    # Detect market
                    market = st.radio("Select Market", ["US", "SG"], horizontal=True)
                    st.session_state.workflow_data['market'] = market
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        with col2:
            st.markdown("**Financial Data (Optional)**")
            financials_file = st.file_uploader(
                "Upload Financials XLS",
                type=['xls', 'xlsx'],
                key="workflow_financials",
                help="Companies with anomalies XLS file for detailed analysis"
            )

            if financials_file:
                try:
                    import xlrd
                    # Save temporarily
                    with open("/tmp/workflow_financials.xls", "wb") as f:
                        f.write(financials_file.getvalue())
                    st.session_state.workflow_data['financials_path'] = "/tmp/workflow_financials.xls"

                    book = xlrd.open_workbook("/tmp/workflow_financials.xls", ignore_workbook_corruption=True)
                    symbols = []
                    for sheet_name in book.sheet_names():
                        parts = sheet_name.split('_')
                        if len(parts) >= 2:
                            symbols.append(parts[1])
                    st.session_state.workflow_data['available_symbols'] = symbols
                    st.success(f"✓ Found {len(symbols)} companies: {', '.join(symbols)}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        # Next button
        if 'screener_df' in st.session_state.workflow_data:
            if st.button("Next: Set Screening Criteria →", type="primary"):
                st.session_state.workflow_step = 2
                st.rerun()

    # Step 2: Screen Stocks
    elif current_step == 2:
        st.subheader("Step 2: Set Screening Criteria")

        df = st.session_state.workflow_data.get('screener_df')
        if df is None:
            st.warning("No data loaded. Please go back to Step 1.")
            if st.button("← Back to Upload"):
                st.session_state.workflow_step = 1
                st.rerun()
            return

        st.markdown("*Adjust the sliders to filter stocks based on fundamental criteria*")

        col1, col2, col3 = st.columns(3)

        with col1:
            gross_margin = st.slider("Min Gross Margin %", 0, 100, 20)
            net_margin = st.slider("Min Net Margin %", -50, 100, 5)
            roa = st.slider("Min ROA %", -20, 50, 5)
            roe = st.slider("Min ROE %", -20, 100, 10)

        with col2:
            revenue_growth = st.slider("Min 5Y Revenue Growth %", -50, 100, 0)
            eps_growth = st.slider("Min 5Y EPS Growth %", -50, 100, 0)
            debt_equity = st.slider("Max Debt-to-Equity", 0.0, 5.0, 1.5, 0.1)

        with col3:
            fcf_margin = st.slider("Min FCF Margin %", -50, 100, 0)
            roic_wacc = st.slider("Min ROIC-WACC", -20, 50, 0)
            rote_wacc = st.slider("Min ROTE-WACC", -50, 100, 0)

        # Store criteria
        criteria = {
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
        st.session_state.workflow_data['criteria'] = criteria

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

        # Apply filters and show preview
        if st.button("🔍 Preview Screening Results"):
            filtered_df = df.copy()

            for col, (op, threshold) in criteria_mapping.items():
                if col in filtered_df.columns:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    if op == '>=':
                        filtered_df = filtered_df[filtered_df[col] >= threshold]
                    else:
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

            st.session_state.workflow_data['filtered_df'] = filtered_df

            # Show summary
            st.success(f"Found {len(filtered_df)} stocks matching your criteria")

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

            # Show table
            display_cols = ['Symbol', 'Company', 'Sector', 'Gross Margin %', 'ROE %',
                          'Debt-to-Equity', 'Market Cap ($M)', 'EPV/MC Ratio', 'Valuation']
            available_cols = [c for c in display_cols if c in filtered_df.columns]

            if 'Valuation' in filtered_df.columns:
                val_order = {'Undervalued': 0, 'Fair Value': 1, 'Overvalued': 2, 'N/A': 3, 'N/A (Negative EPV)': 4}
                filtered_df['_sort'] = filtered_df['Valuation'].map(val_order)
                filtered_df = filtered_df.sort_values('_sort').drop('_sort', axis=1)

            st.dataframe(filtered_df[available_cols].head(50), use_container_width=True, height=300)

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.workflow_step = 1
                st.rerun()
        with col2:
            if 'filtered_df' in st.session_state.workflow_data and len(st.session_state.workflow_data['filtered_df']) > 0:
                if st.button("Next: Select Companies →", type="primary"):
                    st.session_state.workflow_step = 3
                    st.rerun()

    # Step 3: Select Companies for Deep Analysis
    elif current_step == 3:
        st.subheader("Step 3: Select Companies for Deep Analysis")

        filtered_df = st.session_state.workflow_data.get('filtered_df')
        if filtered_df is None or len(filtered_df) == 0:
            st.warning("No stocks found. Please go back and adjust criteria.")
            if st.button("← Back to Screening"):
                st.session_state.workflow_step = 2
                st.rerun()
            return

        st.markdown("*Select the companies you want to include in your final analysis report*")

        # Get list of symbols
        symbols = filtered_df['Symbol'].tolist()

        # Multi-select for companies
        selected_symbols = st.multiselect(
            "Select companies to analyze",
            options=symbols,
            default=symbols[:min(5, len(symbols))],  # Default to first 5
            help="Choose which companies to include in the detailed report"
        )

        st.session_state.workflow_data['selected_symbols'] = selected_symbols

        if selected_symbols:
            st.markdown(f"**Selected {len(selected_symbols)} companies for analysis**")

            # Show selected companies
            selected_df = filtered_df[filtered_df['Symbol'].isin(selected_symbols)]
            display_cols = ['Symbol', 'Company', 'Sector', 'Valuation', 'ROE %', 'Debt-to-Equity']
            available_cols = [c for c in display_cols if c in selected_df.columns]
            st.dataframe(selected_df[available_cols], use_container_width=True)

            # Check if anomaly data is available for these symbols
            available_for_anomaly = st.session_state.workflow_data.get('available_symbols', [])
            can_analyze = [s for s in selected_symbols if s in available_for_anomaly]

            if can_analyze:
                st.info(f"Anomaly detection available for: {', '.join(can_analyze)}")
            else:
                st.warning("No financial data uploaded for anomaly detection. Report will include fundamentals only.")

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.workflow_step = 2
                st.rerun()
        with col2:
            if selected_symbols:
                if st.button("Next: Generate Report →", type="primary"):
                    st.session_state.workflow_step = 4
                    st.rerun()

    # Step 4: Enhanced Analysis and Report Generation
    elif current_step == 4:
        st.subheader("Step 4: Enhanced Analysis & Report")

        selected_symbols = st.session_state.workflow_data.get('selected_symbols', [])
        filtered_df = st.session_state.workflow_data.get('filtered_df')
        criteria = st.session_state.workflow_data.get('criteria', {})
        market = st.session_state.workflow_data.get('market', 'US')

        if not selected_symbols or filtered_df is None:
            st.warning("No companies selected. Please go back.")
            if st.button("← Back"):
                st.session_state.workflow_step = 3
                st.rerun()
            return

        st.markdown(f"**Analyzing {len(selected_symbols)} companies with enhanced features...**")

        # Analysis Options
        st.markdown("### Analysis Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            include_anomaly = st.checkbox("Anomaly Detection", value=True,
                                         help="Run M-Score, Z-Score, F-Score analysis")
            include_anomaly_deepdive = st.checkbox("Deep Dive Anomaly Validation", value=True,
                                                   help="Validate if anomalies are excusable")

        with col2:
            include_charts = st.checkbox("Generate Charts", value=True,
                                        help="Create interactive visualizations")
            include_ceo_tracking = st.checkbox("CEO Commitment Tracking", value=False,
                                              help="Track CEO promises vs fulfillment (requires Tavily API)")

        with col3:
            include_ai = st.checkbox("AI Analysis", value=False,
                                    help="Generate AI-powered insights")
            use_rigorous_selection = st.checkbox("Rigorous Selection Scoring", value=True,
                                                 help="Apply multi-dimensional scoring")

        if st.button("🔬 Run Enhanced Analysis", type="primary"):
            with st.spinner("Running enhanced analysis..."):
                analyses = []
                anomaly_reports = {}
                anomaly_validations = {}
                ceo_reports = {}
                company_scores = []

                # Progress tracking
                progress = st.progress(0)
                status_text = st.empty()
                total_steps = len(selected_symbols) * (1 + int(include_anomaly_deepdive) + int(include_ceo_tracking))
                current_step_num = 0

                # Initialize enhanced analysis tools
                chart_gen = None
                ceo_tracker = None
                anomaly_validator = None
                company_selector = None

                if ENHANCED_AVAILABLE:
                    if include_charts:
                        try:
                            chart_gen = ChartGenerator()
                        except ImportError:
                            st.warning("Plotly not available for charts")

                    if include_ceo_tracking:
                        ceo_tracker = CEOCommitmentTracker()

                    if include_anomaly_deepdive:
                        anomaly_validator = AnomalyValidator()

                    if use_rigorous_selection:
                        company_selector = CompanySelector()

                # Analyze each company
                for i, symbol in enumerate(selected_symbols):
                    status_text.text(f"Analyzing {symbol}...")
                    current_step_num += 1
                    progress.progress(current_step_num / total_steps)

                    # Get stock data
                    stock_data = filtered_df[filtered_df['Symbol'] == symbol].iloc[0].to_dict()
                    company_name = stock_data.get('Company', symbol)
                    sector = stock_data.get('Sector', '')

                    # Run anomaly detection
                    anomaly_report = None
                    if include_anomaly:
                        available_symbols = st.session_state.workflow_data.get('available_symbols', [])
                        if symbol in available_symbols:
                            try:
                                loader = DataLoader("/tmp")
                                detector = AnomalyDetector(loader)
                                anomaly_report = detector.analyze(symbol)
                                anomaly_reports[symbol] = {
                                    'm_score': anomaly_report.m_score,
                                    'z_score': anomaly_report.z_score,
                                    'f_score': anomaly_report.f_score,
                                    'sloan_ratio': anomaly_report.sloan_ratio,
                                    'risk_level': anomaly_report.risk_level,
                                    'anomalies': [{'category': a.category, 'description': a.description,
                                                   'severity': a.severity.value, 'year': a.year}
                                                  for a in anomaly_report.anomalies]
                                }
                            except Exception as e:
                                st.warning(f"Could not analyze {symbol} for anomalies: {e}")

                    # Deep dive anomaly validation
                    if include_anomaly_deepdive and anomaly_report and anomaly_validator:
                        status_text.text(f"Validating anomalies for {symbol}...")
                        current_step_num += 1
                        progress.progress(current_step_num / total_steps)

                        validations = []
                        for anomaly in anomaly_reports.get(symbol, {}).get('anomalies', []):
                            validation = anomaly_validator.validate_anomaly(
                                symbol, company_name, anomaly, sector
                            )
                            validations.append(validation)
                        anomaly_validations[symbol] = validations

                    # CEO Commitment Tracking
                    if include_ceo_tracking and ceo_tracker:
                        status_text.text(f"Tracking CEO commitments for {symbol}...")
                        current_step_num += 1
                        progress.progress(current_step_num / total_steps)

                        commitments = ceo_tracker.search_ceo_commitments(symbol, company_name)
                        if commitments:
                            ceo_reports[symbol] = ceo_tracker.generate_commitment_report(
                                symbol, company_name, commitments
                            )

                    # Rigorous company scoring
                    if use_rigorous_selection and company_selector:
                        score = company_selector.score_company(
                            stock_data,
                            anomaly_reports.get(symbol)
                        )
                        company_scores.append(score)

                    # Create analysis object
                    if REPORT_AVAILABLE:
                        analysis = create_company_analysis_from_data(stock_data, anomaly_report, market)
                    else:
                        analysis = type('Analysis', (), {
                            'symbol': symbol,
                            'company_name': company_name,
                            'sector': sector,
                            'market': market,
                            'valuation_status': stock_data.get('Valuation', 'N/A'),
                            'risk_level': anomaly_report.risk_level if anomaly_report else 'N/A'
                        })()

                    # Add AI analysis if requested
                    if include_ai and AI_AVAILABLE:
                        try:
                            agent = get_ai_agent("general")
                            if agent:
                                prompt = f"Provide a brief investment analysis for {symbol} ({company_name}). Include key strengths, risks, and recommendation."
                                response = agent.chat(prompt)
                                analysis.ai_summary = response.content
                        except Exception as e:
                            st.warning(f"AI analysis failed for {symbol}: {e}")

                    analyses.append(analysis)

                progress.progress(1.0)
                status_text.text("Analysis complete!")

                # Store results
                st.session_state.workflow_data['analyses'] = analyses
                st.session_state.workflow_data['anomaly_reports'] = anomaly_reports
                st.session_state.workflow_data['anomaly_validations'] = anomaly_validations
                st.session_state.workflow_data['ceo_reports'] = ceo_reports
                st.session_state.workflow_data['company_scores'] = company_scores

            # ============================================
            # RESULTS DISPLAY
            # ============================================
            st.success(f"✓ Analyzed {len(analyses)} companies")

            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Summary", "📈 Charts", "🔍 Anomaly Deep Dive",
                "👔 CEO Tracking", "🏆 Selection Ranking"
            ])

            # Tab 1: Summary
            with tab1:
                st.markdown("### Analysis Summary")

                summary_data = []
                for a in analyses:
                    row = {
                        'Symbol': a.symbol,
                        'Company': getattr(a, 'company_name', 'N/A'),
                        'Sector': getattr(a, 'sector', 'N/A'),
                        'Valuation': getattr(a, 'valuation_status', 'N/A'),
                        'Risk Level': getattr(a, 'risk_level', 'N/A'),
                    }

                    # Add scores if available
                    if company_scores:
                        score = next((s for s in company_scores if s.symbol == a.symbol), None)
                        if score:
                            row['Total Score'] = f"{score.total_score:.0f}"
                            row['Passed'] = "✅" if score.passed_screening else "❌"

                    summary_data.append(row)

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

            # Tab 2: Charts
            with tab2:
                if chart_gen and ENHANCED_AVAILABLE:
                    st.markdown("### Valuation Comparison")
                    try:
                        selected_df = filtered_df[filtered_df['Symbol'].isin(selected_symbols)]
                        fig = chart_gen.create_valuation_comparison(selected_df)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate valuation chart: {e}")

                    # Peer comparison
                    st.markdown("### Peer Comparison")
                    metric = st.selectbox("Select metric", ['ROE %', 'Gross Margin %', 'Net Margin %', 'ROIC-WACC'])
                    try:
                        fig = chart_gen.create_peer_comparison(selected_df, metric)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate peer comparison: {e}")

                    # Risk heatmap
                    if anomaly_reports:
                        st.markdown("### Risk Heatmap")
                        try:
                            companies_for_heatmap = []
                            for symbol, report in anomaly_reports.items():
                                companies_for_heatmap.append({
                                    'symbol': symbol,
                                    'm_score': report.get('m_score'),
                                    'z_score': report.get('z_score'),
                                    'f_score': report.get('f_score'),
                                    'sloan_ratio': report.get('sloan_ratio'),
                                    'debt_to_equity': filtered_df[filtered_df['Symbol'] == symbol].iloc[0].get('Debt-to-Equity') if symbol in filtered_df['Symbol'].values else None
                                })
                            fig = chart_gen.create_risk_heatmap(companies_for_heatmap)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate risk heatmap: {e}")

                    # Fundamental radar for selected company
                    st.markdown("### Fundamental Profile")
                    selected_for_radar = st.selectbox("Select company for radar chart", selected_symbols)
                    if selected_for_radar:
                        try:
                            stock_data = filtered_df[filtered_df['Symbol'] == selected_for_radar].iloc[0].to_dict()
                            fig = chart_gen.create_fundamentals_radar(stock_data)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate radar chart: {e}")
                else:
                    st.info("Enable 'Generate Charts' option and ensure plotly is installed")

            # Tab 3: Anomaly Deep Dive
            with tab3:
                if anomaly_validations:
                    st.markdown("### Anomaly Validation Results")

                    for symbol, validations in anomaly_validations.items():
                        with st.expander(f"**{symbol}** - {len(validations)} anomalies analyzed"):
                            valid_concerns = sum(1 for v in validations if v.is_valid_concern)
                            excusable = len(validations) - valid_concerns

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Valid Concerns", valid_concerns)
                            with col2:
                                st.metric("Excusable", excusable)

                            for v in validations:
                                severity_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(v.adjusted_severity, "⚪")

                                st.markdown(f"**{severity_color} {v.anomaly_type}**")
                                st.markdown(f"- Original: {v.original_flag[:100]}...")
                                st.markdown(f"- Severity: {v.severity} → {v.adjusted_severity}")

                                if v.is_valid_concern:
                                    st.error(f"⚠️ Valid Concern: {v.recommendation}")
                                else:
                                    st.success(f"✅ Excusable: {v.recommendation}")

                                if v.mitigating_factors:
                                    st.markdown(f"- Mitigating: {', '.join(v.mitigating_factors)}")
                                if v.one_time_event:
                                    st.info("Likely one-time event")
                                if v.management_explanation:
                                    st.markdown(f"- Management explanation: {v.management_explanation[:200]}...")

                                st.divider()
                else:
                    st.info("Enable 'Deep Dive Anomaly Validation' to see detailed analysis")

            # Tab 4: CEO Tracking
            with tab4:
                if ceo_reports:
                    st.markdown("### CEO Commitment Analysis")
                    for symbol, report in ceo_reports.items():
                        with st.expander(f"**{symbol}** - CEO Commitments"):
                            st.markdown(report)
                else:
                    st.info("Enable 'CEO Commitment Tracking' and ensure Tavily API key is configured")

            # Tab 5: Selection Ranking
            with tab5:
                if company_scores:
                    st.markdown("### Company Selection Ranking")

                    # Selection funnel
                    if ENHANCED_AVAILABLE and chart_gen:
                        try:
                            stages = [
                                ("Initial Screen", len(filtered_df)),
                                ("Manual Selection", len(selected_symbols)),
                                ("Passed Criteria", sum(1 for s in company_scores if s.passed_screening)),
                            ]
                            fig = chart_gen.create_selection_funnel(stages)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            pass

                    # Ranking table
                    ranking_data = []
                    for s in sorted(company_scores, key=lambda x: x.total_score, reverse=True):
                        ranking_data.append({
                            'Rank': s.rank,
                            'Symbol': s.symbol,
                            'Company': s.company_name[:25],
                            'Total': f"{s.total_score:.0f}",
                            'Valuation': f"{s.valuation_score:.0f}",
                            'Quality': f"{s.quality_score:.0f}",
                            'Growth': f"{s.growth_score:.0f}",
                            'Safety': f"{s.safety_score:.0f}",
                            'Passed': "✅" if s.passed_screening else "❌"
                        })

                    st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)

                    # Disqualification reasons
                    disqualified = [s for s in company_scores if not s.passed_screening]
                    if disqualified:
                        st.markdown("### Disqualification Reasons")
                        for s in disqualified:
                            with st.expander(f"**{s.symbol}** - Disqualified"):
                                for reason in s.disqualification_reasons:
                                    st.markdown(f"- ❌ {reason}")
                else:
                    st.info("Enable 'Rigorous Selection Scoring' to see rankings")

            # Download Reports Section
            st.markdown("---")
            st.subheader("📥 Download Reports")

            col1, col2, col3 = st.columns(3)

            with col1:
                if REPORT_AVAILABLE:
                    generator = ReportGenerator()
                    md_report = generator.generate_markdown_report(analyses, criteria)
                    st.download_button(
                        "📄 Download Markdown Report",
                        md_report,
                        "investment_analysis_report.md",
                        "text/markdown"
                    )
                else:
                    st.info("Report generator not available")

            with col2:
                if REPORT_AVAILABLE:
                    try:
                        word_buffer = generator.generate_word_document(analyses, criteria)
                        st.download_button(
                            "📝 Download Word Document",
                            word_buffer,
                            "investment_analysis_report.docx",
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as e:
                        st.warning(f"Word export error: {e}")

            with col3:
                csv_data = pd.DataFrame(summary_data).to_csv(index=False)
                st.download_button(
                    "📊 Download CSV Summary",
                    csv_data,
                    "investment_analysis_summary.csv",
                    "text/csv"
                )

            # Full report view
            if REPORT_AVAILABLE:
                with st.expander("📖 View Full Report"):
                    st.markdown(md_report)

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back"):
                st.session_state.workflow_step = 3
                st.rerun()
        with col2:
            if st.button("🔄 Start New Analysis"):
                st.session_state.workflow_step = 1
                st.session_state.workflow_data = {}
                st.rerun()


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
        # Save uploaded file temporarily
        with open("/tmp/anomaly_data.xls", "wb") as f:
            f.write(anomaly_file.getvalue())

        book = xlrd.open_workbook("/tmp/anomaly_data.xls", ignore_workbook_corruption=True)
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
