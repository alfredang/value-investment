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
    from agents import ValueInvestmentAgent, ScreeningAgent, AnomalyAgent
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Value Investment Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .ai-analysis {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)


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


def main():
    st.title("📈 Value Investment Tool")
    st.markdown("*Screen stocks, analyze valuations, and detect financial anomalies with AI*")

    # Sidebar
    st.sidebar.title("Navigation")

    # AI Configuration in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Agent Settings")

    # Check for env variable first
    env_api_key = os.getenv('OPENAI_API_KEY', '')

    if AI_AVAILABLE:
        if env_api_key:
            st.sidebar.success("✓ AI Enabled (from .env)")
            st.session_state['openai_api_key'] = env_api_key
        else:
            api_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                help="Enter your OpenAI API key or add to .env file"
            )
            if api_key:
                st.sidebar.success("✓ AI Enabled")
            else:
                st.sidebar.info("Enter API key or add to .env")
    else:
        st.sidebar.warning("Install openai package for AI features")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Feature",
        ["Stock Screener", "Anomaly Detector", "About"]
    )

    if page == "Stock Screener":
        show_screener_page()
    elif page == "Anomaly Detector":
        show_anomaly_page()
    else:
        show_about_page()


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
            undervalued = len(filtered_df[filtered_df.get('Valuation', pd.Series()) == 'Undervalued'])
            st.metric("Undervalued", undervalued)
        with col3:
            fair = len(filtered_df[filtered_df.get('Valuation', pd.Series()) == 'Fair Value'])
            st.metric("Fair Value", fair)
        with col4:
            overvalued = len(filtered_df[filtered_df.get('Valuation', pd.Series()) == 'Overvalued'])
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
            # Chat interface
            user_question = st.text_input(
                "Ask the AI agent about your screening results:",
                placeholder="e.g., Which stocks look most promising? What are the key risks?",
                key="screening_question"
            )

            if st.button("✨ Ask AI Agent", type="secondary"):
                if user_question:
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
                # Chat interface for anomaly analysis
                user_question = st.text_input(
                    "Ask the AI agent about these anomalies:",
                    placeholder="e.g., Should I be concerned? What should I investigate further?",
                    key="anomaly_question"
                )

                if st.button("✨ Ask AI Agent", type="secondary", key="anomaly_btn"):
                    question = user_question or f"Analyze {report.symbol} for financial anomalies and explain the key risks."

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


if __name__ == "__main__":
    main()
