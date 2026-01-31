"""
Admin Module - Authentication, API Key Management, and Configuration

Provides admin login, API key storage, LLM model selection, and analysis options.
"""
import os
import json
import hashlib
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


# Config file path
CONFIG_DIR = Path.home() / ".value_investment"
CONFIG_FILE = CONFIG_DIR / "config.json"
ADMIN_PASSWORD_HASH_KEY = "admin_password_hash"


@dataclass
class APIKeys:
    """API Keys configuration."""
    openai_api_key: str = ""
    tavily_api_key: str = ""
    news_api_key: str = ""
    twelve_data_api_key: str = ""
    morningstar_api_key: str = ""


@dataclass
class DataProviderConfig:
    """Data provider configuration."""
    quote_provider: str = "twelve_data"  # morningstar, twelve_data
    fundamentals_provider: str = "morningstar"  # morningstar, twelve_data
    news_provider: str = "news_api"  # morningstar, news_api
    historical_provider: str = "twelve_data"  # morningstar, twelve_data

    @staticmethod
    def available_providers() -> Dict[str, list]:
        """Available providers by data type."""
        return {
            "quote": ["morningstar", "twelve_data"],
            "fundamentals": ["morningstar", "twelve_data"],
            "news": ["morningstar", "news_api"],
            "historical": ["morningstar", "twelve_data"]
        }


@dataclass
class LLMConfig:
    """LLM Model configuration."""
    model: str = "gpt-4o"  # Default model
    temperature: float = 0.7
    max_tokens: int = 4096

    @staticmethod
    def available_models() -> list:
        """List of available OpenAI models."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
        ]


@dataclass
class AnalysisConfig:
    """Value investment analysis configuration."""
    # Valuation thresholds
    undervalued_threshold: float = 1.3  # EPV/MC ratio above this = undervalued
    overvalued_threshold: float = 0.7   # EPV/MC ratio below this = overvalued

    # Risk thresholds
    m_score_threshold: float = -1.78    # M-Score above this = manipulation risk
    z_score_threshold: float = 1.8      # Z-Score below this = bankruptcy risk
    f_score_threshold: int = 3          # F-Score below this = weak financials
    sloan_ratio_threshold: float = 10.0 # Sloan Ratio above this = quality issues

    # Default screening criteria
    default_gross_margin: float = 20.0
    default_net_margin: float = 5.0
    default_roe: float = 10.0
    default_roa: float = 5.0
    default_debt_to_equity: float = 1.5
    default_fcf_margin: float = 0.0
    default_revenue_growth: float = 0.0
    default_eps_growth: float = 0.0
    default_roic_wacc: float = 0.0
    default_rote_wacc: float = 0.0


@dataclass
class AppConfig:
    """Complete application configuration."""
    api_keys: APIKeys = field(default_factory=APIKeys)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    data_provider_config: DataProviderConfig = field(default_factory=DataProviderConfig)
    admin_password_hash: str = ""


def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_config() -> AppConfig:
    """Load configuration from file."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)

            # Parse nested configs
            api_keys_data = data.get("api_keys", {})
            # Handle missing morningstar key for backward compatibility
            api_keys = APIKeys(
                openai_api_key=api_keys_data.get("openai_api_key", ""),
                tavily_api_key=api_keys_data.get("tavily_api_key", ""),
                news_api_key=api_keys_data.get("news_api_key", ""),
                twelve_data_api_key=api_keys_data.get("twelve_data_api_key", ""),
                morningstar_api_key=api_keys_data.get("morningstar_api_key", "")
            )
            llm_config = LLMConfig(**data.get("llm_config", {}))
            analysis_config = AnalysisConfig(**data.get("analysis_config", {}))

            # Handle missing data_provider_config for backward compatibility
            dp_data = data.get("data_provider_config", {})
            data_provider_config = DataProviderConfig(
                quote_provider=dp_data.get("quote_provider", "twelve_data"),
                fundamentals_provider=dp_data.get("fundamentals_provider", "morningstar"),
                news_provider=dp_data.get("news_provider", "news_api"),
                historical_provider=dp_data.get("historical_provider", "twelve_data")
            )

            return AppConfig(
                api_keys=api_keys,
                llm_config=llm_config,
                analysis_config=analysis_config,
                data_provider_config=data_provider_config,
                admin_password_hash=data.get("admin_password_hash", "")
            )
    except Exception as e:
        print(f"Error loading config: {e}")

    return AppConfig()


def save_config(config: AppConfig) -> bool:
    """Save configuration to file."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "api_keys": asdict(config.api_keys),
            "llm_config": asdict(config.llm_config),
            "analysis_config": asdict(config.analysis_config),
            "data_provider_config": asdict(config.data_provider_config),
            "admin_password_hash": config.admin_password_hash
        }

        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def is_admin_setup() -> bool:
    """Check if admin password has been set up."""
    config = load_config()
    return bool(config.admin_password_hash)


def setup_admin_password(password: str) -> bool:
    """Set up initial admin password."""
    config = load_config()
    config.admin_password_hash = hash_password(password)
    return save_config(config)


def verify_admin_password(password: str) -> bool:
    """Verify admin password."""
    config = load_config()
    return config.admin_password_hash == hash_password(password)


def get_api_keys() -> APIKeys:
    """Get API keys from config or environment."""
    config = load_config()

    # Override with environment variables if set
    return APIKeys(
        openai_api_key=os.getenv("OPENAI_API_KEY", "") or config.api_keys.openai_api_key,
        tavily_api_key=os.getenv("TAVILY_API_KEY", "") or config.api_keys.tavily_api_key,
        news_api_key=os.getenv("NEWS_API_KEY", "") or config.api_keys.news_api_key,
        twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", "") or config.api_keys.twelve_data_api_key,
        morningstar_api_key=os.getenv("MORNINGSTAR_API_KEY", "") or config.api_keys.morningstar_api_key
    )


def get_data_provider_config() -> DataProviderConfig:
    """Get data provider configuration."""
    config = load_config()
    return config.data_provider_config


def apply_api_keys_to_env(api_keys: APIKeys):
    """Apply API keys to environment variables."""
    if api_keys.openai_api_key:
        os.environ["OPENAI_API_KEY"] = api_keys.openai_api_key
    if api_keys.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = api_keys.tavily_api_key
    if api_keys.news_api_key:
        os.environ["NEWS_API_KEY"] = api_keys.news_api_key
    if api_keys.twelve_data_api_key:
        os.environ["TWELVE_DATA_API_KEY"] = api_keys.twelve_data_api_key
    if api_keys.morningstar_api_key:
        os.environ["MORNINGSTAR_API_KEY"] = api_keys.morningstar_api_key


def show_admin_login() -> bool:
    """Show admin login page. Returns True if authenticated."""
    st.header("🔐 Admin Login")

    # Check if first time setup
    if not is_admin_setup():
        st.info("Welcome! Please set up your admin password.")

        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input("Create Password", type="password", key="setup_password")
        with col2:
            confirm = st.text_input("Confirm Password", type="password", key="confirm_password")

        if st.button("Create Admin Account", type="primary"):
            if not password:
                st.error("Password cannot be empty")
            elif password != confirm:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                if setup_admin_password(password):
                    st.success("Admin account created! Please login.")
                    st.session_state['admin_authenticated'] = False
                    st.rerun()
                else:
                    st.error("Failed to create admin account")
        return False

    # Login form
    password = st.text_input("Admin Password", type="password", key="login_password")

    if st.button("Login", type="primary"):
        if verify_admin_password(password):
            st.session_state['admin_authenticated'] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid password")

    return False


def show_admin_panel():
    """Show the full admin panel."""
    st.header("⚙️ Admin Panel")

    # Load current config
    config = load_config()

    # Logout button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🚪 Logout"):
            st.session_state['admin_authenticated'] = False
            st.rerun()

    # Tabs for different settings
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 API Keys", "📡 Data Providers", "🤖 LLM Settings",
        "📊 Analysis Options", "🔒 Security"
    ])

    with tab1:
        show_api_keys_tab(config)

    with tab2:
        show_data_providers_tab(config)

    with tab3:
        show_llm_settings_tab(config)

    with tab4:
        show_analysis_options_tab(config)

    with tab5:
        show_security_tab(config)


def show_api_keys_tab(config: AppConfig):
    """API Keys management tab."""
    st.subheader("API Keys Management")
    st.markdown("*Configure API keys for various services. Keys are stored securely and can also be set via environment variables.*")

    # Get current keys (from config or env)
    current_keys = get_api_keys()

    # OpenAI
    st.markdown("### OpenAI API Key")
    st.markdown("Required for AI agent features. Get from [platform.openai.com](https://platform.openai.com)")
    new_openai = st.text_input(
        "OpenAI API Key",
        value=current_keys.openai_api_key,
        type="password",
        key="admin_openai_key",
        help="Your OpenAI API key (sk-...)"
    )
    if current_keys.openai_api_key:
        st.success("✓ OpenAI API key configured")

    # Tavily
    st.markdown("### Tavily API Key")
    st.markdown("For web search capabilities. Get from [tavily.com](https://tavily.com)")
    new_tavily = st.text_input(
        "Tavily API Key",
        value=current_keys.tavily_api_key,
        type="password",
        key="admin_tavily_key",
        help="Your Tavily API key (tvly-...)"
    )
    if current_keys.tavily_api_key:
        st.success("✓ Tavily API key configured")

    # NewsAPI
    st.markdown("### NewsAPI Key")
    st.markdown("For financial news. Get from [newsapi.org](https://newsapi.org)")
    new_news = st.text_input(
        "NewsAPI Key",
        value=current_keys.news_api_key,
        type="password",
        key="admin_news_key"
    )
    if current_keys.news_api_key:
        st.success("✓ NewsAPI key configured")

    # Twelve Data
    st.markdown("### Twelve Data API Key")
    st.markdown("For real-time stock prices. Get from [twelvedata.com](https://twelvedata.com)")
    new_twelve = st.text_input(
        "Twelve Data API Key",
        value=current_keys.twelve_data_api_key,
        type="password",
        key="admin_twelve_key"
    )
    if current_keys.twelve_data_api_key:
        st.success("✓ Twelve Data API key configured")

    # Morningstar
    st.markdown("### Morningstar API Key")
    st.markdown("For comprehensive fundamentals and fair value estimates. Get from [developer.morningstar.com](https://developer.morningstar.com)")
    new_morningstar = st.text_input(
        "Morningstar API Key",
        value=current_keys.morningstar_api_key,
        type="password",
        key="admin_morningstar_key"
    )
    if current_keys.morningstar_api_key:
        st.success("✓ Morningstar API key configured")

    # Save button
    if st.button("💾 Save API Keys", type="primary"):
        config.api_keys = APIKeys(
            openai_api_key=new_openai,
            tavily_api_key=new_tavily,
            news_api_key=new_news,
            twelve_data_api_key=new_twelve,
            morningstar_api_key=new_morningstar
        )
        if save_config(config):
            apply_api_keys_to_env(config.api_keys)
            st.success("API keys saved successfully!")
        else:
            st.error("Failed to save API keys")


def show_data_providers_tab(config: AppConfig):
    """Data providers configuration tab."""
    st.subheader("Market Data Providers")
    st.markdown("*Select which data provider to use for different types of market data.*")

    st.info("""
    **Available Providers:**
    - **Morningstar**: Comprehensive fundamentals, fair value estimates, analyst ratings
    - **Twelve Data**: Real-time quotes, historical prices, technical data
    - **NewsAPI**: Financial news and headlines
    """)

    current_dp = config.data_provider_config
    providers = DataProviderConfig.available_providers()

    # Quote Provider
    st.markdown("### Real-Time Quotes")
    new_quote = st.selectbox(
        "Quote Provider",
        options=providers["quote"],
        index=providers["quote"].index(current_dp.quote_provider) if current_dp.quote_provider in providers["quote"] else 0,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Provider for real-time stock quotes"
    )

    # Fundamentals Provider
    st.markdown("### Company Fundamentals")
    new_fundamentals = st.selectbox(
        "Fundamentals Provider",
        options=providers["fundamentals"],
        index=providers["fundamentals"].index(current_dp.fundamentals_provider) if current_dp.fundamentals_provider in providers["fundamentals"] else 0,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Provider for company fundamentals (P/E, margins, etc.)"
    )
    if new_fundamentals == "morningstar":
        st.caption("Morningstar provides: Fair Value estimates, Economic Moat ratings, Star ratings")

    # Historical Data Provider
    st.markdown("### Historical Prices")
    new_historical = st.selectbox(
        "Historical Provider",
        options=providers["historical"],
        index=providers["historical"].index(current_dp.historical_provider) if current_dp.historical_provider in providers["historical"] else 0,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Provider for historical price data"
    )

    # News Provider
    st.markdown("### Financial News")
    new_news = st.selectbox(
        "News Provider",
        options=providers["news"],
        index=providers["news"].index(current_dp.news_provider) if current_dp.news_provider in providers["news"] else 0,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Provider for financial news"
    )

    # Provider status
    st.markdown("---")
    st.markdown("### Provider Status")

    api_keys = get_api_keys()
    col1, col2 = st.columns(2)

    with col1:
        if api_keys.morningstar_api_key:
            st.success("✓ Morningstar: Configured")
        else:
            st.warning("⚠ Morningstar: No API key")

        if api_keys.twelve_data_api_key:
            st.success("✓ Twelve Data: Configured")
        else:
            st.warning("⚠ Twelve Data: No API key")

    with col2:
        if api_keys.news_api_key:
            st.success("✓ NewsAPI: Configured")
        else:
            st.warning("⚠ NewsAPI: No API key")

    # Save button
    if st.button("💾 Save Data Provider Settings", type="primary"):
        config.data_provider_config = DataProviderConfig(
            quote_provider=new_quote,
            fundamentals_provider=new_fundamentals,
            news_provider=new_news,
            historical_provider=new_historical
        )
        if save_config(config):
            st.success("Data provider settings saved!")
        else:
            st.error("Failed to save settings")


def show_llm_settings_tab(config: AppConfig):
    """LLM settings tab."""
    st.subheader("LLM Model Settings")
    st.markdown("*Configure the AI model used for analysis and chat features.*")

    # Model selection
    current_model = config.llm_config.model
    new_model = st.selectbox(
        "Model",
        options=LLMConfig.available_models(),
        index=LLMConfig.available_models().index(current_model) if current_model in LLMConfig.available_models() else 0,
        help="Select the OpenAI model to use"
    )

    # Model descriptions
    model_info = {
        "gpt-4o": "Most capable model. Best for complex analysis. Fast responses.",
        "gpt-4o-mini": "Smaller, faster, cheaper version of GPT-4o. Good balance.",
        "gpt-4-turbo": "Powerful model with large context. Good for detailed analysis.",
        "gpt-4": "Original GPT-4. Very capable but slower.",
        "gpt-3.5-turbo": "Fast and cheap. Good for simple tasks.",
        "o1": "Advanced reasoning model. Best for complex problems.",
        "o1-mini": "Smaller reasoning model. Good for focused analysis."
    }
    if new_model in model_info:
        st.info(model_info[new_model])

    # Temperature
    new_temp = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=config.llm_config.temperature,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )

    # Max tokens
    new_max_tokens = st.slider(
        "Max Response Tokens",
        min_value=256,
        max_value=8192,
        value=config.llm_config.max_tokens,
        step=256,
        help="Maximum length of AI responses"
    )

    # Save button
    if st.button("💾 Save LLM Settings", type="primary"):
        config.llm_config = LLMConfig(
            model=new_model,
            temperature=new_temp,
            max_tokens=new_max_tokens
        )
        if save_config(config):
            st.success("LLM settings saved successfully!")
            # Update session state
            st.session_state['llm_model'] = new_model
            st.session_state['llm_temperature'] = new_temp
            st.session_state['llm_max_tokens'] = new_max_tokens
        else:
            st.error("Failed to save LLM settings")


def show_analysis_options_tab(config: AppConfig):
    """Analysis configuration tab."""
    st.subheader("Value Investment Analysis Options")
    st.markdown("*Configure thresholds and default criteria for stock analysis.*")

    # Valuation thresholds
    st.markdown("### Valuation Thresholds (EPV/MC Ratio)")
    col1, col2 = st.columns(2)

    with col1:
        new_undervalued = st.number_input(
            "Undervalued Threshold",
            min_value=1.0,
            max_value=3.0,
            value=config.analysis_config.undervalued_threshold,
            step=0.1,
            help="EPV/MC ratio above this = Undervalued (default: 1.3)"
        )

    with col2:
        new_overvalued = st.number_input(
            "Overvalued Threshold",
            min_value=0.1,
            max_value=1.0,
            value=config.analysis_config.overvalued_threshold,
            step=0.1,
            help="EPV/MC ratio below this = Overvalued (default: 0.7)"
        )

    # Risk thresholds
    st.markdown("### Risk Detection Thresholds")
    col1, col2 = st.columns(2)

    with col1:
        new_m_score = st.number_input(
            "M-Score Threshold",
            min_value=-5.0,
            max_value=0.0,
            value=config.analysis_config.m_score_threshold,
            step=0.1,
            help="M-Score above this = manipulation risk (default: -1.78)"
        )

        new_z_score = st.number_input(
            "Z-Score Threshold",
            min_value=0.0,
            max_value=5.0,
            value=config.analysis_config.z_score_threshold,
            step=0.1,
            help="Z-Score below this = bankruptcy risk (default: 1.8)"
        )

    with col2:
        new_f_score = st.number_input(
            "F-Score Threshold",
            min_value=0,
            max_value=9,
            value=config.analysis_config.f_score_threshold,
            step=1,
            help="F-Score below this = weak financials (default: 3)"
        )

        new_sloan = st.number_input(
            "Sloan Ratio Threshold %",
            min_value=0.0,
            max_value=50.0,
            value=config.analysis_config.sloan_ratio_threshold,
            step=1.0,
            help="Sloan Ratio above this = quality issues (default: 10%)"
        )

    # Default screening criteria
    st.markdown("### Default Screening Criteria")
    st.markdown("*These values will be used as defaults in the Stock Screener.*")

    col1, col2, col3 = st.columns(3)

    with col1:
        new_gross_margin = st.number_input("Default Gross Margin %", 0.0, 100.0,
                                           config.analysis_config.default_gross_margin, 1.0)
        new_net_margin = st.number_input("Default Net Margin %", -50.0, 100.0,
                                         config.analysis_config.default_net_margin, 1.0)
        new_roe = st.number_input("Default ROE %", -20.0, 100.0,
                                  config.analysis_config.default_roe, 1.0)
        new_roa = st.number_input("Default ROA %", -20.0, 50.0,
                                  config.analysis_config.default_roa, 1.0)

    with col2:
        new_debt_equity = st.number_input("Default Max Debt/Equity", 0.0, 5.0,
                                          config.analysis_config.default_debt_to_equity, 0.1)
        new_fcf = st.number_input("Default FCF Margin %", -50.0, 100.0,
                                  config.analysis_config.default_fcf_margin, 1.0)
        new_rev_growth = st.number_input("Default 5Y Revenue Growth %", -50.0, 100.0,
                                         config.analysis_config.default_revenue_growth, 1.0)

    with col3:
        new_eps_growth = st.number_input("Default 5Y EPS Growth %", -50.0, 100.0,
                                         config.analysis_config.default_eps_growth, 1.0)
        new_roic_wacc = st.number_input("Default ROIC-WACC", -20.0, 50.0,
                                        config.analysis_config.default_roic_wacc, 1.0)
        new_rote_wacc = st.number_input("Default ROTE-WACC", -50.0, 100.0,
                                        config.analysis_config.default_rote_wacc, 1.0)

    # Save button
    if st.button("💾 Save Analysis Options", type="primary"):
        config.analysis_config = AnalysisConfig(
            undervalued_threshold=new_undervalued,
            overvalued_threshold=new_overvalued,
            m_score_threshold=new_m_score,
            z_score_threshold=new_z_score,
            f_score_threshold=new_f_score,
            sloan_ratio_threshold=new_sloan,
            default_gross_margin=new_gross_margin,
            default_net_margin=new_net_margin,
            default_roe=new_roe,
            default_roa=new_roa,
            default_debt_to_equity=new_debt_equity,
            default_fcf_margin=new_fcf,
            default_revenue_growth=new_rev_growth,
            default_eps_growth=new_eps_growth,
            default_roic_wacc=new_roic_wacc,
            default_rote_wacc=new_rote_wacc
        )
        if save_config(config):
            st.success("Analysis options saved successfully!")
        else:
            st.error("Failed to save analysis options")


def show_security_tab(config: AppConfig):
    """Security settings tab."""
    st.subheader("Security Settings")

    # Change password
    st.markdown("### Change Admin Password")

    current_password = st.text_input("Current Password", type="password", key="current_pwd")
    new_password = st.text_input("New Password", type="password", key="new_pwd")
    confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pwd")

    if st.button("🔐 Change Password", type="primary"):
        if not verify_admin_password(current_password):
            st.error("Current password is incorrect")
        elif not new_password:
            st.error("New password cannot be empty")
        elif len(new_password) < 6:
            st.error("New password must be at least 6 characters")
        elif new_password != confirm_password:
            st.error("New passwords do not match")
        else:
            config.admin_password_hash = hash_password(new_password)
            if save_config(config):
                st.success("Password changed successfully!")
            else:
                st.error("Failed to change password")

    st.markdown("---")

    # Config location
    st.markdown("### Configuration Storage")
    st.info(f"Configuration is stored at: `{CONFIG_FILE}`")

    # Export/Import config
    st.markdown("### Export/Import Configuration")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Export Config (without secrets)"):
            export_config = {
                "llm_config": asdict(config.llm_config),
                "analysis_config": asdict(config.analysis_config)
            }
            st.download_button(
                "Download Config",
                json.dumps(export_config, indent=2),
                "value_investment_config.json",
                "application/json"
            )

    with col2:
        uploaded_config = st.file_uploader("Import Config", type=['json'], key="import_config")
        if uploaded_config:
            try:
                imported = json.load(uploaded_config)
                if "llm_config" in imported:
                    config.llm_config = LLMConfig(**imported["llm_config"])
                if "analysis_config" in imported:
                    config.analysis_config = AnalysisConfig(**imported["analysis_config"])
                if save_config(config):
                    st.success("Configuration imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to import config: {e}")
