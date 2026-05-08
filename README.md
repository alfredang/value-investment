# Value Investment Academy

An AI-powered Python tool for value investors with stock screening, in-house DCF valuation, AI forensic anomaly detection, competitor analysis, and professional report generation for US and Singapore markets.

**[Try the Live Demo](https://value-investment-academy.streamlit.app)**

![VIA Financial Analysis Platform](screenshot.png)

## Features

### 1. Stock Screener (10 Criteria)
Filter stocks across US and Singapore markets with synced slider + text input controls:
- **Profitability**: Gross Margin %, Net Margin %, ROA %, ROE %, FCF Margin %
- **Growth**: 5-Year Revenue Growth, 5-Year EPS Growth
- **Efficiency**: ROIC - WACC, ROTE - WACC
- **Balance Sheet**: Debt-to-Equity Ratio
- **Exchange Filter**: Multi-select dropdown (NASDAQ, NYSE, SGX, etc.)
- **Multi-file Upload**: Combine US + SG CSV files with auto exchange detection

### 2. In-House DCF Valuation Formula
Two-stage DCF model implementing the client's proprietary Excel formula:
- **Low IV (Conservative)**: historical EPS growth dampened ÷ 8 (capped at 3%)
- **High IV (Aggressive)**: historical EPS growth dampened ÷ 2 (capped at 12%)
- **Verdict**: Undervalued (price < Low IV) / Fair Value (within range) / Overvalued (price > High IV)
- **Defaults** (per client spec): 10% discount rate, 2% terminal growth, 30% margin of safety, 10y growth phase, 10y terminal phase
- **Inputs auto-derived** from CSV: Current Price, PE Ratio (or EPV × WACC fallback), 5-Year EPS Growth Rate

### 3. AI Anomaly Detection (Claude-Powered)
Replaces traditional M-Score/Z-Score checks with AI forensic analysis:
- Analyzes 10-year financial history from XLS, FMP API, or CSV data
- Detects one-off income/expenses, business model changes, margin distortions
- Rates each company: **CLEAN**, **MINOR**, or **MATERIAL**
- Provides specific year-by-year findings and valuation impact assessment

### 4. Professional Summary Report Dashboard
Bloomberg/FactSet-inspired Step 3 report with:
- **Executive Summary**: KPI stat cards, valuation/anomaly distribution charts, top picks
- **Portfolio Analytics**: Interactive Plotly radar chart, valuation scatter plot, metrics comparison bars
- **Competitor Comparison (live)**: Pick any final candidate → app finds peers in same Sector + Industry → side-by-side metrics table + 5 interactive Plotly bar charts
- **Company Deep Dive Cards**: 10-metric grid with color coding, progress bars, valuation row
- **Recommendation Table**: Conviction scoring (0-100), STRONG BUY/BUY/HOLD/WATCH ratings
- **DOCX Export**: AI-enhanced professional report generation via Claude

### 5. AI-Enhanced DOCX Report (Standardized Template)
Each company section follows a fixed 6-part research template:
- **A. Company Profile** — Name, Sector, Industry, AI-generated background
- **B. Valuation Range** — Low IV / Fair Value / High IV with verdict
- **C. Key Financial Metrics** — full metrics table with notes
- **D. 10-Year Financial Trends** — embedded line graphs (Revenue / Net Income / FCF)
- **E. Competitor Comparison** — 5 embedded bar charts vs peers
- **F. Analyst Assessment** — AI-generated business quality, risk, and investment thesis

### 6. Company Detail Popup (FMP API)
Click "View Trends" on any stock card to see:
- Business description, CEO, location (via Financial Modeling Prep API)
- 10-year Revenue vs Net Income chart (Plotly)
- 10-year Free Cash Flow trend chart
- Graceful degradation when API key is not configured

### 7. AI Agent Team (Claude Agent SDK)
Multi-agent system with subagent dispatch and specialized roles:

| Agent | Specialization | Tools |
|-------|---------------|-------|
| **Coordinator** | Routes tasks to specialists | All tools + Task dispatch |
| **Screening Agent** | Finds value stocks | Stock screening, comparison |
| **Anomaly Agent** | Forensic analysis | AI-powered detection |
| **Research Agent** | Investment thesis | Web search, news, real-time prices |

### 8. Real-Time Data Integration
- **Financial Modeling Prep**: Company profiles, financial statements, charts
- **Tavily**: Web search for latest company news
- **NewsAPI**: Financial headlines and market news
- **Twelve Data**: Real-time stock prices and historical data

## Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js** (for Claude Code CLI)
- **Claude Code CLI** logged in via `claude /login` (uses your Claude subscription — no API key needed)

### Using pip

```bash
git clone https://github.com/alfredang/value-investment.git
cd value-investment
pip install -r requirements.txt

# One-time Claude Code CLI auth (if not already set up)
npm install -g @anthropic-ai/claude-code
claude /login

streamlit run app.py
```

### Using Docker

```bash
docker pull tertiaryinfotech/value-investment:latest

docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=your-key \
  -e FMP_API_KEY=your-key \
  tertiaryinfotech/value-investment:latest
```

Then open http://localhost:8501 in your browser.

> **Note:** Docker deployment uses the API key path since Claude Code CLI is not bundled in the container. For local use without an API key, install Claude Code CLI on your host and run with pip instead.

### Command Line Interface

```bash
python main.py screen --market US --roe 15 --gross-margin 30
python main.py analyze DDI
python main.py --help
```

## 3-Step Workflow

```
Step 1: Companies Screening     Step 2: AI Anomaly Analysis     Step 3: Summary Report
+------------------------+      +------------------------+      +------------------------+
| Upload CSV files       |      | AI analyzes 10-year    |      | Executive Summary      |
| Set 10 criteria        | ---> | financial history      | ---> | Portfolio Analytics     |
| Filter by exchange     |      | Rates CLEAN/MINOR/     |      | Competitor Comparison  |
| In-house DCF verdict   |      | MATERIAL               |      | Company Deep Dives     |
| Select companies       |      |                        |      | Recommendations        |
+------------------------+      +------------------------+      | Export DOCX Report     |
                                                                 +------------------------+
```

## Configuration

### Authentication

This app uses **`claude-agent-sdk`** which authenticates two ways:

1. **Claude Code CLI subscription (recommended, free with Claude Pro/Max)**
   - Run `claude /login` once
   - No `.env` setup needed for AI calls

2. **Anthropic API key (optional, paid)**
   - Get key from [console.anthropic.com](https://console.anthropic.com)
   - Either paste it into the app's UI input field, OR set in `.env`:
   ```bash
   ANTHROPIC_API_KEY=sk-ant-your-api-key-here
   ```

### Optional Environment Variables

Copy `.env.example` to `.env` and fill in any of:

```bash
# Optional - for enhanced features
FMP_API_KEY=your-key              # Company profiles, charts (financialmodelingprep.com)
TAVILY_API_KEY=tvly-your-key      # Web search (tavily.com)
NEWS_API_KEY=your-newsapi-key     # Market news (newsapi.org)
TWELVE_DATA_API_KEY=your-key      # Real-time prices (twelvedata.com)
```

All four data API keys are optional — the app works without them, just with reduced functionality.

## Project Structure

```
value-investment/
├── app.py                  # Streamlit web app (main)
├── valuation.py            # In-house two-stage DCF formula (Low IV / High IV)
├── screener.py             # Stock screening logic (applies in-house valuation)
├── peer_finder.py          # Peer discovery by Sector + Industry
├── chart_engine.py         # Matplotlib (DOCX) + Plotly (Streamlit) chart factories
├── llm.py                  # claude-agent-sdk wrapper (one-shot completions, retry logic)
├── ai_agents.py            # Claude Agent SDK multi-agent system
├── ai_analyzer.py          # High-level AI analysis helpers
├── enhanced_report.py      # AI-enhanced DOCX report generator (standardized template)
├── anomaly_detector.py     # Rule-based financial anomaly detection
├── data_loader.py          # CSV/XLS file parsing
├── fmp_api.py              # Financial Modeling Prep API client
├── market_data.py          # Real-time price/news integrations
├── summary_report.py       # Professional Step 3 dashboard (Plotly + HTML/CSS)
├── report_generator.py     # Markdown/Word report generation
├── admin.py                # API key management + LLM config UI
├── main.py                 # CLI interface
├── requirements.txt        # pip dependencies
└── .env.example            # Environment variables template
```

## Dependencies

Core:
- pandas, xlrd - Data processing
- streamlit - Web UI
- plotly - Interactive charts (radar, scatter, bar)
- matplotlib - Static chart images for DOCX
- claude-agent-sdk - All Claude calls (chat, anomaly, DOCX, multi-agent system)
- python-docx - Word document generation
- requests - FMP API integration
- nest-asyncio - Streamlit async support

Optional:
- tavily-python - Web search
- httpx - HTTP client for real-time data APIs

CLI:
- click, rich, tabulate - Command line interface

## Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your fork
4. Add secrets in app settings:
   - `ANTHROPIC_API_KEY` (required for cloud — Claude Code CLI is not available in Streamlit Cloud)
   - `FMP_API_KEY` (optional, for company profiles)
   - `TAVILY_API_KEY` (optional, for web search)
   - `NEWS_API_KEY` (optional, for news)
   - `TWELVE_DATA_API_KEY` (optional, for real-time prices)
5. Deploy!

## License

MIT License

## Legacy Documentation

The previous version of this README (describing the OpenAI-based architecture before the Claude Agent SDK migration) is preserved at [`README_LEGACY.md`](./README_LEGACY.md) for historical reference.
