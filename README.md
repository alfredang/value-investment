# Value Investment Tool

An AI-powered Python tool for value investors with stock screening, automated valuation, and financial anomaly detection for US and Singapore markets.

**[🚀 Try the Live Demo](https://value-investment.streamlit.app)** *(Deploy your own to use)*

## Features

### 1. Stock Screener
Filter stocks based on 10 fundamental criteria:
- Gross Margin %, Net Margin %, ROA %, ROE %
- 5-Year Revenue & EPS Growth Rates
- Debt-to-Equity Ratio, Free Cash Flow Margin
- ROIC - WACC, ROTE - WACC

### 2. Automated Valuation (EPV vs Market Cap)
- **Undervalued**: EPV/MC > 1.3 (30%+ margin of safety)
- **Fair Value**: 0.7 <= EPV/MC <= 1.3
- **Overvalued**: EPV/MC < 0.7

### 3. Anomaly Detection
- **M-Score**: Earnings manipulation probability
- **Z-Score**: Bankruptcy/distress risk
- **F-Score**: Financial strength
- **Sloan Ratio**: Accrual-based earnings quality
- One-off events, cash flow mismatches, balance sheet red flags

### 4. AI Agent Team (OpenAI Agents SDK)
Multi-agent system with handoffs and specialized roles:

| Agent | Specialization | Tools |
|-------|---------------|-------|
| **Coordinator** | Routes tasks to specialists | All tools + handoffs |
| **Screening Agent** | Finds value stocks | Stock screening, comparison |
| **Anomaly Agent** | Forensic analysis | M-Score, Z-Score detection |
| **Research Agent** | Investment thesis | Web search, news, real-time prices |

### 5. Real-Time Data Integration
- **Tavily**: Web search for latest company news
- **NewsAPI**: Financial headlines and market news
- **Twelve Data**: Real-time stock prices and historical data

### 6. Full Analysis Workflow
Step-by-step guided process:
1. Upload data files (CSV/XLS)
2. Set screening criteria
3. Select companies for deep analysis
4. Generate comprehensive reports (Markdown/Word/CSV)

### 7. Agent Skills (skills.sh)
Pre-installed open standard agent skills:
- `stock-analysis` - Yahoo Finance-based stock/crypto analysis
- `stock-research-executor` - 8-phase investment due diligence
- `startup-financial-modeling` - 3-5 year financial projections
- `analyze` - Investment analysis patterns
- `documentation-templates` - Report generation templates

## Quick Start

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/alfredang/value-investment.git
cd value-investment

# Install with uv
uv sync

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys

# Run the web app
uv run streamlit run app.py
```

### Using pip

```bash
git clone https://github.com/alfredang/value-investment.git
cd value-investment
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys

streamlit run app.py
```

### Command Line Interface

```bash
# Screen US stocks
python main.py screen --market US --roe 15 --gross-margin 30

# Analyze company for anomalies
python main.py analyze DDI

# Show help
python main.py --help
```

## AI Agent Usage

```python
from ai_agents import ValueInvestmentAgent, ScreeningAgent, AnomalyAgent, ResearchAgent

# Use the coordinator agent (routes to specialists)
agent = ValueInvestmentAgent()
response = agent.chat("Find undervalued US stocks with ROE > 15% and analyze for red flags")
print(response.content)

# Direct specialist usage
screening = ScreeningAgent()
response = screening.chat("Screen US stocks with gross margin > 30%")

anomaly = AnomalyAgent()
response = anomaly.chat("Analyze DDI for financial anomalies")

research = ResearchAgent()
response = research.chat("Get latest news on AAPL and provide investment thesis")

# Sequential multi-agent analysis
import asyncio
from ai_agents import run_full_analysis

results = asyncio.run(run_full_analysis("AAPL", market="US"))
print(results["fundamentals"])
print(results["anomalies"])
print(results["thesis"])
```

## Using Agent Skills

### Stock Analysis (Yahoo Finance)
```bash
# Analyze a single stock
uv run .agents/skills/stock-analysis/scripts/analyze_stock.py AAPL

# Compare multiple stocks
uv run .agents/skills/stock-analysis/scripts/analyze_stock.py AAPL MSFT GOOGL
```

### Install More Skills
```bash
# Search for skills
npx skills search financial

# Install a skill
npx skills add -y wshobson/agents@market-sizing-analysis
```

## Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your fork
4. Add secrets in app settings:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY` (optional, for web search)
   - `NEWS_API_KEY` (optional, for news)
   - `TWELVE_DATA_API_KEY` (optional, for real-time prices)
5. Deploy!

## Configuration

### Environment Variables

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional - for enhanced features
TAVILY_API_KEY=tvly-your-key        # Web search (tavily.com)
NEWS_API_KEY=your-newsapi-key       # Market news (newsapi.org)
TWELVE_DATA_API_KEY=your-key        # Real-time prices (twelvedata.com)
```

### Screening Criteria Config

```json
{
  "gross_margin": 20,
  "net_margin": 5,
  "roe": 10,
  "debt_to_equity": 1.5,
  "fcf_margin": 5,
  "roic_wacc": 0
}
```

## Project Structure

```
value-investment/
├── app.py                  # Streamlit web app
├── main.py                 # CLI interface
├── ai_agents.py            # OpenAI Agents SDK implementation
├── screener.py             # Stock screening logic
├── valuation.py            # EPV-based valuation
├── anomaly_detector.py     # Financial anomaly detection
├── report_generator.py     # Markdown/Word report generation
├── data_loader.py          # CSV/XLS file parsing
├── pyproject.toml          # uv project config
├── requirements.txt        # pip dependencies
├── .env.example            # Environment variables template
├── .agents/skills/         # Installed agent skills
│   ├── stock-analysis/
│   ├── stock-research-executor/
│   ├── startup-financial-modeling/
│   ├── analyze/
│   └── documentation-templates/
└── .claude/skills/         # Symlinks for Claude Code
```

## Dependencies

Core:
- pandas, xlrd - Data processing
- streamlit - Web UI
- openai, openai-agents - AI agents with SDK
- python-docx - Word document generation

Real-time data:
- tavily-python - Web search
- httpx - HTTP client for APIs

CLI:
- click, rich, tabulate - Command line interface

## Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              ValueInvestmentCoordinator                      │
│   (Routes requests to specialist agents via handoffs)        │
└───────────────┬──────────────┬──────────────┬───────────────┘
                │              │              │
    ┌───────────▼───┐  ┌───────▼───────┐  ┌──▼──────────────┐
    │ScreeningAgent │  │ AnomalyAgent  │  │  ResearchAgent  │
    │               │  │               │  │                 │
    │• screen_stocks│  │• detect_      │  │• web_search     │
    │• compare_     │  │  anomalies    │  │• get_market_news│
    │  stocks       │  │• get_stock_   │  │• get_realtime_  │
    │• analyze_     │  │  fundamentals │  │  price          │
    │  valuation    │  │               │  │• get_price_     │
    │               │  │               │  │  history        │
    └───────────────┘  └───────────────┘  └─────────────────┘
```

## License

MIT License
