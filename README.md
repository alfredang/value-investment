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

### 4. AI Agents (Powered by OpenAI)
- **Screening Agent**: Finds value stocks and explains reasoning
- **Anomaly Agent**: Interprets financial red flags forensically
- **Research Agent**: Generates investment theses
- Agents use tools to screen stocks, detect anomalies, compare companies

## Quick Start

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/alfredang/value-investment.git
cd value-investment

# Install with uv
uv sync

# Set up API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run the web app
uv run streamlit run app.py
```

### Using pip

```bash
git clone https://github.com/alfredang/value-investment.git
cd value-investment
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your OpenAI API key

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
from agents import ScreeningAgent, AnomalyAgent

# Stock screening with AI
agent = ScreeningAgent()
response = agent.chat("Find undervalued US stocks with ROE > 15% and low debt")
print(response.content)

# Anomaly detection with AI
agent = AnomalyAgent()
response = agent.chat("Analyze TEX for financial red flags")
print(response.content)
```

## Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your fork
4. Add `OPENAI_API_KEY` as a secret in app settings
5. Deploy!

## Configuration

### Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-api-key-here
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
├── app.py              # Streamlit web app
├── main.py             # CLI interface
├── agents.py           # AI agents with tools
├── screener.py         # Stock screening logic
├── valuation.py        # EPV-based valuation
├── anomaly_detector.py # Financial anomaly detection
├── data_loader.py      # CSV/XLS file parsing
├── pyproject.toml      # uv project config
└── requirements.txt    # pip dependencies
```

## Dependencies

- pandas, xlrd - Data processing
- streamlit - Web UI
- openai - AI agents
- click, rich, tabulate - CLI

## License

MIT License
