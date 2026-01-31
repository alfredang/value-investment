# Value Investment Tool

A Python tool for value investors that provides criteria-based stock screening, automated valuation, and financial anomaly detection for US and Singapore markets.

**[🚀 Try the Live Demo](https://value-investment.streamlit.app)** *(Deploy your own to use)*

## Features

### 1. Stock Screener
Filter stocks based on 10 fundamental criteria:
- Gross Margin %
- Net Margin %
- Return on Assets (ROA) %
- Return on Equity (ROE) %
- 5-Year Revenue Growth Rate
- 5-Year EPS Growth Rate
- Debt-to-Equity Ratio
- Free Cash Flow Margin %
- ROIC - WACC (Return on Invested Capital minus Weighted Average Cost of Capital)
- ROTE - WACC (Return on Tangible Equity minus WACC)

### 2. Automated Valuation
Classifies stocks using Earnings Power Value (EPV) vs Market Cap:
- **Undervalued**: EPV/MC > 1.3 (EPV is 30%+ above Market Cap)
- **Fair Value**: 0.7 <= EPV/MC <= 1.3
- **Overvalued**: EPV/MC < 0.7 (EPV is 30%+ below Market Cap)

### 3. Anomaly Detection
Detects financial distortions and red flags:
- **Beneish M-Score**: Earnings manipulation probability
- **Altman Z-Score**: Bankruptcy/distress risk
- **Piotroski F-Score**: Financial strength
- **Sloan Ratio**: Accrual-based earnings quality
- **One-off Events**: Unusual YoY changes in revenue, margins, EPS
- **Cash Flow Mismatches**: Positive earnings with negative operating cash flow
- **Balance Sheet Anomalies**: Receivables/inventory growth vs revenue

## Quick Start

### Option 1: Web App (Streamlit)

```bash
# Clone and run locally
git clone https://github.com/alfredang/value-investment.git
cd value-investment
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Command Line Interface

```bash
# Screen US stocks
python main.py screen --market US --roe 15 --gross-margin 30

# Analyze company for anomalies
python main.py analyze DDI

# Show help
python main.py --help
```

## Deploy to Streamlit Cloud (Free)

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repository
5. Set main file path: `app.py`
6. Click "Deploy"

Your app will be live at `https://[your-app-name].streamlit.app`

## CLI Usage

### Screen Stocks
```bash
# Screen US stocks with specific criteria
python main.py screen --market US --roe 15 --gross-margin 30 --debt-equity 1.0

# Screen Singapore stocks
python main.py screen --market SG --roe 10 --gross-margin 25

# Use a config file for criteria
python main.py screen --market US --config config/default_criteria.json

# Export results to CSV
python main.py screen --market US --roe 15 --output results.csv
```

### Analyze for Anomalies
```bash
# Analyze a specific company
python main.py analyze DDI
python main.py analyze TEX
python main.py analyze CHCI

# Export report to file
python main.py analyze DDI --output ddi_report.txt
```

### Full Pipeline
```bash
# Screen + valuation + filter undervalued only
python main.py full --market US --roe 15 --undervalued-only

# With output file
python main.py full --market US --config config/default_criteria.json --output results.csv
```

### Other Commands
```bash
# Show available screening criteria
python main.py criteria

# Show available symbols for anomaly analysis
python main.py symbols
```

## Screening Criteria

| Parameter | Description | CLI Option |
|-----------|-------------|------------|
| gross_margin | Gross Margin % (>= threshold) | --gross-margin |
| net_margin | Net Margin % (>= threshold) | --net-margin |
| roa | Return on Assets % (>= threshold) | --roa |
| roe | Return on Equity % (>= threshold) | --roe |
| revenue_growth_5y | 5-Year Revenue Growth (>= threshold) | --revenue-growth |
| eps_growth_5y | 5-Year EPS Growth (>= threshold) | --eps-growth |
| debt_to_equity | Debt-to-Equity Ratio (<= threshold) | --debt-equity |
| fcf_margin | Free Cash Flow Margin % (>= threshold) | --fcf-margin |
| roic_wacc | ROIC minus WACC (>= threshold) | --roic-wacc |
| rote_wacc | ROTE minus WACC (>= threshold) | --rote-wacc |

## Config File Format

Create a JSON file with your preferred criteria:

```json
{
  "gross_margin": 20,
  "net_margin": 5,
  "roa": 5,
  "roe": 10,
  "revenue_growth_5y": 5,
  "eps_growth_5y": 5,
  "debt_to_equity": 1.5,
  "fcf_margin": 5,
  "roic_wacc": 0,
  "rote_wacc": 0
}
```

## Data Files

The tool accepts:
- **Screener CSV**: Stock fundamentals data with columns for margins, ratios, growth rates
- **Anomaly XLS**: 30-year financial history for detailed analysis

## Dependencies

- pandas >= 2.0.0
- xlrd >= 2.0.0
- click >= 8.0.0
- tabulate >= 0.9.0
- rich >= 13.0.0
- streamlit >= 1.28.0

## License

MIT License
