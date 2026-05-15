"""Generate a SAMPLE Excel report for the user to review before any code
integration. Standalone — does NOT modify app.py or any existing file.

Run:  python _make_sample_excel.py
Output: Value_Investment_Sample_Report.xlsx in this directory
"""
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import pandas as pd

from valuation import Valuator
from excel_report import generate_excel_report


def _to_company_data(row, ai_rating="MINOR", ai_text=""):
    """Map a screener CSV row into the shape generate_excel_report expects."""
    def _num(v):
        try:
            f = float(v)
            return f if f == f else None
        except (TypeError, ValueError):
            return None

    return {
        "symbol": row.get("Symbol"),
        "company": row.get("Company"),
        "sector": row.get("Sector"),
        "industry": row.get("Industry"),
        "subindustry": row.get("Subindustry"),
        "exchange": row.get("Exchange"),
        "currency": row.get("Currency") or "USD",
        "current_price": _num(row.get("Current Price")),
        "epv": _num(row.get("Earnings Power Value (EPV)")),
        "market_cap": _num(row.get("Market Cap ($M)")),
        "low_iv": _num(row.get("Low IV")),
        "high_iv": _num(row.get("High IV")),
        "valuation": row.get("Valuation"),
        "gross_margin": _num(row.get("Gross Margin %")),
        "net_margin": _num(row.get("Net Margin %")),
        "roe": _num(row.get("ROE %")),
        "roa": _num(row.get("ROA %")),
        "fcf_margin": _num(row.get("FCF Margin %")),
        "roic": _num(row.get("ROIC %")),
        "wacc": _num(row.get("WACC %")),
        "roic_wacc": _num(row.get("ROIC-WACC")),
        "rote_wacc": _num(row.get("ROTE-WACC")),
        "debt_equity": _num(row.get("Debt-to-Equity")),
        "rev_growth": _num(row.get("5-Year Revenue Growth Rate (Per Share)")),
        "eps_growth": _num(row.get("5-Year EPS without NRI Growth Rate")),
        "ai_rating": ai_rating,
        "ai_analysis": ai_text,
    }


def main():
    us_path = r"C:\Users\Afzaana Jaffar\Downloads\US All In One Screeners 2026-01-12 04_55_25.csv"
    sg_path = r"C:\Users\Afzaana Jaffar\Downloads\SG All In One Screeners 2026-01-12 04_29_45.csv"

    us = pd.read_csv(us_path, encoding="utf-8-sig")
    sg = pd.read_csv(sg_path, encoding="utf-8-sig")
    us.columns = us.columns.str.strip()
    sg.columns = sg.columns.str.strip()
    us["Market"] = "US"
    sg["Market"] = "SG"
    combined = pd.concat([us, sg], ignore_index=True)

    # Run the in-house DCF so Low IV / High IV / Valuation columns exist
    combined = Valuator().analyze_dataframe(combined)

    # Pick 3 representative companies for the sample —
    # CALM = US-GAAP filer, JFIN = IFRS Chinese ADR, AFYA = IFRS Brazilian filer.
    # This proves Section D works across both XBRL taxonomies (us-gaap + ifrs-full).
    sample_tickers = ["CALM", "JFIN", "AFYA"]
    picks = combined[combined["Symbol"].isin(sample_tickers)]
    print(f"Building sample Excel for: {list(picks['Symbol'])}")

    report_data = []
    fake_anomaly_excerpts = {
        "CALM": ("MATERIAL", "FY2023 net income jumped 5.7x ($133M -> $758M) driven by "
                              "avian flu industry-supply collapse and resulting wholesale "
                              "egg-price spike. FY2024 normalized as supply recovered; FY2025 "
                              "shows another cycle. Reported metrics are accurate but "
                              "investors should treat the FY2023 figure as non-recurring."),
        "JFIN": ("MINOR", "Operating cash flow has tracked closely with reported net income "
                            "across 2022-2024, suggesting earnings quality is intact. ROE "
                            "trajectory normalising as equity base grows."),
        "AFYA": ("MINOR", "Brazilian education sector consolidator. Revenue and EPS have "
                            "grown consistently 2019-2024 with no material non-recurring items "
                            "in the past three fiscal years."),
    }
    for _, r in picks.iterrows():
        rating, excerpt = fake_anomaly_excerpts.get(r["Symbol"], ("MINOR", ""))
        report_data.append(_to_company_data(r, ai_rating=rating, ai_text=excerpt))

    criteria = {
        "exchanges": ["NASDAQ", "NYSE", "SGX"],
        "gross_margin": 20,
        "net_margin": 5,
        "roe": 10,
        "roa": 5,
        "fcf_margin": 0,
        "revenue_growth_5y": 0,
        "eps_growth_5y": 0,
        "roic_wacc": 0,
        "rote_wacc": 0,
        "debt_to_equity": 1.5,
    }

    def _progress(msg):
        print(f"  {msg}")

    print("\nGenerating Excel sample (this calls Firecrawl + Claude for some sections — please wait)...")
    buf = generate_excel_report(
        report_data=report_data,
        criteria=criteria,
        progress_callback=_progress,
        universe_df=combined,
    )

    out_path = r"C:\Users\Afzaana Jaffar\Downloads\Value_Investment_Sample_Report_SEC_EDGAR.xlsx"
    with open(out_path, "wb") as f:
        f.write(buf.getvalue())

    print(f"\nSample Excel saved to: {out_path}")
    print(f"   File size: {os.path.getsize(out_path) / 1024:.1f} KB")
    print(f"\nOpen this file to review the layout, then tell me:")
    print(f"   (a) yes integrate it as an Export button in Step 3, or")
    print(f"   (b) what you want changed before I integrate.")


if __name__ == "__main__":
    main()
