"""
Enhanced AI-Powered Report Generator for Value Investment Academy
Generates professional investment reports with deep AI analysis
"""

import os
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def get_openai_client(api_key: str):
    """Get OpenAI client with provided API key."""
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def generate_ai_analysis(client, prompt: str, max_tokens: int = 1500) -> str:
    """Generate AI analysis using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a senior equity research analyst at a top investment bank.
                    Write professional, insightful investment analysis. Be specific, data-driven,
                    and provide actionable insights. Use clear, concise financial language.
                    Do NOT use markdown formatting - write plain text suitable for a Word document."""
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Analysis generation failed: {str(e)}]"


def generate_executive_summary(client, report_data: List[Dict], criteria: Dict) -> str:
    """Generate AI-powered executive summary."""
    undervalued = [d for d in report_data if d.get('valuation') == 'Undervalued']
    clean_companies = [d for d in report_data if d.get('ai_rating') == 'CLEAN']
    minor_companies = [d for d in report_data if d.get('ai_rating') == 'MINOR']
    material_companies = [d for d in report_data if d.get('ai_rating') == 'MATERIAL']

    # Calculate average metrics
    avg_roe = sum(d.get('roe', 0) or 0 for d in report_data) / max(len(report_data), 1)
    avg_margin = sum(d.get('gross_margin', 0) or 0 for d in report_data) / max(len(report_data), 1)

    symbols = [d['symbol'] for d in undervalued[:5]]

    prompt = f"""Write a compelling executive summary (3-4 paragraphs) for a value investment research report.

PORTFOLIO OVERVIEW:
- Total companies analyzed: {len(report_data)}
- Undervalued opportunities (EPV/MC > 1.3): {len(undervalued)} companies
- Clean financials (no anomalies): {len(clean_companies)} companies
- Minor anomalies detected: {len(minor_companies)} companies
- Material distortions found: {len(material_companies)} companies
- Average ROE across universe: {avg_roe:.1f}%
- Average Gross Margin: {avg_margin:.1f}%

TOP UNDERVALUED PICKS: {', '.join(symbols) if symbols else 'None identified'}

SCREENING CRITERIA USED:
- Minimum Gross Margin: {criteria.get('gross_margin', 'N/A')}%
- Minimum Net Margin: {criteria.get('net_margin', 'N/A')}%
- Minimum ROE: {criteria.get('roe', 'N/A')}%
- Maximum Debt-to-Equity: {criteria.get('debt_to_equity', 'N/A')}

Write a professional executive summary that:
1. Opens with the key investment thesis and market opportunity
2. Highlights the most compelling investment candidates with specific reasoning
3. Summarizes the risk-reward profile of the screened universe
4. Provides clear actionable recommendations for investors

Write in a sophisticated, institutional investor style. Be specific about the value proposition."""

    return generate_ai_analysis(client, prompt)


def generate_company_deep_dive(client, company_data: Dict) -> Dict[str, str]:
    """Generate comprehensive AI analysis for a single company."""

    symbol = company_data.get('symbol', 'Unknown')
    company = company_data.get('company', 'Unknown Company')

    # Extract all available metrics
    metrics = {
        'valuation': company_data.get('valuation', 'N/A'),
        'epv': company_data.get('epv'),
        'market_cap': company_data.get('market_cap'),
        'ai_rating': company_data.get('ai_rating', 'N/A'),
        'ai_analysis': company_data.get('ai_analysis', ''),
        'roe': company_data.get('roe'),
        'gross_margin': company_data.get('gross_margin'),
        'net_margin': company_data.get('net_margin'),
        'debt_equity': company_data.get('debt_equity'),
        'fcf_margin': company_data.get('fcf_margin'),
        'roic_wacc': company_data.get('roic_wacc'),
        'rev_growth': company_data.get('rev_growth'),
        'eps_growth': company_data.get('eps_growth')
    }

    # Calculate EPV/MC ratio if available
    epv_mc_ratio = None
    margin_of_safety = None
    if metrics['epv'] and metrics['market_cap']:
        try:
            epv_mc_ratio = float(metrics['epv']) / float(metrics['market_cap'])
            margin_of_safety = (epv_mc_ratio - 1) * 100
        except:
            pass

    # Risk flags from AI anomaly analysis
    ai_rating = metrics.get('ai_rating', 'N/A')
    anomaly_risk = "LOW" if ai_rating == 'CLEAN' else "MODERATE" if ai_rating == 'MINOR' else "ELEVATED"

    # Business Analysis
    business_prompt = f"""Provide a concise business analysis (2-3 paragraphs) for {symbol} ({company}).

FINANCIAL PROFILE:
- Return on Equity: {metrics['roe']}%
- Gross Margin: {metrics['gross_margin']}%
- Net Margin: {metrics['net_margin']}%
- Debt-to-Equity: {metrics['debt_equity']}
- Free Cash Flow Margin: {metrics['fcf_margin']}%
- 5Y Revenue Growth: {metrics['rev_growth']}%
- 5Y EPS Growth: {metrics['eps_growth']}%

Based on these metrics, analyze:
1. The quality and sustainability of the business model
2. Competitive positioning and moat indicators
3. Capital allocation efficiency and management quality signals

Write as a senior analyst providing insights to portfolio managers. Be specific about what the numbers tell us."""

    business_analysis = generate_ai_analysis(client, business_prompt, max_tokens=800)

    # Valuation Analysis
    valuation_prompt = f"""Provide a valuation analysis (2 paragraphs) for {symbol}.

VALUATION METRICS:
- Earnings Power Value (EPV): ${metrics['epv']}M
- Market Capitalization: ${metrics['market_cap']}M
- EPV/Market Cap Ratio: {f'{epv_mc_ratio:.2f}' if epv_mc_ratio else 'N/A'}
- Implied Margin of Safety: {f'{margin_of_safety:+.1f}%' if margin_of_safety else 'N/A'}
- Current Valuation Status: {metrics['valuation']}

Analyze:
1. Whether the current valuation is justified by fundamentals
2. Key assumptions and risks to the valuation thesis
3. Potential catalysts that could close any valuation gap

Be specific about upside/downside scenarios."""

    valuation_analysis = generate_ai_analysis(client, valuation_prompt, max_tokens=600)

    # Risk Assessment
    ai_analysis_excerpt = metrics.get('ai_analysis', '')[:500] if metrics.get('ai_analysis') else 'No AI anomaly analysis available.'

    risk_prompt = f"""Provide a comprehensive risk assessment (2-3 paragraphs) for {symbol}.

RISK INDICATORS:
- AI Anomaly Rating: {ai_rating} (Anomaly Risk: {anomaly_risk})
- Debt-to-Equity: {metrics['debt_equity']}
- ROIC-WACC Spread: {metrics.get('roic_wacc', 'N/A')}

AI ANOMALY ANALYSIS FINDINGS:
{ai_analysis_excerpt}

Analyze:
1. What the AI anomaly findings tell us about earnings quality and financial consistency
2. Specific red flags or concerns investors should monitor
3. Mitigating factors or reasons for confidence

Be balanced but highlight genuine concerns."""

    risk_analysis = generate_ai_analysis(client, risk_prompt, max_tokens=700)

    # Investment Thesis
    thesis_prompt = f"""Write a clear investment thesis (2 paragraphs) for {symbol} ({company}).

KEY FACTS:
- Valuation: {metrics['valuation']} (EPV/MC: {f"{epv_mc_ratio:.2f}" if epv_mc_ratio else 'N/A'})
- ROE: {metrics['roe']}% | Gross Margin: {metrics['gross_margin']}%
- AI Anomaly Rating: {ai_rating} ({anomaly_risk} risk)
- Growth: Revenue {metrics['rev_growth']}% | EPS {metrics['eps_growth']}%

Write a compelling investment thesis that:
1. Summarizes the bull case in clear terms
2. Identifies key risks and monitoring points
3. Provides a clear recommendation (Strong Buy / Buy / Hold / Avoid)

Be decisive and specific about why an investor should or should not own this stock."""

    investment_thesis = generate_ai_analysis(client, thesis_prompt, max_tokens=600)

    return {
        'business_analysis': business_analysis,
        'valuation_analysis': valuation_analysis,
        'risk_analysis': risk_analysis,
        'investment_thesis': investment_thesis
    }


def generate_ceo_analysis(client, report_data: List[Dict]) -> str:
    """Generate AI analysis of management quality indicators."""

    # Aggregate management quality signals from the data
    high_roe_companies = [d for d in report_data if d.get('roe') and d['roe'] > 15]
    clean_financials = [d for d in report_data if d.get('ai_rating') == 'CLEAN']
    low_debt = [d for d in report_data if d.get('debt_equity') and d['debt_equity'] < 0.5]

    prompt = f"""Write a professional analysis (3-4 paragraphs) of management quality indicators across the analyzed companies.

DATA POINTS:
- Companies with ROE > 15% (indicates capital discipline): {len(high_roe_companies)}
- Companies with clean financials (no anomalies): {len(clean_financials)}
- Companies with D/E < 0.5 (indicates conservative financing): {len(low_debt)}
- Total companies analyzed: {len(report_data)}

Top performers by ROE: {', '.join([d['symbol'] for d in sorted(report_data, key=lambda x: x.get('roe') or 0, reverse=True)[:3]])}
Companies with cleanest financials: {', '.join([d['symbol'] for d in clean_financials[:3]]) if clean_financials else 'None'}

Provide analysis on:
1. What these quantitative signals tell us about management quality across the portfolio
2. Which companies show the strongest evidence of shareholder-aligned management
3. Red flags that might indicate poor capital allocation or governance
4. Recommendations for further qualitative due diligence

Write as a governance analyst providing institutional investors with management quality insights."""

    return generate_ai_analysis(client, prompt, max_tokens=1000)


def generate_portfolio_recommendations(client, report_data: List[Dict]) -> str:
    """Generate AI-powered portfolio construction recommendations."""

    # Rank companies by combined score
    ranked = []
    for d in report_data:
        score = 0
        if d.get('valuation') == 'Undervalued':
            score += 3
        elif d.get('valuation') == 'Fair':
            score += 1
        # AI anomaly rating scoring
        ai_rating = d.get('ai_rating', '')
        if ai_rating == 'CLEAN':
            score += 3
        elif ai_rating == 'MINOR':
            score += 1
        if d.get('roe') and d['roe'] > 15:
            score += 2
        elif d.get('roe') and d['roe'] > 10:
            score += 1
        if d.get('roic_wacc') and d['roic_wacc'] > 5:
            score += 1
        ranked.append({**d, 'composite_score': score})

    ranked.sort(key=lambda x: x['composite_score'], reverse=True)

    top_picks = ranked[:5]
    top_picks_summary = "\n".join([
        f"- {d['symbol']}: Score {d['composite_score']}/10, {d.get('valuation', 'N/A')}, ROE {d.get('roe')}%, AI Rating: {d.get('ai_rating', 'N/A')}"
        for d in top_picks
    ])

    avoid_list = [d for d in ranked if d['composite_score'] <= 2]
    avoid_summary = ", ".join([d['symbol'] for d in avoid_list[:5]]) if avoid_list else "None"

    prompt = f"""Write professional portfolio recommendations (4-5 paragraphs) for a value-focused portfolio.

TOP RANKED PICKS (Composite Score out of 10):
{top_picks_summary}

COMPANIES TO AVOID OR MONITOR: {avoid_summary}

UNIVERSE STATISTICS:
- Total companies: {len(report_data)}
- Strong conviction picks (Score ≥ 7): {len([d for d in ranked if d['composite_score'] >= 7])}
- Moderate conviction (Score 4-6): {len([d for d in ranked if 4 <= d['composite_score'] < 7])}
- Low conviction (Score < 4): {len([d for d in ranked if d['composite_score'] < 4])}

Write recommendations covering:
1. Model portfolio construction with specific allocations for top picks
2. Sector/concentration considerations
3. Risk management and position sizing guidance
4. Monitoring triggers and rebalancing criteria
5. Timeline for position building

Write as a portfolio strategist advising a value-focused investment committee."""

    return generate_ai_analysis(client, prompt, max_tokens=1200)


def set_document_styles(doc: Document):
    """Configure professional document styling."""

    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(8)
    style.paragraph_format.line_spacing = 1.15

    # Heading 1 style
    h1_style = doc.styles['Heading 1']
    h1_style.font.name = 'Calibri'
    h1_style.font.size = Pt(18)
    h1_style.font.bold = True
    h1_style.font.color.rgb = RGBColor(0, 51, 102)  # Dark blue

    # Heading 2 style
    h2_style = doc.styles['Heading 2']
    h2_style.font.name = 'Calibri'
    h2_style.font.size = Pt(14)
    h2_style.font.bold = True
    h2_style.font.color.rgb = RGBColor(0, 76, 153)

    # Heading 3 style
    h3_style = doc.styles['Heading 3']
    h3_style.font.name = 'Calibri'
    h3_style.font.size = Pt(12)
    h3_style.font.bold = True
    h3_style.font.color.rgb = RGBColor(51, 102, 153)


def add_styled_table(doc: Document, headers: List[str], rows: List[List[str]],
                     header_color: RGBColor = RGBColor(0, 51, 102)):
    """Add a professionally styled table."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Style header row
    hdr_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        cell = hdr_cells[i]
        cell.text = header
        # Make header bold
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
        # Set header background
        shading = OxmlElement('w:shd')
        shading.set(qn('w:fill'), '003366')  # Dark blue
        cell._tc.get_or_add_tcPr().append(shading)
        # Set header text color to white
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)

    # Add data rows
    for row_data in rows:
        row = table.add_row().cells
        for i, value in enumerate(row_data):
            row[i].text = str(value)

    return table


def fmt_score(val, decimals=2):
    """Format a numeric score."""
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def fmt_pct(val):
    """Format a percentage."""
    return f"{val}%" if val is not None else "N/A"


def fmt_money(val):
    """Format money in millions."""
    return f"${val}M" if val is not None else "N/A"


def generate_professional_report(
    report_data: List[Dict],
    criteria: Dict,
    api_key: str,
    progress_callback=None
) -> BytesIO:
    """
    Generate a professional AI-enhanced investment report.

    Args:
        report_data: List of company data dictionaries
        criteria: Screening criteria used
        api_key: OpenAI API key
        progress_callback: Optional callback function for progress updates

    Returns:
        BytesIO buffer containing the DOCX file
    """

    client = get_openai_client(api_key)
    doc = Document()

    # Apply professional styling
    set_document_styles(doc)

    def update_progress(message: str):
        if progress_callback:
            progress_callback(message)

    # ========================================
    # COVER PAGE
    # ========================================
    update_progress("Creating cover page...")

    doc.add_paragraph("")
    doc.add_paragraph("")
    doc.add_paragraph("")

    title = doc.add_heading('VALUE INVESTMENT ACADEMY', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_heading('Professional Investment Research Report', level=1)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    doc.add_paragraph("")

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run(f"Report Date: {datetime.now().strftime('%B %d, %Y')}\n")
    run.bold = True
    run.font.size = Pt(12)
    info.add_run(f"\nCompanies Analyzed: {len(report_data)}\n")
    info.add_run("Analysis Framework: EPV Valuation + Forensic Risk Assessment\n")
    info.add_run("\nPowered by AI-Enhanced Analysis")

    doc.add_page_break()

    # ========================================
    # TABLE OF CONTENTS
    # ========================================
    doc.add_heading('Table of Contents', level=1)

    toc_items = [
        ("1.", "Executive Summary", "AI-generated investment thesis and key findings"),
        ("2.", "Screening Results", "Quantitative screening results and valuation overview"),
        ("3.", "Detailed Company Analysis", "Deep-dive analysis of each company"),
        ("4.", "Management Quality Assessment", "CEO commitment and governance indicators"),
        ("5.", "Portfolio Recommendations", "Position sizing and allocation strategy"),
        ("6.", "Risk Disclosure", "Methodology notes and limitations")
    ]

    for num, title, desc in toc_items:
        p = doc.add_paragraph()
        p.add_run(f"{num} {title}").bold = True
        p.add_run(f"\n    {desc}")

    doc.add_page_break()

    # ========================================
    # 1. EXECUTIVE SUMMARY (AI-Generated)
    # ========================================
    update_progress("Generating AI executive summary...")

    doc.add_heading('1. Executive Summary', level=1)

    executive_summary = generate_executive_summary(client, report_data, criteria)

    # Add the AI-generated summary
    for paragraph in executive_summary.split('\n\n'):
        if paragraph.strip():
            doc.add_paragraph(paragraph.strip())

    doc.add_paragraph("")

    # Key metrics summary table
    doc.add_heading('Key Metrics at a Glance', level=2)

    undervalued = [d for d in report_data if d.get('valuation') == 'Undervalued']
    clean = [d for d in report_data if d.get('ai_rating') == 'CLEAN']
    minor = [d for d in report_data if d.get('ai_rating') == 'MINOR']
    material = [d for d in report_data if d.get('ai_rating') == 'MATERIAL']

    add_styled_table(doc,
        ['Metric', 'Count', 'Percentage'],
        [
            ['Total Companies Screened', str(len(report_data)), '100%'],
            ['Undervalued (EPV/MC > 1.3)', str(len(undervalued)), f'{len(undervalued)*100//max(len(report_data),1)}%'],
            ['Clean Financials (No Anomalies)', str(len(clean)), f'{len(clean)*100//max(len(report_data),1)}%'],
            ['Minor Anomalies Detected', str(len(minor)), f'{len(minor)*100//max(len(report_data),1)}%'],
            ['Material Distortions Found', str(len(material)), f'{len(material)*100//max(len(report_data),1)}%']
        ]
    )

    doc.add_page_break()

    # ========================================
    # 2. SCREENING RESULTS OVERVIEW
    # ========================================
    update_progress("Building screening results section...")

    doc.add_heading('2. Screening Results Overview', level=1)

    doc.add_heading('Screening Criteria Applied', level=2)
    criteria_table = add_styled_table(doc,
        ['Criterion', 'Threshold', 'Description'],
        [
            ['Gross Margin', f"≥ {criteria.get('gross_margin', 'N/A')}%", 'Pricing power indicator'],
            ['Net Margin', f"≥ {criteria.get('net_margin', 'N/A')}%", 'Profitability measure'],
            ['Return on Equity', f"≥ {criteria.get('roe', 'N/A')}%", 'Capital efficiency'],
            ['Debt-to-Equity', f"≤ {criteria.get('debt_to_equity', 'N/A')}", 'Financial leverage'],
            ['FCF Margin', f"≥ {criteria.get('fcf_margin', 'N/A')}%", 'Cash generation ability'],
            ['ROIC - WACC', f"≥ {criteria.get('roic_wacc', 'N/A')}%", 'Value creation spread']
        ]
    )

    doc.add_paragraph("")

    doc.add_heading('Valuation Distribution', level=2)

    fair_value = [d for d in report_data if d.get('valuation') == 'Fair']
    overvalued = [d for d in report_data if d.get('valuation') == 'Overvalued']

    add_styled_table(doc,
        ['Valuation Category', 'Count', 'Companies'],
        [
            ['UNDERVALUED (EPV/MC > 1.3)', str(len(undervalued)),
             ', '.join([d['symbol'] for d in undervalued][:6]) + ('...' if len(undervalued) > 6 else '')],
            ['FAIR VALUE (0.7 ≤ EPV/MC ≤ 1.3)', str(len(fair_value)),
             ', '.join([d['symbol'] for d in fair_value][:6]) + ('...' if len(fair_value) > 6 else '')],
            ['OVERVALUED (EPV/MC < 0.7)', str(len(overvalued)),
             ', '.join([d['symbol'] for d in overvalued][:6]) + ('...' if len(overvalued) > 6 else '')]
        ]
    )

    doc.add_paragraph("")

    doc.add_heading('Complete Results Summary', level=2)

    # Prepare rows for results table
    results_rows = []
    for d in report_data:
        results_rows.append([
            d.get('symbol', 'N/A'),
            str(d.get('company', 'N/A'))[:20],
            d.get('valuation', 'N/A'),
            d.get('ai_rating', 'N/A'),
            fmt_pct(d.get('roe')),
            fmt_pct(d.get('gross_margin')),
            fmt_pct(d.get('fcf_margin'))
        ])

    add_styled_table(doc,
        ['Symbol', 'Company', 'Valuation', 'AI Rating', 'ROE', 'Gross Margin', 'FCF Margin'],
        results_rows
    )

    doc.add_page_break()

    # ========================================
    # 3. DETAILED COMPANY ANALYSIS (AI-Enhanced)
    # ========================================
    doc.add_heading('3. Detailed Company Analysis', level=1)

    doc.add_paragraph(
        "The following section provides AI-enhanced deep-dive analysis for each company "
        "in the screened universe. Analysis covers business quality, valuation, risk assessment, "
        "and investment thesis."
    )

    for idx, company_data in enumerate(report_data):
        symbol = company_data.get('symbol', 'Unknown')
        company = company_data.get('company', 'Unknown Company')

        update_progress(f"Analyzing {symbol} ({idx + 1}/{len(report_data)})...")

        doc.add_heading(f"3.{idx + 1} {symbol} - {company}", level=2)

        # Company snapshot table
        epv_mc_ratio = None
        if company_data.get('epv') and company_data.get('market_cap'):
            try:
                epv_mc_ratio = float(company_data['epv']) / float(company_data['market_cap'])
            except:
                pass

        add_styled_table(doc,
            ['EPV', 'Market Cap', 'EPV/MC Ratio', 'Valuation', 'AI Rating'],
            [[
                fmt_money(company_data.get('epv')),
                fmt_money(company_data.get('market_cap')),
                f"{epv_mc_ratio:.2f}" if epv_mc_ratio else 'N/A',
                company_data.get('valuation', 'N/A'),
                company_data.get('ai_rating', 'N/A')
            ]]
        )

        doc.add_paragraph("")

        # Generate AI analysis for this company
        ai_analysis = generate_company_deep_dive(client, company_data)

        # Business Analysis Section
        doc.add_heading('Business Quality Analysis', level=3)
        for para in ai_analysis['business_analysis'].split('\n\n'):
            if para.strip():
                doc.add_paragraph(para.strip())

        # Key Metrics Table
        doc.add_heading('Financial Metrics', level=3)
        add_styled_table(doc,
            ['ROE', 'Gross Margin', 'Net Margin', 'D/E', 'FCF Margin'],
            [[
                fmt_pct(company_data.get('roe')),
                fmt_pct(company_data.get('gross_margin')),
                fmt_pct(company_data.get('net_margin')),
                str(company_data.get('debt_equity')) if company_data.get('debt_equity') else 'N/A',
                fmt_pct(company_data.get('fcf_margin'))
            ]]
        )

        doc.add_paragraph("")

        # Valuation Analysis Section
        doc.add_heading('Valuation Analysis', level=3)
        for para in ai_analysis['valuation_analysis'].split('\n\n'):
            if para.strip():
                doc.add_paragraph(para.strip())

        # Risk Assessment Section
        doc.add_heading('Risk Assessment', level=3)

        # AI anomaly analysis summary
        ai_rating = company_data.get('ai_rating', 'N/A')
        rating_desc = {
            'CLEAN': 'No significant distortions detected',
            'MINOR': 'Minor one-offs detected, not material to valuation',
            'MATERIAL': 'Material distortions found, EPV adjustment may be needed'
        }
        add_styled_table(doc,
            ['Assessment', 'Rating', 'Description'],
            [
                ['AI Anomaly Analysis', ai_rating,
                 rating_desc.get(ai_rating, 'Analysis not available')],
                ['Debt-to-Equity', str(company_data.get('debt_equity', 'N/A')),
                 'Balance sheet leverage indicator'],
                ['ROIC-WACC Spread', str(company_data.get('roic_wacc', 'N/A')),
                 'Value creation above cost of capital']
            ]
        )

        doc.add_paragraph("")

        # Include AI anomaly findings if available
        ai_anomaly_text = company_data.get('ai_analysis', '')
        if ai_anomaly_text:
            doc.add_heading('AI Anomaly Findings', level=3)
            for para in ai_anomaly_text.split('\n\n'):
                if para.strip():
                    doc.add_paragraph(para.strip())
            doc.add_paragraph("")

        for para in ai_analysis['risk_analysis'].split('\n\n'):
            if para.strip():
                doc.add_paragraph(para.strip())

        # Investment Thesis Section
        doc.add_heading('Investment Thesis', level=3)
        for para in ai_analysis['investment_thesis'].split('\n\n'):
            if para.strip():
                p = doc.add_paragraph(para.strip())

        doc.add_paragraph("")
        doc.add_paragraph("─" * 50)  # Section divider
        doc.add_paragraph("")

    doc.add_page_break()

    # ========================================
    # 4. MANAGEMENT QUALITY ASSESSMENT
    # ========================================
    update_progress("Generating management quality assessment...")

    doc.add_heading('4. Management Quality Assessment', level=1)

    doc.add_paragraph(
        "Management quality is a critical determinant of long-term shareholder returns. "
        "While quantitative metrics provide signals, qualitative assessment of leadership "
        "commitment and capital allocation discipline is essential for value investors."
    )

    doc.add_heading('Quantitative Signals of Management Quality', level=2)

    # Generate AI management analysis
    ceo_analysis = generate_ceo_analysis(client, report_data)

    for para in ceo_analysis.split('\n\n'):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.add_paragraph("")

    # Management quality ranking table
    doc.add_heading('Management Quality Indicators by Company', level=2)

    mgmt_rows = []
    for d in sorted(report_data, key=lambda x: (x.get('roe') or 0), reverse=True):
        roe = d.get('roe')
        ai_rating = d.get('ai_rating', 'N/A')

        if ai_rating == 'CLEAN' and roe and roe > 15:
            quality = "HIGH"
            signal = "Strong execution, clean financials"
        elif ai_rating in ('CLEAN', 'MINOR') and roe and roe > 10:
            quality = "ABOVE AVG"
            signal = "Good capital discipline"
        elif ai_rating == 'MATERIAL':
            quality = "CAUTION"
            signal = "Material distortions, requires scrutiny"
        else:
            quality = "AVERAGE"
            signal = "Monitor execution"

        mgmt_rows.append([
            d.get('symbol', 'N/A'),
            quality,
            ai_rating,
            fmt_pct(roe),
            signal
        ])

    add_styled_table(doc,
        ['Symbol', 'Quality Rating', 'AI Rating', 'ROE', 'Key Signal'],
        mgmt_rows
    )

    doc.add_paragraph("")

    doc.add_heading('Due Diligence Recommendations', level=2)
    doc.add_paragraph("For each company, we recommend reviewing:")
    doc.add_paragraph("• Proxy statements for executive compensation alignment")
    doc.add_paragraph("• Insider trading patterns over the past 12 months")
    doc.add_paragraph("• Management tenure and track record of guidance accuracy")
    doc.add_paragraph("• Capital allocation history (M&A returns, buyback timing, dividend policy)")
    doc.add_paragraph("• Related party transactions and governance structure")

    doc.add_page_break()

    # ========================================
    # 5. PORTFOLIO RECOMMENDATIONS
    # ========================================
    update_progress("Generating portfolio recommendations...")

    doc.add_heading('5. Portfolio Recommendations', level=1)

    # Generate AI portfolio recommendations
    portfolio_recs = generate_portfolio_recommendations(client, report_data)

    for para in portfolio_recs.split('\n\n'):
        if para.strip():
            doc.add_paragraph(para.strip())

    doc.add_paragraph("")

    # Final recommendation table
    doc.add_heading('Summary Recommendations', level=2)

    rec_rows = []
    for d in report_data:
        ai_clean = d.get('ai_rating') == 'CLEAN'
        ai_minor = d.get('ai_rating') == 'MINOR'
        ai_ok = ai_clean or ai_minor
        underval = d.get('valuation') == 'Undervalued'
        strong_roe = bool(d.get('roe') and d['roe'] > 15)

        score = sum([ai_clean, ai_ok, underval, strong_roe])

        if underval and ai_clean and strong_roe:
            rating = "STRONG BUY"
            action = "Initiate 3-5% position"
        elif underval and ai_ok:
            rating = "BUY"
            action = "Initiate 2-3% position"
        elif d.get('valuation') == 'Fair' and ai_ok:
            rating = "HOLD"
            action = "Monitor for entry"
        else:
            rating = "WATCH"
            action = "Further research needed"

        rec_rows.append([
            d.get('symbol', 'N/A'),
            d.get('valuation', 'N/A'),
            rating,
            f"{score}/4",
            action
        ])

    # Sort by conviction
    rec_rows.sort(key=lambda x: (
        0 if 'STRONG' in x[2] else 1 if 'BUY' == x[2] else 2 if 'HOLD' in x[2] else 3
    ))

    add_styled_table(doc,
        ['Symbol', 'Valuation', 'Rating', 'Conviction', 'Suggested Action'],
        rec_rows
    )

    doc.add_page_break()

    # ========================================
    # 6. RISK DISCLOSURE
    # ========================================
    doc.add_heading('6. Risk Disclosure & Methodology', level=1)

    doc.add_heading('Analysis Methodology', level=2)

    doc.add_paragraph("This report employs a multi-factor value investing framework:")
    doc.add_paragraph("")

    add_styled_table(doc,
        ['Metric', 'Purpose', 'Threshold', 'Interpretation'],
        [
            ['EPV/Market Cap', 'Intrinsic value assessment', '> 1.3', 'Undervalued with margin of safety'],
            ['AI Anomaly Rating', 'Detect one-off financial distortions', 'CLEAN', 'No material distortions in 10Y history'],
            ['ROIC - WACC', 'Value creation spread', '> 0', 'Creating value above cost of capital'],
            ['ROE', 'Capital efficiency', '> 15%', 'Strong returns on equity']
        ]
    )

    doc.add_paragraph("")

    doc.add_heading('Important Limitations', level=2)
    doc.add_paragraph("• Analysis is based on historical financial data and may not reflect current conditions")
    doc.add_paragraph("• AI-generated insights are supplementary and should not replace professional judgment")
    doc.add_paragraph("• Quantitative screens may miss qualitative factors critical to investment success")
    doc.add_paragraph("• Market conditions can change rapidly, affecting valuations and risk profiles")
    doc.add_paragraph("• Past performance and financial metrics do not guarantee future results")

    doc.add_paragraph("")

    doc.add_heading('Disclaimer', level=2)

    disclaimer_para = doc.add_paragraph()
    disclaimer_para.add_run("IMPORTANT: ").bold = True
    disclaimer_para.add_run(
        "This report is for informational and educational purposes only and does not constitute "
        "investment advice, an offer to sell, or a solicitation of an offer to buy any securities. "
        "The analysis and recommendations contained herein are based on publicly available information "
        "and AI-assisted analysis, which may contain errors or omissions. Investors should conduct "
        "their own due diligence and consult with qualified financial advisors before making any "
        "investment decisions. Past performance is not indicative of future results. All investments "
        "involve risk, including the possible loss of principal."
    )

    doc.add_paragraph("")
    doc.add_paragraph("")

    # Footer
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("─" * 40)
    footer.add_run(f"\nGenerated by Value Investment Academy | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    footer.add_run("\nPowered by AI-Enhanced Analysis")

    # Save to buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    update_progress("Report generation complete!")

    return buffer
