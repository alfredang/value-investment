"""
Summary Report Module – Professional Step 3 Dashboard
Bloomberg/FactSet-inspired investment report visualization.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
from datetime import datetime

# ─── Shared Plotly dark theme ───────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family="'Segoe UI', Tahoma, sans-serif", color='#cbd5e1', size=12),
    margin=dict(t=36, b=28, l=40, r=20),
    legend=dict(orientation='h', y=-0.18, bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', size=11)),
    hoverlabel=dict(bgcolor='#1e293b', font_color='#f8fafc', bordercolor='#334155'),
)

CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444',
                '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16']

RATING_BADGE = {
    'STRONG BUY': ('#052e16', '#10b981'),
    'BUY':        ('rgba(16,185,129,0.15)', '#10b981'),
    'HOLD':       ('rgba(245,158,11,0.15)', '#f59e0b'),
    'WATCH':      ('rgba(239,68,68,0.15)', '#ef4444'),
}

VAL_BADGE = {
    'Undervalued': ('#052e16', '#10b981'),
    'Fair':        ('rgba(245,158,11,0.15)', '#f59e0b'),
    'Overvalued':  ('rgba(239,68,68,0.15)', '#ef4444'),
}

AI_BADGE = {
    'CLEAN':    ('#052e16', '#10b981'),
    'MINOR':    ('rgba(245,158,11,0.15)', '#f59e0b'),
    'MATERIAL': ('rgba(239,68,68,0.15)', '#ef4444'),
}


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(val, suffix='%', decimals=1):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    try:
        return f"{float(val):.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return 'N/A'


def _metric_color(val, good, ok=None, lower_better=False):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return '#cbd5e1'
    try:
        v = float(val)
    except (ValueError, TypeError):
        return '#cbd5e1'
    if lower_better:
        return '#10b981' if v < good else '#f59e0b' if ok and v < ok else '#ef4444'
    return '#10b981' if v > good else '#f59e0b' if ok and v > ok else '#ef4444'


def _badge(text, bg, fg):
    return (f"<span style='display:inline-block;padding:2px 8px;border-radius:4px;"
            f"font-size:11px;font-weight:600;background:{bg};color:{fg};"
            f"border:1px solid {fg}30;letter-spacing:0.3px;'>{text}</span>")


def _safe(val, default=0):
    if val is None:
        return default
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (ValueError, TypeError):
        return default


def _progress_bar_html(value, label, min_val=-20, max_val=50):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"<div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.3px'>{label}</div><div style='font-size:12px;color:#cbd5e1;'>N/A</div>"
    v = _safe(value)
    pct = max(0, min(100, (v - min_val) / (max_val - min_val) * 100))
    bar_c = '#10b981' if v > 0 else '#ef4444'
    sign = '+' if v > 0 else ''
    return f"""<div>
      <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.3px'>{label}</div>
      <div style='display:flex;align-items:center;gap:6px;margin-top:2px'>
        <div style='flex:1;height:6px;background:rgba(51,65,85,.5);border-radius:3px;overflow:hidden'>
          <div style='width:{pct:.0f}%;height:100%;background:{bar_c};border-radius:3px'></div>
        </div>
        <span style='color:{bar_c};font-weight:600;font-size:12px;min-width:50px;text-align:right;font-variant-numeric:tabular-nums'>{sign}{v:.1f}%</span>
      </div>
    </div>"""


def _metric_cell(label, value, suffix='%', good=None, ok=None, lower_better=False):
    color = _metric_color(value, good, ok, lower_better) if good is not None else '#cbd5e1'
    display = _fmt(value, suffix)
    return f"""<div>
      <div style='font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.3px'>{label}</div>
      <div style='font-size:13px;font-weight:600;color:{color};font-variant-numeric:tabular-nums'>{display}</div>
    </div>"""


# ─── Recommendation scoring ────────────────────────────────────────────────────

def compute_recommendation(d: dict) -> Tuple[str, int, str]:
    score = 0
    if d.get('valuation') == 'Undervalued':
        score += 40
    elif d.get('valuation') == 'Fair':
        score += 20
    r = d.get('ai_rating', '')
    if r == 'CLEAN':
        score += 25
    elif r == 'MINOR':
        score += 15
    roe = _safe(d.get('roe'))
    gm = _safe(d.get('gross_margin'))
    if roe > 15 and gm > 30:
        score += 20
    elif roe > 10 and gm > 20:
        score += 10
    de = d.get('debt_equity')
    fcf = _safe(d.get('fcf_margin'))
    if de is not None:
        dev = _safe(de, 999)
        if dev < 1.0 and fcf > 10:
            score += 15
        elif dev < 1.5 and fcf > 5:
            score += 8
    if score >= 80:
        return ('STRONG BUY', score, 'Initiate position at current levels')
    elif score >= 60:
        return ('BUY', score, 'Accumulate on weakness')
    elif score >= 40:
        return ('HOLD', score, 'Monitor for catalysts')
    return ('WATCH', score, 'Add to watchlist only')


# ─── CSS injection ──────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""<style>
    .sr-header{border-bottom:2px solid #10b981;padding-bottom:12px;margin-bottom:16px}
    .sr-title{font-size:22px;font-weight:700;color:#f8fafc;letter-spacing:1.5px;margin:0}
    .sr-subtitle{font-size:13px;color:#64748b;margin-top:4px}
    .sr-kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
    .sr-kpi{background:linear-gradient(135deg,rgba(30,41,59,.95),rgba(15,23,42,.95));
      border:1px solid rgba(51,65,85,.4);border-radius:8px;padding:12px 14px;
      border-left:3px solid var(--accent-color,#3b82f6)}
    .sr-kpi-label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
    .sr-kpi-value{font-size:22px;font-weight:700;color:var(--accent-color,#f8fafc);
      font-variant-numeric:tabular-nums;margin-top:2px}
    .sr-kpi-delta{font-size:11px;color:#94a3b8;margin-top:2px}
    .sr-section{margin-top:20px;margin-bottom:10px;padding-bottom:6px;
      border-bottom:1px solid rgba(51,65,85,.3)}
    .sr-section-title{font-size:14px;font-weight:600;color:#94a3b8;
      text-transform:uppercase;letter-spacing:1px}
    .sr-picks-row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 16px}
    .sr-pick{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);
      border-radius:6px;padding:6px 12px;display:inline-flex;align-items:center;gap:8px}
    .sr-pick-sym{font-size:13px;font-weight:700;color:#10b981}
    .sr-pick-price{font-size:12px;color:#cbd5e1;font-variant-numeric:tabular-nums}
    .sr-card{background:linear-gradient(135deg,rgba(30,41,59,.95),rgba(15,23,42,.95));
      border:1px solid rgba(51,65,85,.4);border-radius:8px;padding:14px 16px;margin-bottom:10px}
    .sr-card-header{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}
    .sr-card-sym{font-size:17px;font-weight:700;color:#f8fafc;letter-spacing:.5px}
    .sr-card-company{font-size:12px;color:#64748b;flex:1;min-width:100px;overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap}
    .sr-card-price{font-size:15px;font-weight:600;color:#f8fafc;font-variant-numeric:tabular-nums}
    .sr-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:6px 12px}
    .sr-valuation-row{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;padding-top:8px;
      border-top:1px solid rgba(51,65,85,.3);font-size:12px;color:#94a3b8}
    .sr-valuation-row span{font-weight:600;font-variant-numeric:tabular-nums}
    .sr-rec-table{width:100%;border-collapse:collapse;font-size:13px}
    .sr-rec-table th{text-align:left;padding:8px 10px;font-size:11px;color:#64748b;
      text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid rgba(51,65,85,.4)}
    .sr-rec-table td{padding:8px 10px;border-bottom:1px solid rgba(51,65,85,.2);
      color:#cbd5e1;font-variant-numeric:tabular-nums}
    .sr-rec-table tr:hover td{background:rgba(59,130,246,.05)}
    .sr-conviction{display:inline-flex;align-items:center;gap:6px}
    .sr-conv-bar{width:80px;height:6px;background:rgba(51,65,85,.5);border-radius:3px;overflow:hidden}
    .sr-conv-fill{height:100%;border-radius:3px}
    </style>""", unsafe_allow_html=True)


# ─── Section A: Executive Summary ──────────────────────────────────────────────

def render_executive_summary(data: List[Dict]):
    total = len(data)
    undervalued = [d for d in data if d.get('valuation') == 'Undervalued']
    fair = [d for d in data if d.get('valuation') == 'Fair']
    overvalued = [d for d in data if d.get('valuation') == 'Overvalued']
    clean = [d for d in data if d.get('ai_rating') == 'CLEAN']
    minor = [d for d in data if d.get('ai_rating') == 'MINOR']
    material = [d for d in data if d.get('ai_rating') == 'MATERIAL']
    avg_roe = sum(_safe(d.get('roe')) for d in data) / max(total, 1)

    # Header
    st.markdown(f"""<div class='sr-header'>
      <div class='sr-title'>INVESTMENT ANALYSIS REPORT</div>
      <div class='sr-subtitle'>{datetime.now().strftime('%d %b %Y').upper()} &nbsp;|&nbsp; {total} COMPANIES ANALYZED &nbsp;|&nbsp; AI-ENHANCED FORENSIC SCREENING</div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    roe_color = '#10b981' if avg_roe > 15 else '#f59e0b' if avg_roe > 10 else '#ef4444'
    st.markdown(f"""<div class='sr-kpi-row'>
      <div class='sr-kpi' style='--accent-color:#3b82f6'>
        <div class='sr-kpi-label'>Total Analyzed</div>
        <div class='sr-kpi-value'>{total}</div>
        <div class='sr-kpi-delta'>Passed all screening criteria</div>
      </div>
      <div class='sr-kpi' style='--accent-color:#10b981'>
        <div class='sr-kpi-label'>Undervalued</div>
        <div class='sr-kpi-value'>{len(undervalued)}</div>
        <div class='sr-kpi-delta'>EPV/MC &gt; threshold</div>
      </div>
      <div class='sr-kpi' style='--accent-color:#10b981'>
        <div class='sr-kpi-label'>Clean Financials</div>
        <div class='sr-kpi-value'>{len(clean)}</div>
        <div class='sr-kpi-delta'>No material distortions</div>
      </div>
      <div class='sr-kpi' style='--accent-color:{roe_color}'>
        <div class='sr-kpi-label'>Avg ROE</div>
        <div class='sr-kpi-value'>{avg_roe:.1f}%</div>
        <div class='sr-kpi-delta'>Portfolio average</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Distribution charts side by side
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        cats = ['Undervalued', 'Fair', 'Overvalued']
        vals = [len(undervalued), len(fair), len(overvalued)]
        colors = ['#10b981', '#f59e0b', '#ef4444']
        fig.add_trace(go.Bar(y=cats, x=vals, orientation='h',
                             marker_color=colors, text=vals, textposition='auto',
                             textfont=dict(color='#f8fafc', size=13, family="'Segoe UI'")))
        fig.update_layout(height=160, title=dict(text='VALUATION DISTRIBUTION', font=dict(size=12, color='#64748b')),
                          xaxis=dict(showgrid=False, showticklabels=False),
                          yaxis=dict(autorange='reversed', tickfont=dict(size=12, color='#cbd5e1')),
                          **PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        cats2 = ['CLEAN', 'MINOR', 'MATERIAL']
        vals2 = [len(clean), len(minor), len(material)]
        colors2 = ['#10b981', '#f59e0b', '#ef4444']
        fig2.add_trace(go.Bar(y=cats2, x=vals2, orientation='h',
                              marker_color=colors2, text=vals2, textposition='auto',
                              textfont=dict(color='#f8fafc', size=13, family="'Segoe UI'")))
        fig2.update_layout(height=160, title=dict(text='AI ANOMALY ASSESSMENT', font=dict(size=12, color='#64748b')),
                           xaxis=dict(showgrid=False, showticklabels=False),
                           yaxis=dict(autorange='reversed', tickfont=dict(size=12, color='#cbd5e1')),
                           **PLOTLY_DARK)
        st.plotly_chart(fig2, use_container_width=True)

    # Top Picks strip
    top_picks = [d for d in data if d.get('valuation') == 'Undervalued' and d.get('ai_rating') in ('CLEAN', 'MINOR')]
    if top_picks:
        st.markdown("<div class='sr-section'><div class='sr-section-title'>TOP PICKS</div></div>", unsafe_allow_html=True)
        picks_html = "<div class='sr-picks-row'>"
        for d in top_picks[:6]:
            price_str = _fmt(d.get('current_price'), '$', 2) if d.get('current_price') else ''
            ai_bg, ai_fg = AI_BADGE.get(d.get('ai_rating', ''), ('rgba(148,163,184,.15)', '#94a3b8'))
            picks_html += f"""<div class='sr-pick'>
              <span class='sr-pick-sym'>{d['symbol']}</span>
              <span class='sr-pick-price'>{price_str}</span>
              {_badge(d.get('ai_rating', ''), ai_bg, ai_fg)}
            </div>"""
        picks_html += "</div>"
        st.markdown(picks_html, unsafe_allow_html=True)


# ─── Section B: Portfolio Comparison Charts ────────────────────────────────────

def render_comparison_charts(data: List[Dict]):
    st.markdown("<div class='sr-section'><div class='sr-section-title'>PORTFOLIO ANALYTICS</div></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Radar Comparison", "Valuation Map", "Metrics Comparison"])

    with tab1:
        _render_radar(data)
    with tab2:
        _render_scatter(data)
    with tab3:
        _render_grouped_bar(data)


def _render_radar(data: List[Dict]):
    categories = ['ROE', 'Gross Margin', 'Net Margin', 'FCF Margin', 'Rev Growth', 'Value Score']
    fig = go.Figure()
    for i, d in enumerate(data[:8]):
        epv_mc = _safe(d.get('epv')) / max(_safe(d.get('market_cap'), 1), 1)
        vals = [
            min(_safe(d.get('roe')) / 50 * 100, 100),
            min(_safe(d.get('gross_margin')) / 80 * 100, 100),
            min(_safe(d.get('net_margin')) / 40 * 100, 100),
            min(_safe(d.get('fcf_margin')) / 40 * 100, 100),
            min(max(_safe(d.get('rev_growth')), 0) / 30 * 100, 100),
            min(epv_mc * 50, 100),
        ]
        c = CHART_COLORS[i % len(CHART_COLORS)]
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill='toself', fillcolor=f'rgba({r},{g},{b},0.08)',
            line=dict(color=c, width=2), name=d['symbol'],
        ))
    fig.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor='rgba(51,65,85,.3)',
                                   tickfont=dict(color='#64748b', size=10)),
                   angularaxis=dict(gridcolor='rgba(51,65,85,.3)',
                                    tickfont=dict(color='#cbd5e1', size=11))),
        height=440, showlegend=True, **PLOTLY_DARK)
    st.plotly_chart(fig, use_container_width=True)


def _render_scatter(data: List[Dict]):
    fig = go.Figure()
    rating_map = {'CLEAN': '#10b981', 'MINOR': '#f59e0b', 'MATERIAL': '#ef4444'}
    for rating, color in rating_map.items():
        subset = [d for d in data if d.get('ai_rating') == rating]
        if not subset:
            continue
        mc_vals = [max(_safe(d.get('market_cap'), 1), 1) for d in subset]
        epv_mc = [_safe(d.get('epv')) / mc for d, mc in zip(subset, mc_vals)]
        roe_sizes = [max(8, min(28, _safe(d.get('roe'), 10))) for d in subset]
        fig.add_trace(go.Scatter(
            x=mc_vals, y=epv_mc, mode='markers+text',
            name=f'AI: {rating}',
            text=[d['symbol'] for d in subset], textposition='top center',
            textfont=dict(color='#cbd5e1', size=11),
            marker=dict(size=roe_sizes, color=color, opacity=0.85,
                        line=dict(width=1, color='rgba(255,255,255,.2)')),
            hovertemplate='<b>%{text}</b><br>MCap: $%{x:,.0f}M<br>EPV/MC: %{y:.2f}x<extra></extra>',
        ))
    fig.add_hline(y=1.0, line_dash='dash', line_color='#64748b', opacity=0.5,
                  annotation_text='Fair Value', annotation_font_color='#64748b')
    fig.add_hline(y=1.3, line_dash='dot', line_color='#10b981', opacity=0.4,
                  annotation_text='Undervalued', annotation_font_color='#10b981')
    fig.update_layout(xaxis_title='Market Cap ($M)', yaxis_title='EPV / Market Cap',
                      xaxis_type='log', height=400, **PLOTLY_DARK)
    st.plotly_chart(fig, use_container_width=True)


def _render_grouped_bar(data: List[Dict]):
    symbols = [d['symbol'] for d in data]
    metrics = [
        ('ROE %', 'roe', '#3b82f6'),
        ('Gross Margin %', 'gross_margin', '#10b981'),
        ('Net Margin %', 'net_margin', '#f59e0b'),
        ('FCF Margin %', 'fcf_margin', '#8b5cf6'),
    ]
    fig = go.Figure()
    for label, key, color in metrics:
        vals = [_safe(d.get(key)) for d in data]
        fig.add_trace(go.Bar(
            x=symbols, y=vals, name=label, marker_color=color, opacity=0.85,
            text=[f'{v:.1f}' for v in vals], textposition='outside',
            textfont=dict(size=10, color='#94a3b8'),
        ))
    fig.add_hline(y=15, line_dash='dot', line_color='#3b82f6', opacity=0.3,
                  annotation_text='ROE 15%', annotation_font_color='#3b82f6', annotation_font_size=9)
    fig.add_hline(y=30, line_dash='dot', line_color='#10b981', opacity=0.3,
                  annotation_text='GM 30%', annotation_font_color='#10b981', annotation_font_size=9)
    fig.update_layout(barmode='group', height=400,
                      xaxis=dict(tickfont=dict(size=12, color='#f8fafc')),
                      **PLOTLY_DARK)
    st.plotly_chart(fig, use_container_width=True)


# ─── Section C: Company Deep Dive Cards ────────────────────────────────────────

def render_company_cards(data: List[Dict]):
    st.markdown("<div class='sr-section'><div class='sr-section-title'>COMPANY ANALYSIS</div></div>", unsafe_allow_html=True)

    for d in data:
        sym = d.get('symbol', '')
        company = d.get('company', '')
        val = d.get('valuation', 'N/A')
        ai = d.get('ai_rating', 'N/A')
        price = d.get('current_price')

        # Badges
        val_bg, val_fg = VAL_BADGE.get(val, ('rgba(148,163,184,.15)', '#94a3b8'))
        ai_bg, ai_fg = AI_BADGE.get(ai, ('rgba(148,163,184,.15)', '#94a3b8'))

        price_str = f"${float(price):,.2f}" if price else ''

        # Metrics grid row 1
        row1 = ''.join([
            _metric_cell('ROE', d.get('roe'), '%', good=15, ok=10),
            _metric_cell('Gross Margin', d.get('gross_margin'), '%', good=30, ok=20),
            _metric_cell('Net Margin', d.get('net_margin'), '%', good=10, ok=5),
            _metric_cell('ROA', d.get('roa'), '%', good=10, ok=5),
            _metric_cell('FCF Margin', d.get('fcf_margin'), '%', good=10, ok=5),
        ])

        # Metrics grid row 2
        row2_left = ''.join([
            _metric_cell('D/E', d.get('debt_equity'), 'x', good=1.0, ok=1.5, lower_better=True),
            _metric_cell('5Y Rev Gr', d.get('rev_growth'), '%', good=10, ok=5),
            _metric_cell('5Y EPS Gr', d.get('eps_growth'), '%', good=10, ok=5),
        ])
        bar1 = _progress_bar_html(d.get('roic_wacc'), 'ROIC-WACC')
        bar2 = _progress_bar_html(d.get('rote_wacc'), 'ROTE-WACC')

        # Valuation row
        epv = d.get('epv')
        mc = d.get('market_cap')
        epv_mc = float(epv) / max(float(mc), 1) if epv and mc else None
        mos = (epv_mc - 1) * 100 if epv_mc else None
        epv_str = f"${float(epv):,.0f}M" if epv else 'N/A'
        mc_str = f"${float(mc):,.0f}M" if mc else 'N/A'
        epv_mc_str = f"{epv_mc:.2f}x" if epv_mc else 'N/A'
        mos_color = '#10b981' if mos and mos > 0 else '#ef4444' if mos else '#cbd5e1'
        mos_str = f"{mos:+.1f}%" if mos else 'N/A'

        card_html = f"""<div class='sr-card'>
          <div class='sr-card-header'>
            <span class='sr-card-sym'>{sym}</span>
            <span class='sr-card-company'>{company}</span>
            <span class='sr-card-price'>{price_str}</span>
            {_badge(val, val_bg, val_fg)}
            {_badge(ai, ai_bg, ai_fg)}
          </div>
          <div class='sr-grid'>{row1}</div>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr 1fr;gap:6px 12px;margin-top:6px'>
            {row2_left}
            <div>{bar1}</div>
            <div>{bar2}</div>
          </div>
          <div class='sr-valuation-row'>
            <div>EPV: <span style='color:#f8fafc'>{epv_str}</span></div>
            <div>MCap: <span style='color:#f8fafc'>{mc_str}</span></div>
            <div>EPV/MC: <span style='color:#f8fafc'>{epv_mc_str}</span></div>
            <div>MoS: <span style='color:{mos_color}'>{mos_str}</span></div>
          </div>
        </div>"""
        st.markdown(card_html, unsafe_allow_html=True)

        # AI Analysis expander
        if d.get('ai_analysis'):
            with st.expander(f"AI Anomaly Analysis — {sym}", expanded=False):
                st.markdown(d['ai_analysis'])


# ─── Section D: Recommendation Table ──────────────────────────────────────────

def render_recommendation_table(data: List[Dict]):
    st.markdown("<div class='sr-section'><div class='sr-section-title'>INVESTMENT RECOMMENDATIONS</div></div>", unsafe_allow_html=True)

    # Score and sort
    scored = []
    for d in data:
        rating, score, action = compute_recommendation(d)
        scored.append({**d, '_rating': rating, '_score': score, '_action': action})
    scored.sort(key=lambda x: x['_score'], reverse=True)

    # Build table
    rows_html = ''
    for i, s in enumerate(scored):
        rb, rf = RATING_BADGE.get(s['_rating'], ('rgba(148,163,184,.15)', '#94a3b8'))
        conv_color = '#10b981' if s['_score'] >= 60 else '#f59e0b' if s['_score'] >= 40 else '#ef4444'
        val = s.get('valuation', 'N/A')
        vb, vf = VAL_BADGE.get(val, ('rgba(148,163,184,.15)', '#94a3b8'))
        ab, af = AI_BADGE.get(s.get('ai_rating', ''), ('rgba(148,163,184,.15)', '#94a3b8'))

        rows_html += f"""<tr>
          <td style='color:#64748b;font-weight:600'>{i+1}</td>
          <td style='font-weight:700;color:#f8fafc'>{s['symbol']}</td>
          <td style='color:#94a3b8;max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>{s.get('company','')}</td>
          <td>{_badge(val, vb, vf)}</td>
          <td>{_badge(s.get('ai_rating','N/A'), ab, af)}</td>
          <td><div class='sr-conviction'>
            <div class='sr-conv-bar'><div class='sr-conv-fill' style='width:{s["_score"]}%;background:{conv_color}'></div></div>
            <span style='font-size:12px;color:{conv_color};font-weight:600'>{s["_score"]}</span>
          </div></td>
          <td>{_badge(s['_rating'], rb, rf)}</td>
          <td style='font-size:12px;color:#94a3b8'>{s['_action']}</td>
        </tr>"""

    st.markdown(f"""<table class='sr-rec-table'>
      <thead><tr>
        <th>#</th><th>Symbol</th><th>Company</th><th>Valuation</th>
        <th>AI Quality</th><th>Conviction</th><th>Rating</th><th>Action</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ─── Main entry point ─────────────────────────────────────────────────────────

def render_summary_report(report_data: List[Dict]):
    """Render the full professional summary report dashboard."""
    inject_css()
    render_executive_summary(report_data)
    render_comparison_charts(report_data)
    render_company_cards(report_data)
    render_recommendation_table(report_data)
