"""Shared display styles and formatting; saved data and calculations stay unchanged."""
from html import escape
import math
import numbers

import pandas as pd
import streamlit as st

TEAL = '#0f766e'
NAVY = '#18354a'
BLUE = '#2563a6'
AMBER = '#b65d13'

LABELS = {
    'timestamp': 'Hour', 'period': 'Period', 'start': 'Start date', 'end': 'End date',
    'expected_hours': 'Expected hours', 'observed_hours': 'Observed hours',
    'missing_hours': 'Missing hours', 'duplicate_rows': 'Duplicate rows',
    'excluded_rows': 'Excluded rows', 'Model': 'Model', 'model': 'Model',
    'MAE': 'MAE', 'RMSE': 'RMSE', 'Evaluated_Hours': 'Evaluated hours',
    'Training_Time_sec': 'Fit time (s)', 'Poisson_NLL': 'Poisson NLL',
    'cnt': 'Actual rentals', 'scored_forecast': 'Forecast', 'error': 'Signed error',
    'abs_error': 'Absolute error', 'hour': 'Hour of day', 'workingday': 'Day type',
    'n': 'Hours', 'mean_residual': 'Mean signed error', 'mean_rentals': 'Mean rentals',
    'feature': 'Feature', 'coefficient': 'Coefficient', 'input_set': 'Input set',
    'selected_penalty': 'Selected penalty', 'validation_hours': 'Validation hours',
    'validation_MAE': 'Validation MAE', 'inputs': 'Inputs',
    'training_count': 'Training hours', 'validation_count': 'Validation hours',
    'training_proportion': 'Training proportion', 'validation_proportion': 'Validation proportion',
    'absolute_difference': 'Absolute difference', 'bin': 'Count range',
    'coverage_percent': 'Observed coverage (%)', 'nominal_percent': 'Nominal target (%)',
    'mean_width': 'Mean width (rentals)', 'group': 'Group',
}


def install_styles():
    st.html('''<style>
    :root { --ink:#18354a; --muted:#506477; --line:#dce5eb; --teal:#0f766e; }
    .block-container { padding-top:4.5rem; padding-bottom:2.5rem; max-width:1480px; }
    h1 { font-size:1.9rem !important; letter-spacing:-.035em; }
    h2 { font-size:1.65rem !important; letter-spacing:-.025em; padding-top:.35rem !important; }
    h3 { font-size:1.15rem !important; letter-spacing:-.015em; }
    [data-testid="stVerticalBlock"] { gap:.8rem; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
    [data-testid="stMetric"] { padding:.9rem 1rem; border:1px solid var(--line); border-radius:12px; background:#f8fafc; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    [data-testid="stMetricValue"] { font-size:1.7rem; color:var(--ink); font-variant-numeric:tabular-nums; }
    [data-testid="stSidebar"] { background:#f3f7f8; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] h1 { font-size:1.15rem !important; }
    .brand { display:flex; align-items:center; gap:.7rem; padding-bottom:.75rem; margin-bottom:.35rem; border-bottom:1px solid var(--line); }
    .brand-mark { background:var(--teal); color:white; border-radius:9px; padding:.3rem .55rem; font-weight:750; font-size:.85rem; }
    .brand-name { font-size:.9rem; font-weight:650; color:var(--ink); }
    .brand-note { color:var(--muted); font-size:.78rem; margin-left:auto; }
    .scope { display:flex; align-items:center; flex-wrap:wrap; gap:.5rem; margin:.65rem 0 .3rem; }
    .scope-title { font-weight:650; font-size:1rem; color:var(--ink); }
    .scope-tag,.default-tag { display:inline-block; font-size:.75rem; font-weight:650; line-height:1.5; padding:.2rem .55rem; border-radius:6px; }
    .scope-tag { background:#edf3f7; color:#36546b; border:1px solid #d7e2ea; }
    .scope-tag.selected { background:#e8f5f2; color:#0b615b; border-color:#bfdfd7; }
    .default-tag { background:#e2f2ed; color:#0a6158; white-space:nowrap; margin-left:.4rem; }
    .scope-note,.small-note { color:var(--muted); font-size:.83rem; line-height:1.55; margin:.2rem 0 .65rem; }
    .coverage-note { background:#fff6e9; color:#85430e; padding:.75rem 1rem; border:1px solid #efd0a6; border-radius:9px; font-size:.88rem; margin:.65rem 0; }
    .coverage-note.neutral { background:#eef5f8; color:#36546b; border-color:#d7e2ea; }
    .data-table-wrap { max-width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:10px; margin:.25rem 0 .5rem; }
    .display-table { width:100%; border-collapse:collapse; font-size:.84rem; line-height:1.5; color:var(--ink); }
    .display-table th { background:#f1f5f8; color:#3b5264; font-weight:650; text-align:left; }
    .display-table th,.display-table td { padding:.65rem .75rem; border-bottom:1px solid #e7edf1; vertical-align:top; overflow-wrap:anywhere; white-space:pre-line; }
    .display-table tr:last-child td { border-bottom:0; }
    .display-table tbody tr:nth-child(even) { background:#fafcfd; }
    .display-table td.numeric,.display-table th.numeric { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
    .display-table td.source-cell { min-width:170px; max-width:260px; }
    .display-table td:first-child { font-weight:550; }
    .table-help { font-size:.78rem; color:var(--muted); margin-top:.1rem; }
    @media(max-width:640px) {
      .block-container { padding:4.25rem 1rem 2rem; }
      h1 { font-size:1.5rem !important; }
      h2 { font-size:1.35rem !important; }
      h3 { font-size:1.05rem !important; }
      .brand-note { display:none; }
      .brand { margin-bottom:.1rem; padding-bottom:.5rem; }
      [data-testid="stMetric"] { padding:.65rem .8rem; }
      [data-testid="stMetricValue"] { font-size:1.45rem; }
      .stacked-table thead { display:none; }
      .stacked-table,.stacked-table tbody { display:block; }
      .stacked-table tr { display:block; padding:.45rem .7rem; border-bottom:1px solid var(--line); }
      .stacked-table tr:last-child { border-bottom:0; }
      .stacked-table td,.stacked-table td.numeric,.stacked-table td.source-cell { display:grid; grid-template-columns:minmax(100px,42%) minmax(0,1fr); gap:.6rem; min-width:0; max-width:none; border:0; padding:.3rem 0; text-align:left; white-space:pre-line; }
      .stacked-table td::before { content:attr(data-label); color:var(--muted); font-weight:550; font-size:.78rem; }
    }
    @media(prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; } }
    </style>''')


def brand():
    st.html('<div class="brand"><span class="brand-mark">BR</span><span class="brand-name">Bike Rental Dashboard</span><span class="brand-note">Historical forecast replay · 2011–2012</span></div>')


def scope(title, tag, note='', selected=False):
    style = 'scope-tag selected' if selected else 'scope-tag'
    st.html(f'<div class="scope"><span class="scope-title">{escape(title)}</span><span class="{style}">{escape(tag)}</span></div>'
            + (f'<p class="scope-note">{escape(note)}</p>' if note else ''))


def number(value, decimals=2):
    if pd.isna(value):
        return 'Not available'
    if isinstance(value, numbers.Real) and math.isfinite(value):
        return f'{value:,.{decimals}f}'
    return str(value)


def table(frame, *, labels=None, formats=None, stacked=True, default_model=None):
    """Semantic, wrapping tables for compact results, with stacked mobile rows."""
    labels = {**LABELS, **(labels or {})}
    formats = formats or {}
    numeric = {c for c in frame if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])}
    classes = 'display-table stacked-table' if stacked else 'display-table'
    heads = ''.join(f'<th scope="col" class="{"numeric" if c in numeric else ""}">{escape(str(labels.get(c,c)))}</th>' for c in frame)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col in frame:
            value = row[col]
            if pd.isna(value):
                shown = 'Not available'
            elif col in formats:
                shown = formats[col](value) if callable(formats[col]) else number(value, formats[col])
            elif isinstance(value, (bool,)) or pd.api.types.is_bool_dtype(frame[col]):
                shown = 'Yes' if value else 'No'
            elif col in numeric:
                shown = number(value, 0 if pd.api.types.is_integer_dtype(frame[col]) else 2)
            else:
                shown = str(value)
            content = escape(str(shown))
            if col in ['Model','model'] and value == default_model:
                content += '<span class="default-tag">Default</span>'
            css = 'numeric' if col in numeric else ('source-cell' if col in ['source','Source'] else '')
            cells.append(f'<td class="{css}" data-label="{escape(str(labels.get(col,col)),quote=True)}">{content}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    st.html(f'<div class="data-table-wrap"><table class="{classes}"><thead><tr>{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def chart_style(chart):
    return chart.configure_view(stroke=None).configure_axis(
        labelColor='#506477', titleColor=NAVY, gridColor='#e6edf2',
        labelFontSize=11, titleFontSize=12, titlePadding=12,
    ).configure_legend(labelColor=NAVY, titleColor=NAVY, orient='top')


def coverage_summary(overall, level):
    gap = 100*float(overall.coverage)-level
    cols = st.columns(4)
    for col, label, value in zip(cols,
        ['Nominal target','Observed coverage','Gap to target','Mean width (rentals)'],
        [f'{level}%',f'{100*overall.coverage:.2f}%',f'{gap:+.2f} pp',number(overall.mean_width)]):
        col.metric(label,value)
    if pd.isna(gap):
        message, css = 'No observed hours are available for this coverage calculation.', 'neutral'
    elif gap < 0:
        message, css = f'Below target: these intervals covered {abs(gap):.2f} percentage points fewer Test outcomes than the nominal level.', ''
    else:
        message, css = f'Observed coverage is {gap:.2f} percentage points above the nominal level. Consider interval width alongside coverage.', 'neutral'
    st.html(f'<p class="coverage-note {css}">{escape(message)}</p>')


def model_details(result):
    with st.expander('Model details and saved settings'):
        settings = {'Model':result['model'],'Model run':result['model_reference']}
        names = {'alpha':'Penalty (alpha)','alpha_prior':'Prior precision','sigma2_noise':'Observation noise variance','noise_multiplier':'Noise multiplier'}
        settings.update({names.get(k,k): number(v, 4 if k in ['alpha','alpha_prior'] else 2) if isinstance(v,numbers.Real) else str(v) for k,v in result['settings'].items()})
        table(pd.DataFrame(settings.items(),columns=['Parameter','Value']))
