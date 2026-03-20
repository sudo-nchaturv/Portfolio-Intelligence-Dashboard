"""
INDstocks Portfolio Intelligence Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
from typing import Optional
import os

from risk import (
    compute_returns,
    compute_risk_metrics,
    compute_rsi,
    compute_sma,
    compute_momentum,
)
from alerts import send_telegram_alert, build_alert_message, check_alert_conditions

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Top bar */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f0f14;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    color: #6b7280 !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 700;
    color: #f4f4f5 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0a0f;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: #0f0f14 !important;
    border: 1px solid #2a2a3e !important;
    color: #f4f4f5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 6px !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Tab styling */
[data-baseweb="tab-list"] {
    background: #0a0a0f !important;
    border-bottom: 1px solid #1e1e2e !important;
    padding: 0 !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.03em;
    padding: 10px 20px !important;
    color: #6b7280 !important;
    border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
    background: transparent !important;
}

/* Alert badges */
.badge-high   { background:#3b1219; color:#f87171; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'JetBrains Mono',monospace; font-weight:600; }
.badge-medium { background:#2d2208; color:#fbbf24; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'JetBrains Mono',monospace; font-weight:600; }
.badge-low    { background:#0f2118; color:#34d399; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'JetBrains Mono',monospace; font-weight:600; }
.badge-ok     { background:#1a1a2e; color:#818cf8; padding:2px 8px; border-radius:4px; font-size:11px; font-family:'JetBrains Mono',monospace; font-weight:600; }

/* Section header */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 6px;
}

/* Risk flag row */
.flag-row {
    background: #1e0a0a;
    border-left: 3px solid #ef4444;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #fca5a5;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.indstocks.com"
NIFTY_SCRIP = "NSE_13"   # NIFTY 50 index scrip code (verify with instruments API)

RISK_THRESHOLDS = {
    "vol_high":          0.40,
    "drawdown_high":    -0.30,
    "var95_high":       -0.025,
    "corr_high":         0.80,
    "concentration":     0.20,
    "rsi_overbought":   70,
    "rsi_oversold":     30,
    "pnl_alert_pct":    -5.0,
}

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#9ca3af", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
)

# ─── Session state defaults ───────────────────────────────────────────────────

for k, v in {
    "holdings_df": None,
    "close_df": None,
    "risk_df": None,
    "corr_matrix": None,
    "momentum_df": None,
    "last_refresh": None,
    "alert_log": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── API helpers ──────────────────────────────────────────────────────────────

def api_headers(token: str) -> dict:
    return {"Authorization": token}


@st.cache_data(ttl=300)
def fetch_holdings(token: str) -> Optional[pd.DataFrame]:
    try:
        r = requests.get(f"{BASE_URL}/portfolio/holdings",
                         headers=api_headers(token), timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None
        df = pd.DataFrame(data)
        df["weight_pct"] = (df["market_value"] / df["market_value"].sum() * 100).round(2)
        return df
    except Exception as e:
        st.error(f"Holdings API error: {e}")
        return None


@st.cache_data(ttl=3600)
def fetch_ohlcv(scrip_code: str, token: str, days_back: int = 365) -> Optional[pd.DataFrame]:
    """Fetch up to 1 year of daily candles, chunking if needed."""
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - (days_back * 86_400_000)
    try:
        r = requests.get(
            f"{BASE_URL}/market/historical/1day",
            headers=api_headers(token),
            params={"scrip-codes": scrip_code,
                    "start_time": start_ms,
                    "end_time":   end_ms},
            timeout=20,
        )
        r.raise_for_status()
        candles = r.json().get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
        return df.set_index("ts").sort_index()
    except Exception as e:
        st.warning(f"OHLCV error for {scrip_code}: {e}")
        return None


def build_close_matrix(holdings_df: pd.DataFrame, token: str) -> pd.DataFrame:
    """Fetch close prices for all holdings and return a combined DataFrame."""
    frames = {}
    bar = st.progress(0, text="Fetching historical prices…")
    total = len(holdings_df)
    for i, row in holdings_df.iterrows():
        sym   = row["trading_symbol"]
        scrip = row["security_id"]
        bar.progress((list(holdings_df.index).index(i) + 1) / total,
                     text=f"Fetching {sym}…")
        ohlcv = fetch_ohlcv(str(scrip), token)
        if ohlcv is not None and not ohlcv.empty:
            frames[sym] = ohlcv["close"]
    bar.empty()
    if not frames:
        return pd.DataFrame()
    close_df = pd.DataFrame(frames).ffill().dropna(how="all")
    return close_df


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:18px;
         font-weight:700; color:#a78bfa; letter-spacing:0.04em; margin-bottom:4px;'>
    📊 PORTFOLIO<br>INTELLIGENCE
    </div>
    <div style='font-size:11px; color:#4b5563; font-family:"JetBrains Mono",monospace;
         letter-spacing:0.06em; margin-bottom:1.5rem;'>
    POWERED BY INDSTOCKS API
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">API CONFIG</div>', unsafe_allow_html=True)
    _default_token      = st.secrets.get("INDSTOCKS_TOKEN",  "")
    _default_tele_token = st.secrets.get("TELEGRAM_TOKEN",   "")
    _default_tele_chat  = st.secrets.get("TELEGRAM_CHAT_ID", "")

    api_token = st.text_input(
        "Access Token",
        value=_default_token,
        type="password",
        placeholder="YOUR_ACCESS_TOKEN",
        help="Your INDstocks API bearer token",
    )

    st.markdown('<div class="section-header">TELEGRAM ALERTS</div>', unsafe_allow_html=True)
    tele_token  = st.text_input("Bot Token",  value=_default_tele_token,
                                type="password", placeholder="123456:ABC…")
    tele_chatid = st.text_input("Chat ID",    value=_default_tele_chat,
                                placeholder="-100XXXXXXXXXX")
    tele_enabled = st.toggle("Enable alerts", value=False)

    st.markdown('<div class="section-header">THRESHOLDS</div>', unsafe_allow_html=True)
    with st.expander("Customise risk thresholds"):
        RISK_THRESHOLDS["vol_high"] = st.slider(
            "Volatility flag (annualised)", 0.10, 0.80,
            RISK_THRESHOLDS["vol_high"], 0.05,
            format="%.0f%%", help="Flag stocks with annualised vol above this"
        )
        RISK_THRESHOLDS["concentration"] = st.slider(
            "Concentration flag", 0.05, 0.50,
            RISK_THRESHOLDS["concentration"], 0.05,
            format="%.0f%%"
        )
        RISK_THRESHOLDS["rsi_overbought"] = st.slider(
            "RSI overbought", 60, 85,
            int(RISK_THRESHOLDS["rsi_overbought"]), 1
        )

    st.markdown('<div class="section-header">DATA</div>', unsafe_allow_html=True)
    days_back = st.select_slider(
        "History window",
        options=[30, 60, 90, 180, 365],
        value=365,
        format_func=lambda x: f"{x}d",
    )
    if st.button("⟳  Refresh data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state.holdings_df = None
        st.session_state.close_df    = None
        st.session_state.risk_df     = None
        st.rerun()

    st.markdown("""
    <div style='margin-top:auto; padding-top:2rem; font-size:10px;
         color:#374151; font-family:"JetBrains Mono",monospace; line-height:1.8;'>
    Data cached 5 min (holdings)<br>
    OHLCV cached 60 min<br>
    All times IST
    </div>
    """, unsafe_allow_html=True)

# ─── Load data ────────────────────────────────────────────────────────────────

if not api_token:
    st.markdown("""
    <div style='text-align:center; padding:5rem 2rem; color:#4b5563;'>
        <div style='font-size:48px; margin-bottom:1rem;'>🔑</div>
        <div style='font-family:"JetBrains Mono",monospace; font-size:14px;
             color:#6b7280; letter-spacing:0.05em;'>
            ENTER YOUR INDSTOCKS ACCESS TOKEN IN THE SIDEBAR TO BEGIN
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load holdings
if st.session_state.holdings_df is None:
    with st.spinner("Loading portfolio holdings…"):
        st.session_state.holdings_df = fetch_holdings(api_token)

holdings_df = st.session_state.holdings_df

if holdings_df is None or holdings_df.empty:
    st.error("Could not load holdings. Check your token or try refreshing.")
    st.stop()

# Load historical prices
if st.session_state.close_df is None:
    st.session_state.close_df = build_close_matrix(holdings_df, api_token)

close_df = st.session_state.close_df

# Compute risk (re-run if close_df changed)
if st.session_state.risk_df is None and not close_df.empty:
    returns_df = compute_returns(close_df)
    st.session_state.risk_df, st.session_state.corr_matrix = compute_risk_metrics(
        close_df, returns_df, holdings_df
    )
    st.session_state.momentum_df = compute_momentum(close_df)
    st.session_state.last_refresh = datetime.now()

risk_df      = st.session_state.risk_df
corr_matrix  = st.session_state.corr_matrix
momentum_df  = st.session_state.momentum_df

# ─── Header ───────────────────────────────────────────────────────────────────

total_value  = holdings_df["market_value"].sum()
total_cost   = (holdings_df["average_price"] * holdings_df["quantity"]).sum()
total_pnl    = holdings_df["pnl_absolute"].sum()
total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

flagged_count = int(risk_df["flag"].sum()) if risk_df is not None else 0
last_ref = st.session_state.last_refresh.strftime("%H:%M:%S") if st.session_state.last_refresh else "—"

col_title, col_stamp = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:22px;
         font-weight:700; color:#f4f4f5; letter-spacing:0.02em;'>
        Portfolio Intelligence Dashboard
    </div>
    """, unsafe_allow_html=True)
with col_stamp:
    st.markdown(f"""
    <div style='text-align:right; font-family:"JetBrains Mono",monospace;
         font-size:10px; color:#4b5563; padding-top:8px;'>
        LAST REFRESH: {last_ref}
    </div>
    """, unsafe_allow_html=True)

# Summary metrics row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Portfolio Value", f"₹{total_value:,.0f}")
with m2:
    st.metric("Total P&L", f"₹{total_pnl:,.0f}",
              delta=f"{total_pnl_pct:+.2f}%",
              delta_color="normal")
with m3:
    st.metric("Holdings", str(len(holdings_df)))
with m4:
    st.metric("Risk Flags", str(flagged_count),
              delta="⚠ review" if flagged_count > 0 else "✓ clean",
              delta_color="inverse" if flagged_count > 0 else "off")
with m5:
    winners = (holdings_df["pnl_percent"] > 0).sum()
    losers  = (holdings_df["pnl_percent"] <= 0).sum()
    st.metric("Win / Loss", f"{winners} / {losers}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_risk, tab_corr, tab_momentum, tab_alerts = st.tabs([
    "📋  Overview",
    "⚡  Risk Analysis",
    "🔗  Correlation",
    "📈  Momentum",
    "🔔  Alerts",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">HOLDINGS</div>', unsafe_allow_html=True)

        disp = holdings_df[[
            "trading_symbol", "quantity", "average_price",
            "last_traded_price", "market_value", "weight_pct",
            "pnl_absolute", "pnl_percent"
        ]].copy()
        disp.columns = ["Symbol", "Qty", "Avg Price", "LTP",
                        "Mkt Value ₹", "Weight %", "P&L ₹", "P&L %"]
        disp = disp.sort_values("Mkt Value ₹", ascending=False)

        def colour_pnl(val):
            if isinstance(val, float):
                colour = "#34d399" if val >= 0 else "#f87171"
                return f"color: {colour}; font-family: 'JetBrains Mono', monospace; font-size:12px"
            return "font-family: 'JetBrains Mono', monospace; font-size:12px"

        st.dataframe(
            disp.style
                .format({
                    "Avg Price": "₹{:,.2f}",
                    "LTP":       "₹{:,.2f}",
                    "Mkt Value ₹": "₹{:,.0f}",
                    "Weight %":   "{:.2f}%",
                    "P&L ₹":     "₹{:,.0f}",
                    "P&L %":     "{:+.2f}%",
                })
                .applymap(colour_pnl, subset=["P&L ₹", "P&L %"])
                .set_properties(**{
                    "font-family": "'JetBrains Mono', monospace",
                    "font-size":   "12px",
                }),
            use_container_width=True,
            height=420,
        )

    with col_right:
        st.markdown('<div class="section-header">PORTFOLIO WEIGHTS</div>', unsafe_allow_html=True)

        fig_pie = px.pie(
            holdings_df.sort_values("weight_pct", ascending=False),
            values="weight_pct",
            names="trading_symbol",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig_pie.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont=dict(family="JetBrains Mono, monospace", size=10),
        )
        fig_pie.update_layout(
            **PLOTLY_DARK,
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # P&L waterfall
        st.markdown('<div class="section-header">P&L BY HOLDING</div>', unsafe_allow_html=True)
        pnl_df = holdings_df.sort_values("pnl_absolute")
        colours = ["#34d399" if x >= 0 else "#f87171" for x in pnl_df["pnl_absolute"]]

        fig_bar = go.Figure(go.Bar(
            x=pnl_df["trading_symbol"],
            y=pnl_df["pnl_absolute"],
            marker_color=colours,
            text=[f"₹{v:,.0f}" for v in pnl_df["pnl_absolute"]],
            textposition="outside",
            textfont=dict(size=9, family="JetBrains Mono, monospace"),
        ))
        fig_bar.update_layout(
            **PLOTLY_DARK,
            height=220,
            xaxis_tickfont=dict(size=9),
            yaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Sector / index breakdown row
    if not close_df.empty:
        st.markdown('<div class="section-header">PRICE HISTORY (NORMALISED TO 100)</div>',
                    unsafe_allow_html=True)
        norm = close_df.div(close_df.iloc[0]) * 100
        fig_norm = go.Figure()
        for col in norm.columns:
            fig_norm.add_trace(go.Scatter(
                x=norm.index, y=norm[col], mode="lines",
                name=col,
                line=dict(width=1.5),
                hovertemplate="%{x|%d %b %Y}<br>%{y:.1f}<extra>%{fullData.name}</extra>",
            ))
        fig_norm.update_layout(
            **PLOTLY_DARK,
            height=280,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
            legend=dict(orientation="h", y=-0.2, font=dict(size=9)),
        )
        st.plotly_chart(fig_norm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab_risk:
    if risk_df is None:
        st.info("Risk data not available — check API connectivity and reload.")
    else:
        # Flagged stocks callout
        flagged = risk_df[risk_df["flag"]]
        if not flagged.empty:
            st.markdown(f"""
            <div style='background:#1e0a0a; border:1px solid #7f1d1d; border-radius:8px;
                 padding:1rem 1.25rem; margin-bottom:1rem;'>
                <div style='font-family:"JetBrains Mono",monospace; font-size:11px;
                     color:#f87171; letter-spacing:0.08em; margin-bottom:6px;'>
                    ⚠ {len(flagged)} STOCK{'S' if len(flagged)>1 else ''} FLAGGED FOR REVIEW
                </div>
                <div style='font-family:"JetBrains Mono",monospace; font-size:13px;
                     color:#fca5a5;'>
                    {" · ".join(flagged.index.tolist())}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk metrics table
        st.markdown('<div class="section-header">RISK METRICS PER HOLDING</div>',
                    unsafe_allow_html=True)

        display_risk = risk_df[[
            "ann_vol", "max_drawdown", "var95_1d", "sharpe", "beta", "flag"
        ]].copy()
        display_risk.columns = [
            "Ann. Vol", "Max Drawdown", "VaR 95% (1d)", "Sharpe", "Beta (NIFTY)", "⚠ Flag"
        ]

        def style_risk(val):
            s = "font-family:'JetBrains Mono',monospace; font-size:12px;"
            if isinstance(val, float):
                if val < 0:
                    return s + " color:#f87171;"
                elif val > 0:
                    return s + " color:#34d399;"
            if isinstance(val, bool):
                return s + (" color:#f87171;" if val else " color:#6b7280;")
            return s

        st.dataframe(
            display_risk.style
                .format({
                    "Ann. Vol":       "{:.1%}",
                    "Max Drawdown":   "{:.1%}",
                    "VaR 95% (1d)":  "{:.2%}",
                    "Sharpe":         "{:.2f}",
                    "Beta (NIFTY)":   "{:.2f}",
                    "⚠ Flag":         lambda x: "YES" if x else "—",
                })
                .applymap(style_risk),
            use_container_width=True,
        )

        col_v, col_d = st.columns(2)

        with col_v:
            st.markdown('<div class="section-header">ANNUALISED VOLATILITY</div>',
                        unsafe_allow_html=True)
            vol_sorted = risk_df["ann_vol"].sort_values(ascending=True)
            colours_v  = ["#f87171" if v > RISK_THRESHOLDS["vol_high"] else "#818cf8"
                          for v in vol_sorted]
            fig_vol = go.Figure(go.Bar(
                y=vol_sorted.index,
                x=vol_sorted.values,
                orientation="h",
                marker_color=colours_v,
                text=[f"{v:.1%}" for v in vol_sorted],
                textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono, monospace"),
            ))
            fig_vol.add_vline(
                x=RISK_THRESHOLDS["vol_high"],
                line_dash="dash", line_color="#ef4444", line_width=1,
                annotation_text="threshold",
                annotation_font=dict(size=9, color="#ef4444"),
            )
            fig_vol.update_layout(**PLOTLY_DARK, height=max(250, len(risk_df)*28))
            st.plotly_chart(fig_vol, use_container_width=True)

        with col_d:
            st.markdown('<div class="section-header">MAX DRAWDOWN</div>',
                        unsafe_allow_html=True)
            dd_sorted   = risk_df["max_drawdown"].sort_values(ascending=True)
            colours_dd  = ["#ef4444" if v < RISK_THRESHOLDS["drawdown_high"] else "#6b7280"
                           for v in dd_sorted]
            fig_dd = go.Figure(go.Bar(
                y=dd_sorted.index,
                x=dd_sorted.values,
                orientation="h",
                marker_color=colours_dd,
                text=[f"{v:.1%}" for v in dd_sorted],
                textposition="outside",
                textfont=dict(size=9, family="JetBrains Mono, monospace"),
            ))
            fig_dd.add_vline(
                x=RISK_THRESHOLDS["drawdown_high"],
                line_dash="dash", line_color="#ef4444", line_width=1,
                annotation_text="threshold",
                annotation_font=dict(size=9, color="#ef4444"),
            )
            fig_dd.update_layout(**PLOTLY_DARK, height=max(250, len(risk_df)*28))
            st.plotly_chart(fig_dd, use_container_width=True)

        # Drawdown over time for selected stock
        st.markdown('<div class="section-header">UNDERWATER CHART (DRAWDOWN OVER TIME)</div>',
                    unsafe_allow_html=True)
        selected = st.selectbox("Select stock", close_df.columns.tolist(), key="dd_select")
        if selected:
            prices   = close_df[selected].dropna()
            roll_max = prices.expanding().max()
            dd_series = (prices - roll_max) / roll_max * 100

            fig_uw = go.Figure()
            fig_uw.add_trace(go.Scatter(
                x=dd_series.index, y=dd_series.values,
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.15)",
                line=dict(color="#ef4444", width=1.5),
                name="Drawdown %",
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}%<extra></extra>",
            ))
            fig_uw.update_layout(
                **PLOTLY_DARK, height=200,
                yaxis=dict(showgrid=True, gridcolor="#1e1e2e",
                           tickformat=".0f", ticksuffix="%"),
                xaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_uw, use_container_width=True)

        # Sharpe ratio chart
        st.markdown('<div class="section-header">SHARPE RATIO (RISK-ADJUSTED RETURN)</div>',
                    unsafe_allow_html=True)
        sharpe_s = risk_df["sharpe"].sort_values()
        colours_sh = ["#34d399" if v >= 1.0 else "#fbbf24" if v >= 0 else "#f87171"
                      for v in sharpe_s]
        fig_sh = go.Figure(go.Bar(
            x=sharpe_s.index, y=sharpe_s.values,
            marker_color=colours_sh,
            text=[f"{v:.2f}" for v in sharpe_s],
            textposition="outside",
            textfont=dict(size=9, family="JetBrains Mono, monospace"),
        ))
        fig_sh.add_hline(y=1.0, line_dash="dash", line_color="#34d399",
                         line_width=1, annotation_text="good (≥1)",
                         annotation_font=dict(size=9, color="#34d399"))
        fig_sh.add_hline(y=0.0, line_dash="dot", line_color="#6b7280", line_width=1)
        fig_sh.update_layout(
            **PLOTLY_DARK, height=200,
            yaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
        )
        st.plotly_chart(fig_sh, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

with tab_corr:
    if corr_matrix is None:
        st.info("Correlation data unavailable.")
    else:
        st.markdown('<div class="section-header">RETURN CORRELATION MATRIX</div>',
                    unsafe_allow_html=True)

        col_info, col_thresh = st.columns([3, 1])
        with col_thresh:
            high_corr_pairs = []
            tickers = corr_matrix.columns.tolist()
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    c = corr_matrix.iloc[i, j]
                    if abs(c) > RISK_THRESHOLDS["corr_high"]:
                        high_corr_pairs.append((tickers[i], tickers[j], c))

            if high_corr_pairs:
                st.markdown(f"""
                <div style='background:#2d2208; border:1px solid #92400e; border-radius:8px;
                     padding:0.75rem; font-family:"JetBrains Mono",monospace; font-size:11px;'>
                    <div style='color:#fbbf24; margin-bottom:4px;'>
                        ⚡ {len(high_corr_pairs)} HIGH-CORR PAIR{'S' if len(high_corr_pairs)>1 else ''}
                    </div>
                    {''.join(f"<div style='color:#fde68a; margin:2px 0;'>{a} ↔ {b}: {c:.2f}</div>"
                             for a, b, c in high_corr_pairs)}
                </div>
                """, unsafe_allow_html=True)

        # Heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_masked = corr_matrix.copy()
        corr_masked[mask] = np.nan

        fig_heat = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale=[
                [0.0,  "#1a1a3e"],
                [0.25, "#312e81"],
                [0.5,  "#1e1e2e"],
                [0.75, "#064e3b"],
                [1.0,  "#065f46"],
            ],
            zmid=0,
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=9, family="JetBrains Mono, monospace", color="#f4f4f5"),
            hoverongaps=False,
            colorbar=dict(
                tickfont=dict(size=9, family="JetBrains Mono, monospace"),
                thickness=12,
            ),
        ))
        fig_heat.update_layout(
            **PLOTLY_DARK,
            height=max(300, len(corr_matrix) * 45 + 60),
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Diversification score
        n = len(corr_matrix)
        off_diag = corr_matrix.values[~np.eye(n, dtype=bool)]
        avg_corr = np.nanmean(off_diag)
        div_score = max(0, 1 - avg_corr) * 100

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Avg Pairwise Correlation", f"{avg_corr:.2f}")
        with c2:
            st.metric("Diversification Score", f"{div_score:.0f} / 100")
        with c3:
            hhi = ((holdings_df["weight_pct"] / 100) ** 2).sum()
            st.metric("HHI Concentration Index",
                      f"{hhi:.3f}",
                      delta="concentrated" if hhi > 0.18 else "diversified",
                      delta_color="inverse" if hhi > 0.18 else "off")

        # Rolling 30-day correlation between two selected stocks
        if not close_df.empty and len(close_df.columns) >= 2:
            st.markdown('<div class="section-header">ROLLING 30-DAY CORRELATION</div>',
                        unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                sel_a = st.selectbox("Stock A", close_df.columns.tolist(),
                                     index=0, key="corr_a")
            with cb:
                sel_b = st.selectbox("Stock B", close_df.columns.tolist(),
                                     index=min(1, len(close_df.columns)-1), key="corr_b")

            if sel_a != sel_b:
                ret_a = np.log(close_df[sel_a] / close_df[sel_a].shift(1)).dropna()
                ret_b = np.log(close_df[sel_b] / close_df[sel_b].shift(1)).dropna()
                roll_corr = ret_a.rolling(30).corr(ret_b).dropna()

                fig_rc = go.Figure()
                fig_rc.add_trace(go.Scatter(
                    x=roll_corr.index, y=roll_corr.values,
                    mode="lines",
                    line=dict(color="#a78bfa", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(167,139,250,0.08)",
                    hovertemplate="%{x|%d %b %Y}<br>corr: %{y:.3f}<extra></extra>",
                ))
                fig_rc.add_hline(y=RISK_THRESHOLDS["corr_high"],
                                 line_dash="dash", line_color="#fbbf24",
                                 line_width=1,
                                 annotation_text="high corr threshold",
                                 annotation_font=dict(size=9, color="#fbbf24"))
                fig_rc.add_hline(y=0, line_dash="dot", line_color="#374151", line_width=1)
                fig_rc.update_layout(
                    **PLOTLY_DARK, height=200,
                    yaxis=dict(showgrid=True, gridcolor="#1e1e2e",
                               range=[-1.1, 1.1]),
                )
                st.plotly_chart(fig_rc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

with tab_momentum:
    if momentum_df is None:
        st.info("Momentum data unavailable.")
    else:
        st.markdown('<div class="section-header">MOMENTUM SIGNALS</div>',
                    unsafe_allow_html=True)

        disp_mom = momentum_df[[
            "rsi_14", "roc_30", "roc_90",
            "vs_52w_high_pct", "vs_52w_low_pct",
            "sma20_signal"
        ]].copy()
        disp_mom.columns = [
            "RSI(14)", "ROC 30d", "ROC 90d",
            "vs 52W High", "vs 52W Low", "SMA20 Signal"
        ]

        def style_momentum(val):
            s = "font-family:'JetBrains Mono',monospace; font-size:12px;"
            if isinstance(val, float):
                if val < 0:  return s + " color:#f87171;"
                elif val > 0: return s + " color:#34d399;"
            if isinstance(val, str):
                if "ABOVE" in val: return s + " color:#34d399;"
                if "BELOW" in val: return s + " color:#f87171;"
            return s

        def style_rsi(val):
            s = "font-family:'JetBrains Mono',monospace; font-size:12px;"
            if isinstance(val, float):
                if val > RISK_THRESHOLDS["rsi_overbought"]: return s + " color:#f87171; font-weight:600;"
                if val < RISK_THRESHOLDS["rsi_oversold"]:   return s + " color:#34d399; font-weight:600;"
            return s

        st.dataframe(
            disp_mom.style
                .format({
                    "RSI(14)":    "{:.1f}",
                    "ROC 30d":    "{:+.1%}",
                    "ROC 90d":    "{:+.1%}",
                    "vs 52W High": "{:+.1%}",
                    "vs 52W Low":  "{:+.1%}",
                })
                .applymap(style_rsi,        subset=["RSI(14)"])
                .applymap(style_momentum,   subset=["ROC 30d","ROC 90d",
                                                    "vs 52W High","vs 52W Low","SMA20 Signal"]),
            use_container_width=True,
        )

        # RSI gauge row
        st.markdown('<div class="section-header">RSI DASHBOARD</div>', unsafe_allow_html=True)
        rsi_cols = st.columns(min(len(momentum_df), 6))
        for idx, (sym, row) in enumerate(momentum_df.iterrows()):
            if idx >= 6: break
            rsi_val = row["rsi_14"]
            with rsi_cols[idx]:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rsi_val,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": sym[:10], "font": {"size": 10,
                           "family": "JetBrains Mono, monospace", "color": "#9ca3af"}},
                    number={"font": {"size": 16,
                                     "family": "JetBrains Mono, monospace",
                                     "color": "#f4f4f5"}},
                    gauge={
                        "axis": {"range": [0, 100],
                                 "tickfont": {"size": 8, "family": "JetBrains Mono, monospace"}},
                        "bar":  {"color": (
                            "#f87171" if rsi_val > RISK_THRESHOLDS["rsi_overbought"] else
                            "#34d399" if rsi_val < RISK_THRESHOLDS["rsi_oversold"]   else
                            "#818cf8"
                        )},
                        "bgcolor": "#0f0f14",
                        "steps": [
                            {"range": [0, 30],  "color": "rgba(52,211,153,0.1)"},
                            {"range": [30, 70], "color": "rgba(0,0,0,0)"},
                            {"range": [70, 100],"color": "rgba(248,113,113,0.1)"},
                        ],
                        "threshold": {
                            "line": {"color": "#fbbf24", "width": 1},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig_g.update_layout(
                    **PLOTLY_DARK, height=140,
                    margin=dict(l=5, r=5, t=25, b=5),
                )
                st.plotly_chart(fig_g, use_container_width=True)

        # Price with SMA for selected stock
        st.markdown('<div class="section-header">PRICE + MOVING AVERAGES</div>',
                    unsafe_allow_html=True)
        sel_ma = st.selectbox("Select stock", close_df.columns.tolist(), key="ma_select")

        if sel_ma:
            prices = close_df[sel_ma].dropna()
            sma20  = prices.rolling(20).mean()
            sma50  = prices.rolling(50).mean()
            vol    = (
                pd.DataFrame(
                    fetch_ohlcv(
                        str(holdings_df[holdings_df["trading_symbol"] == sel_ma]["security_id"].values[0]),
                        api_token
                    ) or {}
                ).get("volume", pd.Series(dtype=float))
            )

            fig_ma = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.04,
            )
            fig_ma.add_trace(go.Candlestick(
                x=prices.index, open=prices, high=prices,
                low=prices, close=prices,
                name="Price",
                increasing_line_color="#34d399",
                decreasing_line_color="#f87171",
                showlegend=False,
            ), row=1, col=1)
            # Use line chart for cleaner look
            fig_ma.data = []
            fig_ma.add_trace(go.Scatter(
                x=prices.index, y=prices.values,
                name="Price", line=dict(color="#f4f4f5", width=1.2),
            ), row=1, col=1)
            fig_ma.add_trace(go.Scatter(
                x=sma20.index, y=sma20.values,
                name="SMA 20", line=dict(color="#a78bfa", width=1.5, dash="dash"),
            ), row=1, col=1)
            fig_ma.add_trace(go.Scatter(
                x=sma50.index, y=sma50.values,
                name="SMA 50", line=dict(color="#fbbf24", width=1.5, dash="dot"),
            ), row=1, col=1)

            fig_ma.update_layout(
                **PLOTLY_DARK,
                height=340,
                legend=dict(orientation="h", y=1.05, font=dict(size=9)),
                xaxis2=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e1e2e"),
            )
            st.plotly_chart(fig_ma, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ALERTS
# ══════════════════════════════════════════════════════════════════════════════

with tab_alerts:
    col_cfg, col_log = st.columns([1, 2])

    with col_cfg:
        st.markdown('<div class="section-header">ALERT CONFIGURATION</div>',
                    unsafe_allow_html=True)

        alert_conditions = {
            "pnl_drop":        st.checkbox("P&L drop today > threshold", value=True),
            "concentration":   st.checkbox("Holding weight > threshold", value=True),
            "high_vol":        st.checkbox("Annualised volatility spike", value=True),
            "big_drawdown":    st.checkbox("Max drawdown exceeded",       value=True),
            "rsi_extreme":     st.checkbox("RSI overbought / oversold",   value=True),
            "high_corr":       st.checkbox("New high correlation pair",   value=False),
        }

        st.markdown('<div class="section-header">SEND ALERTS</div>', unsafe_allow_html=True)

        btn_test = st.button("📨 Send test message", use_container_width=True)
        btn_run  = st.button("⚡ Run checks now & alert", use_container_width=True,
                             type="primary")

        if not tele_enabled:
            st.warning("Telegram alerts are disabled. Enable in the sidebar.")
        elif not tele_token or not tele_chatid:
            st.warning("Enter your Telegram bot token and chat ID.")

        st.markdown('<div class="section-header">ALERT STATUS</div>', unsafe_allow_html=True)

        if risk_df is not None:
            checks = check_alert_conditions(
                holdings_df, risk_df, momentum_df, corr_matrix,
                RISK_THRESHOLDS, alert_conditions
            )
            for check in checks:
                sev   = check["severity"]
                badge = f'<span class="badge-{sev}">{sev.upper()}</span>'
                st.markdown(
                    f'{badge} <span style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:12px; color:#d1d5db;"> {check["message"]}</span>',
                    unsafe_allow_html=True,
                )

    with col_log:
        st.markdown('<div class="section-header">ALERT LOG</div>', unsafe_allow_html=True)

        # Run checks on button press
        if btn_test and tele_enabled and tele_token and tele_chatid:
            ok = send_telegram_alert(
                tele_token, tele_chatid,
                "✅ *Portfolio Intelligence* — test message OK"
            )
            if ok:
                st.success("Test message sent!")
                st.session_state.alert_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "TEST", "message": "Test message sent", "ok": True
                })
            else:
                st.error("Failed to send — check token and chat ID.")

        if btn_run and risk_df is not None:
            if not tele_enabled or not tele_token or not tele_chatid:
                st.warning("Configure Telegram first.")
            else:
                checks = check_alert_conditions(
                    holdings_df, risk_df, momentum_df, corr_matrix,
                    RISK_THRESHOLDS, alert_conditions
                )
                msg = build_alert_message(checks, holdings_df, total_pnl_pct)
                ok  = send_telegram_alert(tele_token, tele_chatid, msg)
                status = "✅ Sent" if ok else "❌ Failed"
                st.session_state.alert_log.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "MANUAL",
                    "message": f"{len(checks)} conditions • {status}",
                    "ok": ok,
                })
                if ok:
                    st.success(f"Alert sent — {len(checks)} conditions reported.")
                else:
                    st.error("Send failed.")

        if st.session_state.alert_log:
            for entry in reversed(st.session_state.alert_log[-20:]):
                colour = "#34d399" if entry["ok"] else "#f87171"
                st.markdown(f"""
                <div style='border:0.5px solid #1e1e2e; border-radius:6px;
                     padding:8px 12px; margin-bottom:6px; display:flex;
                     align-items:center; gap:10px;'>
                    <span style='font-family:"JetBrains Mono",monospace;
                         font-size:10px; color:#6b7280; min-width:60px;'>
                        {entry['time']}
                    </span>
                    <span style='font-family:"JetBrains Mono",monospace;
                         font-size:10px; color:#818cf8; min-width:60px;'>
                        {entry['type']}
                    </span>
                    <span style='font-family:"JetBrains Mono",monospace;
                         font-size:12px; color:{colour};'>
                        {entry['message']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='color:#374151; font-family:"JetBrains Mono",monospace;
                 font-size:12px; padding:1rem 0;'>
                No alerts sent this session.
            </div>
            """, unsafe_allow_html=True)

        # Show full alert preview
        if risk_df is not None:
            st.markdown('<div class="section-header">MESSAGE PREVIEW</div>',
                        unsafe_allow_html=True)
            checks = check_alert_conditions(
                holdings_df, risk_df, momentum_df, corr_matrix,
                RISK_THRESHOLDS, alert_conditions
            )
            preview = build_alert_message(checks, holdings_df, total_pnl_pct)
            st.code(preview, language=None)
