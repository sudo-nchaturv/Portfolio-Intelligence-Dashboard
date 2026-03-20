"""
alerts.py — Telegram notification logic and alert condition checks.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional


# ─── Telegram sender ──────────────────────────────────────────────────────────

def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """
    Send a Markdown-formatted message via Telegram Bot API.
    Returns True on success, False on failure.
    """
    if not bot_token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={
                "chat_id":    chat_id,
                "text":       message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ─── Condition checks ─────────────────────────────────────────────────────────

def check_alert_conditions(
    holdings_df:   pd.DataFrame,
    risk_df:       pd.DataFrame,
    momentum_df:   Optional[pd.DataFrame],
    corr_matrix:   Optional[pd.DataFrame],
    thresholds:    Dict[str, float],
    enabled:       Dict[str, bool],
) -> List[Dict[str, Any]]:
    """
    Run all configured alert checks and return a list of triggered conditions.
    Each item: {"severity": "high"|"medium"|"low", "message": str, "symbol": str|None}
    """
    alerts = []

    # ── P&L drop ──────────────────────────────────────────────────────────────
    if enabled.get("pnl_drop", True):
        losers = holdings_df[holdings_df["pnl_percent"] < thresholds["pnl_alert_pct"]]
        for _, row in losers.iterrows():
            alerts.append({
                "severity": "high",
                "symbol":   row["trading_symbol"],
                "message":  (
                    f"{row['trading_symbol']} P&L at "
                    f"{row['pnl_percent']:+.2f}% "
                    f"(₹{row['pnl_absolute']:,.0f})"
                ),
            })

    # ── Concentration ─────────────────────────────────────────────────────────
    if enabled.get("concentration", True):
        heavy = holdings_df[holdings_df["weight_pct"] / 100 > thresholds["concentration"]]
        for _, row in heavy.iterrows():
            alerts.append({
                "severity": "high",
                "symbol":   row["trading_symbol"],
                "message":  (
                    f"{row['trading_symbol']} is {row['weight_pct']:.1f}% of portfolio "
                    f"(threshold: {thresholds['concentration']*100:.0f}%)"
                ),
            })

    # ── Volatility spike ──────────────────────────────────────────────────────
    if enabled.get("high_vol", True) and risk_df is not None:
        hi_vol = risk_df[risk_df["ann_vol"] > thresholds["vol_high"]]
        for sym in hi_vol.index:
            v = hi_vol.loc[sym, "ann_vol"]
            alerts.append({
                "severity": "medium",
                "symbol":   sym,
                "message":  f"{sym} annualised vol is {v:.1%} (threshold: {thresholds['vol_high']:.0%})",
            })

    # ── Max drawdown ──────────────────────────────────────────────────────────
    if enabled.get("big_drawdown", True) and risk_df is not None:
        dd_bad = risk_df[risk_df["max_drawdown"] < thresholds["drawdown_high"]]
        for sym in dd_bad.index:
            d = dd_bad.loc[sym, "max_drawdown"]
            alerts.append({
                "severity": "high",
                "symbol":   sym,
                "message":  f"{sym} max drawdown: {d:.1%}",
            })

    # ── RSI extremes ──────────────────────────────────────────────────────────
    if enabled.get("rsi_extreme", True) and momentum_df is not None and "rsi_14" in momentum_df.columns:
        overbought = momentum_df[momentum_df["rsi_14"] > thresholds["rsi_overbought"]]
        oversold   = momentum_df[momentum_df["rsi_14"] < thresholds["rsi_oversold"]]
        for sym in overbought.index:
            r = overbought.loc[sym, "rsi_14"]
            alerts.append({
                "severity": "medium",
                "symbol":   sym,
                "message":  f"{sym} RSI = {r:.1f} — overbought signal",
            })
        for sym in oversold.index:
            r = oversold.loc[sym, "rsi_14"]
            alerts.append({
                "severity": "low",
                "symbol":   sym,
                "message":  f"{sym} RSI = {r:.1f} — oversold / possible entry",
            })

    # ── High correlation pairs ────────────────────────────────────────────────
    if enabled.get("high_corr", False) and corr_matrix is not None:
        tickers = corr_matrix.columns.tolist()
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                c = corr_matrix.iloc[i, j]
                if abs(c) > thresholds["corr_high"]:
                    alerts.append({
                        "severity": "low",
                        "symbol":   None,
                        "message":  (
                            f"High correlation: {tickers[i]} ↔ {tickers[j]} = {c:.2f}"
                        ),
                    })

    return alerts


# ─── Message builder ──────────────────────────────────────────────────────────

def build_alert_message(
    alerts:        List[Dict[str, Any]],
    holdings_df:   pd.DataFrame,
    pnl_pct:       float,
) -> str:
    """Build a Markdown-formatted Telegram message from a list of alerts."""
    now     = datetime.now().strftime("%d %b %Y, %H:%M IST")
    total_v = holdings_df["market_value"].sum()

    lines = [
        "📊 *Portfolio Intelligence Alert*",
        f"_{now}_",
        "",
        f"*Portfolio value:* ₹{total_v:,.0f}",
        f"*Overall P&L:*    {pnl_pct:+.2f}%",
        "",
    ]

    if not alerts:
        lines.append("✅ No alerts triggered — all checks passed.")
        return "\n".join(lines)

    # Group by severity
    high   = [a for a in alerts if a["severity"] == "high"]
    medium = [a for a in alerts if a["severity"] == "medium"]
    low    = [a for a in alerts if a["severity"] == "low"]

    if high:
        lines.append("🔴 *HIGH PRIORITY*")
        for a in high:
            lines.append(f"• {a['message']}")
        lines.append("")

    if medium:
        lines.append("🟡 *MEDIUM PRIORITY*")
        for a in medium:
            lines.append(f"• {a['message']}")
        lines.append("")

    if low:
        lines.append("🟢 *LOW PRIORITY*")
        for a in low:
            lines.append(f"• {a['message']}")
        lines.append("")

    lines.append(f"_{len(alerts)} condition{'s' if len(alerts)>1 else ''} flagged_")
    return "\n".join(lines)
