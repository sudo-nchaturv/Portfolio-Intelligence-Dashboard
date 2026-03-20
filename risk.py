"""
risk.py — Portfolio risk and momentum calculations.
All functions accept pandas DataFrames and return DataFrames / Series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ─── Return series ────────────────────────────────────────────────────────────

def compute_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns for each column."""
    return np.log(close_df / close_df.shift(1)).dropna(how="all")


# ─── Core risk metrics ────────────────────────────────────────────────────────

def compute_risk_metrics(
    close_df:    pd.DataFrame,
    returns_df:  pd.DataFrame,
    holdings_df: pd.DataFrame,
    risk_free:   float = 0.065,     # 6.5% India risk-free rate (10Y GSec approx)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        risk_df     — per-stock risk DataFrame
        corr_matrix — pairwise Pearson correlation of daily returns
    """
    trading_days = 252
    rf_daily     = risk_free / trading_days

    results = {}
    for sym in close_df.columns:
        prices = close_df[sym].dropna()
        ret    = returns_df[sym].dropna() if sym in returns_df.columns else pd.Series(dtype=float)

        if len(ret) < 20:
            continue

        # Annualised volatility
        ann_vol = ret.std() * np.sqrt(trading_days)

        # Max drawdown
        roll_max    = prices.expanding().max()
        dd_series   = (prices - roll_max) / roll_max
        max_drawdown = dd_series.min()

        # Parametric VaR (95%, 1-day)
        var95_1d = ret.mean() - 1.645 * ret.std()

        # Sharpe ratio (annualised)
        excess_ret = ret - rf_daily
        sharpe     = (excess_ret.mean() / ret.std()) * np.sqrt(trading_days) if ret.std() > 0 else np.nan

        # Sortino ratio
        downside   = ret[ret < rf_daily]
        down_std   = downside.std() * np.sqrt(trading_days) if len(downside) > 0 else np.nan
        sortino    = (ret.mean() - rf_daily) * trading_days / down_std if down_std and down_std > 0 else np.nan

        # CAGR
        n_years = len(prices) / trading_days
        cagr    = (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

        results[sym] = {
            "ann_vol":     ann_vol,
            "max_drawdown": max_drawdown,
            "var95_1d":    var95_1d,
            "sharpe":      sharpe,
            "sortino":     sortino,
            "cagr":        cagr,
        }

    risk_df = pd.DataFrame(results).T

    # Beta vs equal-weighted portfolio proxy
    risk_df = _add_beta(risk_df, returns_df)

    # Flag logic
    risk_df["flag"] = (
        (risk_df["ann_vol"]      > 0.40) |
        (risk_df["max_drawdown"] < -0.30) |
        (risk_df["var95_1d"]     < -0.025)
    )

    # Correlation matrix
    corr_matrix = returns_df[list(results.keys())].corr()

    return risk_df, corr_matrix


def _add_beta(risk_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute beta of each stock vs the portfolio's equal-weighted
    daily return (a market proxy when NIFTY data is unavailable).
    """
    if returns_df.empty or len(returns_df.columns) < 2:
        risk_df["beta"] = np.nan
        return risk_df

    market = returns_df.mean(axis=1)          # equal-weighted portfolio return
    var_m  = market.var()

    betas = {}
    for sym in returns_df.columns:
        if sym not in risk_df.index:
            continue
        cov = returns_df[sym].cov(market)
        betas[sym] = cov / var_m if var_m > 0 else np.nan

    risk_df["beta"] = pd.Series(betas)
    return risk_df


# ─── Technical indicators ─────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Wilder RSI."""
    delta  = prices.diff().dropna()
    gain   = delta.where(delta > 0, 0.0)
    loss   = -delta.where(delta < 0, 0.0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else np.nan


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(period).mean()


def compute_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Returns (MACD line, signal line)."""
    ema12  = prices.ewm(span=12, adjust=False).mean()
    ema26  = prices.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_momentum(close_df: pd.DataFrame) -> pd.DataFrame:
    """Full momentum summary for all holdings."""
    rows = {}
    for sym in close_df.columns:
        prices = close_df[sym].dropna()
        if len(prices) < 50:
            continue

        rsi_14     = compute_rsi(prices, 14)
        sma20      = prices.rolling(20).mean().iloc[-1]
        sma50      = prices.rolling(50).mean().iloc[-1]
        last_price = prices.iloc[-1]

        # Rate of change
        roc_30  = (last_price / prices.iloc[-31] - 1) if len(prices) >= 31 else np.nan
        roc_90  = (last_price / prices.iloc[-91] - 1) if len(prices) >= 91 else np.nan

        # 52-week high / low
        year_data      = prices.iloc[-252:] if len(prices) >= 252 else prices
        high_52w       = year_data.max()
        low_52w        = year_data.min()
        vs_52w_high    = (last_price / high_52w - 1)
        vs_52w_low     = (last_price / low_52w  - 1)

        # SMA signal
        if not np.isnan(sma20) and not np.isnan(sma50):
            sma_signal = "ABOVE SMA20" if last_price > sma20 else "BELOW SMA20"
        else:
            sma_signal = "N/A"

        # MACD
        macd_line, signal_line = compute_macd(prices)
        macd_cross = "BULLISH" if macd_line.iloc[-1] > signal_line.iloc[-1] else "BEARISH"

        rows[sym] = {
            "rsi_14":        rsi_14,
            "roc_30":        roc_30,
            "roc_90":        roc_90,
            "vs_52w_high_pct": vs_52w_high,
            "vs_52w_low_pct":  vs_52w_low,
            "sma20_signal":  sma_signal,
            "macd_signal":   macd_cross,
            "last_price":    last_price,
            "sma20":         sma20,
            "sma50":         sma50,
        }

    return pd.DataFrame(rows).T
