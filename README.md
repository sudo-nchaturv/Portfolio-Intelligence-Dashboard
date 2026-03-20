# Portfolio Intelligence Dashboard

A full-featured portfolio analytics dashboard built on the INDstocks API.  
Tracks your Demat holdings, runs risk analysis, plots a correlation heatmap,  
and fires Telegram alerts when thresholds are breached.

---

## Quick start (local)

```bash
# 1. Clone / download the three files
#    app.py  risk.py  alerts.py  requirements.txt

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Then open http://localhost:8501 and enter your INDstocks access token in the sidebar.

---

## Files

| File             | Purpose                                             |
|------------------|-----------------------------------------------------|
| `app.py`         | Streamlit UI — all tabs, charts, and layout         |
| `risk.py`        | Risk maths: volatility, drawdown, VaR, RSI, SMA     |
| `alerts.py`      | Telegram sender + alert condition checks             |
| `requirements.txt` | Python dependencies                               |

---

## Getting your INDstocks access token

1. Log in to the INDstocks developer portal.
2. Navigate to **API Keys** → generate a new token.
3. Paste it into the **Access Token** field in the sidebar.

The token is never stored to disk — it lives only in the browser session.

---

## Telegram alerts setup

1. Open Telegram, search for **@BotFather**.
2. Send `/newbot` and follow prompts — copy the **bot token**.
3. Add the bot to a group or message it directly.
4. Get your **chat ID** by messaging `https://api.telegram.org/bot<TOKEN>/getUpdates`  
   after sending one message to the bot — look for `"chat":{"id":...}`.
5. Paste both into the sidebar, toggle **Enable alerts**, and run.

---

## Automating daily alerts (cron)

Create `cron_alert.py`:

```python
"""Run once per day via cron or GitHub Actions."""
import os, pandas as pd, time, requests
from risk import compute_returns, compute_risk_metrics, compute_momentum
from alerts import (
    send_telegram_alert, check_alert_conditions,
    build_alert_message
)

TOKEN      = os.environ["INDSTOCKS_TOKEN"]
TELE_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELE_CHAT  = os.environ["TELEGRAM_CHAT_ID"]
BASE       = "https://api.indstocks.com"

def fetch(path, **kw):
    return requests.get(f"{BASE}{path}",
                        headers={"Authorization": TOKEN},
                        **kw).json()

holdings = pd.DataFrame(fetch("/portfolio/holdings")["data"])
holdings["weight_pct"] = holdings["market_value"] / holdings["market_value"].sum() * 100

# ... fetch OHLCV, compute risk, run checks ...
# (same logic as app.py)
```

Add to crontab:
```
0 16 * * 1-5  cd /path/to/app && python cron_alert.py
```

---

## Free deployment on Streamlit Cloud

1. Push all four files to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → select your repo.
3. Set **Main file path** to `app.py`.
4. Add secrets in **Advanced settings** if you want to pre-fill credentials:

```toml
# .streamlit/secrets.toml  (DO NOT commit this file)
INDSTOCKS_TOKEN  = "your_token"
TELEGRAM_TOKEN   = "bot_token"
TELEGRAM_CHAT_ID = "-100XXXXXXXXXX"
```

Then read them in `app.py` with `st.secrets["INDSTOCKS_TOKEN"]`.

---

## Risk metrics explained

| Metric             | What it measures                                              | Flag threshold |
|--------------------|---------------------------------------------------------------|----------------|
| Annualised vol     | Std dev of daily returns × √252                               | > 40%          |
| Max drawdown       | Worst peak-to-trough decline in history                       | < −30%         |
| VaR 95% (1-day)    | Expected worst daily loss on 19/20 trading days (parametric) | < −2.5%        |
| Sharpe ratio       | Excess return per unit of risk (> 1 = good)                  | < 0 = bad      |
| Sortino ratio      | Like Sharpe, but penalises only downside volatility           | — info only    |
| Beta               | Sensitivity vs equal-weighted portfolio                       | — info only    |
| RSI (14-day)       | Momentum oscillator: > 70 overbought, < 30 oversold          | 70 / 30        |
| Correlation        | Pearson correlation of daily log-returns                      | > 0.80         |
| HHI                | Portfolio concentration index (0 = diverse, 1 = all in one)  | > 0.18         |

---

## Notes

- All API calls are cached (holdings: 5 min, OHLCV: 60 min) via `@st.cache_data`.
- Beta is computed vs the portfolio's own equal-weighted return proxy, not NIFTY 50,  
  because the NIFTY scrip code may differ across accounts. Swap `market` in `risk.py`  
  with a NIFTY OHLCV series for true index beta.
- The parametric VaR assumes normally distributed returns — not always accurate for  
  small/mid-cap Indian stocks. Use as directional guidance, not precise risk management.
