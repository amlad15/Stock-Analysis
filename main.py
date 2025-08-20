from typing import Optional, Dict, Any
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas_ta as ta

# -----------------------------
# App & Middleware
# -----------------------------
app = FastAPI(title="Stock Analysis API", version="1.0.0")

# Enable CORS (allow all origins; adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key (set env var API_KEY; defaults to 'changeme')
EXPECTED_API_KEY = os.getenv("API_KEY", "changeme")


def require_api_key(x_api_key: Optional[str]):
    if x_api_key is None or x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True


# -----------------------------
# Helper Functions
# -----------------------------
def safe_last(series: pd.Series) -> Optional[float]:
    if series is None or len(series.dropna()) == 0:
        return None
    return float(series.dropna().iloc[-1])


def score_rsi(rsi_value: Optional[float]) -> int:
    # Centered around 50, linear drop to 0 at 0 or 100
    if rsi_value is None:
        return 50
    score = 100 - abs(rsi_value - 50) * 2
    return int(max(0, min(100, score)))


def score_macd(macd_hist: pd.Series) -> int:
    # Score based on percentile of the last hist value within recent distribution
    if macd_hist is None or macd_hist.dropna().empty:
        return 50
    last_val = macd_hist.dropna().iloc[-1]
    dist = macd_hist.dropna()
    # Avoid degenerate distribution
    if dist.std() == 0:
        return 50
    percentile = (dist <= last_val).mean()
    return int(max(0, min(100, round(percentile * 100))))


def score_sma_cross(sma50: pd.Series, sma200: pd.Series) -> (bool, int):
    if sma50 is None or sma200 is None or sma50.dropna().empty or sma200.dropna().empty:
        return False, 50
    s50 = sma50.dropna()
    s200 = sma200.dropna()
    # Align indexes
    joined = pd.concat([s50, s200], axis=1).dropna()
    joined.columns = ["sma50", "sma200"]
    if joined.empty:
        return False, 50
    curr_above = joined["sma50"].iloc[-1] > joined["sma200"].iloc[-1]
    prev_above = joined["sma50"].iloc[-2] > joined["sma200"].iloc[-2] if len(joined) > 1 else curr_above
    golden_cross = curr_above and not prev_above
    if golden_cross:
        score = 80
    elif curr_above:
        score = 65
    else:
        # If death cross happened today (crossed down)
        death_cross = (not curr_above) and prev_above
        score = 35 if death_cross else 45
    return bool(golden_cross), int(score)


def score_bollinger(position: str) -> int:
    if position == "below_lower":
        return 70
    if position == "above_upper":
        return 30
    return 50  # neutral / within bands


def compute_debt_to_equity(tkr: yf.Ticker) -> Optional[float]:
    # Try quarterly_balance_sheet then balance_sheet
    for sheet_name in ["quarterly_balance_sheet", "balance_sheet"]:
        try:
            sheet = getattr(tkr, sheet_name)
            if isinstance(sheet, pd.DataFrame) and not sheet.empty:
                # Common fields
                # yfinance frames: rows are labels, columns are dates
                def get_field(frame: pd.DataFrame, key_options):
                    for k in key_options:
                        if k in frame.index:
                            return frame.loc[k].dropna().iloc[0]
                    return None
                total_debt = get_field(sheet, ["Total Debt", "Short Long Term Debt", "Long Term Debt"])
                total_equity = get_field(sheet, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"])
                if total_debt is not None and total_equity not in (None, 0):
                    return float(total_debt) / float(total_equity)
        except Exception:
            continue
    return None


def compute_revenue_growth_yoy(tkr: yf.Ticker) -> Optional[float]:
    # Use yearly income statement when possible, else quarterly annualized
    for stmt_name in ["income_stmt", "quarterly_income_stmt"]:
        try:
            stmt = getattr(tkr, stmt_name)
            if isinstance(stmt, pd.DataFrame) and not stmt.empty:
                if "Total Revenue" in stmt.index:
                    rev = stmt.loc["Total Revenue"].dropna()
                elif "TotalRevenue" in stmt.index:
                    rev = stmt.loc["TotalRevenue"].dropna()
                else:
                    continue
                if len(rev) >= 2:
                    # Most recent vs one year prior (or previous quarter)
                    newest = float(rev.iloc[0])
                    prior = float(rev.iloc[1])
                    if prior != 0:
                        return (newest - prior) / abs(prior)
        except Exception:
            continue
    return None


def score_pe(pe: Optional[float]) -> int:
    if pe is None or pe <= 0:
        return 20
    # Favor ~10-25, taper outside
    if pe <= 10:
        return 80
    if 10 < pe <= 25:
        # map 10->80 to 25->90
        return int(80 + (pe - 10) * (10 / 15))
    if 25 < pe <= 40:
        # map 25->90 to 40->60
        return int(90 - (pe - 25) * (30 / 15))
    if 40 < pe <= 80:
        # map 40->60 to 80->30
        return int(60 - (pe - 40) * (30 / 40))
    return 20  # very high P/E


def score_de(de: Optional[float]) -> int:
    if de is None:
        return 50
    if de < 0:
        return 40
    if de <= 0.3:
        return 90
    if de <= 0.5:
        return 80
    if de <= 1.0:
        return 60
    if de <= 2.0:
        return 40
    return 20


def score_rev_growth(g: Optional[float]) -> int:
    if g is None:
        return 50
    if g <= -0.2:
        return 10
    if -0.2 < g < 0:
        return 30
    if 0 <= g < 0.05:
        return 40
    if 0.05 <= g < 0.10:
        return 60
    if 0.10 <= g < 0.20:
        return 75
    if 0.20 <= g < 0.40:
        return 90
    return 95  # very strong


def monte_carlo_sim(returns: pd.Series, trials: int = 10_000, days: int = 252) -> Dict[str, float]:
    ret = returns.dropna().values
    if ret.size == 0:
        return {"mean_return": 0.0, "worst_case": 0.0, "prob_profit": 0.0}
    rng = np.random.default_rng()
    # Sample with replacement for each trial, compound returns
    sims = np.empty(trials, dtype=float)
    for i in range(trials):
        path = rng.choice(ret, size=days, replace=True)
        sims[i] = float(np.prod(1 + path) - 1)
    mean_ret = float(np.mean(sims))
    worst_case = float(np.percentile(sims, 5))
    prob_profit = float((sims > 0).mean())
    return {"mean_return": mean_ret, "worst_case": worst_case, "prob_profit": prob_profit}


def bollinger_position(close: pd.Series, length: int = 20, stds: float = 2.0) -> (str, pd.DataFrame):
    bb = ta.bbands(close, length=length, std=stds)
    if bb is None or bb.dropna().empty:
        return "neutral", pd.DataFrame()
    bb = bb.dropna()
    last = bb.iloc[-1]
    last_close = float(close.dropna().iloc[-1])
    lower = float(last["BBL_20_2.0"])
    upper = float(last["BBU_20_2.0"])
    if last_close < lower:
        return "below_lower", bb
    if last_close > upper:
        return "above_upper", bb
    return "within_bands", bb


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/analyze")
def analyze(
    ticker: str = Query(..., description="Stock ticker symbol"),
    period: str = Query("1y", description="Historical period (e.g., 6mo, 1y, 5y)"),
    x_api_key: str = Header(..., alias="x-api-key"),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # Fetch data
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for ticker")

    # --- Technical indicators ---
    rsi = ta.rsi(df["Close"], length=14).iloc[-1]
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    macd_val = macd["MACD_12_26_9"].iloc[-1] - macd["MACDs_12_26_9"].iloc[-1]

    sma50 = ta.sma(df["Close"], length=50).iloc[-1]
    sma200 = ta.sma(df["Close"], length=200).iloc[-1]
    golden_cross = sma50 > sma200

    bbands = ta.bbands(df["Close"], length=20, std=2)
    last_close = df["Close"].iloc[-1]
    upper, lower = bbands["BBU_20_2.0"].iloc[-1], bbands["BBL_20_2.0"].iloc[-1]
    if last_close > upper:
        bb_pos = "overbought"
    elif last_close < lower:
        bb_pos = "oversold"
    else:
        bb_pos = "neutral"

    # --- Fundamentals (limited by yfinance availability) ---
    pe_ratio = stock.info.get("trailingPE", None)
    debt_equity = stock.info.get("debtToEquity", None)
    revenue_growth = stock.info.get("revenueGrowth", None)

    # --- Scoring (simplified linear mapping) ---
    def scale(val, low, high):
        if val is None or np.isnan(val):
            return 50
        return int(np.clip(100 * (val - low) / (high - low), 0, 100))

    rsi_score = 100 - abs(rsi - 50) * 2 if not np.isnan(rsi) else 50
    macd_score = scale(macd_val, -2, 2)
    sma_score = 80 if golden_cross else 40
    bb_score = {"overbought": 30, "oversold": 70, "neutral": 50}[bb_pos]

    pe_score = scale(pe_ratio, 5, 30)
    de_score = scale(debt_equity, 0, 200)
    rev_score = scale(revenue_growth, -0.1, 0.3)

    # --- Weighted gradient ---
    gradient = int(
        rsi_score * 0.15
        + macd_score * 0.15
        + sma_score * 0.20
        + bb_score * 0.10
        + pe_score * 0.15
        + de_score * 0.10
        + rev_score * 0.15
    )

    # --- Monte Carlo simulation ---
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()
    trials, days = 10000, 252
    simulations = np.exp(
        np.cumsum(np.random.normal(mu, sigma, (days, trials)), axis=0)
    )
    final_returns = simulations[-1] - 1

    mean_return = float(final_returns.mean())
    worst_case = float(np.percentile(final_returns, 5))
    prob_profit = float((final_returns > 0).mean())

    # Sample paths for charting
    paths_sample = simulations[:, :10].tolist()  # 10 paths

    # Histogram for distribution
    hist, bins = np.histogram(final_returns, bins=50)
    distribution = {
        "bins": bins.tolist(),
        "freq": hist.tolist(),
    }

    return {
        "gradient": gradient,
        "technicals": {
            "RSI": {"value": float(rsi), "score": rsi_score},
            "MACD": {"value": float(macd_val), "score": macd_score},
            "SMA_Cross": {"golden_cross": golden_cross, "score": sma_score},
            "Bollinger": {"position": bb_pos, "score": bb_score},
        },
        "fundamentals": {
            "PE": {"value": pe_ratio, "score": pe_score},
            "DebtEquity": {"value": debt_equity, "score": de_score},
            "RevenueGrowth": {"value": revenue_growth, "score": rev_score},
        },
        "monte_carlo": {
            "mean_return": mean_return,
            "worst_case": worst_case,
            "prob_profit": prob_profit,
            "paths_sample": paths_sample,
            "distribution": distribution,
        },
    }

    return response
