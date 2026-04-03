import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURATION  ← only section you need to edit
# ==============================================================================

# ── Universe ───────────────────────────────────────────────────────────────────
CSV_FILE = ""   # Path to CSV with columns: Ticker, Weight, Sector
                # Leave "" to use the TICKERS list below

TICKERS = [
    "SGSN.SW", "GJF.OL",   "NTGY.MC",  "ANDR.VI",
    "DPLM.L",  "IMI.L",    "ELISA.HE", "SAND.ST",
    "VALMT.HE","KOG.OL",   "RIO.L",    "RXL.PA",
    "KBX.DE",  "METSO.HE", "ASSA-B.ST","MONC.MI",
    "HM-B.ST", "GMAB.CO",
]

SECTOR_MAP = {
    "SGSN.SW":  "Healthcare",  "GJF.OL":   "Financials",
    "NTGY.MC":  "Utilities",   "ANDR.VI":  "Industrials",
    "DPLM.L":   "Industrials", "IMI.L":    "Industrials",
    "ELISA.HE": "Telecom",     "SAND.ST":  "Materials",
    "VALMT.HE": "Industrials", "KOG.OL":   "Energy",
    "RIO.L":    "Materials",   "RXL.PA":   "Healthcare",
    "KBX.DE":   "Financials",  "METSO.HE": "Industrials",
    "ASSA-B.ST":"Industrials", "MONC.MI":  "Luxury",
    "HM-B.ST":  "ConsDisc",    "GMAB.CO":  "Healthcare",
}

# ── Benchmark ──────────────────────────────────────────────────────────────────
BENCHMARK = "^STOXX"  

# ── Date range & walk-forward split ───────────────────────────────────────────
START     = "2010-01-01"
TRAIN_END = "2019-01-01"  # In-sample boundary — tune parameters on this window
END       = "2024-01-01"  # Full period end
RUN_END   = END           # ← swap to TRAIN_END to run in-sample only

# ── Signal parameters ──────────────────────────────────────────────────────────
SMA_LIST     = [100, 150, 200, 250]   # Ensemble trend SMAs
RSI_PERIOD   = 14
RSI_BUY_CAP  = 65                     # Block entry when RSI ≥ this (overextended)
VOL_LOOKBACK = 60                     # Realised vol window in days (60–120 recommended)
REBAL_FREQ   = 21                     # Trading days between rebalances (~monthly)

# ── Active overlay ─────────────────────────────────────────────────────────────
ACTIVE_RISK = 0.20   # Scaling of active bets vs equal-weight benchmark
TE_TARGET   = 0.05   # Target tracking error (5%)

# ── Position sizing ────────────────────────────────────────────────────────────
MIN_WEIGHT  = 0.015   # 1.5% floor per holding (only applied to non-zero positions)
MAX_WEIGHT  = 0.05    # 5.0% cap per holding
MAX_SECTOR  = 0.25    # Max 25% per sector

# ── Execution costs ────────────────────────────────────────────────────────────
INIT_CASH = 100_000
FEES      = 0.0015
SLIPPAGE  = 0.005

# ── Stress periods ─────────────────────────────────────────────────────────────
STRESS_PERIODS = {
    "Dot-com Bust 2001-02": ("2001-01-01", "2002-12-31"),
    "GFC 2008-09":          ("2008-09-01", "2009-03-31"),
    "COVID Crash 2020":     ("2020-02-01", "2020-04-30"),
    "Rate Hike Shock 2022": ("2022-01-01", "2022-12-31"),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def extract_close(raw, tickers):
    """Robustly extract Close prices from yfinance (handles MultiIndex)."""
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.xs("Close", axis=1, level=0)
    else:
        df = raw[["Close"]] if "Close" in raw.columns else raw
    available = [t for t in tickers if t in df.columns]
    skipped   = set(tickers) - set(available)
    if skipped:
        print(f"  ⚠  No data for: {sorted(skipped)}")
    return df[available] if available else pd.DataFrame()


def compute_rsi(prices, period=14):
    """Vectorized RSI for a DataFrame of prices."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def apply_sector_cap(weights, sector_map, max_sector):
    """Vectorized sector cap — no slow row-by-row loop."""
    sectors = pd.Series(sector_map)
    for sector in sectors.unique():
        stocks = [s for s in sectors[sectors == sector].index if s in weights.columns]
        if not stocks:
            continue
        sector_total = weights[stocks].sum(axis=1)
        over = sector_total > max_sector
        if over.any():
            factor = (max_sector / sector_total[over]).clip(upper=1.0)
            weights.loc[over, stocks] = weights.loc[over, stocks].mul(factor, axis=0)
    return weights


# ==============================================================================
# 1. LOAD PORTFOLIO DEFINITION
# ==============================================================================
print("=" * 62)
print("   FUND BACKTESTER")
print("=" * 62)

sector_map   = SECTOR_MAP.copy()
base_weights = {}

if CSV_FILE:
    try:
        df_csv       = pd.read_csv(CSV_FILE)
        tickers      = df_csv["Ticker"].tolist()
        base_weights = dict(zip(df_csv["Ticker"], df_csv["Weight"]))
        if "Sector" in df_csv.columns:
            sector_map = dict(zip(df_csv["Ticker"], df_csv["Sector"]))
        print(f"  Loaded {len(tickers)} holdings from '{CSV_FILE}'")
    except FileNotFoundError:
        print(f"  ⚠  '{CSV_FILE}' not found — using TICKERS list.")
        tickers = TICKERS
else:
    tickers = TICKERS

mode = "IN-SAMPLE" if RUN_END == TRAIN_END else "FULL PERIOD"
print(f"  Universe  : {len(tickers)} tickers")
print(f"  Period    : {START} → {RUN_END}  [{mode}]")
if RUN_END != TRAIN_END:
    print(f"  Split     : in-sample → {TRAIN_END} | out-of-sample → {END}")
print(f"  Benchmark : {BENCHMARK}")

# ==============================================================================
# 2. DATA DOWNLOAD
# ==============================================================================
print("\n[1/5] Downloading market data...")

raw    = yf.download(tickers, start=START, end=RUN_END, auto_adjust=True, progress=False)
prices = extract_close(raw, tickers)

# Drop tickers with insufficient history
min_bars = max(SMA_LIST) + VOL_LOOKBACK
valid    = [t for t in prices.columns if prices[t].dropna().shape[0] >= min_bars]
dropped  = [t for t in prices.columns if t not in valid]
if dropped:
    print(f"  ⚠  Dropped (insufficient history): {dropped}")

prices = prices[valid].ffill().dropna()
if prices.empty:
    raise RuntimeError("No valid price data. Check tickers and date range.")

# Benchmark — downloaded AFTER prices so index alignment is correct
bench_raw    = yf.download(BENCHMARK, start=START, end=RUN_END, auto_adjust=True, progress=False)
bench_prices = extract_close(bench_raw, [BENCHMARK])
if bench_prices.empty:
    raise RuntimeError(f"Could not download benchmark '{BENCHMARK}'. Check the ticker.")
bench_prices  = bench_prices.squeeze().reindex(prices.index).ffill()
bench_returns = np.log(bench_prices / bench_prices.shift(1))
returns       = np.log(prices / prices.shift(1))

print(f"  ✓  {len(prices.columns)} tickers | {len(prices)} trading days")
print(f"  ✓  Benchmark '{BENCHMARK}' loaded")

# ==============================================================================
# 3. SIGNAL ENGINE
# ==============================================================================
print("\n[2/5] Computing signals...")

# Trend: ensemble SMA  (0 = fully bearish → 1 = fully bullish)
signal_sum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
for sma in SMA_LIST:
    signal_sum += (prices > prices.rolling(sma).mean()).astype(float)
trend_signal = signal_sum / len(SMA_LIST)

# RSI filter: block new entries when price is overextended
rsi        = compute_rsi(prices, period=RSI_PERIOD)
rsi_filter = (rsi < RSI_BUY_CAP).astype(float)

# Combined signal
signals = trend_signal * rsi_filter

print(f"  ✓  Ensemble SMA {SMA_LIST} + RSI({RSI_PERIOD} < {RSI_BUY_CAP})")

# ==============================================================================
# 4. POSITION SIZING
# ==============================================================================
print("\n[3/5] Sizing positions...")

# Inverse-volatility base weights
vol     = returns.rolling(VOL_LOOKBACK).std() * np.sqrt(252)
inv_vol = (1 / vol).replace([np.inf, -np.inf], np.nan)

raw_w   = signals * inv_vol
strat_w = raw_w.div(raw_w.sum(axis=1), axis=0).fillna(0)

# Active overlay vs equal-weight benchmark
bench_w = pd.DataFrame(1 / len(prices.columns), index=prices.index, columns=prices.columns)
active  = strat_w.sub(strat_w.mean(axis=1), axis=0)   # zero-mean active bets
weights = bench_w + ACTIVE_RISK * active

# Tracking error control
implied_ret    = (weights.shift(1) * returns).sum(axis=1)
excess_implied = implied_ret - bench_returns
te_rolling     = excess_implied.rolling(60).std() * np.sqrt(252)
te_scale       = (TE_TARGET / te_rolling).clip(upper=1).fillna(1)
weights        = bench_w + (weights - bench_w).mul(te_scale, axis=0)

# Min/max clamp (only for active positions)
pos_mask          = weights > 0
weights[pos_mask] = weights[pos_mask].clip(lower=MIN_WEIGHT, upper=MAX_WEIGHT)

# Sector cap (vectorized)
weights = apply_sector_cap(weights, sector_map, MAX_SECTOR)

# Rebalancing: hold weights constant between rebalance dates
rebal_mask              = pd.Series(False, index=weights.index)
rebal_mask.iloc[::REBAL_FREQ] = True
weights_rebal           = weights.copy()
weights_rebal.loc[~rebal_mask] = np.nan
weights                 = weights_rebal.ffill().fillna(0)

# Final normalization
weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)

print(f"  ✓  Inv-vol sizing | Active overlay (risk={ACTIVE_RISK}) | TE cap={TE_TARGET:.0%}")
print(f"  ✓  Weight range [{MIN_WEIGHT:.1%} – {MAX_WEIGHT:.1%}] | Sector cap {MAX_SECTOR:.0%}")
print(f"  ✓  Rebalancing every {REBAL_FREQ} trading days")

# ==============================================================================
# 5. BACKTEST
# ==============================================================================
print("\n[4/5] Running backtest...")

portfolio = vbt.Portfolio.from_orders(
    close        = prices,
    size         = weights,
    size_type    = "targetpercent",
    init_cash    = INIT_CASH,
    fees         = FEES,
    slippage     = SLIPPAGE,
    freq         = "1D",
    group_by     = True,    # one combined portfolio, not N separate ones
    cash_sharing = True,    # shared cash pool across all assets
)

print("  ✓  Backtest complete")

# ==============================================================================
# 6. PERFORMANCE METRICS
# ==============================================================================
print("\n[5/5] Computing metrics...")

port_value    = portfolio.value()
port_ret_real = portfolio.returns()

bench_aligned   = bench_returns.reindex(port_value.index).fillna(0)
excess_realized = port_ret_real - bench_aligned
annual_excess   = excess_realized.mean() * 252
tracking_err    = excess_realized.std() * np.sqrt(252)
info_ratio      = annual_excess / tracking_err if tracking_err != 0 else np.nan

bench_value     = (1 + bench_aligned).cumprod() * INIT_CASH

port_total_ret  = portfolio.total_return()
port_sharpe     = portfolio.sharpe_ratio()
port_maxdd      = portfolio.max_drawdown()
bench_total_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0]) - 1
bench_sharpe    = bench_returns.mean() / bench_returns.std() * np.sqrt(252)
bench_maxdd     = (bench_prices / bench_prices.cummax() - 1).min()
avg_vol_exp     = (weights * vol).sum(axis=1).mean()

print("\n" + "=" * 62)
print("  PERFORMANCE vs BENCHMARK")
print("=" * 62)
print(f"  {'Metric':<28} {'Strategy':>12} {'Benchmark':>12}")
print("  " + "-" * 54)
print(f"  {'Total Return':<28} {port_total_ret:>11.2%} {bench_total_ret:>11.2%}")
print(f"  {'Sharpe Ratio':<28} {port_sharpe:>12.2f} {bench_sharpe:>12.2f}")
print(f"  {'Max Drawdown':<28} {port_maxdd:>11.2%} {bench_maxdd:>11.2%}")

print("\n" + "=" * 62)
print("  RELATIVE PERFORMANCE")
print("=" * 62)
print(f"  {'Annual Excess Return':<28} {annual_excess:>+11.2%}")
print(f"  {'Tracking Error':<28} {tracking_err:>11.2%}")
print(f"  {'Information Ratio':<28} {info_ratio:>12.2f}")

print("\n" + "=" * 62)
print("  FACTOR DIAGNOSTICS")
print("=" * 62)
print(f"  {'Avg Volatility Exposure':<28} {avg_vol_exp:>12.4f}")

print("\n" + "=" * 62)
print("  STRESS PERIOD ANALYSIS")
print("=" * 62)
print(f"  {'Period':<24} {'Strategy':>10} {'Benchmark':>10} {'Alpha':>10}")
print("  " + "-" * 57)

for name, (s, e) in STRESS_PERIODS.items():
    try:
        pv = port_value.loc[s:e]
        bv = bench_prices.loc[s:e]
        if len(pv) < 2 or len(bv) < 2:
            print(f"  {name:<24} {'N/A (out of date range)':>33}")
            continue
        p_ret = (pv.iloc[-1] / pv.iloc[0]) - 1
        b_ret = (bv.iloc[-1] / bv.iloc[0]) - 1
        alpha = p_ret - b_ret
        flag  = " ✓" if alpha > 0 else " ✗"
        print(f"  {name:<24} {p_ret:>9.2%}  {b_ret:>9.2%}  {alpha:>+9.2%}{flag}")
    except Exception:
        print(f"  {name:<24} {'N/A':>33}")

print("=" * 62)

# ==============================================================================
# 7. CHARTS
# ==============================================================================
port_dd  = portfolio.drawdown()
bench_dd = bench_prices / bench_prices.cummax() - 1

# ── Chart 1: Cumulative value vs benchmark ────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=port_value.index, y=port_value,
    name="Strategy", line=dict(color="royalblue", width=2)))
fig1.add_trace(go.Scatter(
    x=bench_value.index, y=bench_value,
    name=BENCHMARK, line=dict(color="darkorange", width=2, dash="dash")))

for name, (s, e) in STRESS_PERIODS.items():
    fig1.add_vrect(
        x0=s, x1=e, fillcolor="red", opacity=0.07,
        layer="below", line_width=0,
        annotation_text=name, annotation_position="top left",
        annotation=dict(font_size=9, font_color="crimson"))

fig1.update_layout(
    title="Portfolio Value vs Benchmark (stress periods shaded)",
    xaxis_title="Date", yaxis_title="Portfolio Value (€)",
    legend=dict(x=0.01, y=0.99),
    template="plotly_white", hovermode="x unified")
fig1.write_html("portfolio_value.html")

# ── Chart 2: Drawdown comparison ──────────────────────────────────────────────
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=port_dd.index, y=port_dd,
    name="Strategy", fill="tozeroy",
    line=dict(color="royalblue", width=1.5)))
fig2.add_trace(go.Scatter(
    x=bench_dd.index, y=bench_dd,
    name=BENCHMARK, fill="tozeroy", opacity=0.5,
    line=dict(color="darkorange", width=1.5, dash="dash")))

for name, (s, e) in STRESS_PERIODS.items():
    fig2.add_vrect(
        x0=s, x1=e, fillcolor="red", opacity=0.07,
        layer="below", line_width=0)

fig2.update_layout(
    title="Drawdown: Strategy vs Benchmark",
    xaxis_title="Date", yaxis_title="Drawdown",
    yaxis_tickformat=".0%", legend=dict(x=0.01, y=0.01),
    template="plotly_white", hovermode="x unified")
fig2.write_html("portfolio_drawdown.html")

# ── Chart 3: Stacked portfolio weights over time ──────────────────────────────
fig3 = go.Figure()
for ticker in prices.columns:
    fig3.add_trace(go.Scatter(
        x=weights.index, y=weights[ticker],
        name=ticker, stackgroup="one", mode="none"))

fig3.update_layout(
    title="Portfolio Weights Over Time",
    xaxis_title="Date", yaxis_title="Weight",
    yaxis_tickformat=".0%",
    template="plotly_white", hovermode="x unified")
fig3.write_html("portfolio_weights.html")

print("\nCharts saved:")
print("  → portfolio_value.html    (cumulative value vs benchmark)")
print("  → portfolio_drawdown.html (drawdown comparison)")
print("  → portfolio_weights.html  (stacked weights over time)")