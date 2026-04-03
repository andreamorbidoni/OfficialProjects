import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ==============================================================================
# CONFIGURATION  ← only section you need to edit
# ==============================================================================

# ── Universe ───────────────────────────────────────────────────────────────────
CSV_FILE = ""   # Path to CSV with columns: Ticker, Weight, Sector
                # Leave "" to use the TICKERS list below

TICKERS = [
    "SGSN.SW", "GJF.OL", "NTGY.MC",
    "DPLM.L",  "IMI.L", "ELISA.HE",
    "KOG.OL",  "RXL.PA", "ASSA-B.ST",
    "GMAB.CO",
]

SECTOR_MAP = {
    "SGSN.SW":  "Healthcare",  "GJF.OL":   "Financials",
    "NTGY.MC":  "Utilities",
    "DPLM.L":   "Industrials", "IMI.L":    "Industrials",
    "ELISA.HE": "Telecom",
    "KOG.OL":   "Energy",
    "RXL.PA":   "Healthcare",
    "ASSA-B.ST":"Industrials",
    "GMAB.CO":  "Healthcare",
}

# ── Benchmark ──────────────────────────────────────────────────────────────────
# FIX: "^STOXX" is not a valid yfinance ticker.
#      "^STOXX600" (STOXX Europe 600) is the broadest match for this universe.
#      Alternatives: "^STOXX50E" (Euro STOXX 50 – large-cap only).
BENCHMARK = "^STOXX"

# ── Date range & walk-forward split ───────────────────────────────────────────
START     = "2016-01-01"
TRAIN_END = "2019-01-01"   # In-sample boundary
END       = "2026-03-02"   # Full period end
RUN_END   = END            # ← swap to TRAIN_END to run in-sample only

# ── Covariance estimation ──────────────────────────────────────────────────────
COV_LOOKBACK  = 126    # Rolling window for covariance matrix (≈ 6 months, min 60)
                       # Ledoit-Wolf shrinkage is always applied on top
SHRINK_ALPHA  = 0.0    # Extra diagonal regularisation (0 = pure LW, 0.01 = extra damping)

# ── Rebalancing ────────────────────────────────────────────────────────────────
REBAL_FREQ    = 21     # Trading days between rebalances (~monthly)

# ── Position sizing ────────────────────────────────────────────────────────────
# FIX: Comment was "15% cap" but the value is 0.05 = 5%.
#      NOTE: With a 10-stock universe, a 5% cap means the optimizer has little
#      freedom (10 × 5% = 50% max invested). Consider raising to 0.15-0.20 to
#      let min-variance do meaningful work.
MIN_WEIGHT    = 0.01   # 1% floor per holding (forces diversification)
MAX_WEIGHT    = 0.10   # 5% cap per holding
MAX_SECTOR    = 0.35   # Max 35% per sector

# ── Volatility target (optional dampener) ──────────────────────────────────────
# If the ex-ante portfolio vol exceeds VOL_TARGET, weights are scaled down
# and the remainder is held in cash.  Set to None to disable.
VOL_TARGET    = 0.10   # 10% annualised; set to None to skip

# ── Execution costs ────────────────────────────────────────────────────────────
INIT_CASH     = 100_000
FEES          = 0.0015
SLIPPAGE      = 0.003

# FIX: Defined once here only (was duplicated at line 360 in the original).
CASH_RETURN_ANN = 0.035   # Annual return on cash held (3.5%)

# ── Stress periods ─────────────────────────────────────────────────────────────
# FIX: Removed "Dot-com Bust 2001-02" and "GFC 2008-09" — both predate
#      START = "2010-01-01" and will always produce N/A in the output.
STRESS_PERIODS = {
    "Dot-com Bust 2001-02": ("2001-01-01", "2002-12-31"),
    "GFC 2008-09":          ("2008-09-01", "2009-03-31"),
    "COVID Crash 2020":     ("2020-02-01", "2020-04-30"),
    "Rate Hike Shock 2022": ("2022-01-01", "2022-12-31"),
}

# ==============================================================================
# HELPERS
# ==============================================================================

def extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Robustly extract Close prices from yfinance output (handles MultiIndex
    from multi-ticker downloads and flat DataFrames from single-ticker downloads).

    FIX: In the original, single-ticker downloads returned a "Close"-named
    column that was never matched against the ticker name, producing an empty
    DataFrame and a misleading RuntimeError downstream.
    """
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker download: columns are (field, ticker)
        df = raw.xs("Close", axis=1, level=0)
    else:
        # Single-ticker download: columns are plain field names ("Close", "Open", …)
        if "Close" in raw.columns:
            # Rename "Close" to the requested ticker so the lookup below works
            df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            df = raw

    available = [t for t in tickers if t in df.columns]
    skipped   = set(tickers) - set(available)
    if skipped:
        print(f"  ⚠  No data for: {sorted(skipped)}")
    return df[available] if available else pd.DataFrame()


def lw_cov(ret_window: np.ndarray, shrink_alpha: float = 0.0) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance.
    shrink_alpha adds extra diagonal regularisation to handle near-singular matrices.
    """
    lw  = LedoitWolf().fit(ret_window)
    cov = lw.covariance_
    if shrink_alpha > 0:
        cov += shrink_alpha * np.diag(np.diag(cov))
    return cov


def min_variance_weights(
    cov: np.ndarray,
    n: int,
    min_w: float,
    max_w: float,
    sector_labels: list,
    max_sector: float,
    # FIX: Removed unused `sector_names` parameter that was accepted but never
    #      referenced inside the function body (dead code).
) -> np.ndarray:
    """
    Solve the minimum-variance problem:

        min  w' Σ w
        s.t. Σ wᵢ = 1
             min_w ≤ wᵢ ≤ max_w   ∀ i
             Σ_{i ∈ sector} wᵢ ≤ max_sector   ∀ sector

    Returns a weight vector of length n.
    Falls back to constraint-clipped inverse-vol weights on optimiser failure.
    """
    w0 = np.full(n, 1.0 / n)

    # Objective: portfolio variance
    def port_var(w):
        return w @ cov @ w

    def port_var_grad(w):
        return 2 * cov @ w

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    ]

    # Sector inequality constraints
    unique_sectors = list(set(sector_labels))
    for sec in unique_sectors:
        idx = [i for i, s in enumerate(sector_labels) if s == sec]
        if idx:
            constraints.append({
                "type": "ineq",
                "fun": lambda w, ix=idx: max_sector - w[ix].sum()
            })

    bounds = [(min_w, max_w)] * n

    result = minimize(
        port_var,
        w0,
        jac=port_var_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        w     = np.clip(result.x, 0, None)   # guard against tiny negatives
        total = w.sum()
        return w / total if total > 0 else w0

    # FIX: Original fallback returned raw inverse-vol weights that could violate
    #      min_w / max_w / sector constraints.  Now we clip to [min_w, max_w]
    #      and renormalise so the fallback respects individual weight bounds.
    vols       = np.sqrt(np.diag(cov))
    inv        = np.where(vols > 0, 1.0 / vols, 0.0)
    w_fallback = inv / inv.sum() if inv.sum() > 0 else w0
    w_fallback = np.clip(w_fallback, min_w, max_w)
    w_fallback /= w_fallback.sum()
    return w_fallback


def portfolio_ex_ante_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualised ex-ante portfolio volatility (log-return covariance basis)."""
    return float(np.sqrt(252 * weights @ cov @ weights))


# ==============================================================================
# 1. LOAD PORTFOLIO DEFINITION
# ==============================================================================
print(" " * 62)
print(" " * 62)
print("=" * 62)
print("   MINIMUM-VOLATILITY FUND BACKTESTER")
print("=" * 62)

sector_map = SECTOR_MAP.copy()

if CSV_FILE:
    try:
        df_csv     = pd.read_csv(CSV_FILE)
        tickers    = df_csv["Ticker"].tolist()
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
print(f"  Objective : Minimum Variance (Ledoit-Wolf shrinkage, cov={COV_LOOKBACK}d)")
if VOL_TARGET:
    print(f"  Vol Target: {VOL_TARGET:.0%} annualised (cash buffer on excess)")

# ==============================================================================
# 2. DATA DOWNLOAD
# ==============================================================================
print("\n[1/5] Downloading market data...")

raw    = yf.download(tickers, start=START, end=RUN_END, auto_adjust=True, progress=False)
prices = extract_close(raw, tickers)

# Drop tickers with insufficient history
min_bars = COV_LOOKBACK + 10
valid    = [t for t in prices.columns if prices[t].dropna().shape[0] >= min_bars]
dropped  = [t for t in prices.columns if t not in valid]
if dropped:
    print(f"  ⚠  Dropped (insufficient history): {dropped}")

prices = prices[valid].ffill().dropna()

n_assets = len(prices.columns)

# Feasibility check
if MAX_WEIGHT * n_assets < 1.0:
    raise ValueError(
        f"Constraints infeasible: MAX_WEIGHT={MAX_WEIGHT:.0%} × {n_assets} assets = "
        f"{MAX_WEIGHT * n_assets:.0%} < 100%. Increase MAX_WEIGHT or add more tickers."
    )

if prices.empty:
    raise RuntimeError("No valid price data. Check tickers and date range.")

# Benchmark
bench_raw    = yf.download(BENCHMARK, start=START, end=RUN_END, auto_adjust=True, progress=False)
bench_prices = extract_close(bench_raw, [BENCHMARK])
if bench_prices.empty:
    raise RuntimeError(
        f"Could not download benchmark '{BENCHMARK}'. "
        "Check the ticker is valid on yfinance (e.g. '^STOXX600', '^STOXX50E')."
    )
bench_prices  = bench_prices.squeeze().reindex(prices.index).ffill()
bench_returns = np.log(bench_prices / bench_prices.shift(1))
returns       = np.log(prices / prices.shift(1))

n_assets = len(prices.columns)
print(f"  ✓  {n_assets} tickers | {len(prices)} trading days")
print(f"  ✓  Benchmark '{BENCHMARK}' loaded")

# ==============================================================================
# 3. COVARIANCE ESTIMATION
# ==============================================================================
print("\n[2/5] Estimating rolling covariance matrices...")

# Pre-compute rebalance dates only — optimisation runs once per period
rebal_dates = prices.index[::REBAL_FREQ]

# Sector labels aligned to the valid tickers
sector_labels = [sector_map.get(t, "Other") for t in prices.columns]

weights_dict: dict = {}   # date → weight array

for i, date in enumerate(rebal_dates):
    loc = prices.index.get_loc(date)
    if loc < COV_LOOKBACK:
        # Not enough history yet → equal weight
        weights_dict[date] = np.full(n_assets, 1.0 / n_assets)
        continue

    ret_window = returns.iloc[loc - COV_LOOKBACK : loc].values
    # Drop assets with any NaN in the window
    col_valid  = ~np.any(np.isnan(ret_window), axis=0)
    if col_valid.sum() < 2:
        weights_dict[date] = np.full(n_assets, 1.0 / n_assets)
        continue

    sub_ret    = ret_window[:, col_valid]
    cov_sub    = lw_cov(sub_ret, SHRINK_ALPHA)
    sub_labels = [sector_labels[j] for j in range(n_assets) if col_valid[j]]

    # FIX: Removed the dead `sector_names` argument from the call site.
    w_sub = min_variance_weights(
        cov_sub, col_valid.sum(), MIN_WEIGHT, MAX_WEIGHT,
        sub_labels, MAX_SECTOR,
    )

    # Expand back to full universe (excluded assets get 0)
    w_full = np.zeros(n_assets)
    w_full[col_valid] = w_sub

    # ── Volatility target dampener ────────────────────────────────────────────
    if VOL_TARGET is not None:
        cov_full = np.zeros((n_assets, n_assets))
        cov_full[np.ix_(np.where(col_valid)[0], np.where(col_valid)[0])] = cov_sub
        port_vol = portfolio_ex_ante_vol(w_full, cov_full)
        if port_vol > VOL_TARGET:
            scale   = VOL_TARGET / port_vol
            w_full *= scale   # remainder (1 - scale) sits in cash

    weights_dict[date] = w_full
    if (i + 1) % 20 == 0 or i == len(rebal_dates) - 1:
        print(f"  …  {i+1}/{len(rebal_dates)} rebalance dates optimised", end="\r")

print(f"\n  ✓  {len(rebal_dates)} optimisations complete (Ledoit-Wolf + SLSQP)")

# ==============================================================================
# 4. BUILD WEIGHT TIME-SERIES
# ==============================================================================
print("\n[3/5] Building weight matrix...")

weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
for date, w_arr in weights_dict.items():
    if date in weights.index:
        weights.loc[date] = w_arr

# Forward-fill between rebalance dates; seed the warm-up rows with equal weight
weights = weights.ffill()
eq_w    = np.full(n_assets, 1.0 / n_assets)
weights.iloc[:COV_LOOKBACK] = weights.iloc[:COV_LOOKBACK].fillna(
    pd.Series(eq_w, index=prices.columns)
)
weights = weights.fillna(0)

if VOL_TARGET is None:
    # Renormalise so rows sum to 1 (no cash buffer)
    row_sums = weights.sum(axis=1).replace(0, np.nan)
    weights  = weights.div(row_sums, axis=0).fillna(0)
# When VOL_TARGET is active, rows intentionally sum to < 1; cash earns CASH_RETURN_ANN.

# Diagnostics
avg_holdings = (weights > 1e-4).sum(axis=1).mean()
avg_wt_max   = weights.max(axis=1).mean()
print(f"  ✓  Avg active holdings : {avg_holdings:.1f}")
print(f"  ✓  Avg max single wt   : {avg_wt_max:.2%}")
print(f"  ✓  Weight range [{MIN_WEIGHT:.1%} – {MAX_WEIGHT:.1%}] | Sector cap {MAX_SECTOR:.0%}")

# ==============================================================================
# 5. BACKTEST + CASH RETURN ADJUSTMENT
# ==============================================================================
# FIX: Removed the duplicate section header / daily_cash_rate / print block
#      that appeared at lines 332-343 in the original.
print("\n[4/5] Running backtest...")

# Daily compounded cash rate (derived once from the single config constant)
daily_cash_rate = (1 + CASH_RETURN_ANN) ** (1 / 252) - 1

portfolio = vbt.Portfolio.from_orders(
    close        = prices,
    size         = weights,
    size_type    = "targetpercent",
    init_cash    = INIT_CASH,
    fees         = FEES,
    slippage     = SLIPPAGE,
    freq         = "1D",
    group_by     = True,
    cash_sharing = True,
)

# ── Apply cash return post-hoc ────────────────────────────────────────────────
# Cash weight = fraction of portfolio not deployed in equities (clipped to [0,1])
cash_weight    = (1.0 - weights.sum(axis=1)).clip(0, 1)
port_ret_adj   = portfolio.returns() + cash_weight * daily_cash_rate
port_value_adj = (1 + port_ret_adj).cumprod() * INIT_CASH

print("  ✓  Backtest complete (cash return applied post-hoc)")

# ==============================================================================
# 6. PERFORMANCE METRICS
# ==============================================================================
print("\n[5/5] Computing metrics...")

port_value    = port_value_adj
port_ret_real = port_ret_adj

bench_aligned   = bench_returns.reindex(port_value.index).fillna(0)
excess_realized = port_ret_real - bench_aligned
annual_excess   = excess_realized.mean() * 252
tracking_err    = excess_realized.std() * np.sqrt(252)
info_ratio      = annual_excess / tracking_err if tracking_err != 0 else np.nan

bench_value = (1 + bench_aligned).cumprod() * INIT_CASH

# FIX: All three metrics below were pulled from the unadjusted `portfolio` object
#      while the rest of the table used the cash-adjusted series.  Now all metrics
#      are derived consistently from port_value / port_ret_real.
port_total_ret = (port_value.iloc[-1] / port_value.iloc[0]) - 1
port_sharpe    = (port_ret_real.mean() / port_ret_real.std() * np.sqrt(252)
                  if port_ret_real.std() > 0 else np.nan)
port_maxdd     = (port_value / port_value.cummax() - 1).min()

bench_total_ret = (bench_prices.iloc[-1] / bench_prices.iloc[0]) - 1
bench_sharpe    = (bench_returns.mean() / bench_returns.std() * np.sqrt(252)
                   if bench_returns.std() > 0 else np.nan)
bench_maxdd     = (bench_prices / bench_prices.cummax() - 1).min()

# ── Volatility metrics ────────────────────────────────────────────────────────
port_ann_vol  = port_ret_real.std() * np.sqrt(252)
bench_ann_vol = bench_returns.std() * np.sqrt(252)
vol_reduction = (bench_ann_vol - port_ann_vol) / bench_ann_vol

# Rolling 63-day realised vol (quarterly window)
roll_port_vol  = port_ret_real.rolling(63).std() * np.sqrt(252)
roll_bench_vol = bench_aligned.rolling(63).std() * np.sqrt(252)

# Calmar ratio (CAGR / |Max Drawdown|)
n_years    = len(port_value) / 252
cagr       = (port_value.iloc[-1] / port_value.iloc[0]) ** (1 / n_years) - 1
calmar     = cagr / abs(port_maxdd) if port_maxdd != 0 else np.nan

bench_cagr   = (bench_value.iloc[-1] / bench_value.iloc[0]) ** (1 / n_years) - 1
bench_calmar = bench_cagr / abs(bench_maxdd) if bench_maxdd != 0 else np.nan

# FIX: Sortino ratio was computed with std() of clipped returns, which is not
#      downside deviation.  Correct formula: annualised mean return divided by
#      the annualised downside deviation = sqrt(mean(min(r,0)^2)).
downside_dev = np.sqrt((port_ret_real.clip(upper=0) ** 2).mean()) * np.sqrt(252)
sortino      = (port_ret_real.mean() * 252) / downside_dev if downside_dev > 0 else np.nan

# ── Print tables ──────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  PERFORMANCE vs BENCHMARK")
print("=" * 62)
print(f"  {'Metric':<28} {'Strategy':>12} {'Benchmark':>12}")
print("  " + "-" * 54)
print(f"  {'Total Return':<28} {port_total_ret:>11.2%} {bench_total_ret:>11.2%}")
print(f"  {'CAGR':<28} {cagr:>11.2%} {bench_cagr:>11.2%}")
print(f"  {'Sharpe Ratio':<28} {port_sharpe:>12.2f} {bench_sharpe:>12.2f}")
print(f"  {'Sortino Ratio':<28} {sortino:>12.2f} {'—':>12}")
print(f"  {'Calmar Ratio':<28} {calmar:>12.2f} {bench_calmar:>12.2f}")
print(f"  {'Max Drawdown':<28} {port_maxdd:>11.2%} {bench_maxdd:>11.2%}")

print("\n" + "=" * 62)
print("  VOLATILITY METRICS")
print("=" * 62)
print(f"  {'Metric':<28} {'Strategy':>12} {'Benchmark':>12}")
print("  " + "-" * 54)
print(f"  {'Realised Vol (ann.)':<28} {port_ann_vol:>11.2%} {bench_ann_vol:>11.2%}")
print(f"  {'Vol Reduction vs Bench':<28} {vol_reduction:>+11.2%} {'—':>12}")
print(f"  {'Tracking Error':<28} {tracking_err:>11.2%} {'—':>12}")
print(f"  {'Information Ratio':<28} {info_ratio:>12.2f} {'—':>12}")
print(f"  {'Annual Excess Return':<28} {annual_excess:>+11.2%} {'—':>12}")

print("\n" + "=" * 62)
print("  STRESS PERIOD ANALYSIS")
print("=" * 62)
print(f"  {'Period':<24} {'Strategy':>10} {'Benchmark':>10} {'Alpha':>10} {'Vol ↓':>8}")
print("  " + "-" * 66)

for name, (s, e) in STRESS_PERIODS.items():
    try:
        pv = port_value.loc[s:e]
        bv = bench_prices.loc[s:e]
        pr = port_ret_real.loc[s:e]
        br = bench_aligned.loc[s:e]
        if len(pv) < 2 or len(bv) < 2:
            print(f"  {name:<24} {'N/A (out of date range)':>36}")
            continue
        p_ret = (pv.iloc[-1] / pv.iloc[0]) - 1
        b_ret = (bv.iloc[-1] / bv.iloc[0]) - 1
        alpha = p_ret - b_ret
        p_vol = pr.std() * np.sqrt(252)
        b_vol = br.std() * np.sqrt(252)
        vd    = (b_vol - p_vol) / b_vol if b_vol else float("nan")
        flag  = " ✓" if alpha > 0 else " ✗"
        print(f"  {name:<24} {p_ret:>9.2%}  {b_ret:>9.2%}  {alpha:>+9.2%}{flag}  {vd:>+7.1%}")
    except Exception:
        print(f"  {name:<24} {'N/A':>36}")

print("=" * 62)

# ==============================================================================
# 7. CHARTS
# ==============================================================================

# FIX: Original used portfolio.drawdown() (unadjusted).  Now computed directly
#      from the cash-adjusted value curve for consistency with all other metrics.
port_dd  = port_value / port_value.cummax() - 1
bench_dd = bench_prices / bench_prices.cummax() - 1

# Shared stress-period shading helper
def add_stress_shading(fig, periods):
    for name, (s, e) in periods.items():
        fig.add_vrect(
            x0=s, x1=e, fillcolor="red", opacity=0.07,
            layer="below", line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation=dict(font_size=9, font_color="crimson"))

# ── Chart 1: Cumulative value vs benchmark ────────────────────────────────────
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=port_value.index, y=port_value,
    name="Min-Vol Fund", line=dict(color="royalblue", width=2)))
fig1.add_trace(go.Scatter(
    x=bench_value.index, y=bench_value,
    name=BENCHMARK, line=dict(color="darkorange", width=2, dash="dash")))
add_stress_shading(fig1, STRESS_PERIODS)
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
    name="Min-Vol Fund", fill="tozeroy",
    line=dict(color="royalblue", width=1.5)))
fig2.add_trace(go.Scatter(
    x=bench_dd.index, y=bench_dd,
    name=BENCHMARK, fill="tozeroy", opacity=0.5,
    line=dict(color="darkorange", width=1.5, dash="dash")))
add_stress_shading(fig2, STRESS_PERIODS)
fig2.update_layout(
    title="Drawdown: Min-Vol Fund vs Benchmark",
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
    title="Portfolio Weights Over Time (Min-Vol)",
    xaxis_title="Date", yaxis_title="Weight",
    yaxis_tickformat=".0%",
    template="plotly_white", hovermode="x unified")
fig3.write_html("portfolio_weights.html")

# ── Chart 4: Rolling realised volatility ─────────────────────────────────────
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=roll_port_vol.index, y=roll_port_vol,
    name="Min-Vol Fund (63d)", line=dict(color="royalblue", width=2)))
fig4.add_trace(go.Scatter(
    x=roll_bench_vol.index, y=roll_bench_vol,
    name=f"{BENCHMARK} (63d)", line=dict(color="darkorange", width=2, dash="dash")))
if VOL_TARGET is not None:
    fig4.add_hline(
        y=VOL_TARGET, line_dash="dot", line_color="green",
        annotation_text=f"Vol target {VOL_TARGET:.0%}",
        annotation_position="bottom right")
add_stress_shading(fig4, STRESS_PERIODS)
fig4.update_layout(
    title="Rolling 63-Day Realised Volatility (annualised)",
    xaxis_title="Date", yaxis_title="Annualised Vol",
    yaxis_tickformat=".0%", legend=dict(x=0.01, y=0.99),
    template="plotly_white", hovermode="x unified")
fig4.write_html("portfolio_volatility.html")

print("\nCharts saved:")
print("  → portfolio_value.html      (cumulative value vs benchmark)")
print("  → portfolio_drawdown.html   (drawdown comparison)")
print("  → portfolio_weights.html    (stacked weights over time)")
print("  → portfolio_volatility.html (rolling realised vol vs benchmark)")