""" streamlit run "/Users/andrea/Library/Mobile Documents/com~apple~CloudDocs/GitHub/OfficialProjects/MinVarAPP.py" """
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import date

st.set_page_config(page_title="Min-Vol Backtester", layout="wide")

# ==============================================================================
# DEFAULTS & CONFIGURATION
# ==============================================================================
DEFAULT_TICKERS = [
    "AM.PA", "KOG.OL", "DTE.DE", "ANDR.VI", "VALMT.HE", "RIO.L", "GJF.OL",
    "NTGY.MC", "METSO.HE", "LOGN.SW", "RXL.PA", "ADM.L", "GMAB.CO",
    "ELISA.HE", "GET.PA", "SAMPO.HE", "ASML.AS", "SBMO.AS", "IMI.L",
    "SAND.ST", "KBX.DE"
]

DEFAULT_SECTOR_MAP = {
    "AM.PA": "Industrials", "KOG.OL": "Industrials", "DTE.DE": "Communication Services",
    "ANDR.VI": "Industrials", "VALMT.HE": "Industrials", "RIO.L": "Basic Materials",
    "GJF.OL": "Financial Services", "NTGY.MC": "Utilities", "METSO.HE": "Industrials",
    "LOGN.SW": "Technology", "RXL.PA": "Industrials", "ADM.L": "Financial Services",
    "GMAB.CO": "Healthcare", "ELISA.HE": "Communication Services", "GET.PA": "Industrials",
    "SAMPO.HE": "Financial Services", "ASML.AS": "Technology", "SBMO.AS": "Energy",
    "IMI.L": "Industrials", "SAND.ST": "Industrials", "KBX.DE": "Industrials"
}

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
    """Robustly extract Close prices from yfinance output."""
    if raw.empty: return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.xs("Close", axis=1, level=0)
    else:
        if "Close" in raw.columns:
            df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            df = raw
    available = [t for t in tickers if t in df.columns]
    skipped = set(tickers) - set(available)
    if skipped: st.warning(f"No data for: {sorted(skipped)}")
    return df[available] if available else pd.DataFrame()

def lw_cov(ret_window: np.ndarray, shrink_alpha: float = 0.0) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance."""
    lw = LedoitWolf().fit(ret_window)
    cov = lw.covariance_
    if shrink_alpha > 0:
        cov += shrink_alpha * np.diag(np.diag(cov))
    return cov

def min_variance_weights(cov: np.ndarray, n: int, min_w: float, max_w: float,
                         sector_labels: list, max_sector: float, c_bench: np.ndarray = None,
                         te_penalty: float = 0.0) -> np.ndarray:
    """Solve the penalised minimum-variance problem."""
    w0 = np.full(n, 1.0 / n)
    use_te = (te_penalty > 0.0) and (c_bench is not None)
    lam = te_penalty if use_te else 0.0
    cb = c_bench if use_te else np.zeros(n)

    def objective(w): return (1.0 + lam) * (w @ cov @ w) - 2.0 * lam * (w @ cb)
    def objective_grad(w): return 2.0 * (1.0 + lam) * (cov @ w) - 2.0 * lam * cb

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    for sec in set(sector_labels):
        idx = [i for i, s in enumerate(sector_labels) if s == sec]
        if idx: constraints.append({"type": "ineq", "fun": lambda w, ix=idx: max_sector - w[ix].sum()})

    bounds = [(min_w, max_w)] * n

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = minimize(objective, w0, jac=objective_grad, method="SLSQP",
                          bounds=bounds, constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})

    if result.success:
        w = np.clip(result.x, 0, None)
        return w / w.sum() if w.sum() > 0 else w0

    vols = np.sqrt(np.diag(cov))
    inv = np.where(vols > 0, 1.0 / vols, 0.0)
    w_fallback = np.clip(inv / inv.sum() if inv.sum() > 0 else w0, min_w, max_w)
    return w_fallback / w_fallback.sum()

def portfolio_ex_ante_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualised ex-ante portfolio volatility."""
    return float(np.sqrt(252 * weights @ cov @ weights))

def add_stress_shading(fig, periods):
    for name, (s, e) in periods.items():
        fig.add_vrect(
            x0=s, x1=e, fillcolor="red", opacity=0.07, layer="below", line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation=dict(font_size=9, font_color="crimson")
        )

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("📈 Min-Variance VectorBT Backtester")
st.caption("Historical simulator for portfolio variance optimization, featuring Ledoit-Wolf shrinkage, tracking error penalties, and volatility targeting.")

with st.sidebar:
    st.header("⚙️ Universe Definition")
    target_csv = st.file_uploader("Upload Portfolio CSV (Ticker, Weight, Sector)", type=["csv"], help="Use the same CSV as your Fund Manager. The Weight column will be ignored during the backtest as historical weights are calculated dynamically.")
    benchmark = st.text_input("Benchmark Ticker", "^STOXX")

    st.divider()
    st.header("📅 Timeline")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-03-02"))

    st.divider()
    st.header("📊 Rebalancing & Optimization")
    cov_lookback = st.number_input("Covariance Lookback (Days)", 30, 1260, 252, 10)
    rebal_freq = st.number_input("Rebalance Frequency (Days)", 1, 252, 21, 1)
    min_weight = st.slider("Min Weight per Asset", 0.0, 0.10, 0.01, 0.01)
    max_weight = st.slider("Max Weight per Asset", 0.05, 0.50, 0.15, 0.01)
    max_sector = st.slider("Max Weight per Sector", 0.10, 1.0, 0.25, 0.05)
    shrink_alpha = st.slider("Shrinkage Alpha", 0.0, 0.1, 0.0, 0.01,
                             help="Extra diagonal regularisation on top of Ledoit-Wolf. Keep at 0 unless you have a specific reason to double-shrink.")

    st.divider()
    st.header("🎯 Advanced Risk Targets")
    use_vol_target = st.checkbox("Enable Volatility Target", value=True)
    vol_target = st.slider("Max Volatility Target (%)", 1.0, 30.0, 10.0, 1.0) / 100.0 if use_vol_target else None
    te_penalty = st.number_input("TE Penalty (λ)", 0.0, 50.0, 10.0, 1.0, help="Higher values track the benchmark tighter.")

    st.divider()
    st.header("💸 Execution & Fees")
    init_cash = st.number_input("Initial Cash (€)", value=100000)
    fees = st.number_input("Trading Fees (%)", value=0.15, step=0.01) / 100.0
    slippage = st.number_input("Slippage (%)", value=0.30, step=0.01) / 100.0
    cash_return = st.number_input("Cash Return (Annual %)", value=3.5, step=0.1) / 100.0

    st.divider()
    run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)


if run_btn:
    st.divider()

    # 1. Load Portfolio
    if target_csv:
        df_csv = pd.read_csv(target_csv)

        required_cols = {"Ticker", "Weight", "Sector"}
        if not required_cols.issubset(df_csv.columns):
            st.error(f"❌ CSV must contain exact columns: {', '.join(required_cols)}")
            st.stop()

        df_csv["Ticker"] = df_csv["Ticker"].astype(str).str.strip()
        df_csv["Sector"] = df_csv["Sector"].astype(str).str.strip()

        tickers = df_csv["Ticker"].tolist()
        sector_map = dict(zip(df_csv["Ticker"], df_csv["Sector"]))
    else:
        tickers = DEFAULT_TICKERS
        sector_map = DEFAULT_SECTOR_MAP

    with st.spinner("Downloading Market Data..."):
        # 2. Download Data
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        raw = yf.download(tickers, start=start_str, end=end_str, auto_adjust=True, progress=False)
        prices = extract_close(raw, tickers)

        min_bars = cov_lookback + 10
        valid = [t for t in prices.columns if prices[t].dropna().shape[0] >= min_bars]
        prices = prices[valid].ffill().dropna()
        n_assets = len(prices.columns)

        if prices.empty:
            st.error("❌ No valid price data found. Check tickers and date range.")
            st.stop()

        if max_weight * n_assets < 1.0:
            st.error(f"❌ Constraints infeasible: Max Weight ({max_weight:.0%}) × {n_assets} assets = {max_weight * n_assets:.0%} < 100%. Increase max weight.")
            st.stop()

        bench_raw = yf.download(benchmark, start=start_str, end=end_str, auto_adjust=True, progress=False)
        bench_prices = extract_close(bench_raw, [benchmark]).squeeze().reindex(prices.index).ffill()

        # FIX: Use arithmetic (pct_change) returns for the benchmark so that
        # (1 + bench_aligned).cumprod() compounds correctly.
        # Log returns used here previously caused slight understatement of benchmark
        # performance, making the strategy look better than it actually was vs the index.
        bench_returns = bench_prices.pct_change()

        # Keep log returns only for the covariance matrix (they are more appropriate there)
        returns = np.log(prices / prices.shift(1))

    with st.spinner("Running Rolling Optimization..."):
        # 3. Covariance & Optimization
        rebal_dates = prices.index[::int(rebal_freq)]
        sector_labels = [sector_map.get(t, "Other") for t in prices.columns]
        weights_dict = {}

        progress_bar = st.progress(0)

        # Renamed loop variable from `date` to `rebal_date` to avoid shadowing
        # the `date` class imported from the datetime module.
        for i, rebal_date in enumerate(rebal_dates):
            loc = prices.index.get_loc(rebal_date)
            if loc < cov_lookback:
                weights_dict[rebal_date] = np.full(n_assets, 1.0 / n_assets)
                continue

            ret_window = returns.iloc[loc - cov_lookback : loc].values
            col_valid = ~np.any(np.isnan(ret_window), axis=0)

            if col_valid.sum() < 2:
                weights_dict[rebal_date] = np.full(n_assets, 1.0 / n_assets)
                continue

            sub_ret = ret_window[:, col_valid]
            cov_sub = lw_cov(sub_ret, shrink_alpha)
            sub_labels = [sector_labels[j] for j in range(n_assets) if col_valid[j]]

            c_bench_sub = None
            if te_penalty > 0.0:
                bench_window = bench_returns.reindex(returns.index).iloc[loc - cov_lookback : loc].values
                valid_rows = ~np.isnan(bench_window)
                if valid_rows.sum() > 10:
                    b = bench_window[valid_rows]
                    a = sub_ret[valid_rows]
                    c_bench_sub = ((a - a.mean(axis=0)) * (b - b.mean())[:, None]).mean(axis=0)

            w_sub = min_variance_weights(cov_sub, col_valid.sum(), min_weight, max_weight,
                                         sub_labels, max_sector, c_bench_sub, te_penalty)

            w_full = np.zeros(n_assets)
            w_full[col_valid] = w_sub

            if vol_target is not None:
                cov_full = np.zeros((n_assets, n_assets))
                cov_full[np.ix_(np.where(col_valid)[0], np.where(col_valid)[0])] = cov_sub
                port_vol = portfolio_ex_ante_vol(w_full, cov_full)
                if port_vol > vol_target:
                    w_full *= (vol_target / port_vol)

            weights_dict[rebal_date] = w_full
            progress_bar.progress((i + 1) / len(rebal_dates))

        progress_bar.empty()

        # 4. Build Weight Matrix
        weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        for d, w_arr in weights_dict.items():
            if d in weights.index: weights.loc[d] = w_arr

        weights = weights.ffill()
        weights.iloc[:int(cov_lookback)] = weights.iloc[:int(cov_lookback)].fillna(
            pd.Series(np.full(n_assets, 1.0 / n_assets), index=prices.columns)
        )
        weights = weights.fillna(0)

        if vol_target is None:
            weights = weights.div(weights.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    with st.spinner("Executing VectorBT Simulation..."):
        # 5. Backtest & Cash Return
        daily_cash_rate = (1 + cash_return) ** (1 / 252) - 1
        portfolio = vbt.Portfolio.from_orders(
            close=prices, size=weights, size_type="targetpercent",
            init_cash=init_cash, fees=fees, slippage=slippage, freq="1D", group_by=True, cash_sharing=True
        )

        # portfolio.returns() is arithmetic — consistent with bench_returns (pct_change)
        cash_weight = (1.0 - weights.sum(axis=1)).clip(0, 1)
        port_ret_real = portfolio.returns() + cash_weight * daily_cash_rate
        port_value = (1 + port_ret_real).cumprod() * init_cash

        bench_aligned = bench_returns.reindex(port_value.index).fillna(0)
        bench_value = (1 + bench_aligned).cumprod() * init_cash

        # 6. Metrics Calculation
        excess_realized = port_ret_real - bench_aligned
        annual_excess = excess_realized.mean() * 252
        tracking_err = excess_realized.std() * np.sqrt(252)
        info_ratio = annual_excess / tracking_err if tracking_err != 0 else np.nan

        n_years = len(port_value) / 252

        def calc_cagr(val): return (val.iloc[-1] / val.iloc[0]) ** (1 / n_years) - 1
        def calc_maxdd(val): return (val / val.cummax() - 1).min()
        def calc_sharpe(ret): return (ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan)

        p_cagr, b_cagr = calc_cagr(port_value), calc_cagr(bench_value)
        p_dd, b_dd = calc_maxdd(port_value), calc_maxdd(bench_value)
        p_sharpe, b_sharpe = calc_sharpe(port_ret_real), calc_sharpe(bench_aligned)

        p_vol = port_ret_real.std() * np.sqrt(252)
        b_vol = bench_aligned.std() * np.sqrt(252)

        downside_dev = np.sqrt((port_ret_real.clip(upper=0) ** 2).mean()) * np.sqrt(252)
        sortino = (port_ret_real.mean() * 252) / downside_dev if downside_dev > 0 else np.nan
        calmar = p_cagr / abs(p_dd) if p_dd != 0 else np.nan

    # ==============================================================================
    # OUTPUT RENDER
    # ==============================================================================
    st.success("✅ Backtest Simulation Complete")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Strategy CAGR", f"{p_cagr:.2%}", delta=f"{(p_cagr - b_cagr):.2%} vs Bench")
    col2.metric("Strategy Volatility", f"{p_vol:.2%}", delta=f"{(p_vol - b_vol):.2%} vs Bench", delta_color="inverse")
    col3.metric("Max Drawdown", f"{p_dd:.2%}", delta=f"{(p_dd - b_dd):.2%} vs Bench")
    col4.metric("Sharpe Ratio", f"{p_sharpe:.2f}", delta=f"{(p_sharpe - b_sharpe):.2f} vs Bench")

    st.divider()

    # Tables
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Volatility Metrics", "Stress Periods"])

    with tab1:
        perf_df = pd.DataFrame({
            "Metric": ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown"],
            "Strategy": [
                f"{((port_value.iloc[-1] / port_value.iloc[0]) - 1):.2%}",
                f"{p_cagr:.2%}", f"{p_sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}", f"{p_dd:.2%}"
            ],
            "Benchmark": [
                f"{((bench_value.iloc[-1] / bench_value.iloc[0]) - 1):.2%}",
                f"{b_cagr:.2%}", f"{b_sharpe:.2f}", "—",
                f"{(b_cagr / abs(b_dd) if b_dd != 0 else np.nan):.2f}", f"{b_dd:.2%}"
            ]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    with tab2:
        vol_df = pd.DataFrame({
            "Metric": ["Realised Vol (ann.)", "Vol Reduction vs Bench", "Tracking Error", "Information Ratio", "Annual Excess Return"],
            "Strategy": [
                f"{p_vol:.2%}", f"{((b_vol - p_vol) / b_vol):+11.2%}",
                f"{tracking_err:.2%}", f"{info_ratio:.2f}", f"{annual_excess:+11.2%}"
            ],
            "Benchmark": [f"{b_vol:.2%}", "—", "—", "—", "—"]
        })
        st.dataframe(vol_df, use_container_width=True, hide_index=True)

    with tab3:
        stress_rows = []
        for name, (s, e) in STRESS_PERIODS.items():
            try:
                pv = port_value.loc[s:e]
                bv = bench_prices.loc[s:e]
                pr = port_ret_real.loc[s:e]
                br = bench_aligned.loc[s:e]
                if len(pv) > 1 and len(bv) > 1:
                    p_ret = (pv.iloc[-1] / pv.iloc[0]) - 1
                    b_ret = (bv.iloc[-1] / bv.iloc[0]) - 1
                    b_vol_stress = br.std() * np.sqrt(252)
                    p_vol_stress = pr.std() * np.sqrt(252)
                    stress_rows.append({
                        "Period": name,
                        "Strategy Return": f"{p_ret:.2%}",
                        "Benchmark Return": f"{b_ret:.2%}",
                        "Alpha": f"{(p_ret - b_ret):+.2%}",
                        "Vol Reduction": f"{((b_vol_stress - p_vol_stress) / b_vol_stress):+.1%}"
                    })
            except Exception as ex:
                st.warning(f"Could not compute stress period '{name}': {ex}")
        if stress_rows:
            st.dataframe(pd.DataFrame(stress_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Stress periods fall outside the selected date range.")

    st.divider()

    # Charts
    st.subheader("Cumulative Value vs Benchmark")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=port_value.index, y=port_value, name="Min-Vol Fund", line=dict(color="royalblue", width=2)))
    fig1.add_trace(go.Scatter(x=bench_value.index, y=bench_value, name=benchmark, line=dict(color="darkorange", width=2, dash="dash")))
    add_stress_shading(fig1, STRESS_PERIODS)
    fig1.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown Comparison")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=port_value.index, y=(port_value / port_value.cummax() - 1),
                                  name="Min-Vol", fill="tozeroy", line=dict(color="royalblue", width=1.5)))
        fig2.add_trace(go.Scatter(x=bench_value.index, y=(bench_value / bench_value.cummax() - 1),
                                  name=benchmark, fill="tozeroy", opacity=0.5, line=dict(color="darkorange", width=1.5, dash="dash")))
        fig2.update_layout(template="plotly_white", yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("Rolling 63-Day Realised Volatility")
        roll_p = port_ret_real.rolling(63).std() * np.sqrt(252)
        roll_b = bench_aligned.rolling(63).std() * np.sqrt(252)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=roll_p.index, y=roll_p, name="Min-Vol", line=dict(color="royalblue", width=2)))
        fig4.add_trace(go.Scatter(x=roll_b.index, y=roll_b, name=benchmark, line=dict(color="darkorange", width=2, dash="dash")))
        if vol_target:
            fig4.add_hline(y=vol_target, line_dash="dot", line_color="green", annotation_text=f"Target {vol_target:.0%}")
        fig4.update_layout(template="plotly_white", yaxis_tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Portfolio Weights Over Time")
    fig3 = go.Figure()
    for tick in prices.columns:
        fig3.add_trace(go.Scatter(x=weights.index, y=weights[tick], name=tick, stackgroup="one", mode="none"))
    fig3.update_layout(template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)