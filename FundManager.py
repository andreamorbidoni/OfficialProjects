""" streamlit run "/Users/andrea/Library/Mobile Documents/com~apple~CloudDocs/GitHub/OfficialProjects/FundManager.py" """
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from datetime import date, timedelta
import warnings

st.set_page_config(page_title="Portfolio Rebalancer", layout="wide")

# ==============================================================================
# HELPERS & MATH
# ==============================================================================
def extract_close(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Safely extracts closing prices from yfinance dataframe (handles single & multi-tickers)."""
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
    """Computes the Ledoit-Wolf shrinkage covariance matrix."""
    lw = LedoitWolf().fit(ret_window)
    cov = lw.covariance_
    if shrink_alpha > 0:
        cov += shrink_alpha * np.diag(np.diag(cov))
    return cov

def min_variance_weights(cov, n, min_w, max_w, sector_labels, max_sector, c_bench=None, te_penalty=0.0, target_equity=1.0):
    """Calculates the minimum variance portfolio weights with optional TE penalty and cash constraint."""
    # If the user wants 100% cash, bypass the optimizer and return 0 for all equities
    if target_equity <= 0.0:
        return np.zeros(n)
        
    w0 = np.full(n, target_equity / n)
    
    use_te = (te_penalty > 0.0) and (c_bench is not None)
    lam = te_penalty if use_te else 0.0
    cb = c_bench if use_te else np.zeros(n)
    
    # Combined Objective: (1+λ)w'Σw - 2λw'cb
    def objective(w): return (1.0 + lam) * (w @ cov @ w) - 2.0 * lam * (w @ cb)
    def objective_grad(w): return 2.0 * (1.0 + lam) * (cov @ w) - 2.0 * lam * cb

    # Constraint: sum of weights equals the target equity (1.0 - target cash)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - target_equity}]
    
    for sec in set(sector_labels):
        idx = [i for i, s in enumerate(sector_labels) if s == sec]
        if idx:
            constraints.append({"type": "ineq", "fun": lambda w, ix=idx: max_sector - w[ix].sum()})

    bounds = [(min_w, max_w)] * n
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = minimize(objective, w0, jac=objective_grad, method="SLSQP",
                          bounds=bounds, constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})

    if result.success:
        w = np.clip(result.x, 0, None)
        return w * (target_equity / w.sum()) if w.sum() > 0 else w0
        
    # Fallback to inverse volatility if SLSQP fails
    vols = np.sqrt(np.diag(cov))
    inv = np.where(vols > 0, 1.0 / vols, 0.0)
    w_fb = inv / inv.sum() if inv.sum() > 0 else w0
    w_fb = np.clip(w_fb, min_w, max_w)
    return w_fb * (target_equity / w_fb.sum()) if w_fb.sum() > 0 else w0

def portfolio_ex_ante_vol(w, cov):
    """Calculates annualized ex-ante volatility."""
    return float(np.sqrt(252 * w @ cov @ w))

def compute_rebalance_trades(starting_weights, target_weights, prices, portfolio_value, sector_map, starting_cash_pct, target_cash_pct):
    """Calculates trades including cash reserves."""
    last_prices = prices.iloc[-1]
    rows = []
    
    # Process Equities (Looping over target weights ensures we don't miss valid optimization outputs)
    for ticker in target_weights.index:
        cw = starting_weights.get(ticker, 0.0)  
        if pd.isna(cw): cw = 0.0 # Failsafe for missing data/NaNs
        
        tw = target_weights.get(ticker, 0.0)    
        if pd.isna(tw): tw = 0.0
        
        price = last_prices.get(ticker, np.nan)
        if np.isnan(price): continue
        
        current_shares = (cw * portfolio_value) / price if price > 0 else 0
        target_shares = (tw * portfolio_value) / price if price > 0 else 0
        delta_shares = target_shares - current_shares
        
        rows.append({
            "Ticker": ticker, 
            "Sector": sector_map.get(ticker, "Other"),
            "Last Price": round(price, 2), 
            "Starting Weight": f"{cw:.2%}",
            "Target Weight": f"{tw:.2%}",
            "Current Shares": round(current_shares, 2),
            "Target Shares": round(target_shares, 2),
            "Δ Shares": round(delta_shares, 2),
            "Action": "BUY" if delta_shares > 0.5 else ("SELL" if delta_shares < -0.5 else "HOLD")
        })
        
    # Process Cash
    start_cash_val = starting_cash_pct * portfolio_value
    target_cash_val = target_cash_pct * portfolio_value
    delta_cash = target_cash_val - start_cash_val
    
    rows.append({
        "Ticker": "CASH",
        "Sector": "Reserve",
        "Last Price": 1.00,
        "Starting Weight": f"{starting_cash_pct:.2%}",
        "Target Weight": f"{target_cash_pct:.2%}",
        "Current Shares": round(start_cash_val, 2),
        "Target Shares": round(target_cash_val, 2),
        "Δ Shares": round(delta_cash, 2),
        "Action": "DEPOSIT" if delta_cash > 0.5 else ("WITHDRAW" if delta_cash < -0.5 else "HOLD")
    })
    
    return pd.DataFrame(rows).sort_values("Target Weight", ascending=False)

# ==============================================================================
# STREAMLIT UI
# ==============================================================================
st.title("📊 Min-Variance Portfolio Rebalancer")
st.caption("Upload your current portfolio weights. The optimizer will calculate minimum variance target weights and output your required trades.")

with st.sidebar:
    st.header("⚙️ Inputs")
    target_csv = st.file_uploader("Upload Portfolio CSV (Ticker, Weight, Sector)", type=["csv"])
    
    st.divider()
    st.header("💰 Investment Size")
    port_value = st.number_input("Total Portfolio Value ($)", min_value=1000.0, value=100000.0, step=1000.0)
    starting_cash = st.slider("Starting Cash Allocation (%)", 0.0, 100.0, 0.0, 1.0, help="Percentage of your total portfolio currently sitting in cash.")
    
    st.divider()
    st.header("📈 Optimization Settings")
    target_cash = st.slider("Target Cash Allocation (%)", 0.0, 100.0, 0.0, 1.0, help="The optimizer will leave this percentage uninvested.")
    cov_lookback = st.number_input("Covariance Lookback (Trading Days)", min_value=30, max_value=1260, value=252, step=10)
    min_weight = st.slider("Min Weight per Asset", 0.0, 0.10, 0.01, 0.01)
    max_weight = st.slider("Max Weight per Asset", 0.05, 0.50, 0.15, 0.01)
    max_sector = st.slider("Max Weight per Sector", 0.10, 1.0, 0.25, 0.05)
    
    st.divider()
    st.header("🎯 Tracking Error Penalty")
    benchmark_ticker = st.text_input("Benchmark Ticker", "^STOXX")
    te_penalty = st.number_input("TE Penalty (λ)", min_value=0.0, max_value=50.0, value=0.0, step=1.0, help="Higher values pull the portfolio closer to the benchmark. 0 disables the penalty.")
    
    st.divider()
    run_btn = st.button("🚀 Run Optimizer & Rebalance", type="primary", use_container_width=True)

if run_btn:
    if not target_csv:
        st.error("❌ Please upload the Portfolio CSV to proceed.")
    else:
        st.divider()
        with st.spinner("Fetching market data & optimizing portfolio..."):
            try:
                # 1. Parse Current Portfolio
                df_port = pd.read_csv(target_csv)
                
                required_cols = {"Ticker", "Weight", "Sector"}
                if not required_cols.issubset(df_port.columns):
                    st.error(f"❌ CSV must contain exact columns: {', '.join(required_cols)}")
                    st.stop()
                    
                df_port["Ticker"] = df_port["Ticker"].astype(str).str.strip()
                df_port["Sector"] = df_port["Sector"].astype(str).str.strip()
                
                # FIX: Safely normalize weights to avoid Division by Zero if all inputs are 0
                starting_weights_raw = df_port.set_index("Ticker")["Weight"].astype(float)
                raw_sum = starting_weights_raw.sum()
                
                if raw_sum > 0:
                    starting_weights_raw = starting_weights_raw / raw_sum
                else:
                    starting_weights_raw = starting_weights_raw * 0.0
                    
                starting_weights = starting_weights_raw * (1.0 - (starting_cash / 100.0))
                
                sector_map = dict(zip(df_port["Ticker"], df_port["Sector"]))
                tickers = starting_weights.index.tolist()

                # 2. Fetch Historical Prices
                today = date.today()
                start = today - timedelta(days=int(cov_lookback * 1.6)) 
                
                raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"),
                                  auto_adjust=True, progress=False)
                prices = extract_close(raw, tickers).ffill().dropna()
                
                valid_tickers = [t for t in prices.columns if t in tickers]
                prices = prices[valid_tickers]
                starting_weights = starting_weights[valid_tickers]

                # 3. Calculate Covariance Matrix
                returns = np.log(prices / prices.shift(1)).iloc[-cov_lookback:].values
                col_valid = ~np.any(np.isnan(returns), axis=0)
                
                if not any(col_valid):
                    st.error("❌ Not enough historical data to calculate covariance for these assets.")
                    st.stop()

                cov = lw_cov(returns[:, col_valid])
                valid_cols = prices.columns[col_valid]
                sector_labels = [sector_map.get(t, "Other") for t in valid_cols]

                # 4. Handle Benchmark & Tracking Error
                c_bench = None
                if te_penalty > 0.0 and benchmark_ticker:
                    raw_bench = yf.download(benchmark_ticker, start=start.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"), 
                                            auto_adjust=True, progress=False)
                    bench_prices = extract_close(raw_bench, [benchmark_ticker])
                    
                    if not bench_prices.empty:
                        bench_prices = bench_prices.squeeze().reindex(prices.index).ffill()
                        bench_returns = np.log(bench_prices / bench_prices.shift(1)).iloc[-cov_lookback:].values
                        
                        valid_bench = ~np.isnan(bench_returns)
                        if valid_bench.sum() > 10:
                            b = bench_returns[valid_bench]
                            a = returns[:, col_valid][valid_bench]
                            b_dm = b - b.mean()
                            a_dm = a - a.mean(axis=0)
                            c_bench = (a_dm * b_dm[:, None]).mean(axis=0)
                    else:
                        st.warning(f"⚠️ Could not download Benchmark '{benchmark_ticker}'. TE Penalty ignored.")

                # 5. Run Optimizer for TARGET Weights
                target_equity = 1.0 - (target_cash / 100.0)
                w_arr = min_variance_weights(cov, len(valid_cols), min_weight, max_weight, sector_labels, max_sector, 
                                             c_bench, te_penalty, target_equity)
                target_weights = pd.Series(w_arr, index=valid_cols)
                
                # Align starting weights with valid columns
                starting_weights = starting_weights.loc[valid_cols]
                
                # Calculate new volatility
                port_vol = portfolio_ex_ante_vol(w_arr, cov)

                # --- 6. Calculate Ex-Ante Statistics ---
                valid_returns = returns[:, col_valid]
                port_hist_returns = valid_returns @ w_arr
                
                p_exp_ret = port_hist_returns.mean() * 252
                p_var_95 = np.percentile(port_hist_returns, 5)
                p_cvar_95 = port_hist_returns[port_hist_returns <= p_var_95].mean()
                
                port_hist_series = pd.Series(port_hist_returns)
                p_skew = port_hist_series.skew()
                p_kurt = port_hist_series.kurt()
                
                p_sharpe = p_exp_ret / port_vol if port_vol > 0 else np.nan

                st.success("✅ Optimization complete")
                
                # 7. Display Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Optimized Ex-Ante Volatility", f"{port_vol:.2%}")
                c2.metric("Assets Analyzed", len(target_weights))
                c3.metric("Optimized Target Cash", f"{(target_cash / 100.0):.2%}")

                st.divider()

                # --- Display Advanced Stats ---
                st.subheader("📈 Ex-Ante Portfolio Statistics")
                st.caption(f"Estimated forward-looking performance based on the last {int(cov_lookback)} trading days.")
                
                s1, s2, s3, s4, s5, s6 = st.columns(6)
                s1.metric("Expected Return (Ann.)", f"{p_exp_ret:.2%}")
                s2.metric("Sharpe Ratio", f"{p_sharpe:.2f}")
                s3.metric("Daily VaR (95%)", f"{p_var_95:.2%}")
                s4.metric("Daily CVaR (95%)", f"{p_cvar_95:.2%}")
                s5.metric("Skewness", f"{p_skew:.2f}")
                s6.metric("Kurtosis", f"{p_kurt:.2f}")

                st.divider()

                # 8. Process Trades
                st.subheader("📦 Rebalance Trade Sheet")
                trades = compute_rebalance_trades(starting_weights, target_weights, prices, port_value, sector_map, 
                                                  starting_cash / 100.0, target_cash / 100.0)
                
                # Custom styling to highlight BUY/SELL actions
                def color_action(val):
                    color = '#4CAF50' if val in ['BUY', 'DEPOSIT'] else '#F44336' if val in ['SELL', 'WITHDRAW'] else 'gray'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(trades.style.map(color_action, subset=['Action']), use_container_width=True)
                
                csv_dl = trades.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Trades CSV", csv_dl, f"rebalance_trades_{today}.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")