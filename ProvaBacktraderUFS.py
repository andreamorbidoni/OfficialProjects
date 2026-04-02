import backtrader as bt
import yfinance as yf
import pandas as pd

# --- 1. READ THE CSV ---
try:
    # Read the CSV file into pandas
    portfolio_df = pd.read_csv('HoldingsProvaBacktrader.csv')
    
    # Convert the columns into a Python dictionary: {'AAPL': 0.15, 'MSFT': 0.20...}
    target_weights = dict(zip(portfolio_df['Ticker'], portfolio_df['Weight']))
    
    # Extract just the tickers to feed to yfinance
    fund_tickers = list(target_weights.keys())
    print(f"Successfully loaded {len(fund_tickers)} tickers from CSV.")

except FileNotFoundError:
    print("CRITICAL ERROR: 'portfolio.csv' not found.")
    print("Please ensure the file exists in the same folder as this script.")
    exit() # Stops the script from crashing further down


# --- 2. DEFINE THE STRATEGY ---
class FundStrategy(bt.Strategy):
    # This allows us to pass the weights dictionary from the CSV into the rules
    params = (
        ('weights', {}), 
    )

    def __init__(self):
        self.moving_averages = {}
        for stock_data in self.datas:
            self.moving_averages[stock_data._name] = bt.indicators.SimpleMovingAverage(
                stock_data.close, period=200 
            )

    def next(self):
        for stock_data in self.datas:
            ticker = stock_data._name
            price_today = stock_data.close[0]
            moving_avg = self.moving_averages[ticker][0]
            
            # Fetch the specific target weight for this ticker from the CSV data
            weight = self.params.weights.get(ticker, 0)

            # BUY RULE
            if not self.getposition(stock_data).size: 
                if price_today > moving_avg:
                    # UPGRADE: Buy exactly enough shares to hit the target weight percentage
                    self.order_target_percent(data=stock_data, target=weight)
                    print(f"[{self.data.datetime.date(0)}] BUY: Allocating {weight*100}% of fund to {ticker}")
            
            # SELL RULE
            elif price_today < moving_avg:
                # Only try to sell if we currently own shares
                if self.getposition(stock_data).size > 0: 
                    # Set target to 0.0 to sell all shares of this stock
                    self.order_target_percent(data=stock_data, target=0.0)
                    print(f"[{self.data.datetime.date(0)}] SELL: Liquidating position in {ticker}")


# --- 3. INITIALIZE THE BRAIN ---
cerebro = bt.Cerebro()
cerebro.broker.setcash(100000.0)

# (Notice we removed the global PercentSizer here because the strategy now handles exact sizing)

# --- 4. GET THE DATA ---
for ticker in fund_tickers:
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
    
    if df.empty:
        print(f"Warning: No data downloaded for {ticker}. Skipping.")
        continue

    data_feed = bt.feeds.PandasData(dataname=df, name=ticker)
    cerebro.adddata(data_feed)

# Add the strategy, and pass in our dictionary of weights from the CSV
cerebro.addstrategy(FundStrategy, weights=target_weights)

# --- 5. RUN ---
print(f"\nStarting Fund Value: ${cerebro.broker.getvalue():,.2f}")
cerebro.run()
print(f"Ending Fund Value: ${cerebro.broker.getvalue():,.2f}")