import torch
import yfinance as yf
import pandas as pd
import numpy as np
from chronos import BaseChronosPipeline
from datetime import datetime, timedelta

# 1. Configuration
TICKER = "AAPL"
CONTEXT_LENGTH = 30
PREDICTION_LENGTH = 5
STEP_SIZE = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Starting backtest for {TICKER}...")
print(f"Using device: {DEVICE}")

# 2. Data Acquisition
# Fetch 6 months of daily data
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
df = yf.download(TICKER, start=start_date, end=end_date, interval="1d", progress=False)

# Clean up yfinance MultiIndex if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

prices = df['Close'].values
dates = df.index.values

print(f"Downloaded {len(prices)} days of price data.")

# 3. Initialize Pipeline
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map=DEVICE,
    dtype=torch.bfloat16,
)

# 4. Backtest Loop
results = []
total_mae = []
directional_hits = 0
total_windows = 0

# We need at least CONTEXT_LENGTH + PREDICTION_LENGTH data points
for i in range(0, len(prices) - CONTEXT_LENGTH - PREDICTION_LENGTH + 1, STEP_SIZE):
    # Slice context and target
    context_prices = prices[i : i + CONTEXT_LENGTH]
    actual_prices = prices[i + CONTEXT_LENGTH : i + CONTEXT_LENGTH + PREDICTION_LENGTH]
    forecast_dates = dates[i + CONTEXT_LENGTH : i + CONTEXT_LENGTH + PREDICTION_LENGTH]
    
    # Predict
    context_tensor = torch.tensor(context_prices, device=DEVICE, dtype=torch.float32)
    forecast = pipeline.predict(context_tensor, PREDICTION_LENGTH)
    
    # Extract median forecast
    samples = forecast[0].numpy()
    median_forecast = np.median(samples, axis=0)
    
    # Calculate Metrics for this window
    mae = np.mean(np.abs(actual_prices - median_forecast))
    total_mae.append(mae)
    
    # Directional Accuracy: Did the 5-day move direction match?
    # Actual direction: Last price of context vs Last price of target
    actual_dir = 1 if actual_prices[-1] > context_prices[-1] else -1
    pred_dir = 1 if median_forecast[-1] > context_prices[-1] else -1
    
    if actual_dir == pred_dir:
        directional_hits += 1
    
    total_windows += 1
    
    # Save first and last date of forecast window for display
    window_start = pd.to_datetime(forecast_dates[0]).strftime('%Y-%m-%d')
    
    results.append({
        "Date": window_start,
        "Actual_End": actual_prices[-1],
        "Pred_End": median_forecast[-1],
        "MAE": mae
    })

# 5. Output Results
print("\n" + "="*60)
print(f"{'Start Date':<12} | {'Actual End':<10} | {'Pred End':<10} | {'MAE':<8}")
print("-" * 60)
for r in results:
    print(f"{r['Date']:<12} | {r['Actual_End']:<10.2f} | {r['Pred_End']:<10.2f} | {r['MAE']:<8.2f}")

avg_mae = np.mean(total_mae)
dir_acc = (directional_hits / total_windows) * 100

print("="*60)
print(f"BACKTEST SUMMARY ({TICKER})")
print(f"Total Windows:      {total_windows}")
print(f"Average MAE:        {avg_mae:.4f}")
print(f"Directional Acc:    {dir_acc:.2f}%")
print("="*60)
