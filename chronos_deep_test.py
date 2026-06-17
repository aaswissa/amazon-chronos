import torch
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
from datetime import datetime, timedelta

# 1. Configuration
TICKER = "BTC-USD"
CONTEXT_LENGTH = 60
HORIZONS = [5, 10]
STEP_SIZE = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Starting Deep Backtest for {TICKER}...")
print(f"Context: {CONTEXT_LENGTH} days | Horizons: {HORIZONS} | Device: {DEVICE}")

# 2. Data Acquisition (1 Year)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
df = yf.download(TICKER, start=start_date, end=end_date, interval="1d", progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

prices = df['Close'].values
dates = df.index.values
print(f"Downloaded {len(prices)} days of data.")

# 3. Initialize Pipeline
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map=DEVICE,
    dtype=torch.bfloat16,
)

# 4. Deep Backtest Loop
all_results = {h: [] for h in HORIZONS}

# To avoid re-running the same context for different horizons, we loop once
for i in range(0, len(prices) - CONTEXT_LENGTH - max(HORIZONS) + 1, STEP_SIZE):
    context_prices = prices[i : i + CONTEXT_LENGTH]
    context_tensor = torch.tensor(context_prices, device=DEVICE, dtype=torch.float32)
    
    for h in HORIZONS:
        target_prices = prices[i + CONTEXT_LENGTH : i + CONTEXT_LENGTH + h]
        
        # Predict
        forecast = pipeline.predict(context_tensor, h)
        samples = forecast[0].numpy()
        
        low, median, high = np.percentile(samples, [10, 50, 90], axis=0)
        
        # Metrics
        mae = np.mean(np.abs(target_prices - median))
        mape = np.mean(np.abs((target_prices - median) / target_prices)) * 100
        
        # Directional: Did it correctly predict the direction of the final price in the horizon relative to context end?
        actual_dir = 1 if target_prices[-1] > context_prices[-1] else -1
        pred_dir = 1 if median[-1] > context_prices[-1] else -1
        hit = 1 if actual_dir == pred_dir else 0
        
        all_results[h].append({
            "start_date": dates[i + CONTEXT_LENGTH],
            "mae": mae,
            "mape": mape,
            "hit": hit,
            "actual": target_prices,
            "pred": median,
            "low": low,
            "high": high
        })

# 5. Summary and Plotting
plt.figure(figsize=(15, 10))

for idx, h in enumerate(HORIZONS):
    horizon_data = all_results[h]
    maes = [r['mae'] for r in horizon_data]
    mapes = [r['mape'] for r in horizon_data]
    hits = [r['hit'] for r in horizon_data]
    
    avg_mae = np.mean(maes)
    avg_mape = np.mean(mapes)
    acc = np.mean(hits) * 100
    
    print(f"\n--- Horizon: {h} Days ---")
    print(f"Avg MAE:  {avg_mae:.2f}")
    print(f"Avg MAPE: {avg_mape:.2f}%")
    print(f"Accuracy: {acc:.2f}%")
    
    # Plotting for the first horizon (5 days) to avoid clutter
    if h == 5:
        plt.subplot(2, 1, 1)
        # Flatten actuals and predictions for a continuous look
        all_actual_dates = []
        all_actual_values = []
        all_pred_values = []
        
        for r in horizon_data:
            d_range = pd.date_range(start=r['start_date'], periods=h)
            all_actual_dates.extend(d_range)
            all_actual_values.extend(r['actual'])
            all_pred_values.extend(r['pred'])
            
        plt.plot(dates, prices, label="Actual Price", color='black', alpha=0.3)
        plt.plot(all_actual_dates, all_pred_values, label=f"Predicted Median ({h}d)", color='blue', linestyle='--')
        plt.title(f"BTC-USD Deep Backtest - {h} Day Horizon")
        plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.savefig("btc_backtest_results.png")
print("\nResults saved to btc_backtest_results.png")
