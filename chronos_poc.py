import torch
import numpy as np
from chronos import BaseChronosPipeline

# 1. Hardcode mock list of 30 recent stock prices
historical_prices = [
    150.2, 151.5, 152.1, 151.8, 153.2, 154.5, 154.0, 155.2, 156.8, 157.1,
    156.5, 158.2, 159.4, 158.8, 157.9, 158.5, 159.2, 160.1, 161.5, 162.2,
    161.8, 163.5, 164.8, 164.2, 165.5, 166.8, 167.1, 166.5, 168.2, 169.4
]

# 2. Target Apple Silicon GPU (MPS) if available, fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 3. Initialize the pipeline
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map=device,
    dtype=torch.bfloat16,
)

# 4. Forecast the next 5 steps
context = torch.tensor(historical_prices, device=device)
prediction_length = 5
forecast = pipeline.predict(context, prediction_length)

# 5. Extract 10th, 50th (median), and 90th percentiles
samples = forecast[0].numpy()  # (num_samples, prediction_length)
low, median, high = np.percentile(samples, [10, 50, 90], axis=0)

# 6. Print results
print("\nHistorical Prices (last 10):", historical_prices[-10:])
print("\n5-Day Forecast:")
print(f"{'Day':<5} {'Low (10th)':<12} {'Median (50th)':<15} {'High (90th)':<12}")
for i in range(prediction_length):
    print(f"{i+1:<5} {low[i]:<12.2f} {median[i]:<15.2f} {high[i]:<12.2f}")
