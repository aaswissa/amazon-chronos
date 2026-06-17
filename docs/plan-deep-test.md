# Plan: Deep Backtest for BTC-USD with Chronos Bolt

## Objective
Perform a rigorous, "deep" evaluation of the `amazon/chronos-bolt-base` model on Bitcoin (BTC-USD) to determine its reliability for high-volatility assets.

## Technical Requirements
1.  **Data Acquisition:** Use `yfinance` to fetch **1 year** of daily data for `BTC-USD`.
2.  **Extended Context:** Use a **60-day context window** (instead of 30) to give the model more historical patterns.
3.  **Multi-Horizon Evaluation:**
    *   Test 5-day forecasts.
    *   Test 10-day forecasts (to see if the model holds up over longer horizons).
4.  **Deep Metrics:**
    *   **MAE (Mean Absolute Error):** Standard error measurement.
    *   **MAPE (Mean Absolute Percentage Error):** Better for BTC given its price range.
    *   **Directional Accuracy:** % of correct up/down predictions.
    *   **Volatility Catch:** Measure if the model predicts higher volatility during high-volatility periods.
5.  **Visualization:** Generate a plot comparing Actual vs. Predicted prices for the entire backtest period to visually identify "lag" or "overshooting".

## Implementation Steps
1.  Create `chronos_deep_test.py`.
2.  Implement the multi-horizon backtest loop.
3.  Add `matplotlib` for visualization (verify if installed).
4.  Calculate and print comprehensive summary statistics.
5.  Save the visualization as `btc_backtest_results.png`.
