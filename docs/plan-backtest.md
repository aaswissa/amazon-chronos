# Plan: Real-World Stock Backtesting with Chronos Bolt

## Objective
Evaluate the accuracy of the `amazon/chronos-bolt-base` model by performing a backtest on real historical stock data fetched from Yahoo Finance.

## Technical Requirements
1.  **Data Acquisition:** Use `yfinance` to fetch the last 6 months of daily closing prices for a popular stock (e.g., Apple - `AAPL`).
2.  **Backtest Strategy:**
    *   **Window Type:** Rolling Window.
    *   **Context Length:** 30 days (historical prices provided to the model).
    *   **Prediction Horizon:** 5 days (forecast length).
    *   **Step Size:** 5 days (move the window forward by the prediction length to avoid overlapping forecasts in the evaluation).
3.  **Model Configuration:**
    *   Model: `amazon/chronos-bolt-base`.
    *   Device: `mps` (Apple Silicon) or `cpu`.
    *   Dtype: `torch.bfloat16`.
4.  **Evaluation Metrics:**
    *   Calculate **Mean Absolute Error (MAE)** for each 5-day window.
    *   Calculate the **Directional Accuracy** (did the model correctly predict if the price would go up or down over the 5-day period?).
5.  **Output:**
    *   A clean table showing the Date, Actual Price, Predicted Median, and the Error for each backtest step.
    *   A final summary of the model's performance (Total MAE and Directional Accuracy %).

## Implementation Steps
1.  Install `yfinance` (if not already present).
2.  Create `chronos_backtest.py`.
3.  Implement data fetching and normalization (Chronos handles scale internally, but we need to manage the pandas series).
4.  Implement the backtest loop.
5.  Print the formatted results to the console.
