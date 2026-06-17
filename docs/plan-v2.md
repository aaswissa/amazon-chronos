Act as an expert Python developer and AI engineer. 

Create a short, simple, and self-contained Proof of Concept (PoC) Python script that uses Amazon's `chronos-bolt-base` model from Hugging Face to forecast financial data.

The script must fulfill the following technical requirements:
1. Hardcode a mock list of 30 recent stock prices (e.g., [150.2, 151.5, ...]) as the historical context, so no external API keys or heavy downloads (like yfinance) are required for this basic PoC.
2. Explicitly target Apple Silicon GPU acceleration by setting the PyTorch device to "mps" if available, falling back to "cpu".
3. Initialize the pipeline using `BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", torch_dtype=torch.bfloat16)`.
4. Forecast the next 5 steps (prediction_length=5).
5. Extract the 10th, 50th (median), and 90th percentiles from the forecast samples.
6. Print the historical prices and the 5-day predicted median, low, and high bounds cleanly to the console.

Keep the code minimal, readable, clean, and fully contained in a single script block with brief inline comments. Do not overcomplicate it with matplotlib charts or complex error handling.