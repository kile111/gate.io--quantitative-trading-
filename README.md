# Gate.io Trading Bot (Streamlit Version)

A modern, offline-capable cryptocurrency trading bot and backtesting platform for Gate.io, built with Python, Streamlit, and ccxt. This tool provides a user-friendly web interface for strategy research, backtesting, live trading (paper & real), and visual analytics.

## Features

- **Backtesting**: Test EMA/RSI/ATR-based strategies on historical data with customizable parameters.
- **Live Trading**: Trade in real-time using Gate.io API, with support for both paper (simulated) and real trading modes.
- **Visual Analytics**: Interactive candlestick charts, EMA overlays, real-time price lines, equity curves, and grid trading visualization.
- **Parameter Tuning**: Adjust all key strategy and risk parameters via the sidebar.
- **Quick Pair Switching**: Instantly switch between popular trading pairs.
- **Session Logging**: All trades and actions are logged and downloadable as CSV.
- **Offline Installation**: Fully supports offline deployment using pre-downloaded wheels and a batch script.
- **API Key Management**: Securely manage API keys via `.streamlit/secrets.toml` or environment variables.

## Project Structure

```
.
├── app.py                  # Main Streamlit app (Chinese UI, stable)
├── trading_core.py         # Core trading logic, indicators, backtester, live trader
├── requirements.txt        # Python dependencies
├── run_app.bat             # Offline startup script (Windows)
├── wheels/                 # Pre-downloaded Python wheels for offline install
└── .streamlit/
    └── secrets.toml        # API keys (not tracked by git)
```

## Quick Start

### 1. Prepare Environment

- Ensure Python 3.10+ is installed.
- For offline use, copy the `wheels/` directory from a prepared machine.

### 2. Configure API Keys

- Edit `.streamlit/secrets.toml` and fill in your Gate.io API credentials:
  ```
  GATEIO_API_KEY="your_api_key"
  GATEIO_SECRET="your_api_secret"
  ```

### 3. Install Dependencies

- **Offline (Recommended):**
  - Run `run_app.bat` (Windows). This will:
    - Create a virtual environment
    - Install all dependencies from `wheels/`
    - Launch the Streamlit app in your browser

- **Online (Manual):**
  ```shell
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  pip install streamlit-autorefresh
  streamlit run app.py
  ```

### 4. Using the App

- Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser.
- Use the sidebar to connect, set parameters, and switch between backtest, live trading, and logs.
- All trading is in paper mode by default. To enable real trading, disable "模拟盘" and confirm risk acknowledgment.

## Strategy Logic

- **Entry**: Long when fast EMA crosses above slow EMA and RSI exceeds threshold.
- **Exit**: On EMA cross-down, RSI drop, or ATR-based stop-loss/take-profit.
- **Risk Management**: Position sizing by risk per trade, min notional, slippage, and fees.

## Security

- API keys are never logged or exposed in the UI.
- Real trading requires explicit confirmation and disabling of paper mode.

## Notes

- This tool is for educational and research purposes only. Crypto trading is risky.
- The UI is in Chinese; code and comments are in English for clarity.
- For offline use, ensure all wheels match your Python version.

## License

MIT License
