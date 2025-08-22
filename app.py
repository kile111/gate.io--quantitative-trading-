# app_en.py — English UI (Stable: second-level refresh / always-on real-time chart / quick pairs / paper trading auto+manual)
import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from dotenv import load_dotenv

# Optional: Auto-refresh (recommended)
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

from trading_core import (
    GateIO, StrategyConfig, RiskConfig, RuntimeConfig,
    fetch_history, compute_indicators, generate_signals, backtest,
    LiveTrader,
)

# -----------------------------
# Helpers (cached history & timeframe limits)
# -----------------------------
@st.cache_data(ttl=15, show_spinner=False)
def _fetch_history_cached(_g, symbol: str, timeframe: str, days: int):
    """Cache historical K-lines only; _g underscore lets Streamlit skip object hashing."""
    return fetch_history(_g, symbol, timeframe, days)


def _tf_minutes(tf: str) -> int:
    table = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    return table.get(tf, 60)


def _max_days_for_tf(tf: str, cap: int = 10_000) -> int:
    """Gate.io returns at most cap K-lines per request; convert to max days."""
    mins = _tf_minutes(tf)
    return max(1, int(cap * mins / (60 * 24)))


def _push_live_price(symbol: str, price: float, window_min: int = 30) -> pd.DataFrame:
    row = {"dt": pd.Timestamp.utcnow(), "price": float(price)}
    df = st.session_state.live_prices.get(symbol)
    if df is None or df.empty:
        df = pd.DataFrame([row])
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=window_min)
    df = df[df["dt"] >= cutoff]
    st.session_state.live_prices[symbol] = df
    return df


def _plot_live_line(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["dt"], y=df["price"], mode="lines+markers", name="Last"))
    fig.update_layout(height=360, xaxis_rangeslider_visible=False, title=title)
    st.plotly_chart(fig, use_container_width=True)


def _plot_equity(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["dt"], y=df["equity"], mode="lines", name="Equity"))
    fig.update_layout(height=300, xaxis_rangeslider_visible=False, title=title)
    st.plotly_chart(fig, use_container_width=True)


def _build_market_fig(
    df_hist: pd.DataFrame,
    df_live: Optional[pd.DataFrame],
    timeframe: str,
    title: str,
    show_candle: bool = True,
    show_ema: bool = True,
    show_live: bool = True,
    use_rangeslider: bool = True,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    dca_on: bool = False,
    dca_minutes: int = 60,
    grid_on: bool = False,
    grid_center: Optional[float] = None,
    grid_step_pct: float = 0.5,
    grid_levels: int = 4,
):
    fig = go.Figure()

    # Main K-line
    if show_candle and (df_hist is not None) and (not df_hist.empty):
        fig.add_trace(go.Candlestick(
            x=df_hist["dt"], open=df_hist["open"], high=df_hist["high"],
            low=df_hist["low"], close=df_hist["close"], name="Candles"
        ))

    # EMA overlay
    if show_ema and ("ema_fast" in df_hist.columns) and ("ema_slow" in df_hist.columns):
        fig.add_trace(go.Scatter(x=df_hist["dt"], y=df_hist["ema_fast"], name="EMA Fast", mode="lines"))
        fig.add_trace(go.Scatter(x=df_hist["dt"], y=df_hist["ema_slow"], name="EMA Slow", mode="lines"))

    # DCA simulated average (rolling window)
    if dca_on and (df_hist is not None) and (not df_hist.empty):
        win = max(1, int(dca_minutes / max(1, _tf_minutes(timeframe))))
        dca = df_hist["close"].rolling(window=win, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_hist["dt"], y=dca, name=f"DCA({dca_minutes}m)", mode="lines"))

    # Real-time points
    if show_live and df_live is not None and not df_live.empty:
        fig.add_trace(go.Scatter(x=df_live["dt"], y=df_live["price"], name="Live Price", mode="lines+markers"))

    # Grid lines
    if grid_on and grid_center is not None and grid_levels > 0:
        step = float(grid_step_pct) / 100.0
        for k in range(-int(grid_levels), int(grid_levels) + 1):
            level = grid_center * ((1 + step) ** k)
            fig.add_hline(y=level, line_dash="dot", opacity=0.5,
                          annotation_text=f"Grid {k:+d}", annotation_position="right")

    # Axes and interaction
    xaxis = dict(rangeslider=dict(visible=bool(use_rangeslider)),
                 rangeselector=dict(buttons=[
                     dict(count=15, label="15m", step="minute", stepmode="backward"),
                     dict(count=1, label="1h", step="hour", stepmode="backward"),
                     dict(count=4, label="4h", step="hour", stepmode="backward"),
                     dict(count=1, label="1d", step="day", stepmode="backward"),
                     dict(step="all", label="All")
                 ]))
    fig.update_layout(xaxis=xaxis, height=520, title=title)

    if (y_min is not None) and (y_max is not None) and (y_max > y_min):
        fig.update_yaxes(range=[y_min, y_max])

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Gate.io Trading Bot", page_icon="📈", layout="wide")
st.title("📈 Gate.io Trading Bot — Streamlit")

with st.expander("⚠️ Disclaimer"):
    st.write(
        "This tool is for learning and research only and does not constitute investment advice. Crypto asset trading is risky."
        "Paper mode is enabled by default. For real trading, enable at your own risk."
    )

# -----------------------------
# Session State Initialization
# -----------------------------
ss = st.session_state
ss.setdefault("trader", None)
ss.setdefault("running", False)
ss.setdefault("log_rows", [])
ss.setdefault("live_prices", {})  # {symbol: DataFrame(dt, price)}
ss.setdefault("symbol_input", "BTC/USDT")
ss.setdefault("paper", None)  # Local paper wallet
ss.setdefault("equity_series", pd.DataFrame(columns=["dt", "equity"]))
ss.setdefault("sim_prev_long_cond", None)

# Handle 'Quick Switch' preset — must be before sidebar controls
if "symbol_to_set" in ss:
    ss["symbol_input"] = ss.pop("symbol_to_set")

# -----------------------------
# Sidebar — Connection & Parameters
# -----------------------------
st.sidebar.header("🔐 API / Connection")
load_dotenv()
use_secrets = st.sidebar.checkbox("Use Streamlit secrets (recommended)", value=True)
if use_secrets and "GATEIO_API_KEY" in st.secrets and "GATEIO_SECRET" in st.secrets:
    api_key = st.secrets["GATEIO_API_KEY"]
    api_secret = st.secrets["GATEIO_SECRET"]
else:
    api_key = st.sidebar.text_input("GATEIO_API_KEY", os.getenv("GATEIO_API_KEY", ""), type="password")
    api_secret = st.sidebar.text_input("GATEIO_SECRET", os.getenv("GATEIO_SECRET", ""), type="password")

default_type = st.sidebar.selectbox("Market Type", ["spot"], index=0, help="This app currently focuses on spot. Let me know if you need futures support.")

st.sidebar.header("🧠 Strategy Parameters")
col_a, col_b = st.sidebar.columns(2)
fast_ema = col_a.number_input(
    "EMA Fast", value=20, min_value=1, step=1,
    help="Exponential Moving Average (short-term). Used for trend and entry; smaller is more sensitive. Suggest: 10–20.",
)
slow_ema = col_b.number_input(
    "EMA Slow", value=50, min_value=2, step=1,
    help="Exponential Moving Average (long-term). Smoother; crossover with fast line determines bias. Common: 50–200.",
)
rsi_period = col_a.number_input(
    "RSI Period", value=14, min_value=2, step=1,
    help="Relative Strength Index window for momentum. Shorter is more sensitive; common: 14.",
)
rsi_long = col_b.number_input(
    "RSI Long Threshold", value=55.0, step=0.5,
    help="RSI above this triggers long; too high may chase, too low may miss. Suggest: 50–60.",
)
atr_period = col_a.number_input(
    "ATR Period", value=14, min_value=2, step=1,
    help="Average True Range window for stop/take-profit. Common: 14.",
)
atr_sl_mult = col_b.number_input(
    "ATR Stop Loss Multiplier", value=1.5, step=0.1,
    help="Stop loss = ATR × multiplier. Higher = wider stop, higher win rate but bigger drawdown. Suggest: 1–3.",
)
atr_tp_mult = col_a.number_input(
    "ATR Take Profit Multiplier", value=3.0, step=0.1,
    help="Take profit = ATR × multiplier. Usually > stop loss for positive expectancy. Suggest: 2–4.",
)

st.sidebar.header("🎯 Risk & Execution")
fee_rate = st.sidebar.number_input(
    "Fee Rate (e.g. 0.002 = 0.2%)", value=0.002, step=0.0001, format="%.4f",
    help="Fee per buy/sell, for backtest and risk. Typical spot: 0.1%–0.2%.",
)
slippage = st.sidebar.number_input(
    "Slippage (e.g. 0.0005 = 5bp)", value=0.0005, step=0.0001, format="%.4f",
    help="Estimated price deviation from order book. Suggest: 0.0002–0.001.",
)
risk_per_trade = st.sidebar.number_input(
    "Risk per Trade", value=0.01, step=0.001, format="%.3f",
    help="Max loss per trade as % of equity. Used for position sizing. Common: 0.5%–2%.",
)
min_notional = st.sidebar.number_input(
    "Min Notional (quote)", value=10.0, step=1.0,
    help="Minimum order notional (quote currency), per exchange rules.",
)
poll_sec = st.sidebar.number_input(
    "Polling Seconds", value=600, min_value=1, step=1,
    help="Live loop & chart refresh interval (seconds). 1 = every second (subject to exchange rate limits).",
)

st.sidebar.header("🧱 Market")
st.sidebar.text_input(
    "Symbol", value=ss.get("symbol_input", "BTC/USDT"), key="symbol_input",
    help="e.g. BTC/USDT; must match exchange symbol. Use 'Quick Switch' for one-click change.",
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"], index=0,
    help="K-line interval; affects indicator/signal pace. Keep same for backtest/live.",
)
lookback_days = st.sidebar.number_input(
    "Backtest Window (days)", value=90, min_value=2, step=1,
    help="Download historical K-lines for backtest; longer = more robust but slower. Suggest ≥60 days.",
)

# Build configs
sc = StrategyConfig(
    fast_ema=int(fast_ema), slow_ema=int(slow_ema),
    rsi_period=int(rsi_period), rsi_long=float(rsi_long),
    atr_period=int(atr_period), atr_sl_mult=float(atr_sl_mult), atr_tp_mult=float(atr_tp_mult),
)
rc = RiskConfig(
    fee_rate=float(fee_rate), slippage=float(slippage),
    risk_per_trade=float(risk_per_trade), min_notional=float(min_notional),
)
rt = RuntimeConfig(symbol=ss["symbol_input"], timeframe=timeframe, poll_sec=int(poll_sec), dry_run=True)

connect_btn = st.sidebar.button("Connect / Reconnect")

# -----------------------------
# Tabs: Backtest | Live | Logs
# -----------------------------
tab_bt, tab_live, tab_logs = st.tabs(["📊 Backtest", "🚦 Live Trading", "🧾 Logs"])

# -----------------------------
# Connection
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_exchange(key, secret, default_type):
    return GateIO(api_key=key, api_secret=secret, default_type=default_type)

if connect_btn or ss.trader is None:
    try:
        g = get_exchange(api_key, api_secret, default_type)
        ss.exchange = g
        st.sidebar.success("Connected to Gate.io via ccxt ✅")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

# -----------------------------
# Backtest tab
# -----------------------------
with tab_bt:
    st.subheader("Backtest")
    run_bt = st.button("Run Backtest")
    if run_bt:
        g = ss.get("exchange")
        if not g:
            st.warning("Please connect to exchange first.")
        else:
            with st.spinner("Downloading K-lines and computing indicators..."):
                eff_days = min(int(lookback_days), _max_days_for_tf(timeframe))
                if eff_days < int(lookback_days):
                    st.info(f"Backtest window trimmed to {eff_days} days ({timeframe} max ~10000 K-lines).")
                df = fetch_history(g, ss["symbol_input"], timeframe, eff_days)
                df = compute_indicators(df, sc)
                df = generate_signals(df, sc)
                trades_df, results = backtest(df, sc, rc, starting_equity=10_000.0)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Trades", results["trades"])
            k2.metric("Total Return", f"{results['total_return_pct']:.2f}%")
            k3.metric("Win Rate", f"{results['win_rate_pct']:.2f}%")
            k4.metric("Avg Trade PnL", f"{results['avg_trade_pnl']:.2f}")
            k5.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")

            price_fig = go.Figure()
            price_fig.add_trace(go.Candlestick(
                x=df["dt"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Candles",
            ))
            price_fig.add_trace(go.Scatter(x=df["dt"], y=df["ema_fast"], name=f"EMA {sc.fast_ema}", mode="lines"))
            price_fig.add_trace(go.Scatter(x=df["dt"], y=df["ema_slow"], name=f"EMA {sc.slow_ema}", mode="lines"))
            entries = df[df["long_signal"]]
            exits = df[df["exit_signal"]]
            price_fig.add_trace(go.Scatter(x=entries["dt"], y=entries["close"], mode="markers", name="Entry", marker=dict(symbol="triangle-up", size=9)))
            price_fig.add_trace(go.Scatter(x=exits["dt"], y=exits["close"], mode="markers", name="Exit", marker=dict(symbol="triangle-down", size=9)))
            price_fig.update_layout(height=520, xaxis_rangeslider_visible=False, title=f"{ss['symbol_input']} {timeframe}")
            st.plotly_chart(price_fig, use_container_width=True)

            st.write("Trade List")
            st.dataframe(trades_df, use_container_width=True)
            st.download_button(
                "Download Trades CSV",
                trades_df.to_csv(index=False).encode(),
                file_name="trades_backtest.csv",
                mime="text/csv",
            )

# -----------------------------
# Live tab (always-on: real-time chart + quick switch + paper trading)
# -----------------------------
with tab_live:
    st.subheader("Live Trading Control")

    # Global auto-refresh (updates price/chart every second regardless of running)
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=int(max(1, int(poll_sec)) * 1000), key="global_refresh")

    # Quick switch pairs (reset app, avoid text_input conflict)
    st.markdown("**Quick Switch** (one-click to change left symbol)")
    quick_pairs = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "DOGE/USDT", "TAO/USDT"]

    def _qset(p: str):
        ss["symbol_to_set"] = p

    qcols = st.columns(len(quick_pairs))
    for i, p in enumerate(quick_pairs):
        qcols[i].button(p, use_container_width=True, key=f"q_{p}", on_click=_qset, args=(p,))

    # Real-time chart settings
    live_window_min = st.number_input(
        "Live Chart Window (minutes)", value=30, min_value=1, step=1,
        help="Show price line for the last N minutes.",
    )

    # Always-on: market + historical K-line + chart controls (no need to start)
    ticker = None
    df_live = None
    g = ss.get("exchange")
    if g:
        try:
            ticker = g.fetch_ticker(ss["symbol_input"])  # Latest price
            if ticker.get("last") is not None:
                df_live = _push_live_price(ss["symbol_input"], float(ticker["last"]), window_min=int(live_window_min))
        except Exception as e:
            st.caption(f"Failed to fetch {ss['symbol_input']} ticker: {e}")

        # Fetch history + indicators
        eff_days_vis = min(max(int(lookback_days), 2), _max_days_for_tf(timeframe))
        df_hist = _fetch_history_cached(g, ss["symbol_input"], timeframe, eff_days_vis)
        df_hist = compute_indicators(df_hist, sc)
        df_hist = generate_signals(df_hist, sc)

        # Chart options (with help)
        st.markdown("#### Chart Settings")
        cc1, cc2, cc3 = st.columns(3)
        show_candle = cc1.toggle("Show Candles", True, help="Show OHLC data as candles.")
        show_ema = cc2.toggle("Overlay EMA", True, help="Overlay EMA fast/slow lines for trend.")
        show_live = cc3.toggle("Overlay Live Price", True, help="Overlay latest price for real-time movement.")

        c4, c5, c6 = st.columns(3)
        use_rangeslider = c4.toggle("Enable Range Slider", True, help="Show bottom range slider for zoom.")
        manual_y = c5.toggle("Manual Y-axis Range", False, help="Enable to set min/max price axis.")
        dca_on = c6.toggle("DCA Line", False, help="Simulate DCA cost line with rolling close.")

        ymn, ymx, dca_box = st.columns(3)
        y_min = ymn.number_input("Y-axis Min (0=auto)", value=0.0, help="0 means auto.") if manual_y else None
        y_max = ymx.number_input("Y-axis Max (0=auto)", value=0.0, help="0 means auto.") if manual_y else None
        if manual_y:
            if y_min == 0.0: y_min = None
            if y_max == 0.0: y_max = None
        dca_minutes = dca_box.number_input("DCA Window (minutes)", value=60, min_value=1, step=5,
                                           help="Rolling window for DCA line.") if dca_on else 60

        st.markdown("#### Grid Settings")
        gc1, gc2, gc3, gc4 = st.columns(4)
        grid_on = gc1.toggle("Enable Grid", False, help="Draw horizontal grid lines around center price.")
        default_center = None
        if ticker and ticker.get("last") is not None:
            default_center = float(ticker["last"])
        elif not df_hist.empty:
            default_center = float(df_hist["close"].iloc[-1])
        grid_center = gc2.number_input("Center Price", value=float(default_center or 0.0), format="%.6f",
                                        help="Grid center price, default to latest/close.") if grid_on else None
        grid_step_pct = gc3.number_input("Step (%)", value=0.5, step=0.05, help="Percent gap between grids.") if grid_on else 0.5
        grid_levels = gc4.number_input("Levels (each side)", value=4, min_value=1, step=1, help="How many grids up/down.") if grid_on else 4

        # Compose chart
        _build_market_fig(
            df_hist=df_hist,
            df_live=df_live,
            timeframe=timeframe,
            title=f"{ss['symbol_input']} Market Chart",
            show_candle=show_candle,
            show_ema=show_ema,
            show_live=show_live,
            use_rangeslider=use_rangeslider,
            y_min=y_min,
            y_max=y_max,
            dca_on=dca_on,
            dca_minutes=int(dca_minutes),
            grid_on=grid_on,
            grid_center=grid_center,
            grid_step_pct=float(grid_step_pct),
            grid_levels=int(grid_levels),
        )
    else:
        st.caption("Not connected to exchange, cannot fetch live price or history.")

    # ========= Paper Trading (auto & manual) =========
    st.markdown("---")
    st.markdown("### 🧪 Paper Trading (local, not sent to exchange)")

    colp1, colp2, colp3 = st.columns([1, 1, 1])
    use_paper_wallet = colp1.toggle("Enable Paper Trading", value=True, help="Simulate balance locally with real prices, no real orders.")
    auto_trade_sim = colp2.toggle("Auto Trade by Strategy", value=True, help="Auto open/close by EMA/RSI/ATR signals (ignores min notional).")
    paper_start_equity = colp3.number_input("Initial Equity (USDT)", value=10_000.0, step=100.0)

    alloc_pct = st.slider(
        "Allocation per Trade", min_value=0.05, max_value=1.0, value=0.10, step=0.05,
        help="On signal, use this % of available cash to buy; sell all on exit.",
    )

    # Reset wallet on init/symbol change (no auto buy/sell)
    if use_paper_wallet and (ss.paper is None or ss.paper.get("symbol_bound") != ss["symbol_input"]):
        ss.paper = {"cash": float(paper_start_equity), "base": 0.0, "avg": 0.0, "symbol_bound": ss["symbol_input"]}
        ss.equity_series = pd.DataFrame(columns=["dt", "equity"])  # Clear curve
        ss.sim_prev_long_cond = None

    # Auto strategy (only if paper trading & auto enabled)
    if use_paper_wallet and auto_trade_sim and g:
        try:
            eff_days_sim = min(max(int(lookback_days), 5), _max_days_for_tf(timeframe))
            df_hist = _fetch_history_cached(g, ss["symbol_input"], timeframe, eff_days_sim)
            df_hist = compute_indicators(df_hist, sc)
            df_hist = generate_signals(df_hist, sc)
            if len(df_hist) >= 2:
                last = df_hist.iloc[-1]
                prev = df_hist.iloc[-2]
                long_cond = bool(last.get("long_cond", False))
                prev_long_cond = bool(prev.get("long_cond", False))
                long_sig = long_cond and (not prev_long_cond)
                exit_sig = (not long_cond) and prev_long_cond

                price_ref = None
                if ticker and ticker.get("last") is not None:
                    price_ref = float(ticker["last"])
                elif not pd.isna(last.get("close", np.nan)):
                    price_ref = float(last.get("close"))

                if price_ref is not None:
                    # BUY
                    if long_sig:
                        quote_to_use = ss.paper["cash"] * float(alloc_pct)
                        if quote_to_use > 0:
                            eff_price = price_ref * (1 + float(slippage))
                            qty = (quote_to_use * (1 - float(fee_rate))) / eff_price
                            ss.paper["cash"] -= quote_to_use
                            total_cost = ss.paper["avg"] * ss.paper["base"] + eff_price * qty
                            ss.paper["base"] += qty
                            ss.paper["avg"] = 0.0 if ss.paper["base"] == 0 else total_cost / ss.paper["base"]
                            ss.log_rows.append({
                                "ts": pd.Timestamp.utcnow(), "symbol": ss["symbol_input"], "timeframe": timeframe,
                                "action": f"SIM BUY {qty:.6f}", "entry": json.dumps({"price": eff_price}), "exit": "", "msg": "auto-sim",
                            })
                    # SELL ALL
                    if exit_sig:
                        qty = ss.paper["base"]
                        if qty > 0:
                            eff_price = price_ref * (1 - float(slippage))
                            proceeds = qty * eff_price * (1 - float(fee_rate))
                            ss.paper["cash"] += proceeds
                            ss.paper["base"] = 0.0
                            ss.paper["avg"] = 0.0
                            ss.log_rows.append({
                                "ts": pd.Timestamp.utcnow(), "symbol": ss["symbol_input"], "timeframe": timeframe,
                                "action": "SIM SELL ALL", "entry": "", "exit": json.dumps({"price": eff_price}), "msg": "auto-sim",
                            })
        except Exception as e:
            st.warning(f"Auto paper trading failed: {e}")

    # Manual buttons
    colm1, colm2, colm3, colm4 = st.columns([1, 1, 1, 2])
    if colm1.button("Sim Buy 10%") and use_paper_wallet and ss.paper is not None and ticker and ticker.get("last") is not None:
        price_ref = float(ticker["last"]) * (1 + float(slippage))
        quote_to_use = ss.paper["cash"] * 0.10
        if quote_to_use > 0:
            qty = (quote_to_use * (1 - float(fee_rate))) / price_ref
            ss.paper["cash"] -= quote_to_use
            total_cost = ss.paper["avg"] * ss.paper["base"] + price_ref * qty
            ss.paper["base"] += qty
            ss.paper["avg"] = 0.0 if ss.paper["base"] == 0 else total_cost / ss.paper["base"]
            ss.log_rows.append({
                "ts": pd.Timestamp.utcnow(), "symbol": ss["symbol_input"], "timeframe": timeframe,
                "action": f"SIM BUY 10% {qty:.6f}", "entry": json.dumps({"price": price_ref}), "exit": "", "msg": "manual",
            })
    if colm2.button("Sell All") and use_paper_wallet and ss.paper is not None and ticker and ticker.get("last") is not None:
        price_ref = float(ticker["last"]) * (1 - float(slippage))
        qty = ss.paper["base"]
        if qty > 0:
            proceeds = qty * price_ref * (1 - float(fee_rate))
            ss.paper["cash"] += proceeds
            ss.paper["base"] = 0.0
            ss.paper["avg"] = 0.0
            ss.log_rows.append({
                "ts": pd.Timestamp.utcnow(), "symbol": ss["symbol_input"], "timeframe": timeframe,
                "action": "SIM SELL ALL", "entry": "", "exit": json.dumps({"price": price_ref}), "msg": "manual",
            })
    if colm3.button("Reset Paper Wallet") and use_paper_wallet:
        ss.paper = {"cash": float(paper_start_equity), "base": 0.0, "avg": 0.0, "symbol_bound": ss["symbol_input"]}
        ss.equity_series = pd.DataFrame(columns=["dt", "equity"])  # Clear curve
        ss.sim_prev_long_cond = None
        st.toast("Paper wallet reset.", icon="↩️")

    # Update and show paper equity
    if use_paper_wallet and ss.paper is not None and ticker and ticker.get("last") is not None:
        price_now = float(ticker["last"]) if ticker.get("last") is not None else 0.0
        equity = ss.paper["cash"] + ss.paper["base"] * price_now
        row = pd.DataFrame([[pd.Timestamp.utcnow(), equity]], columns=["dt", "equity"])  # type: ignore
        ss.equity_series = pd.concat([ss.equity_series, row], ignore_index=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=int(live_window_min))
        ss.equity_series = ss.equity_series[ss.equity_series["dt"] >= cutoff]

        e1, e2, e3 = st.columns(3)
        e1.metric("Paper Equity (USDT)", f"{equity:,.2f}")
        e2.metric("Position Size", f"{ss.paper['base']:.6f}")
        upnl = 0.0
        if ss.paper["base"] > 0 and ss.paper["avg"] > 0:
            upnl = (price_now - ss.paper["avg"]) / ss.paper["avg"] * 100
        e3.metric("Unrealized PnL%", f"{upnl:.2f}%")
        _plot_equity(ss.equity_series, title="Paper Equity Curve")

    # ========= Live Trading (optional) =========
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    paper_mode = col1.toggle("Paper Mode (real order switch)", value=True, help="On = no real orders, only local sim; Off + risk confirm = real orders.")
    live_ack = col2.toggle("I understand the risks", value=False)
    start_button = col3.button("Start / Apply", type="primary")
    stop_button = col3.button("Stop")

    if start_button:
        g = ss.get("exchange")
        if not g:
            st.warning("Please connect on the left first.")
        else:
            rt.dry_run = not (live_ack and not paper_mode)
            rt.symbol = ss["symbol_input"]
            ss.trader = LiveTrader(g, sc, rc, rt)
            ss.running = True
            st.toast("Live loop started (see 'Recent Actions' below).", icon="✅")

    if stop_button:
        ss.running = False
        ss.trader = None
        st.toast("Stopped.", icon="🛑")

    if ss.running and ss.trader is not None:
        with st.spinner("Polling..."):
            info = ss.trader.step()
        ss.log_rows.append({
            "ts": pd.Timestamp.utcnow(),
            "symbol": ss["symbol_input"],
            "timeframe": timeframe,
            "action": info.get("action"),
            "entry": json.dumps(info.get("entry")),
            "exit": json.dumps(info.get("exit")),
            "msg": info.get("message", ""),
        })
        st.write("**Recent Action:**", info.get("action") or "—")
        if info.get("entry"):
            st.write("Entry:", info["entry"])
        if info.get("exit"):
            st.write("Exit:", info["exit"])
        if info.get("message"):
            st.info(info["message"])  # trading_core messages (English)

# -----------------------------
# Logs tab
# -----------------------------
with tab_logs:
    st.subheader("Session Logs")
    if ss.log_rows:
        log_df = pd.DataFrame(ss.log_rows)
        st.dataframe(log_df, use_container_width=True, height=320)
        st.download_button(
            "Download Logs CSV",
            log_df.to_csv(index=False).encode(),
            file_name="session_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("No actions yet.")

st.caption("Tip: Put your keys in `.streamlit/secrets.toml` as `GATEIO_API_KEY` and `GATEIO_SECRET` for better security.")
