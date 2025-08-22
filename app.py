# app_zh.py — 中文界面（稳定版：秒级刷新 / 常显实时图 / 快捷交易对 / 模拟资金 自动+手动）
import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from dotenv import load_dotenv

# 可选：自动刷新（建议安装）
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
    """仅缓存历史K线；_g 前下划线让 Streamlit 跳过对象哈希。"""
    return fetch_history(_g, symbol, timeframe, days)


def _tf_minutes(tf: str) -> int:
    table = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    return table.get(tf, 60)


def _max_days_for_tf(tf: str, cap: int = 10_000) -> int:
    """Gate.io 单次最多返回 cap 根K线；换算成天数上限。"""
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

    # 主K线
    if show_candle and (df_hist is not None) and (not df_hist.empty):
        fig.add_trace(go.Candlestick(
            x=df_hist["dt"], open=df_hist["open"], high=df_hist["high"],
            low=df_hist["low"], close=df_hist["close"], name="K线"
        ))

    # EMA 叠加
    if show_ema and ("ema_fast" in df_hist.columns) and ("ema_slow" in df_hist.columns):
        fig.add_trace(go.Scatter(x=df_hist["dt"], y=df_hist["ema_fast"], name="EMA快", mode="lines"))
        fig.add_trace(go.Scatter(x=df_hist["dt"], y=df_hist["ema_slow"], name="EMA慢", mode="lines"))

    # DCA 模拟均线（滚动窗口）
    if dca_on and (df_hist is not None) and (not df_hist.empty):
        win = max(1, int(dca_minutes / max(1, _tf_minutes(timeframe))))
        dca = df_hist["close"].rolling(window=win, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_hist["dt"], y=dca, name=f"DCA({dca_minutes}m)", mode="lines"))

    # 实时点
    if show_live and df_live is not None and not df_live.empty:
        fig.add_trace(go.Scatter(x=df_live["dt"], y=df_live["price"], name="实时价", mode="lines+markers"))

    # 网格线
    if grid_on and grid_center is not None and grid_levels > 0:
        step = float(grid_step_pct) / 100.0
        for k in range(-int(grid_levels), int(grid_levels) + 1):
            level = grid_center * ((1 + step) ** k)
            fig.add_hline(y=level, line_dash="dot", opacity=0.5,
                          annotation_text=f"Grid {k:+d}", annotation_position="right")

    # 轴与交互
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
st.set_page_config(page_title="Gate.io 交易机器人", page_icon="📈", layout="wide")
st.title("📈 Gate.io 交易机器人 — Streamlit")

with st.expander("⚠️ 免责声明"):
    st.write(
        "本工具仅用于学习与研究，不构成投资建议。加密资产交易存在较高风险。"
        "默认启用【模拟盘】（Paper mode）。如需真实交易，请在明知风险的情况下自行开启。"
    )

# -----------------------------
# Session State 初始化
# -----------------------------
ss = st.session_state
ss.setdefault("trader", None)
ss.setdefault("running", False)
ss.setdefault("log_rows", [])
ss.setdefault("live_prices", {})  # {symbol: DataFrame(dt, price)}
ss.setdefault("symbol_input", "BTC/USDT")
ss.setdefault("paper", None)  # 本地模拟资金账本
ss.setdefault("equity_series", pd.DataFrame(columns=["dt", "equity"]))
ss.setdefault("sim_prev_long_cond", None)

# 处理『快捷切换』预设 —— 必须在侧边栏控件渲染之前
if "symbol_to_set" in ss:
    ss["symbol_input"] = ss.pop("symbol_to_set")

# -----------------------------
# Sidebar — 连接 & 参数
# -----------------------------
st.sidebar.header("🔐 API / 连接")
load_dotenv()
use_secrets = st.sidebar.checkbox("使用 Streamlit secrets（推荐）", value=True)
if use_secrets and "GATEIO_API_KEY" in st.secrets and "GATEIO_SECRET" in st.secrets:
    api_key = st.secrets["GATEIO_API_KEY"]
    api_secret = st.secrets["GATEIO_SECRET"]
else:
    api_key = st.sidebar.text_input("GATEIO_API_KEY", os.getenv("GATEIO_API_KEY", ""), type="password")
    api_secret = st.sidebar.text_input("GATEIO_SECRET", os.getenv("GATEIO_SECRET", ""), type="password")

default_type = st.sidebar.selectbox("市场类型", ["spot"], index=0, help="本应用当前专注现货。如需合约支持可告诉我。")

st.sidebar.header("🧠 策略参数")
col_a, col_b = st.sidebar.columns(2)
fast_ema = col_a.number_input(
    "EMA 快线", value=20, min_value=1, step=1,
    help="指数移动平均（短期）。用于与慢线交叉判断趋势与入场；数值越小越灵敏。建议：10–20。",
)
slow_ema = col_b.number_input(
    "EMA 慢线", value=50, min_value=2, step=1,
    help="指数移动平均（长期）。更平滑；与快线交叉决定多空偏向。常见：50–200。",
)
rsi_period = col_a.number_input(
    "RSI 周期", value=14, min_value=2, step=1,
    help="相对强弱指数的计算窗口，用于衡量动量。周期越短越敏感；常见 14。",
)
rsi_long = col_b.number_input(
    "RSI 做多阈值", value=55.0, step=0.5,
    help="当 RSI 高于该值时满足做多条件之一；过高易追高，过低可能错过。建议：50–60。",
)
atr_period = col_a.number_input(
    "ATR 周期", value=14, min_value=2, step=1,
    help="平均真实波幅（波动率）计算窗口，用于确定止损/止盈距离。常见 14。",
)
atr_sl_mult = col_b.number_input(
    "ATR 止损倍数", value=1.5, step=0.1,
    help="止损距离 = ATR × 倍数。倍数越大止损更宽，胜率↑但潜在回撤↑。建议：1–3。",
)
atr_tp_mult = col_a.number_input(
    "ATR 止盈倍数", value=3.0, step=0.1,
    help="止盈距离 = ATR × 倍数。通常大于止损倍数以获得正期望。建议：2–4。",
)

st.sidebar.header("🎯 风控与执行")
fee_rate = st.sidebar.number_input(
    "单边手续费 (例如 0.002 = 0.2%)", value=0.002, step=0.0001, format="%.4f",
    help="每次买或卖的费率，用于回测和风控成本。现货典型范围 0.1%–0.2%。",
)
slippage = st.sidebar.number_input(
    "滑点 (例如 0.0005 = 5bp)", value=0.0005, step=0.0001, format="%.4f",
    help="成交价相对盘口的预估偏差比例，反映冲击成本。建议：0.0002–0.001。",
)
risk_per_trade = st.sidebar.number_input(
    "单笔风险占比", value=0.01, step=0.001, format="%.3f",
    help="允许单笔最大亏损占账户权益的比例，用于头寸规模计算。常见 0.5%–2%。",
)
min_notional = st.sidebar.number_input(
    "最小成交额（计价币）", value=10.0, step=1.0,
    help="订单的最小名义金额（计价币），避免因过小被交易所拒单。按交易所规则设置。",
)
poll_sec = st.sidebar.number_input(
    "轮询秒数", value=600, min_value=1, step=1,
    help="实盘循环与图表刷新间隔（秒）。设置为 1 即每秒刷新（受交易所限频影响）。",
)

st.sidebar.header("🧱 市场")
st.sidebar.text_input(
    "交易对", value=ss.get("symbol_input", "BTC/USDT"), key="symbol_input",
    help="如 BTC/USDT；必须与交易所符号一致。可用『快捷切换』按钮一键更改。",
)

timeframe = st.sidebar.selectbox(
    "周期",
    ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"], index=0,
    help="K线时间框，影响指标与信号节奏；回测与实盘保持一致。",
)
lookback_days = st.sidebar.number_input(
    "回测窗口（天）", value=90, min_value=2, step=1,
    help="下载用于回测的历史K线天数；越长越稳健但耗时更久。建议≥60天。",
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

connect_btn = st.sidebar.button("连接 / 重连")

# -----------------------------
# Tabs: Backtest | Live | Logs
# -----------------------------
tab_bt, tab_live, tab_logs = st.tabs(["📊 回测", "🚦 实盘交易", "🧾 日志"])

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
        st.sidebar.success("已通过 ccxt 连接 Gate.io ✅")
    except Exception as e:
        st.sidebar.error(f"连接失败: {e}")

# -----------------------------
# Backtest tab
# -----------------------------
with tab_bt:
    st.subheader("回测")
    run_bt = st.button("开始回测")
    if run_bt:
        g = ss.get("exchange")
        if not g:
            st.warning("请先连接交易所。")
        else:
            with st.spinner("正在下载K线并计算指标..."):
                eff_days = min(int(lookback_days), _max_days_for_tf(timeframe))
                if eff_days < int(lookback_days):
                    st.info(f"回测窗口已按交易所上限裁剪为 {eff_days} 天 ({timeframe} 最多约 10000 根K线).")
                df = fetch_history(g, ss["symbol_input"], timeframe, eff_days)
                df = compute_indicators(df, sc)
                df = generate_signals(df, sc)
                trades_df, results = backtest(df, sc, rc, starting_equity=10_000.0)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("笔数", results["trades"])
            k2.metric("总收益", f"{results['total_return_pct']:.2f}%")
            k3.metric("胜率", f"{results['win_rate_pct']:.2f}%")
            k4.metric("平均单笔盈亏", f"{results['avg_trade_pnl']:.2f}")
            k5.metric("最大回撤", f"{results['max_drawdown_pct']:.2f}%")

            price_fig = go.Figure()
            price_fig.add_trace(go.Candlestick(
                x=df["dt"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="K线",
            ))
            price_fig.add_trace(go.Scatter(x=df["dt"], y=df["ema_fast"], name=f"EMA {sc.fast_ema}", mode="lines"))
            price_fig.add_trace(go.Scatter(x=df["dt"], y=df["ema_slow"], name=f"EMA {sc.slow_ema}", mode="lines"))
            entries = df[df["long_signal"]]
            exits = df[df["exit_signal"]]
            price_fig.add_trace(go.Scatter(x=entries["dt"], y=entries["close"], mode="markers", name="开仓", marker=dict(symbol="triangle-up", size=9)))
            price_fig.add_trace(go.Scatter(x=exits["dt"], y=exits["close"], mode="markers", name="平仓", marker=dict(symbol="triangle-down", size=9)))
            price_fig.update_layout(height=520, xaxis_rangeslider_visible=False, title=f"{ss['symbol_input']} {timeframe}")
            st.plotly_chart(price_fig, use_container_width=True)

            st.write("交易列表")
            st.dataframe(trades_df, use_container_width=True)
            st.download_button(
                "下载交易CSV",
                trades_df.to_csv(index=False).encode(),
                file_name="trades_backtest.csv",
                mime="text/csv",
            )

# -----------------------------
# Live tab（常显：实时图 + 快捷切换 + 模拟资金）
# -----------------------------
with tab_live:
    st.subheader("实盘交易控制")

    # 全局自动刷新（无论是否启动，都可每秒更新行情/图表）
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=int(max(1, int(poll_sec)) * 1000), key="global_refresh")

    # 快捷切换交易对（回调整体应用，避免与 text_input 冲突）
    st.markdown("**快捷切换**（一键修改左侧交易对）")
    quick_pairs = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "DOGE/USDT", "TAO/USDT"]

    def _qset(p: str):
        ss["symbol_to_set"] = p

    qcols = st.columns(len(quick_pairs))
    for i, p in enumerate(quick_pairs):
        qcols[i].button(p, use_container_width=True, key=f"q_{p}", on_click=_qset, args=(p,))

    # 实时图设置
    live_window_min = st.number_input(
        "实时图窗口（分钟）", value=30, min_value=1, step=1,
        help="实时价格线显示最近多少分钟的走势。",
    )

    # 常显：行情 + 历史K线 + 图表控件（无需启动）
    ticker = None
    df_live = None
    g = ss.get("exchange")
    if g:
        try:
            ticker = g.fetch_ticker(ss["symbol_input"])  # 最新价
            if ticker.get("last") is not None:
                df_live = _push_live_price(ss["symbol_input"], float(ticker["last"]), window_min=int(live_window_min))
        except Exception as e:
            st.caption(f"拉取 {ss['symbol_input']} 行情失败：{e}")

        # 拉历史 + 指标
        eff_days_vis = min(max(int(lookback_days), 2), _max_days_for_tf(timeframe))
        df_hist = _fetch_history_cached(g, ss["symbol_input"], timeframe, eff_days_vis)
        df_hist = compute_indicators(df_hist, sc)
        df_hist = generate_signals(df_hist, sc)

        # 图表选项（带问号）
        st.markdown("#### 图表设置")
        cc1, cc2, cc3 = st.columns(3)
        show_candle = cc1.toggle("显示K线", True, help="以历史K线展示 OHLC 数据。")
        show_ema = cc2.toggle("叠加 EMA", True, help="在图上叠加 EMA 快/慢线，用于趋势研判。")
        show_live = cc3.toggle("叠加实时点", True, help="叠加逐秒最新价，便于观察最新波动。")

        c4, c5, c6 = st.columns(3)
        use_rangeslider = c4.toggle("启用时间滑条", True, help="显示底部范围滑条，可拖动缩放时间窗口。")
        manual_y = c5.toggle("手动Y轴范围", False, help="勾选后可指定价格轴最小/最大值。")
        dca_on = c6.toggle("DCA均线", False, help="用近 N 分钟收盘价的滚动均线模拟 DCA 成本线。")

        ymn,ymx,dca_box = st.columns(3)
        y_min = ymn.number_input("Y轴最小(0=自动)", value=0.0, help="设置为 0 表示自动。") if manual_y else None
        y_max = ymx.number_input("Y轴最大(0=自动)", value=0.0, help="设置为 0 表示自动。") if manual_y else None
        if manual_y:
            if y_min == 0.0: y_min = None
            if y_max == 0.0: y_max = None
        dca_minutes = dca_box.number_input("DCA窗口(分钟)", value=60, min_value=1, step=5,
                                           help="用于计算滚动均线的时间窗口大小。") if dca_on else 60

        st.markdown("#### 网格设置")
        gc1,gc2,gc3,gc4 = st.columns(4)
        grid_on = gc1.toggle("启用网格", False, help="以中心价为基准按步长%%画上下水平价位。")
        default_center = None
        if ticker and ticker.get("last") is not None:
            default_center = float(ticker["last"])
        elif not df_hist.empty:
            default_center = float(df_hist["close"].iloc[-1])
        grid_center = gc2.number_input("中心价", value=float(default_center or 0.0), format="%.6f",
                                        help="网格的中心参考价，默认取最新价/收盘价。") if grid_on else None
        grid_step_pct = gc3.number_input("步长(%)", value=0.5, step=0.05, help="相邻网格之间的百分比间隔。") if grid_on else 0.5
        grid_levels = gc4.number_input("层数(每侧)", value=4, min_value=1, step=1, help="向上/下各画多少层网格。") if grid_on else 4

        # 组合绘图
        _build_market_fig(
            df_hist=df_hist,
            df_live=df_live,
            timeframe=timeframe,
            title=f"{ss['symbol_input']} 市场图",
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
        st.caption("还未连接到交易所，无法拉取实时价格与历史K线。")

    # ========= 模拟资金（自动 & 手动） =========
    st.markdown("---")
    st.markdown("### 🧪 模拟资金（本地记账，不与交易所交互）")

    colp1, colp2, colp3 = st.columns([1, 1, 1])
    use_paper_wallet = colp1.toggle("开启模拟资金", value=True, help="使用真实行情在本地进行资金变化模拟，不会下真实单。")
    auto_trade_sim = colp2.toggle("自动按策略交易", value=True, help="用 EMA/RSI/ATR 信号在模拟资金里开/平仓（不受最小成交额限制）。")
    paper_start_equity = colp3.number_input("初始资金 (USDT)", value=10_000.0, step=100.0)

    alloc_pct = st.slider(
        "每笔投入比例", min_value=0.05, max_value=1.0, value=0.10, step=0.05,
        help="信号出现时，按该比例使用当前可用资金买入；平仓时全量卖出。",
    )

    # 初始化/切换品种时重置账本容器（不自动买卖）
    if use_paper_wallet and (ss.paper is None or ss.paper.get("symbol_bound") != ss["symbol_input"]):
        ss.paper = {"cash": float(paper_start_equity), "base": 0.0, "avg": 0.0, "symbol_bound": ss["symbol_input"]}
        ss.equity_series = pd.DataFrame(columns=["dt", "equity"])  # 清空曲线
        ss.sim_prev_long_cond = None

    # 自动策略（只在开启模拟资金&自动交易时触发）
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
            st.warning(f"自动模拟失败：{e}")

    # 手动操作按钮
    colm1, colm2, colm3, colm4 = st.columns([1, 1, 1, 2])
    if colm1.button("模拟买入 10%") and use_paper_wallet and ss.paper is not None and ticker and ticker.get("last") is not None:
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
    if colm2.button("全部卖出") and use_paper_wallet and ss.paper is not None and ticker and ticker.get("last") is not None:
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
    if colm3.button("重置模拟资金") and use_paper_wallet:
        ss.paper = {"cash": float(paper_start_equity), "base": 0.0, "avg": 0.0, "symbol_bound": ss["symbol_input"]}
        ss.equity_series = pd.DataFrame(columns=["dt", "equity"])  # 清空曲线
        ss.sim_prev_long_cond = None
        st.toast("模拟资金已重置。", icon="↩️")

    # 更新并显示模拟资金权益
    if use_paper_wallet and ss.paper is not None and ticker and ticker.get("last") is not None:
        price_now = float(ticker["last"]) if ticker.get("last") is not None else 0.0
        equity = ss.paper["cash"] + ss.paper["base"] * price_now
        row = pd.DataFrame([[pd.Timestamp.utcnow(), equity]], columns=["dt", "equity"])  # type: ignore
        ss.equity_series = pd.concat([ss.equity_series, row], ignore_index=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=int(live_window_min))
        ss.equity_series = ss.equity_series[ss.equity_series["dt"] >= cutoff]

        e1, e2, e3 = st.columns(3)
        e1.metric("模拟资金权益(USDT)", f"{equity:,.2f}")
        e2.metric("持仓数量", f"{ss.paper['base']:.6f}")
        upnl = 0.0
        if ss.paper["base"] > 0 and ss.paper["avg"] > 0:
            upnl = (price_now - ss.paper["avg"]) / ss.paper["avg"] * 100
        e3.metric("未实现盈亏%", f"{upnl:.2f}%")
        _plot_equity(ss.equity_series, title="模拟资金权益曲线")

    # ========= 实盘下单（可选） =========
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    paper_mode = col1.toggle("模拟盘(真实下单开关)", value=True, help="开启=不发真实单，仅本地模拟；关闭+风险确认=才会发单。")
    live_ack = col2.toggle("我已了解风险", value=False)
    start_button = col3.button("启动 / 应用", type="primary")
    stop_button = col3.button("停止")

    if start_button:
        g = ss.get("exchange")
        if not g:
            st.warning("请先在左侧连接。")
        else:
            rt.dry_run = not (live_ack and not paper_mode)
            rt.symbol = ss["symbol_input"]
            ss.trader = LiveTrader(g, sc, rc, rt)
            ss.running = True
            st.toast("已启动实时循环（下方『最近动作』将更新）。", icon="✅")

    if stop_button:
        ss.running = False
        ss.trader = None
        st.toast("已停止。", icon="🛑")

    if ss.running and ss.trader is not None:
        with st.spinner("轮询中..."):
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
        st.write("**最近动作：**", info.get("action") or "—")
        if info.get("entry"):
            st.write("开仓:", info["entry"])
        if info.get("exit"):
            st.write("平仓:", info["exit"])
        if info.get("message"):
            st.info(info["message"])  # trading_core 中的提示（英文）

# -----------------------------
# Logs tab
# -----------------------------
with tab_logs:
    st.subheader("会话日志")
    if ss.log_rows:
        log_df = pd.DataFrame(ss.log_rows)
        st.dataframe(log_df, use_container_width=True, height=320)
        st.download_button(
            "下载日志CSV",
            log_df.to_csv(index=False).encode(),
            file_name="session_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("暂无动作。")

st.caption("提示：将密钥放入 `.streamlit/secrets.toml`，键名为 `GATEIO_API_KEY` 与 `GATEIO_SECRET`，更安全。")
