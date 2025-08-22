# trading_core.py
import os
import time
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import ccxt


# =========================
# Utils
# =========================
def utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def timeframe_to_ms(tf: str) -> int:
    n = int(tf[:-1])
    unit = tf[-1].lower()
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 60 * 60_000
    if unit == "d":
        return n * 24 * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")


def to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


# =========================
# Indicators
# =========================
def ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if period <= 0:
        return out
    alpha = 2 / (period + 1)
    s = None
    for i, v in enumerate(values):
        if np.isnan(v):
            continue
        s = v if s is None else alpha * v + (1 - alpha) * s
        out[i] = s
    return out


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss != 0)
    return 100 - (100 / (1 + rs))


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    return ema(tr, period)


# =========================
# Configs
# =========================
@dataclass
class StrategyConfig:
    fast_ema: int = 20
    slow_ema: int = 50
    rsi_period: int = 14
    rsi_long: float = 55.0
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 3.0


@dataclass
class RiskConfig:
    fee_rate: float = 0.002      # 0.20% taker per side (adjust to your tier)
    slippage: float = 0.0005     # 5 bps
    risk_per_trade: float = 0.01 # risk 1% of equity per trade
    min_notional: float = 10.0   # overridden by exchange rule if higher


@dataclass
class RuntimeConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "5m"
    poll_sec: int = 5
    dry_run: bool = True


# =========================
# Exchange wrapper
# =========================
class GateIO:
    def __init__(self, api_key: str | None = None, api_secret: str | None = None, default_type: str = "spot"):
        load_dotenv()
        api_key = api_key or os.getenv("GATEIO_API_KEY", "")
        api_secret = api_secret or os.getenv("GATEIO_SECRET", "")

        self.exchange = ccxt.gateio({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": default_type},
            "timeout": 20000,
        })
        self.markets = self.exchange.load_markets()

    def market(self, symbol: str) -> dict:
        return self.exchange.market(symbol)

    def fetch_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        return self.exchange.fetch_balance()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, since: int | None = None) -> list:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)

    def create_order(self, symbol: str, type_: str, side: str, amount: float, price: float | None = None, params: dict | None = None):
        params = params or {}
        # ccxt signature: create_order(symbol, type, side, amount, price=None, params={})
        return self.exchange.create_order(symbol, type_, side, amount, price, params)

    def cancel_all(self, symbol: str):
        try:
            for o in self.exchange.fetch_open_orders(symbol):
                try:
                    self.exchange.cancel_order(o["id"], symbol)
                except Exception as e:
                    print("[cancel error]", e)
        except Exception as e:
            print("[cancel_all error]", e)

    @staticmethod
    def round_amount(mkt: dict, amount: float) -> float:
        precision = mkt.get("precision", {}).get("amount")
        if precision is not None:
            amount = float(f"{amount:.{precision}f}")
        min_amt = (mkt.get("limits", {}).get("amount", {}) or {}).get("min")
        if min_amt is not None:
            amount = max(amount, float(min_amt))
        return amount

    @staticmethod
    def round_price(mkt: dict, price: float) -> float:
        precision = mkt.get("precision", {}).get("price")
        if precision is not None:
            price = float(f"{price:.{precision}f}")
        return price

    @staticmethod
    def min_cost(mkt: dict) -> float:
        return float((mkt.get("limits", {}).get("cost", {}) or {}).get("min", 0.0))


# =========================
# Strategy
# =========================
def compute_indicators(df: pd.DataFrame, sc: StrategyConfig) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    df["ema_fast"] = ema(close, sc.fast_ema)
    df["ema_slow"] = ema(close, sc.slow_ema)
    df["rsi"] = rsi(close, sc.rsi_period)
    df["atr"] = atr(high, low, close, sc.atr_period)
    return df


def generate_signals(df: pd.DataFrame, sc: StrategyConfig) -> pd.DataFrame:
    df["long_cond"] = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] >= sc.rsi_long)
    df["long_signal"] = df["long_cond"] & (~df["long_cond"].shift(1).fillna(False))
    df["exit_signal"] = (~df["long_cond"]) & (df["long_cond"].shift(1).fillna(False))
    return df


# =========================
# History fetch
# =========================
def fetch_history(g: GateIO, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    ms_tf = timeframe_to_ms(timeframe)
    now_ms = utc_ms()
    since = now_ms - days * 24 * 60 * 60_000
    out = []
    while since < now_ms:
        batch = g.fetch_ohlcv(symbol, timeframe, limit=1000, since=since)
        if not batch:
            break
        out.extend(batch)
        since = batch[-1][0] + ms_tf
        # be polite
        time.sleep(g.exchange.rateLimit / 1000)
    df = to_df(out)
    # drop last partial candle
    cutoff = now_ms - ms_tf
    df = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    return df


# =========================
# Backtester
# =========================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    size: float
    stop_price: float
    take_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None


def backtest(df: pd.DataFrame, sc: StrategyConfig, rc: RiskConfig, starting_equity: float = 10_000.0):
    equity = starting_equity
    trades: List[Trade] = []
    in_pos = False
    current: Optional[Trade] = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # require indicators
        if any(np.isnan(row[k]) for k in ["ema_fast", "ema_slow", "rsi", "atr"]):
            continue

        if not in_pos and row["long_signal"]:
            entry = row["close"] * (1 + rc.slippage)
            stop = entry - sc.atr_sl_mult * row["atr"]
            take = entry + sc.atr_tp_mult * row["atr"]
            if stop <= 0:
                continue
            risk_amount = equity * rc.risk_per_trade
            stop_dist = entry - stop
            size = max(risk_amount / stop_dist, 0.0)
            notional = size * entry
            fee_cost = notional * rc.fee_rate
            if notional < rc.min_notional:
                continue
            equity -= fee_cost
            current = Trade(entry_time=row["dt"], entry_price=entry, size=size, stop_price=stop, take_price=take)
            in_pos = True
            continue

        if in_pos and current:
            hit_sl = row["low"] <= current.stop_price
            hit_tp = row["high"] >= current.take_price
            exit_price = None
            if hit_sl and hit_tp:
                exit_price = current.stop_price * (1 - rc.slippage)  # conservative
            elif hit_sl:
                exit_price = current.stop_price * (1 - rc.slippage)
            elif hit_tp:
                exit_price = current.take_price * (1 - rc.slippage)
            elif row["exit_signal"]:
                exit_price = row["close"] * (1 - rc.slippage)

            if exit_price is not None:
                pnl = (exit_price - current.entry_price) * current.size
                fees = (current.entry_price * current.size + exit_price * current.size) * rc.fee_rate
                pnl_after_fees = pnl - fees
                equity += pnl_after_fees
                current.exit_time = row["dt"]
                current.exit_price = exit_price
                current.pnl = pnl_after_fees
                trades.append(current)
                current = None
                in_pos = False

    if in_pos and current:
        last = df.iloc[-1]
        exit_price = last["close"]
        pnl = (exit_price - current.entry_price) * current.size
        fees = (current.entry_price * current.size + exit_price * current.size) * rc.fee_rate
        pnl_after_fees = pnl - fees
        current.exit_time = last["dt"]
        current.exit_price = exit_price
        current.pnl = pnl_after_fees
        trades.append(current)

    pnl_series = pd.Series([t.pnl for t in trades], dtype=float)
    curve = [starting_equity]
    eq = starting_equity
    for t in trades:
        eq += t.pnl or 0.0
        curve.append(eq)
    curve = np.array(curve, dtype=float)
    roll_max = np.maximum.accumulate(curve)
    dd = (curve - roll_max) / roll_max
    max_dd = -dd.min() if len(dd) else 0.0

    results = {
        "starting_equity": starting_equity,
        "ending_equity": float(eq),
        "total_return_pct": float((eq - starting_equity) / starting_equity * 100),
        "trades": int(len(trades)),
        "win_rate_pct": float((pnl_series > 0).mean() * 100) if len(pnl_series) else 0.0,
        "avg_trade_pnl": float(pnl_series.mean()) if len(pnl_series) else 0.0,
        "max_drawdown_pct": float(max_dd * 100),
    }

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    return trades_df, results


# =========================
# Live Trader (single step loop for Streamlit auto-refresh)
# =========================
class LiveTrader:
    def __init__(self, g: GateIO, sc: StrategyConfig, rc: RiskConfig, rt: RuntimeConfig):
        self.g = g
        self.sc = sc
        self.rc = rc
        self.rt = rt
        self.mkt = g.market(rt.symbol)
        # Ensure min notional uses exchange rule if higher
        ex_min_cost = GateIO.min_cost(self.mkt)
        if ex_min_cost and ex_min_cost > self.rc.min_notional:
            self.rc.min_notional = ex_min_cost

        self.open_position: Optional[Trade] = None
        self.last_candle_ts: Optional[int] = None

    def _equity_quote(self) -> float:
        bal = self.g.fetch_balance()
        quote = self.rt.symbol.split("/")[1]
        data = bal.get(quote) or {}
        total = float(data.get("total", data.get("free", 0.0)) or 0.0)
        return total

    def _latest_closed_df(self, limit=400) -> pd.DataFrame:
        ms_tf = timeframe_to_ms(self.rt.timeframe)
        now_ms = utc_ms()
        cutoff = now_ms - ms_tf
        candles = self.g.fetch_ohlcv(self.rt.symbol, self.rt.timeframe, limit=limit)
        df = to_df(candles)
        df = df[df["timestamp"] <= cutoff].reset_index(drop=True)
        return df

    def step(self) -> dict:
        """
        Executes one evaluation step (called on every Streamlit refresh).
        Returns a dict of debug/status info.
        """
        info = {"action": None, "message": "", "entry": None, "exit": None}

        df = self._latest_closed_df()
        df = compute_indicators(df, self.sc)
        df = generate_signals(df, self.sc)
        last = df.iloc[-1]

        # OPEN
        if self.open_position is None and last["long_signal"] and not any(
            np.isnan(last[k]) for k in ["ema_fast", "ema_slow", "rsi", "atr"]
        ):
            ticker = self.g.fetch_ticker(self.rt.symbol)
            bid = float(ticker.get("bid") or last["close"])
            entry = bid * (1 + self.rc.slippage)
            stop = entry - self.sc.atr_sl_mult * last["atr"]
            take = entry + self.sc.atr_tp_mult * last["atr"]
            if stop > 0:
                equity = self._equity_quote()
                risk_amount = equity * self.rc.risk_per_trade
                stop_dist = entry - stop
                size = max(risk_amount / stop_dist, 0.0)
                notional = size * entry
                if notional >= self.rc.min_notional:
                    size = GateIO.round_amount(self.mkt, size)
                    entry = GateIO.round_price(self.mkt, entry)
                    if self.rt.dry_run:
                        self.open_position = Trade(last["dt"], entry, size, stop, take)
                        info["action"] = "PAPER BUY"
                        info["entry"] = {"price": entry, "size": size}
                    else:
                        order = self.g.create_order(self.rt.symbol, "market", "buy", size, None, {})
                        fill = float(order.get("average") or entry)
                        self.open_position = Trade(last["dt"], fill, size, stop, take)
                        info["action"] = "LIVE BUY"
                        info["entry"] = {"price": fill, "size": size}
                else:
                    info["message"] = f"Notional {notional:.2f} < min {self.rc.min_notional:.2f}"

        # CLOSE
        if self.open_position is not None:
            ticker = self.g.fetch_ticker(self.rt.symbol)
            ask = float(ticker.get("ask") or last["close"])
            px = ask
            hit_sl = px <= self.open_position.stop_price
            hit_tp = px >= self.open_position.take_price
            exit_signal = last["exit_signal"]
            do_exit = hit_sl or hit_tp or exit_signal
            if do_exit:
                exit_px = GateIO.round_price(self.mkt, px * (1 - self.rc.slippage))
                size = self.open_position.size
                if self.rt.dry_run:
                    pnl = (exit_px - self.open_position.entry_price) * size
                    fees = (self.open_position.entry_price * size + exit_px * size) * self.rc.fee_rate
                    pnl_after_fees = pnl - fees
                    info["action"] = (info["action"] + " & " if info["action"] else "") + "PAPER SELL"
                    info["exit"] = {"price": exit_px, "pnl_after_fees": pnl_after_fees}
                    self.open_position = None
                else:
                    order = self.g.create_order(self.rt.symbol, "market", "sell", size, None, {})
                    fill = float(order.get("average") or exit_px)
                    pnl = (fill - self.open_position.entry_price) * size
                    fees = (self.open_position.entry_price * size + fill * size) * self.rc.fee_rate
                    pnl_after_fees = pnl - fees
                    info["action"] = (info["action"] + " & " if info["action"] else "") + "LIVE SELL"
                    info["exit"] = {"price": fill, "pnl_after_fees": pnl_after_fees}
                    self.open_position = None

        return info
