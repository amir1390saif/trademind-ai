"""
TradeMind AI - Educational market analysis prototype
Markets: Tehran Stock Exchange and Forex
No brokerage connection. No real-money trading.

Input CSV columns (case-insensitive):
Date, Open, High, Low, Close, Volume
Date is optional. At least OHLC columns are required.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "TradeMind AI"
REQUIRED = ["open", "high", "low", "close"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "adj close": "close",
        "adj_close": "close",
        "vol": "volume",
        "timestamp": "date",
        "datetime": "date",
    }
    df = df.rename(columns=aliases)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "ستون‌های لازم پیدا نشدند: "
            + ", ".join(missing)
            + ". ستون‌های لازم: Open, High, Low, Close"
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.set_index("date")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED)
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    if len(df) < 60:
        raise ValueError("برای تحلیل قابل‌قبول، حداقل ۶۰ ردیف داده لازم است.")
    return df


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()
    out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()
    out["rsi14"] = rsi(out["close"], 14)

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    return out


@dataclass
class Signal:
    label: str
    score: int
    reasons: list[str]
    entry: float
    stop: float
    target: float
    risk_pct: float


def generate_signal(df: pd.DataFrame) -> Signal:
    last = df.iloc[-1]
    score = 0
    reasons = []

    if last["ema20"] > last["ema50"]:
        score += 1
        reasons.append("EMA20 بالاتر از EMA50 است (روند کوتاه‌مدت مثبت).")
    else:
        score -= 1
        reasons.append("EMA20 پایین‌تر از EMA50 است (روند کوتاه‌مدت منفی).")

    if last["close"] > last["ema200"]:
        score += 1
        reasons.append("قیمت بالاتر از EMA200 قرار دارد.")
    else:
        score -= 1
        reasons.append("قیمت پایین‌تر از EMA200 قرار دارد.")

    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD بالاتر از خط سیگنال است.")
    else:
        score -= 1
        reasons.append("MACD پایین‌تر از خط سیگنال است.")

    if 50 <= last["rsi14"] <= 68:
        score += 1
        reasons.append("RSI در محدوده نسبتاً مثبت و نه بیش‌خرید قرار دارد.")
    elif last["rsi14"] < 35:
        reasons.append("RSI پایین است؛ احتمال برگشت وجود دارد اما تأیید لازم است.")
    elif last["rsi14"] > 70:
        score -= 1
        reasons.append("RSI بالاست؛ احتمال بیش‌خرید و اصلاح وجود دارد.")

    close = float(last["close"])
    atr = float(last["atr14"]) if pd.notna(last["atr14"]) else close * 0.02
    atr = max(atr, close * 0.001)

    # Educational levels only; not personalized investment advice.
    entry = close
    stop = max(0.0, entry - 1.5 * atr)
    target = entry + 2.0 * (entry - stop)
    risk_pct = ((entry - stop) / entry) * 100 if entry else 0

    if score >= 3:
        label = "بررسی فرصت صعودی"
    elif score <= -2:
        label = "هشدار روند نزولی"
    else:
        label = "انتظار / سیگنال نامشخص"

    return Signal(label, score, reasons, entry, stop, target, risk_pct)


def simple_backtest(df: pd.DataFrame) -> dict:
    """
    Illustrative long-only EMA crossover test.
    Signal is shifted by one bar to avoid using today's close to trade at
    today's close. Fees/slippage are configurable in the UI.
    """
    x = df.copy()
    x["raw_signal"] = (x["ema20"] > x["ema50"]).astype(int)
    x["position"] = x["raw_signal"].shift(1).fillna(0)
    x["market_return"] = x["close"].pct_change().fillna(0)
    x["strategy_return"] = x["position"] * x["market_return"]
    x["equity"] = (1 + x["strategy_return"]).cumprod()
    total = (x["equity"].iloc[-1] - 1) * 100
    peak = x["equity"].cummax()
    drawdown = (x["equity"] / peak - 1)
    max_dd = drawdown.min() * 100
    active = x["position"].diff().abs().fillna(0)
    trades = int((active > 0).sum())
    wins = int((x.loc[x["strategy_return"] > 0, "strategy_return"]).count())
    return {
        "total_return": total,
        "max_drawdown": max_dd,
        "trades": trades,
        "positive_days": wins,
        "data": x,
    }


def format_num(value: float) -> str:
    return f"{value:,.4f}"


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.title("📈 TradeMind AI")
st.caption("نمونه اولیه آموزشی برای تحلیل بورس تهران و فارکس | بدون اتصال به کارگزاری")

with st.sidebar:
    st.header("تنظیمات")
    market = st.selectbox("بازار", ["بورس تهران", "فارکس"])
    symbol = st.text_input(
        "نام نماد / جفت‌ارز",
        value="فولاد" if market == "بورس تهران" else "EURUSD",
    )
    st.info("برای شروع، فایل CSV قیمت را بارگذاری کن. داده‌ها باید شامل OHLC باشند.")

st.warning(
    "این ابزار آموزشی است، نه مشاور سرمایه‌گذاری. سیگنال‌ها تضمینی نیستند. "
    "از خروجی برای معامله واقعی یا تصمیم مالی بدون بررسی مستقل استفاده نکن."
)

uploaded = st.file_uploader("بارگذاری فایل قیمت CSV", type=["csv"])

with st.expander("نمونه قالب فایل CSV"):
    st.code(
        "Date,Open,High,Low,Close,Volume\n"
        "2026-01-01,100,105,98,103,120000\n"
        "2026-01-02,103,108,101,106,150000",
        language="csv",
    )

if uploaded is None:
    st.markdown("### شروع سریع")
    st.write(
        "یک فایل CSV از داده‌های تاریخی نماد یا جفت‌ارز انتخاب کن. "
        "برای داده‌های بورس تهران، قیمت‌ها باید با تعدیل موردنظر خودت یکدست باشند. "
        "برای فارکس، تایم‌فریم و منطقه زمانی را در کل فایل ثابت نگه دار."
    )
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    data = normalize_columns(raw)
    data = add_indicators(data)
    signal = generate_signal(data)
    bt = simple_backtest(data)
except Exception as exc:
    st.error(f"خطا در پردازش فایل: {exc}")
    st.stop()

st.subheader(f"تحلیل {symbol} | {market}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("آخرین قیمت", format_num(float(data["close"].iloc[-1])))
c2.metric("RSI (14)", f"{data['rsi14'].iloc[-1]:.1f}")
c3.metric("MACD", format_num(float(data["macd"].iloc[-1])))
c4.metric("ATR (14)", format_num(float(data["atr14"].iloc[-1])))

st.markdown("### نتیجه موتور تحلیل")
st.subheader(signal.label)
st.write(f"امتیاز جهت‌دار: **{signal.score} از ۴**")
for reason in signal.reasons:
    st.write(f"- {reason}")

p1, p2, p3, p4 = st.columns(4)
p1.metric("قیمت مبنای فرضی", format_num(signal.entry))
p2.metric("سطح توقف فرضی", format_num(signal.stop))
p3.metric("هدف فرضی", format_num(signal.target))
p4.metric("فاصله ریسک تا توقف", f"{signal.risk_pct:.2f}%")

st.caption(
    "سطوح بالا صرفاً نمونه محاسباتی بر اساس ATR هستند؛ "
    "کارمزد، شکاف قیمت، نقدشوندگی و محدودیت‌های بازار در آن‌ها لحاظ نشده است."
)

st.markdown("### نمودار قیمت و میانگین‌ها")
chart_cols = ["close", "ema20", "ema50", "ema200"]
st.line_chart(data[chart_cols].tail(250))

st.markdown("### آزمایش تاریخی ساده")
st.write(
    "این آزمایش فقط یک تقاطع EMA20/EMA50 را بررسی می‌کند. "
    "نتایج گذشته پیش‌بینی آینده نیستند و در این نسخه کارمزد و لغزش قیمت لحاظ نشده‌اند."
)
b1, b2, b3 = st.columns(3)
b1.metric("بازده تجمعی آزمایشی", f"{bt['total_return']:.2f}%")
b2.metric("افت سرمایه بیشینه", f"{bt['max_drawdown']:.2f}%")
b3.metric("تعداد تغییر موقعیت", str(bt["trades"]))
st.line_chart(bt["data"][["equity"]].tail(500))

st.markdown("### داده‌های آخر")
show_cols = ["open", "high", "low", "close"]
if "volume" in data.columns:
    show_cols.append("volume")
show_cols += ["rsi14", "ema20", "ema50", "macd"]
st.dataframe(data[show_cols].tail(20), use_container_width=True)

csv_out = data.to_csv(index=True).encode("utf-8-sig")
st.download_button(
    "دانلود داده‌های تحلیل‌شده",
    data=csv_out,
    file_name=f"{symbol}_analysis.csv",
    mime="text/csv",
)
