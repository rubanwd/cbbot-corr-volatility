#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Scanner Bot — динамический/статический универс + кастомные фильтры
- Переключение через .env:
    DYNAMIC_UNIVERSE_ENABLED=true|false
- В динамическом режиме можно включить/выключить фильтр «разрешённых/запрещённых» активов:
    DYN_SYMBOL_FILTERS_ENABLED=true|false
    DYN_ALLOWED_BASES=BTC,ETH,BNB          # БАЗОВЫЕ тикеры (до /USDT)
    DYN_BLOCKED_BASES=PEPE,DOGE
    DYN_ALLOWED_SYMBOLS=BTC/USDT,ETH/USDT  # Полные символы "BASE/USDT"
    DYN_BLOCKED_SYMBOLS=SOME/USDT
- Остальные фильтры (объём/ATR/стакан/корр. и опц. капа) — отдельные тумблеры.
- Сохраняет выбранные символы в data/symbols_dynamic_YYYYMMDD_HHMMSS.txt (+latest)
- Считает корреляции, RSI, MACD; пишет снапшот; шлёт телеграм-отчёт.
"""

from __future__ import annotations

import os
import json
import time
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import math
import numpy as np
import pandas as pd
import schedule

# ── .env загрузка (необязательно)
def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
        if os.path.exists(".env"):
            load_dotenv()
    except Exception:
        pass

_load_dotenv_if_available()

try:
    import ccxt
except Exception as e:
    raise SystemExit("ccxt is required. Install with: pip install ccxt") from e

try:
    import requests
except Exception:
    requests = None  # CoinGecko фильтр будет выключен, если requests недоступен

# Telegram-компонент
try:
    from telegram_report import send_report_once as tg_send_report_once
except Exception:
    tg_send_report_once = None

# SciPy (не обязателен; если нет — используем pandas corr)
try:
    from scipy.stats import pearsonr, ConstantInputWarning  # type: ignore
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    SCIPY_OK = True
except Exception:
    pearsonr = None
    SCIPY_OK = False


# ───────────────────────── Helpers: env parsing ─────────────────────────

def _get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v not in (None, "") else default
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v not in (None, "") else default
    except Exception:
        return default

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v2 = str(v).strip().lower()
    return v2 in ("1", "true", "t", "yes", "y", "on")

def _parse_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

# ───────────────────────── Core config ─────────────────────────

# Переключатель динамики/статики
DYNAMIC_UNIVERSE_ENABLED: bool = _get_bool("DYNAMIC_UNIVERSE_ENABLED", True)

# Для статического режима
SYMBOLS_FILE: Optional[str]   = _get_str("SYMBOLS_FILE", "symbols.txt")
STATIC_SYMBOLS_CSV: Optional[str] = _get_str("STATIC_SYMBOLS", None)  # "BTC/USDT,ETH/USDT"

USE_FUTURES: bool = _get_bool("USE_FUTURES", True)
TIMEFRAME: str = _get_str("TIMEFRAME", "1h") or "1h"
LOOKBACK_BARS: int = _get_int("LOOKBACK_BARS", 500)

MIN_OBS_PER_SYMBOL: int = _get_int("MIN_OBS_PER_SYMBOL", 50)
MIN_OBS_PER_PAIR: int = _get_int("MIN_OBS_PER_PAIR", 30)

RSI_PERIOD: int = _get_int("RSI_PERIOD", 14)
WEAK_CORR_ABS_THRESH: float = _get_float("WEAK_CORR_ABS_THRESH", 0.30)

OUTPUT_DIR: str = _get_str("OUTPUT_DIR", "./data") or "./data"
JSON_SNAPSHOTS_FILE: str = os.path.join(OUTPUT_DIR, "correlation_snapshots.jsonl")
LATEST_JSON_FILE: str = os.path.join(OUTPUT_DIR, "latest_correlation.json")
LOG_FILE: str = os.path.join(OUTPUT_DIR, "correlation_bot.log")

POSITION_CASES_JSONL: str = os.path.join(OUTPUT_DIR, "position_entry_cases.jsonl")
POSITION_CASES_JSON: str  = os.path.join(OUTPUT_DIR, "position_entry_cases.json")

ALERT_NEGATIVE_THRESH: float = _get_float("ALERT_NEGATIVE_THRESH", -0.30)
ALERT_NEAR_ZERO_ABS: float = _get_float("ALERT_NEAR_ZERO_ABS", 0.10)
TOP_N_TO_LOG: int = _get_int("TOP_N_TO_LOG", 50)
TOP_N_TO_PERSIST_RAW: int = _get_int("TOP_N_TO_PERSIST", 200)
TOP_N_TO_PERSIST: Optional[int] = None if TOP_N_TO_PERSIST_RAW <= 0 else TOP_N_TO_PERSIST_RAW

RUN_EVERY_MINUTES: int = _get_int("RUN_EVERY_MINUTES", 1)
HOURLY_AT: Optional[str] = _get_str("HOURLY_AT", ":01")
HOURLY_AT = None if (HOURLY_AT or "").strip() == "" else HOURLY_AT
RUN_ONCE: bool = _get_bool("RUN_ONCE", False)

TELEGRAM_ENABLED: bool = _get_bool("TELEGRAM_ENABLED", False)
TELEGRAM_HOURLY_AT: str = _get_str("TELEGRAM_HOURLY_AT", ":05") or ":05"
TELEGRAM_MAX_ROWS: int = _get_int("TELEGRAM_MAX_ROWS", 100)
TELEGRAM_SEND_ON_START: bool = _get_bool("TELEGRAM_SEND_ON_START", True)
TELEGRAM_BOT_TOKEN: Optional[str] = _get_str("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_CHAT_ID: Optional[str] = _get_str("TELEGRAM_CHAT_ID", None)

# Индикаторы для отчёта
INDICATOR_TFS: List[str] = [t.strip() for t in (_get_str("INDICATOR_TFS", "15m,1h,1d") or "15m,1h,1d").split(",") if t.strip()]
INDICATOR_MAX_PAIRS: int = _get_int("INDICATOR_MAX_PAIRS", 30)
INDICATOR_BARS: int = _get_int("INDICATOR_BARS", 200)
MACD_FAST: int = _get_int("MACD_FAST", 12)
MACD_SLOW: int = _get_int("MACD_SLOW", 26)
MACD_SIGNAL: int = _get_int("MACD_SIGNAL", 9)
BB_PERIOD: int = _get_int("BB_PERIOD", 20)
BB_MULT: float = _get_float("BB_MULT", 2.0)

# Динамический отбор тикеров — параметры + тумблеры
# 0) Символьные фильтры (allow/block)
DYN_SYMBOL_FILTERS_ENABLED: bool = _get_bool("DYN_SYMBOL_FILTERS_ENABLED", False)
DYN_ALLOWED_BASES:   List[str] = [x.upper() for x in _parse_csv(_get_str("DYN_ALLOWED_BASES", ""))]
DYN_BLOCKED_BASES:   List[str] = [x.upper() for x in _parse_csv(_get_str("DYN_BLOCKED_BASES", ""))]
DYN_ALLOWED_SYMBOLS: List[str] = [x.upper() for x in _parse_csv(_get_str("DYN_ALLOWED_SYMBOLS", ""))]
DYN_BLOCKED_SYMBOLS: List[str] = [x.upper() for x in _parse_csv(_get_str("DYN_BLOCKED_SYMBOLS", ""))]

# 1) Ликвидность (объём)
DYN_FILTER_VOLUME_ENABLED: bool = _get_bool("DYN_FILTER_VOLUME_ENABLED", True)
MIN_DAILY_QUOTE_VOLUME_USD: float = _get_float("MIN_DAILY_QUOTE_VOLUME_USD", 100_000_000.0)
DYN_PRESELECT_TOP_N_BY_VOLUME: int = _get_int("DYN_PRESELECT_TOP_N_BY_VOLUME", 300)

# 2) (опц.) капа через CoinGecko
CAP_FILTER_ENABLED: bool = _get_bool("CAP_FILTER_ENABLED", False)
MIN_MARKET_CAP_USD: float = _get_float("MIN_MARKET_CAP_USD", 500_000_000.0)

# 3) Волатильность (ATR%)
DYN_FILTER_ATR_ENABLED: bool = _get_bool("DYN_FILTER_ATR_ENABLED", True)
ATR_PERCENT_MIN: float = _get_float("ATR_PERCENT_MIN", 5.0)  # ATR14% 1D
ATR_PERIOD: int = _get_int("ATR_PERIOD", 14)
ATR_LOOKBACK_DAYS: int = _get_int("ATR_LOOKBACK_DAYS", 30)

# 4) Стакан (дырки)
DYN_FILTER_ORDERBOOK_ENABLED: bool = _get_bool("DYN_FILTER_ORDERBOOK_ENABLED", True)
ORDERBOOK_TOP_LEVELS: int = _get_int("ORDERBOOK_TOP_LEVELS", 20)
ORDERBOOK_HOLE_PCT: float = _get_float("ORDERBOOK_HOLE_PCT", 0.01)  # 1%

# 5) Корреляция с BTC/ETH
DYN_FILTER_CORR_ENABLED: bool = _get_bool("DYN_FILTER_CORR_ENABLED", True)
CORR_TF: str = _get_str("CORR_TF", "1h") or "1h"
CORR_LOOKBACK_BARS: int = _get_int("CORR_LOOKBACK_BARS", 200)
CORR_WITH_BTC_ETH_MAX: float = _get_float("CORR_WITH_BTC_ETH_MAX", 0.60)


# ───────────────────────── Utils ─────────────────────────

def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("corr_bot")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_exchange():
    # Futures by default
    cfg = dict(enableRateLimit=True, options={"defaultType": "future" if USE_FUTURES else "spot"})
    ex = ccxt.binance(cfg)
    ex.load_markets()
    return ex

def load_symbols_static() -> List[str]:
    # 1) если задан STATIC_SYMBOLS (CSV), берём его
    if STATIC_SYMBOLS_CSV:
        syms = [x for x in _parse_csv(STATIC_SYMBOLS_CSV)]
        if syms:
            return syms
    # 2) иначе читаем файл
    path = SYMBOLS_FILE or "symbols.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            syms = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if syms:
            return syms
    raise SystemExit("Статический режим: не найден список. Задай STATIC_SYMBOLS или symbols.txt")


# ───────────────────────── Data fetch ─────────────────────────

def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")
    except Exception:
        return None

def fetch_close_series(exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.Series]:
    df = fetch_ohlcv_df(exchange, symbol, timeframe, limit)
    if df is None:
        return None
    return df["close"].astype(float)

def fetch_closes(exchange, symbols: List[str], timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    all_series: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            s = fetch_close_series(exchange, sym, timeframe, limit)
            if s is None or s.empty:
                logger.warning("%s: empty OHLCV", sym); continue
            all_series[sym] = s
        except Exception as e:
            logger.warning("%s: fetch_ohlcv failed: %s", sym, e); continue
    if not all_series:
        raise RuntimeError("Нет OHLCV ни по одному символу.")
    return pd.concat(all_series, axis=1).sort_index()


# ───────────────────────── Indicators ─────────────────────────

def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def compute_bbands_series(close: pd.Series, period: int = 20, mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + mult * std
    lower = mid - mult * std
    width = (upper - lower)
    pb = (close - lower) / width
    return lower, mid, upper, pb

def compute_rsi_last_map(closes: pd.DataFrame, period: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in closes.columns:
        try:
            rsi = compute_rsi_series(closes[col], period=period)
            val = rsi.iloc[-1]
            out[col] = float(val) if pd.notna(val) else float("nan")
        except Exception:
            out[col] = float("nan")
    return out


# ───────────────────────── Correlation core ─────────────────────────

@dataclass
class PairStat:
    symbol_1: str
    symbol_2: str
    corr: float
    p_value: Optional[float]
    n_obs: int
    def as_dict(self) -> Dict[str, Any]:
        return {
            "pair": f"{self.symbol_1}~{self.symbol_2}",
            "symbol_1": self.symbol_1,
            "symbol_2": self.symbol_2,
            "corr": float(self.corr),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "n_obs": int(self.n_obs),
        }

def compute_pairwise_correlations(returns: pd.DataFrame, min_obs_pair: int, use_scipy: bool = True) -> List[PairStat]:
    cols = list(returns.columns)
    pairs: List[PairStat] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            sub = returns[[s1, s2]].dropna()
            n = len(sub)
            if n < min_obs_pair: continue
            if sub[s1].std(skipna=True) == 0 or sub[s2].std(skipna=True) == 0: continue
            r = sub[s1].corr(sub[s2])
            p_val: Optional[float] = None
            if use_scipy and SCIPY_OK:
                try:
                    r2, p_val = pearsonr(sub[s1].values, sub[s2].values)  # type: ignore
                    r = float(r2)
                except Exception:
                    p_val = None
            pairs.append(PairStat(s1, s2, float(r), p_val, n))
    return pairs

def attach_rsi_to_pairs(pair_dicts: List[Dict[str, Any]], rsi_last: Dict[str, float]) -> None:
    for d in pair_dicts:
        s1, s2 = d.get("symbol_1"), d.get("symbol_2")
        v1 = rsi_last.get(s1, float("nan")); v2 = rsi_last.get(s2, float("nan"))
        d["symbol_1_rsi"] = None if (isinstance(v1, float) and np.isnan(v1)) else float(v1)
        d["symbol_2_rsi"] = None if (isinstance(v2, float) and np.isnan(v2)) else float(v2)


# ───────────────────────── Persistence ─────────────────────────

def persist_snapshot(snapshot: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(JSON_SNAPSHOTS_FILE), exist_ok=True)
    with open(JSON_SNAPSHOTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    with open(LATEST_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


# ───────────────────────── Position cases dedup ───────────────────────

def _canonical_pair_key(sym1: str, sym2: str) -> Tuple[str, str]:
    return tuple(sorted((sym1, sym2)))

def _parse_iso(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.strptime(ts.replace("Z", "+00:00"), "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)

def _load_position_json_history(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list): return data
        if isinstance(data, dict) and "position_candidates" in data:
            pcs = data.get("position_candidates") or []
            return pcs if isinstance(pcs, list) else []
        return []
    except Exception:
        return []

def _should_append(case: Dict[str, Any], history: List[Dict[str, Any]], now_dt: datetime) -> bool:
    s1 = case.get("symbol_1"); s2 = case.get("symbol_2")
    if not s1 or not s2: return False
    key = _canonical_pair_key(s1, s2)
    day_ago = now_dt - timedelta(hours=24)
    for prev in history:
        p1, p2 = prev.get("symbol_1"), prev.get("symbol_2")
        if not p1 or not p2: continue
        if _canonical_pair_key(p1, p2) != key: continue
        pts = prev.get("timestamp_utc") or prev.get("date")
        if not pts: continue
        try: pts_dt = _parse_iso(pts)
        except Exception: continue
        if pts_dt.tzinfo is None: pts_dt = pts_dt.replace(tzinfo=timezone.utc)
        if pts_dt >= day_ago:
            return False
    return True

def log_position_cases(candidates: List[Dict[str, Any]], ts: str, logger: Optional[logging.Logger] = None) -> None:
    if not candidates: return
    os.makedirs(os.path.dirname(POSITION_CASES_JSONL), exist_ok=True)
    with open(POSITION_CASES_JSONL, "a", encoding="utf-8") as f:
        for c in candidates:
            out = {"timestamp_utc": ts, **c}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    history = _load_position_json_history(POSITION_CASES_JSON)
    now_dt = _parse_iso(ts)
    if now_dt.tzinfo is None: now_dt = now_dt.replace(tzinfo=timezone.utc)
    appended = 0
    for c in candidates:
        entry = {
            "timestamp_utc": ts,
            "pair": c.get("pair"),
            "symbol_1": c.get("symbol_1"),
            "symbol_2": c.get("symbol_2"),
            "symbol_1_rsi": c.get("symbol_1_rsi"),
            "symbol_2_rsi": c.get("symbol_2_rsi"),
            "corr": c.get("corr"),
        }
        if _should_append(entry, history, now_dt):
            history.append(entry); appended += 1
    with open(POSITION_CASES_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    if logger:
        logger.info("Position cases saved: appended %d new unique entries (24h window).", appended)


# ───────────────────────── Dynamic universe helpers ───────────────────

def _get_usdm_linear_symbols(exchange) -> List[str]:
    """USDT-маржин. перпет. контракты (linear swaps), active=True"""
    out = []
    for m in exchange.markets.values():
        try:
            if not m.get("active", True):
                continue
            if not m.get("contract", False):
                continue
            if not (m.get("linear", False) and m.get("swap", False)):
                continue
            if m.get("quote") != "USDT":
                continue
            out.append(m["symbol"])
        except Exception:
            continue
    return sorted(set(out))

def _fetch_quote_volumes(exchange, symbols: List[str]) -> Dict[str, float]:
    """Возвращает {symbol: 24h_quote_volume_usd} по fetch_tickers()."""
    out: Dict[str, float] = {}
    try:
        tickers = exchange.fetch_tickers()
        for s in symbols:
            t = tickers.get(s) or {}
            qv = None
            if "info" in t and isinstance(t["info"], dict) and "quoteVolume" in t["info"]:
                try:
                    qv = float(t["info"]["quoteVolume"])
                except Exception:
                    qv = None
            if qv is None:
                v = t.get("quoteVolume")
                try:
                    qv = float(v) if v is not None else None
                except Exception:
                    qv = None
            out[s] = qv or 0.0
    except Exception:
        for s in symbols:
            out[s] = 0.0
    return out

def _compute_atr_percent_last(exchange, symbol: str, period: int = 14, days: int = 30) -> Optional[float]:
    """ATR% на дневках (ATR14/Close*100). Возвращает последнее значение."""
    df = fetch_ohlcv_df(exchange, symbol, "1d", limit=period + days + 2)
    if df is None or len(df) < period + 2:
        return None
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    last_close = df["close"].iloc[-1]
    atr_last = atr.iloc[-1]
    if pd.isna(last_close) or pd.isna(atr_last) or last_close == 0:
        return None
    return float((atr_last / last_close) * 100.0)

def _check_orderbook_ok(exchange, symbol: str, top_levels: int = 20, gap_pct: float = 0.01) -> bool:
    """Простой sanity-чек стакана: без больших дыр между соседними уровнями в топ-N."""
    try:
        ob = exchange.fetch_order_book(symbol, limit=max(10, top_levels))
        bids = (ob.get("bids") or [])[:top_levels]
        asks = (ob.get("asks") or [])[:top_levels]
        if len(bids) < 3 or len(asks) < 3:
            return False
        def _max_gap(levels: List[List[float]], descending: bool) -> float:
            mx = 0.0
            for i in range(1, len(levels)):
                p_prev = levels[i-1][0]
                p_cur = levels[i][0]
                if descending:  # bids: p_prev >= p_cur
                    gap = (p_prev - p_cur) / p_prev
                else:          # asks: p_cur >= p_prev
                    gap = (p_cur - p_prev) / p_prev
                mx = max(mx, gap)
            return mx
        return _max_gap(bids, True) <= gap_pct and _max_gap(asks, False) <= gap_pct
    except Exception:
        return False

def _compute_corr_with_btc_eth_ok(exchange, symbol: str, tf: str, bars: int, max_abs: float) -> bool:
    """Макс(|corr с BTC|, |corr с ETH|) <= порога."""
    closes_sym = fetch_close_series(exchange, symbol, tf, bars)
    if closes_sym is None or closes_sym.dropna().shape[0] < 20:
        return False
    btc = fetch_close_series(exchange, "BTC/USDT", tf, bars)
    eth = fetch_close_series(exchange, "ETH/USDT", tf, bars)
    if btc is None or eth is None:
        return False
    df = pd.concat([closes_sym.pct_change(), btc.pct_change(), eth.pct_change()], axis=1)
    df.columns = ["sym", "btc", "eth"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 20:
        return False
    corr_btc = df["sym"].corr(df["btc"])
    corr_eth = df["sym"].corr(df["eth"])
    if pd.isna(corr_btc) or pd.isna(corr_eth):
        return False
    return max(abs(float(corr_btc)), abs(float(corr_eth))) <= max_abs

def _apply_symbol_allow_block(symbols: List[str], logger: logging.Logger) -> List[str]:
    """
    Применяет allow/block списки при включённом DYN_SYMBOL_FILTERS_ENABLED.
    - Allowed срабатывает как whitelist (если задан хоть один из списков Allowed).
    - Затем всегда применяется Block (если задан).
    """
    if not DYN_SYMBOL_FILTERS_ENABLED:
        logger.info("Symbol allow/block filters disabled.")
        return symbols

    allowed_bases   = set(DYN_ALLOWED_BASES)
    blocked_bases   = set(DYN_BLOCKED_BASES)
    allowed_symbols = set(DYN_ALLOWED_SYMBOLS)
    blocked_symbols = set(DYN_BLOCKED_SYMBOLS)

    # нормализуем к аппер-кейсу для сравнения
    def base_of(sym: str) -> str:
        return (sym.split("/")[0]).upper()

    before = len(symbols)
    out = symbols

    # Если есть allow-листы — берём пересечение
    if allowed_bases or allowed_symbols:
        out = [s for s in out if (base_of(s) in allowed_bases) or (s.upper() in allowed_symbols)]

    # Затем вычёркиваем из блок-листов
    if blocked_bases or blocked_symbols:
        out = [s for s in out if (base_of(s) not in blocked_bases) and (s.upper() not in blocked_symbols)]

    logger.info("Symbol filters: %d -> %d (allow=%d bases/%d syms, block=%d bases/%d syms)",
                before, len(out), len(allowed_bases), len(allowed_symbols),
                len(blocked_bases), len(blocked_symbols))
    return out

def _maybe_filter_by_market_cap(symbols: List[str], logger: logging.Logger) -> List[str]:
    """
    Опциональный фильтр капы через CoinGecko (по символам; риск ложных совпадений).
    В проде лучше дать маппинг BASE->coingecko_id.
    """
    if not CAP_FILTER_ENABLED:
        logger.info("CAP filter disabled — пропускаем (CAP_FILTER_ENABLED=false).")
        return symbols
    if requests is None:
        logger.warning("requests не доступен — CAP фильтр пропущен.")
        return symbols
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1},
            timeout=15
        )
        if resp.status_code != 200:
            logger.warning("CoinGecko HTTP %s — капа пропущена.", resp.status_code)
            return symbols
        arr = resp.json()
        caps_by_symbol: Dict[str, float] = {}
        for it in arr:
            try:
                sym = str(it.get("symbol") or "").upper()
                mcap = float(it.get("market_cap") or 0.0)
                if sym and mcap:
                    caps_by_symbol[sym] = mcap
            except Exception:
                continue
        kept = []
        for s in symbols:
            base = s.split("/")[0].upper()
            mcap = caps_by_symbol.get(base)
            if (mcap is None) or (mcap >= MIN_MARKET_CAP_USD):
                kept.append(s)
        logger.info("CAP filter: оставлено %d/%d символов (мин. капа $%.0fM).",
                    len(kept), len(symbols), MIN_MARKET_CAP_USD/1e6)
        return kept
    except Exception as e:
        logger.warning("CAP filter error: %s — пропускаем.", e)
        return symbols

def build_dynamic_universe(exchange, logger: logging.Logger) -> List[str]:
    """
    Пайплайн динамического отбора (каждый шаг можно выключать в .env):
      0) Символьные allow/block-фильтры (если включены)
      1) Объём
      2) (опц.) Market Cap
      3) ATR%
      4) Стакан
      5) Corr с BTC/ETH
    """
    all_syms = _get_usdm_linear_symbols(exchange)
    if not all_syms:
        raise RuntimeError("Не удалось получить список USDT-перов.")

    logger.info("Dynamic: start with %d USDT-perp symbols", len(all_syms))

    # 0) allow/block
    syms = _apply_symbol_allow_block(all_syms, logger)

    # 1) объём
    qvol = _fetch_quote_volumes(exchange, syms)
    if DYN_FILTER_VOLUME_ENABLED:
        vol_kept = [s for s in syms if qvol.get(s, 0.0) >= MIN_DAILY_QUOTE_VOLUME_USD]
        vol_kept_sorted = sorted(vol_kept, key=lambda s: qvol.get(s, 0.0), reverse=True)
        preselect = vol_kept_sorted[:DYN_PRESELECT_TOP_N_BY_VOLUME]
        logger.info("Volume filter: %d pass (>=%.0fM), preselect=%d",
                    len(vol_kept), MIN_DAILY_QUOTE_VOLUME_USD/1e6, len(preselect))
    else:
        # без фильтра — просто отсортируем и ограничим preselect для экономии запросов
        pre = sorted(syms, key=lambda s: qvol.get(s, 0.0), reverse=True)
        preselect = pre[:DYN_PRESELECT_TOP_N_BY_VOLUME]
        logger.info("Volume filter DISABLED. Preselect top-%d by volume out of %d", len(preselect), len(syms))

    # 2) (опц.) CAP
    preselect = _maybe_filter_by_market_cap(preselect, logger)

    # 3) ATR%
    if DYN_FILTER_ATR_ENABLED:
        atr_kept = []
        for s in preselect:
            atrp = _compute_atr_percent_last(exchange, s, period=ATR_PERIOD, days=ATR_LOOKBACK_DAYS)
            if atrp is None:
                continue
            if atrp >= ATR_PERCENT_MIN:
                atr_kept.append(s)
        logger.info("ATR%% filter (>= %.2f%%): %d/%d", ATR_PERCENT_MIN, len(atr_kept), len(preselect))
    else:
        atr_kept = preselect
        logger.info("ATR%% filter DISABLED. Passing %d symbols.", len(atr_kept))

    # 4) Стакан
    if DYN_FILTER_ORDERBOOK_ENABLED:
        depth_kept = []
        for s in atr_kept:
            if _check_orderbook_ok(exchange, s, ORDERBOOK_TOP_LEVELS, ORDERBOOK_HOLE_PCT):
                depth_kept.append(s)
        logger.info("Orderbook filter (gaps <= %.2f%% top %d): %d/%d",
                    ORDERBOOK_HOLE_PCT*100, ORDERBOOK_TOP_LEVELS, len(depth_kept), len(atr_kept))
    else:
        depth_kept = atr_kept
        logger.info("Orderbook filter DISABLED. Passing %d symbols.", len(depth_kept))

    # 5) Корреляция
    if DYN_FILTER_CORR_ENABLED:
        corr_kept = []
        for s in depth_kept:
            if s in ("BTC/USDT", "ETH/USDT"):
                corr_kept.append(s)
                continue
            ok = _compute_corr_with_btc_eth_ok(exchange, s, CORR_TF, CORR_LOOKBACK_BARS, CORR_WITH_BTC_ETH_MAX)
            if ok:
                corr_kept.append(s)
        logger.info("Corr filter (max|corr| <= %.2f @ %s): %d/%d",
                    CORR_WITH_BTC_ETH_MAX, CORR_TF, len(corr_kept), len(depth_kept))
    else:
        corr_kept = depth_kept
        logger.info("Corr filter DISABLED. Passing %d symbols.", len(corr_kept))

    # Отсортируем финальный список по объёму убыв.
    corr_kept = sorted(corr_kept, key=lambda s: qvol.get(s, 0.0), reverse=True)
    return corr_kept

def save_dynamic_symbols(symbols: List[str]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"symbols_dynamic_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")
    latest = os.path.join(OUTPUT_DIR, "symbols_dynamic_latest.txt")
    with open(latest, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")
    return path


# ───────────────────────── Indicators for report ──────────────────────

def compute_indicators_for_symbol(exchange, symbol: str, tfs: List[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for tf in tfs:
        close = fetch_close_series(exchange, symbol, tf, INDICATOR_BARS)
        if close is None or len(close) < max(RSI_PERIOD, MACD_SLOW + MACD_SIGNAL, BB_PERIOD) + 1:
            result[tf] = {"rsi": None, "macd": None, "bb": None}
            continue
        rsi = compute_rsi_series(close, period=RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        macd_line, signal_line, hist = compute_macd_series(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        macd_val = macd_line.iloc[-1] if len(macd_line) else np.nan
        signal_val = signal_line.iloc[-1] if len(signal_line) else np.nan
        hist_val = hist.iloc[-1] if len(hist) else np.nan
        macd_obj = None if (pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val)) else {
            "macd": float(macd_val),
            "signal": float(signal_val),
            "hist": float(hist_val),
        }
        # BB считаем «про запас», в телеге не выводим
        lower, middle, upper, pb = compute_bbands_series(close, BB_PERIOD, BB_MULT)
        last_idx = close.index[-1]
        l = lower.loc[last_idx] if last_idx in lower.index else np.nan
        m = middle.loc[last_idx] if last_idx in middle.index else np.nan
        u = upper.loc[last_idx] if last_idx in upper.index else np.nan
        p = pb.loc[last_idx] if last_idx in pb.index else np.nan
        c = close.iloc[-1]
        bb_obj = None if (pd.isna(l) or pd.isna(m) or pd.isna(u) or pd.isna(p)) else {
            "lower": float(l), "middle": float(m), "upper": float(u),
            "close": float(c), "percent_b": float(p),
        }
        result[tf] = {"rsi": rsi_val, "macd": macd_obj, "bb": bb_obj}
    return result


# ───────────────────────── Core cycle ─────────────────────────

def run_cycle(logger: logging.Logger) -> None:
    try:
        exchange = get_exchange()

        # Универс: динамика или статика
        if DYNAMIC_UNIVERSE_ENABLED:
            symbols = build_dynamic_universe(exchange, logger)
            saved_path = save_dynamic_symbols(symbols)
            logger.info("Dynamic universe saved to %s (%d symbols).", saved_path, len(symbols))
        else:
            symbols = load_symbols_static()
            logger.info("Static universe: %d symbols (from %s or STATIC_SYMBOLS).", len(symbols), SYMBOLS_FILE)

        if len(symbols) < 2:
            ts_now = utc_now_iso()
            logger.warning("Слишком мало символов после выборки — прерываем цикл.")
            snapshot = {
                "timestamp_utc": ts_now,
                "exchange": "binanceusdm" if USE_FUTURES else "binance",
                "market_type": "futures" if USE_FUTURES else "spot",
                "timeframe": TIMEFRAME,
                "lookback_bars": LOOKBACK_BARS,
                "symbols": symbols,
                "alerts": {"negative": [], "near_zero": []},
                "pairs_sorted_by_corr": [],
                "pairs_closest_to_zero": [],
                "weak_corr_pairs": [],
                "position_candidates": [],
                "indicators": {},
                "indicator_tfs": INDICATOR_TFS,
            }
            persist_snapshot(snapshot)
            return

        # Основной расчёт
        closes = fetch_closes(exchange, symbols, TIMEFRAME, LOOKBACK_BARS, logger)

        rsi_last_all = compute_rsi_last_map(closes, period=RSI_PERIOD)

        returns = closes.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")

        counts = returns.count()
        stds = returns.std(skipna=True)
        valid_cols = counts.index[(counts >= MIN_OBS_PER_SYMBOL) & (stds > 0)]
        dropped = sorted(set(returns.columns) - set(valid_cols))
        returns = returns[valid_cols]

        if dropped:
            logger.info("Dropped %d symbols (insufficient data or zero variance): %s",
                        len(dropped), ", ".join(dropped[:20]) + ("..." if len(dropped) > 20 else ""))

        if returns.shape[1] < 2:
            ts_now = utc_now_iso()
            logger.info("Недостаточно валидных символов. Пропускаем корреляции.")
            snapshot = {
                "timestamp_utc": ts_now,
                "exchange": "binanceusdm" if USE_FUTURES else "binance",
                "market_type": "futures" if USE_FUTURES else "spot",
                "timeframe": TIMEFRAME,
                "lookback_bars": LOOKBACK_BARS,
                "symbols": list(returns.columns),
                "alerts": {"negative": [], "near_zero": []},
                "pairs_sorted_by_corr": [],
                "pairs_closest_to_zero": [],
                "weak_corr_pairs": [],
                "position_candidates": [],
                "indicators": {},
                "indicator_tfs": INDICATOR_TFS,
            }
            persist_snapshot(snapshot)
            return

        pair_stats = compute_pairwise_correlations(returns, min_obs_pair=MIN_OBS_PER_PAIR, use_scipy=True)
        pair_stats_sorted = sorted(pair_stats, key=lambda x: x.corr)
        pair_stats_by_abs = sorted(pair_stats, key=lambda x: abs(x.corr))

        ts_now = utc_now_iso()
        logger.info("Cycle %s | timeframe=%s | lookback=%d | symbols=%d | pairs=%d",
                    ts_now, TIMEFRAME, LOOKBACK_BARS, returns.shape[1], len(pair_stats_sorted))

        # Словари (НЕ объекты) — важно для сериализации JSON
        pairs_sorted_dicts = [p.as_dict() for p in pair_stats_sorted]
        pairs_abs_dicts    = [p.as_dict() for p in pair_stats_by_abs]
        attach_rsi_to_pairs(pairs_sorted_dicts, rsi_last_all)
        attach_rsi_to_pairs(pairs_abs_dicts,    rsi_last_all)

        # weak pairs: |corr| < threshold, сортируем по разнице RSI
        weak_pairs = [p.as_dict() for p in pair_stats if abs(p.corr) < WEAK_CORR_ABS_THRESH]
        attach_rsi_to_pairs(weak_pairs, rsi_last_all)
        def rsi_gap(d: Dict[str, Any]) -> float:
            r1 = d.get("symbol_1_rsi"); r2 = d.get("symbol_2_rsi")
            if r1 is None or r2 is None: return -1.0
            return abs(float(r1) - float(r2))
        weak_pairs.sort(key=rsi_gap, reverse=True)

        # позиции: один RSI>70, другой <30
        def is_position_candidate(d: Dict[str, Any]) -> bool:
            r1 = d.get("symbol_1_rsi"); r2 = d.get("symbol_2_rsi")
            if r1 is None or r2 is None: return False
            return (r1 > 70 and r2 < 30) or (r2 > 70 and r1 < 30)
        position_candidates = [d for d in pairs_sorted_dicts if is_position_candidate(d)]

        negative_alerts = [d for d in pairs_sorted_dicts if d["corr"] <= ALERT_NEGATIVE_THRESH][:TOP_N_TO_LOG]
        near_zero_alerts = [d for d in pairs_abs_dicts if abs(d["corr"]) <= ALERT_NEAR_ZERO_ABS][:TOP_N_TO_LOG]

        # Индикаторы только для топ weak_corr_pairs
        symbols_for_ind: List[str] = []
        top_pairs_for_ind = weak_pairs[:INDICATOR_MAX_PAIRS] if INDICATOR_MAX_PAIRS > 0 else weak_pairs
        for item in top_pairs_for_ind:
            s1 = item.get("symbol_1"); s2 = item.get("symbol_2")
            if s1: symbols_for_ind.append(s1)
            if s2: symbols_for_ind.append(s2)
        symbols_for_ind = sorted(set([s for s in symbols_for_ind if s]))

        indicators: Dict[str, Any] = {}
        for sym in symbols_for_ind:
            try:
                indicators[sym] = compute_indicators_for_symbol(exchange, sym, INDICATOR_TFS)
            except Exception as e:
                logger.warning("Indicators failed for %s: %s", sym, e)
                indicators[sym] = {}

        # snapshot (только словари!)
        to_persist_sorted = pairs_sorted_dicts[:TOP_N_TO_PERSIST] if TOP_N_TO_PERSIST else pairs_sorted_dicts
        to_persist_abs    = pairs_abs_dicts[:TOP_N_TO_PERSIST]    if TOP_N_TO_PERSIST else pairs_abs_dicts

        snapshot = {
            "timestamp_utc": ts_now,
            "exchange": "binanceusdm" if USE_FUTURES else "binance",
            "market_type": "futures" if USE_FUTURES else "spot",
            "timeframe": TIMEFRAME,
            "lookback_bars": LOOKBACK_BARS,
            "symbols": list(returns.columns),
            "alerts": {"negative": negative_alerts, "near_zero": near_zero_alerts},
            "pairs_sorted_by_corr": to_persist_sorted,   # dicts
            "pairs_closest_to_zero": to_persist_abs,     # dicts
            "weak_corr_pairs": weak_pairs,
            "position_candidates": position_candidates,
            "indicators": indicators,
            "indicator_tfs": INDICATOR_TFS,
        }
        persist_snapshot(snapshot)

        log_position_cases(position_candidates, ts_now, logger=logger)

    except Exception as e:
        logger.exception("Cycle failed: %s", e)


# ───────────────────────── Main (scheduler) ─────────────────────────

def main():
    logger = setup_logger(LOG_FILE)
    logger.info("Starting Correlation Scanner Bot | dynamic=%s", DYNAMIC_UNIVERSE_ENABLED)

    # первый прогон сразу
    run_cycle(logger)

    if RUN_ONCE:
        logger.info("Run-once mode: exiting.")
        return

    schedule.clear()

    # основной цикл
    if HOURLY_AT:
        schedule.every().hour.at(HOURLY_AT).do(run_cycle, logger=logger)
        logger.info("Scheduled main cycle hourly at %s", HOURLY_AT)
    else:
        schedule.every(RUN_EVERY_MINUTES).minutes.do(run_cycle, logger=logger)
        logger.info("Scheduled main cycle every %d minute(s)", RUN_EVERY_MINUTES)

    # telegram отчёт
    if TELEGRAM_ENABLED:
        if tg_send_report_once is None:
            logger.error("Telegram enabled but telegram_report.py not importable")
        elif not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logger.error("Telegram enabled but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set")
        else:
            def telegram_job():
                try:
                    tg_send_report_once(LATEST_JSON_FILE, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MAX_ROWS)
                except Exception as e:
                    logger.exception("Telegram report failed: %s", e)

            schedule.every().hour.at(TELEGRAM_HOURLY_AT).do(telegram_job)
            logger.info("Telegram report scheduled hourly at %s (max_rows=%d)", TELEGRAM_HOURLY_AT, TELEGRAM_MAX_ROWS)

            if TELEGRAM_SEND_ON_START:
                telegram_job()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down by user request.")
            break
        except Exception as e:
            logger.exception("Scheduler loop error: %s", e)
            time.sleep(2)

if __name__ == "__main__":
    main()
