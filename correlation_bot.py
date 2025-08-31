#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Scanner Bot for Binance via CCXT (ENV-only)
- читает ВСЕ настройки из .env (или переменных окружения)
- считает корреляции/RSI, формирует weak_corr_pairs, position_candidates
- ведёт json/jsonl-логи + dedup 24h для position_entry_cases.json
- и, при TELEGRAM_ENABLED=true, каждый час шлёт отчёт в Telegram
- ДОБАВЛЕНО: расчёт индикаторов (RSI, MACD, Bollinger Bands) на ТФ 15m/1h/1d
  по символам из top weak_corr_pairs и сохранение в snapshot["indicators"].
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

import numpy as np
import pandas as pd
import schedule

# ── .env загрузка (необязательно; если нет python-dotenv, просто пропустим)
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

# Telegram-компонент (лежит рядом в telegram_report.py)
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

# ───────────────────────── Config from ENV ─────────────────────────

SYMBOLS: List[str] = []  # можно игнорировать, используем файл

SYMBOLS_FILE: Optional[str] = _get_str("SYMBOLS_FILE", "symbols.txt")
USE_FUTURES: bool = _get_bool("USE_FUTURES", False)
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
TELEGRAM_MAX_ROWS: int = _get_int("TELEGRAM_MAX_ROWS", 30)
TELEGRAM_SEND_ON_START: bool = _get_bool("TELEGRAM_SEND_ON_START", True)
TELEGRAM_BOT_TOKEN: Optional[str] = _get_str("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_CHAT_ID: Optional[str] = _get_str("TELEGRAM_CHAT_ID", None)

# ── ДОБАВЛЕНО: Индикаторы для репорта
INDICATOR_TFS: List[str] = [t.strip() for t in (_get_str("INDICATOR_TFS", "15m,1h,1d") or "15m,1h,1d").split(",") if t.strip()]
INDICATOR_MAX_PAIRS: int = _get_int("INDICATOR_MAX_PAIRS", 30)   # сколько первых weak_corr_pairs считать
INDICATOR_BARS: int = _get_int("INDICATOR_BARS", 200)
MACD_FAST: int = _get_int("MACD_FAST", 12)
MACD_SLOW: int = _get_int("MACD_SLOW", 26)
MACD_SIGNAL: int = _get_int("MACD_SIGNAL", 9)
BB_PERIOD: int = _get_int("BB_PERIOD", 20)
BB_MULT: float = _get_float("BB_MULT", 2.0)

# ───────────────────────── Core helpers ─────────────────────────

def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("corr_bot")
    logger.setLevel(logging.INFO)
    # не плодим хендлеры при повторном импорте/запуске
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

def load_symbols(symbols_file: Optional[str]) -> List[str]:
    if SYMBOLS:
        return SYMBOLS
    if symbols_file and os.path.exists(symbols_file):
        with open(symbols_file, "r", encoding="utf-8") as f:
            syms = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not syms:
            raise SystemExit("symbols.txt пуст. Добавь символы (BTC/USDT и т.д.).")
        return syms
    raise SystemExit("Не заданы символы. Создай symbols.txt или задай SYMBOLS_FILE.")

def get_exchange():
    cfg = dict(enableRateLimit=True, options={"defaultType": "future" if USE_FUTURES else "spot"})
    ex = ccxt.binance(cfg)
    ex.load_markets()
    return ex

def fetch_closes(exchange, symbols: List[str], timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    all_series: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if not ohlcv:
                logger.warning("%s: empty OHLCV", sym); continue
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)
            all_series[sym] = df["close"].astype(float)
        except Exception as e:
            logger.warning("%s: fetch_ohlcv failed: %s", sym, e); continue
    if not all_series:
        raise RuntimeError("Нет OHLCV ни по одному символу.")
    return pd.concat(all_series, axis=1).sort_index()

def fetch_close_series(exchange, symbol: str, timeframe: str, limit: int) -> Optional[pd.Series]:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
        return df["close"].astype(float)
    except Exception:
        return None

def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    # %b — положение цены в полосах (0 у нижней, 1 у верхней)
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

def persist_snapshot(snapshot: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(JSON_SNAPSHOTS_FILE), exist_ok=True)
    with open(JSON_SNAPSHOTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    with open(LATEST_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

# ── position_entry_cases (24h dedup) ────────────────────────────
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
    # JSONL append
    with open(POSITION_CASES_JSONL, "a", encoding="utf-8") as f:
        for c in candidates:
            out = {"timestamp_utc": ts, **c}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    # JSON dedup 24h
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

# ───────────────────────── Индикаторы для набора символов ─────────────────────

def compute_indicators_for_symbol(exchange, symbol: str, tfs: List[str]) -> Dict[str, Any]:
    """
    Возвращает:
    {
      "15m": {
        "rsi": 62.7,
        "macd": {"macd": ..., "signal": ..., "hist": ...},
        "bb": {"lower": ..., "middle": ..., "upper": ..., "close": ..., "percent_b": ...}
      },
      "1h": {...},
      "1d": {...}
    }
    """
    result: Dict[str, Any] = {}
    for tf in tfs:
        close = fetch_close_series(exchange, symbol, tf, INDICATOR_BARS)
        if close is None or len(close) < max(RSI_PERIOD, MACD_SLOW + MACD_SIGNAL, BB_PERIOD) + 1:
            result[tf] = {"rsi": None, "macd": None, "bb": None}
            continue
        # RSI
        rsi = compute_rsi_series(close, period=RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None

        # MACD
        macd_line, signal_line, hist = compute_macd_series(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        macd_val = macd_line.iloc[-1] if len(macd_line) else np.nan
        signal_val = signal_line.iloc[-1] if len(signal_line) else np.nan
        hist_val = hist.iloc[-1] if len(hist) else np.nan
        macd_obj = None if (pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val)) else {
            "macd": float(macd_val),
            "signal": float(signal_val),
            "hist": float(hist_val),
        }

        # Bollinger
        lower, middle, upper, pb = compute_bbands_series(close, BB_PERIOD, BB_MULT)
        last_idx = close.index[-1]
        l = lower.loc[last_idx] if last_idx in lower.index else np.nan
        m = middle.loc[last_idx] if last_idx in middle.index else np.nan
        u = upper.loc[last_idx] if last_idx in upper.index else np.nan
        p = pb.loc[last_idx] if last_idx in pb.index else np.nan
        c = close.iloc[-1]
        bb_obj = None if (pd.isna(l) or pd.isna(m) or pd.isna(u) or pd.isna(p)) else {
            "lower": float(l),
            "middle": float(m),
            "upper": float(u),
            "close": float(c),
            "percent_b": float(p),
        }

        result[tf] = {"rsi": rsi_val, "macd": macd_obj, "bb": bb_obj}
    return result

# ───────────────────────── Core cycle ─────────────────────────

def run_cycle(logger: logging.Logger) -> None:
    try:
        symbols = load_symbols(SYMBOLS_FILE)
        exchange = get_exchange()

        closes = fetch_closes(exchange, symbols, TIMEFRAME, LOOKBACK_BARS, logger)

        # RSI last
        rsi_last_all = compute_rsi_last_map(closes, period=RSI_PERIOD)

        # returns + cleaning
        returns = closes.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(how="all")

        counts = returns.count()
        stds = returns.std(skipna=True)
        valid_cols = counts.index[(counts >= MIN_OBS_PER_SYMBOL) & (stds > 0)]
        dropped = sorted(set(returns.columns) - set(valid_cols))
        returns = returns[valid_cols]

        if dropped:
            logger.info("Dropped %d symbols (insufficient data or zero variance): %s",
                        len(dropped), ", ".join(dropped[:20]) + ("..." if len(dropped) > 20 else ""))

        if returns.shape[1] < 2:
            ts = utc_now_iso()
            logger.info("Недостаточно валидных символов. Пропускаем корреляции.")
            snapshot = {
                "timestamp_utc": ts,
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

        _ = returns.corr(min_periods=MIN_OBS_PER_PAIR)

        # pairwise
        pair_stats = compute_pairwise_correlations(returns, min_obs_pair=MIN_OBS_PER_PAIR, use_scipy=True)
        pair_stats_sorted = sorted(pair_stats, key=lambda x: x.corr)
        pair_stats_by_abs = sorted(pair_stats, key=lambda x: abs(x.corr))

        ts = utc_now_iso()
        logger.info("Cycle %s | timeframe=%s | lookback=%d | symbols=%d | pairs=%d",
                    ts, TIMEFRAME, LOOKBACK_BARS, returns.shape[1], len(pair_stats_sorted))

        pairs_sorted_dicts = [p.as_dict() for p in pair_stats_sorted]
        pairs_abs_dicts    = [p.as_dict() for p in pair_stats_by_abs]
        attach_rsi_to_pairs(pairs_sorted_dicts, rsi_last_all)
        attach_rsi_to_pairs(pairs_abs_dicts,    rsi_last_all)

        # weak pairs: abs(corr) < threshold, sort by RSI gap desc
        weak_pairs = [p.as_dict() for p in pair_stats if abs(p.corr) < WEAK_CORR_ABS_THRESH]
        attach_rsi_to_pairs(weak_pairs, rsi_last_all)
        def rsi_gap(d: Dict[str, Any]) -> float:
            r1 = d.get("symbol_1_rsi"); r2 = d.get("symbol_2_rsi")
            if r1 is None or r2 is None: return -1.0
            return abs(float(r1) - float(r2))
        weak_pairs.sort(key=rsi_gap, reverse=True)

        # position candidates: one RSI>70 and other<30
        def is_position_candidate(d: Dict[str, Any]) -> bool:
            r1 = d.get("symbol_1_rsi"); r2 = d.get("symbol_2_rsi")
            if r1 is None or r2 is None: return False
            return (r1 > 70 and r2 < 30) or (r2 > 70 and r1 < 30)
        position_candidates = [d for d in pairs_sorted_dicts if is_position_candidate(d)]

        # alerts (logs)
        negative_alerts = [d for d in pairs_sorted_dicts if d["corr"] <= ALERT_NEGATIVE_THRESH][:TOP_N_TO_LOG]
        near_zero_alerts = [d for d in pairs_abs_dicts if abs(d["corr"]) <= ALERT_NEAR_ZERO_ABS][:TOP_N_TO_LOG]

        if negative_alerts:
            logger.info("Top negative correlation pairs (<= %.2f):", ALERT_NEGATIVE_THRESH)
            for p in negative_alerts:
                logger.info("  %s | corr=%.4f | n=%d | p=%s | rsi=(%.2f, %.2f)",
                            p["pair"], p["corr"], p["n_obs"],
                            f"{p['p_value']:.4g}" if p["p_value"] is not None else "NA",
                            p.get("symbol_1_rsi") if p.get("symbol_1_rsi") is not None else float("nan"),
                            p.get("symbol_2_rsi") if p.get("symbol_2_rsi") is not None else float("nan"))

        if near_zero_alerts:
            logger.info("Top near-zero correlation pairs (|corr|<= %.2f):", ALERT_NEAR_ZERO_ABS)
            for p in near_zero_alerts:
                logger.info("  %s | corr=%.4f | n=%d | p=%s | rsi=(%.2f, %.2f)",
                            p["pair"], p["corr"], p["n_obs"],
                            f"{p['p_value']:.4g}" if p["p_value"] is not None else "NA",
                            p.get("symbol_1_rsi") if p.get("symbol_1_rsi") is not None else float("nan"),
                            p.get("symbol_2_rsi") if p.get("symbol_2_rsi") is not None else float("nan"))

        if position_candidates:
            logger.info("Position candidates (RSI one>70 & other<30): %d", len(position_candidates))
            for c in position_candidates:
                logger.info("  %s | rsi=(%.2f, %.2f) | corr=%.4f",
                            c["pair"],
                            c.get("symbol_1_rsi") if c.get("symbol_1_rsi") is not None else float("nan"),
                            c.get("symbol_2_rsi") if c.get("symbol_2_rsi") is not None else float("nan"),
                            c["corr"])

        # snapshot (limit size)
        to_persist_sorted = pairs_sorted_dicts[:TOP_N_TO_PERSIST] if TOP_N_TO_PERSIST else pairs_sorted_dicts
        to_persist_abs    = pairs_abs_dicts[:TOP_N_TO_PERSIST]    if TOP_N_TO_PERSIST else pairs_abs_dicts

        # ── ДОБАВЛЕНО: индикаторы для топ weak_corr_pairs
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

        snapshot = {
            "timestamp_utc": ts,
            "exchange": "binanceusdm" if USE_FUTURES else "binance",
            "market_type": "futures" if USE_FUTURES else "spot",
            "timeframe": TIMEFRAME,
            "lookback_bars": LOOKBACK_BARS,
            "symbols": list(returns.columns),
            "alerts": {"negative": negative_alerts, "near_zero": near_zero_alerts},
            "pairs_sorted_by_corr": to_persist_sorted,
            "pairs_closest_to_zero": to_persist_abs,
            "weak_corr_pairs": weak_pairs,
            "position_candidates": position_candidates,
            "indicators": indicators,          # <── ключ для Telegram
            "indicator_tfs": INDICATOR_TFS,    # <── чтобы знать какие ТФ показывать
        }
        persist_snapshot(snapshot)

        # separate log for position candidates with 24h dedup
        log_position_cases(position_candidates, ts, logger=logger)

    except Exception as e:
        logger.exception("Cycle failed: %s", e)

# ───────────────────────── Main (scheduler) ─────────────────────────

def main():
    logger = setup_logger(LOG_FILE)
    logger.info("Starting Correlation Scanner Bot (ENV mode)")

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
