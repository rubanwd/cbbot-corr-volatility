#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Scanner Bot for Binance via CCXT
- Корреляции доходностей, weak-пары, RSI для обеих legs
- position_candidates: RSI одного > 70 и другого < 30
- Логи в JSON/JSONL, расписание через schedule
- position_entry_cases.json теперь накапливает записи и дедуплицирует по паре в окне 24h

Запуск (примеры):
    python correlation_bot.py --timeframe 1h --hourly-at ":01"
    python correlation_bot.py --timeframe 15m --run-every-minutes 1
    python correlation_bot.py --once
"""

from __future__ import annotations

import os
import json
import time
import argparse
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import schedule

try:
    import ccxt
except Exception as e:
    raise SystemExit("ccxt is required. Install with: pip install ccxt") from e

# Мягко приглушаем только спец-предупреждение SciPy
try:
    from scipy.stats import pearsonr, ConstantInputWarning  # type: ignore
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    SCIPY_OK = True
except Exception:
    pearsonr = None
    SCIPY_OK = False


# ========================== CONFIG (переопределяется из CLI) ==========================

SYMBOLS: List[str] = []
SYMBOLS_FILE: Optional[str] = "symbols.txt"

USE_FUTURES: bool = False               # False=spot, True=futures
TIMEFRAME: str = "1h"
LOOKBACK_BARS: int = 500

MIN_OBS_PER_SYMBOL: int = 50
MIN_OBS_PER_PAIR: int = 30

# RSI
RSI_PERIOD: int = 14

# слабая корреляция: близко к нулю
WEAK_CORR_ABS_THRESH: float = 0.30

# Вывод
OUTPUT_DIR: str = "./data"
JSON_SNAPSHOTS_FILE: str = os.path.join(OUTPUT_DIR, "correlation_snapshots.jsonl")
LATEST_JSON_FILE: str = os.path.join(OUTPUT_DIR, "latest_correlation.json")
LOG_FILE: str = os.path.join(OUTPUT_DIR, "correlation_bot.log")

# Доп. лог по кандидатам входа (RSI >70 / <30)
POSITION_CASES_JSONL: str = os.path.join(OUTPUT_DIR, "position_entry_cases.jsonl")
POSITION_CASES_JSON: str  = os.path.join(OUTPUT_DIR, "position_entry_cases.json")

# Алерты/сортировки
ALERT_NEGATIVE_THRESH: float = -0.30
ALERT_NEAR_ZERO_ABS: float = 0.10
TOP_N_TO_LOG: int = 50
TOP_N_TO_PERSIST: Optional[int] = 200

# Расписание
RUN_EVERY_MINUTES: int = 1
HOURLY_AT: Optional[str] = None
RUN_ONCE: bool = False


# ========================== HELPERS ==========================

def setup_logger(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("corr_bot")
    logger.setLevel(logging.INFO)
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
    if SYMBOLS and len(SYMBOLS) > 0:
        return SYMBOLS
    if symbols_file and os.path.exists(symbols_file):
        with open(symbols_file, "r", encoding="utf-8") as f:
            syms = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        if not syms:
            raise SystemExit("symbols.txt пуст. Заполни символы по одному в строке, например BTC/USDT")
        return syms
    raise SystemExit("Не заданы символы. Укажи их в SYMBOLS или создай symbols.txt.")


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
                logger.warning("%s: empty OHLCV", sym)
                continue
            df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("ts", inplace=True)
            s = df["close"].astype(float)
            all_series[sym] = s
        except Exception as e:
            logger.warning("%s: fetch_ohlcv failed: %s", sym, e)
            continue

    if not all_series:
        raise RuntimeError("Не удалось получить OHLCV ни для одного символа. Проверь список/таймфрейм.")
    closes = pd.concat(all_series, axis=1).sort_index()
    return closes


def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI через EWM(alpha=1/period)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_rsi_last_map(closes: pd.DataFrame, period: int) -> Dict[str, float]:
    rsi_last: Dict[str, float] = {}
    for col in closes.columns:
        try:
            rsi = compute_rsi_series(closes[col], period=period)
            val = rsi.iloc[-1]
            rsi_last[col] = float(val) if pd.notna(val) else float("nan")
        except Exception:
            rsi_last[col] = float("nan")
    return rsi_last


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


def compute_pairwise_correlations(
    returns: pd.DataFrame,
    min_obs_pair: int,
    use_scipy: bool = True
) -> List[PairStat]:
    cols = list(returns.columns)
    pairs: List[PairStat] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            sub = returns[[s1, s2]].dropna()
            n = len(sub)
            if n < min_obs_pair:
                continue
            if sub[s1].std(skipna=True) == 0 or sub[s2].std(skipna=True) == 0:
                continue
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
        v1 = rsi_last.get(s1, float("nan"))
        v2 = rsi_last.get(s2, float("nan"))
        d["symbol_1_rsi"] = None if (v1 is None or (isinstance(v1, float) and np.isnan(v1))) else float(v1)
        d["symbol_2_rsi"] = None if (v2 is None or (isinstance(v2, float) and np.isnan(v2))) else float(v2)


def persist_snapshot(snapshot: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(JSON_SNAPSHOTS_FILE), exist_ok=True)
    with open(JSON_SNAPSHOTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    with open(LATEST_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


# ----------------------- Position cases accumulation -----------------------

def _canonical_pair_key(sym1: str, sym2: str) -> Tuple[str, str]:
    """Не зависит от порядка: ('BTC/USDT','ETH/USDT') == ('ETH/USDT','BTC/USDT')."""
    return tuple(sorted((sym1, sym2)))

def _parse_iso(ts: str) -> datetime:
    try:
        # Python 3.11+ читает ISO с оффсетом
        return datetime.fromisoformat(ts)
    except Exception:
        # На всякий случай — принудительно к UTC
        return datetime.strptime(ts.replace("Z", "+00:00"), "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)

def _load_position_json_history(path: str) -> List[Dict[str, Any]]:
    """
    Грузим массив объектов из position_entry_cases.json.
    Совместимость: если там старый формат { "timestamp_utc": ..., "position_candidates": [...] } — расплющим.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "position_candidates" in data:
            pcs = data.get("position_candidates") or []
            if isinstance(pcs, list):
                return pcs
        # если что-то иное — начнем новую историю
        return []
    except Exception:
        return []

def _should_append(case: Dict[str, Any], history: List[Dict[str, Any]], now_dt: datetime) -> bool:
    """
    True если в истории НЕТ записи с той же парой за последние 24 часа.
    """
    s1 = case.get("symbol_1")
    s2 = case.get("symbol_2")
    if not s1 or not s2:
        return False
    key = _canonical_pair_key(s1, s2)
    day_ago = now_dt - timedelta(hours=24)

    for prev in history:
        p1, p2 = prev.get("symbol_1"), prev.get("symbol_2")
        if not p1 or not p2:
            continue
        if _canonical_pair_key(p1, p2) != key:
            continue
        pts = prev.get("timestamp_utc") or prev.get("date")  # на случай другого поля
        if not pts:
            continue
        try:
            pts_dt = _parse_iso(pts)
        except Exception:
            continue
        if pts_dt.tzinfo is None:
            pts_dt = pts_dt.replace(tzinfo=timezone.utc)
        if pts_dt >= day_ago:
            return False  # уже есть свежая запись этой пары
    return True

def log_position_cases(candidates: List[Dict[str, Any]], ts: str, logger: Optional[logging.Logger] = None) -> None:
    """
    - Всегда аппендим сырые кандидаты в JSONL (построчно).
    - В position_entry_cases.json накапливаем массив, но дедуплицируем по паре в окне 24h.
    """
    if not candidates:
        return

    os.makedirs(os.path.dirname(POSITION_CASES_JSONL), exist_ok=True)

    # 1) JSONL — просто дописываем строки
    with open(POSITION_CASES_JSONL, "a", encoding="utf-8") as f:
        for c in candidates:
            out = {"timestamp_utc": ts, **c}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # 2) JSON — накапливаем уникальные по паре за 24h
    history = _load_position_json_history(POSITION_CASES_JSON)
    now_dt = _parse_iso(ts)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

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
            history.append(entry)
            appended += 1

    with open(POSITION_CASES_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if logger:
        logger.info("Position cases saved: appended %d new unique entries (24h window).", appended)


# ----------------------- CORE -----------------------

def run_cycle(logger: logging.Logger) -> None:
    try:
        symbols = load_symbols(SYMBOLS_FILE)
        exchange = get_exchange()

        closes = fetch_closes(exchange, symbols, TIMEFRAME, LOOKBACK_BARS, logger)

        # ---- RSI по последним значениям ----
        rsi_last_all = compute_rsi_last_map(closes, period=RSI_PERIOD)

        # ---- Доходности + чистка ----
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
                        len(dropped),
                        ", ".join(dropped[:20]) + ("..." if len(dropped) > 20 else ""))

        if returns.shape[1] < 2:
            ts = utc_now_iso()
            logger.info("Недостаточно валидных символов после фильтрации. Пропускаем корреляции.")
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
            }
            persist_snapshot(snapshot)
            return

        _ = returns.corr(min_periods=MIN_OBS_PER_PAIR)

        # ---- Попарные корреляции ----
        pair_stats = compute_pairwise_correlations(returns, min_obs_pair=MIN_OBS_PER_PAIR, use_scipy=True)
        pair_stats_sorted = sorted(pair_stats, key=lambda x: x.corr)      # возрастание (сильнее отрицательные сверху)
        pair_stats_by_abs = sorted(pair_stats, key=lambda x: abs(x.corr)) # ближе к нулю сверху

        ts = utc_now_iso()
        logger.info("Cycle %s | timeframe=%s | lookback=%d | symbols=%d | pairs=%d",
                    ts, TIMEFRAME, LOOKBACK_BARS, returns.shape[1], len(pair_stats_sorted))

        # ---- Преобразуем в dict и приклеим RSI ----
        pairs_sorted_dicts = [p.as_dict() for p in pair_stats_sorted]
        pairs_abs_dicts    = [p.as_dict() for p in pair_stats_by_abs]
        attach_rsi_to_pairs(pairs_sorted_dicts, rsi_last_all)
        attach_rsi_to_pairs(pairs_abs_dicts,    rsi_last_all)

        # ---- weak_corr_pairs: |corr| < WEAK_CORR_ABS_THRESH, сортировка по разрыву RSI (по убыванию) ----
        weak_pairs = [p.as_dict() for p in pair_stats if abs(p.corr) < WEAK_CORR_ABS_THRESH]
        attach_rsi_to_pairs(weak_pairs, rsi_last_all)
        def rsi_gap(d: Dict[str, Any]) -> float:
            r1 = d.get("symbol_1_rsi")
            r2 = d.get("symbol_2_rsi")
            if r1 is None or r2 is None:
                return -1.0
            return abs(float(r1) - float(r2))
        weak_pairs.sort(key=rsi_gap, reverse=True)

        # ---- position_candidates: один RSI > 70, другой < 30 ----
        def is_position_candidate(d: Dict[str, Any]) -> bool:
            r1 = d.get("symbol_1_rsi")
            r2 = d.get("symbol_2_rsi")
            if r1 is None or r2 is None:
                return False
            return (r1 > 70 and r2 < 30) or (r2 > 70 and r1 < 30)

        position_candidates = [d for d in pairs_sorted_dicts if is_position_candidate(d)]

        # ---- Логи-алерты в stdout ----
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

        # ---- Ограничение объёма в снапшоте ----
        to_persist_sorted = pairs_sorted_dicts[:TOP_N_TO_PERSIST] if TOP_N_TO_PERSIST else pairs_sorted_dicts
        to_persist_abs    = pairs_abs_dicts[:TOP_N_TO_PERSIST]    if TOP_N_TO_PERSIST else pairs_abs_dicts
        to_persist_weak   = weak_pairs
        to_persist_pos    = position_candidates

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
            "weak_corr_pairs": to_persist_weak,
            "position_candidates": to_persist_pos,
        }
        persist_snapshot(snapshot)

        # ---- Отдельный лог кандидатов входа: JSONL + JSON (накапливающий с дедупликацией 24h) ----
        log_position_cases(position_candidates, ts, logger=logger)

    except Exception as e:
        logger.exception("Cycle failed: %s", e)


def apply_cli_overrides():
    global SYMBOLS_FILE, USE_FUTURES, TIMEFRAME, LOOKBACK_BARS
    global RUN_EVERY_MINUTES, HOURLY_AT, RUN_ONCE
    global MIN_OBS_PER_SYMBOL, MIN_OBS_PER_PAIR
    global ALERT_NEGATIVE_THRESH, ALERT_NEAR_ZERO_ABS
    global RSI_PERIOD, WEAK_CORR_ABS_THRESH

    parser = argparse.ArgumentParser(description="Correlation Scanner Bot")
    parser.add_argument("--symbols-file", type=str, default=SYMBOLS_FILE)
    parser.add_argument("--use-futures", action="store_true", help="Use USDⓈ-M futures instead of spot")
    parser.add_argument("--timeframe", type=str, default=TIMEFRAME)
    parser.add_argument("--lookback", type=int, default=LOOKBACK_BARS)
    parser.add_argument("--run-every-minutes", type=int, default=RUN_EVERY_MINUTES)
    parser.add_argument("--hourly-at", type=str, default=HOURLY_AT, help='e.g. ":01" to run once per hour')
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    parser.add_argument("--min-obs-symbol", type=int, default=MIN_OBS_PER_SYMBOL)
    parser.add_argument("--min-obs-pair", type=int, default=MIN_OBS_PER_PAIR)
    parser.add_argument("--negative-thresh", type=float, default=ALERT_NEGATIVE_THRESH)
    parser.add_argument("--near-zero-abs", type=float, default=ALERT_NEAR_ZERO_ABS)
    parser.add_argument("--rsi-period", type=int, default=RSI_PERIOD)
    parser.add_argument("--weak-abs", type=float, default=WEAK_CORR_ABS_THRESH, help="abs(corr) < this is WEAK")
    args = parser.parse_args()

    SYMBOLS_FILE = args.symbols_file
    USE_FUTURES = args.use_futures
    TIMEFRAME = args.timeframe
    LOOKBACK_BARS = args.lookback
    RUN_EVERY_MINUTES = args.run_every_minutes
    HOURLY_AT = args.hourly_at
    RUN_ONCE = args.once
    MIN_OBS_PER_SYMBOL = args.min_obs_symbol
    MIN_OBS_PER_PAIR = args.min_obs_pair
    ALERT_NEGATIVE_THRESH = args.negative_thresh
    ALERT_NEAR_ZERO_ABS = args.near_zero_abs
    RSI_PERIOD = args.rsi_period
    WEAK_CORR_ABS_THRESH = args.weak_abs


def main():
    apply_cli_overrides()
    logger = setup_logger(LOG_FILE)
    logger.info("Starting Correlation Scanner Bot")

    run_cycle(logger)

    if RUN_ONCE:
        logger.info("Run-once mode: exiting.")
        return

    schedule.clear()
    if HOURLY_AT:
        schedule.every().hour.at(HOURLY_AT).do(run_cycle, logger=logger)
        logger.info("Scheduled hourly at %s", HOURLY_AT)
    else:
        schedule.every(RUN_EVERY_MINUTES).minutes.do(run_cycle, logger=logger)
        logger.info("Scheduled every %d minute(s)", RUN_EVERY_MINUTES)

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
