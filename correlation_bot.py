#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Scanner Bot for Binance via CCXT (updated with weak pairs + RSI)

Новое:
- В snapshot добавлен массив week_corr_pairs: пары с "слабой" корреляцией (|corr| < 0.30).
- В объекты пар добавлены поля symbol_1_rsi и symbol_2_rsi (RSI(14) по ценам закрытия).
- Остальные улучшения по чистке данных и фильтрации оставлены.

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
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import schedule

try:
    import ccxt
except Exception as e:
    raise SystemExit("ccxt is required. Install with: pip install ccxt") from e

# Мягко приглушаем только спец-предупреждение SciPy, если оно вдруг где-то проскочит.
try:
    from scipy.stats import pearsonr, ConstantInputWarning  # type: ignore
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    SCIPY_OK = True
except Exception:
    pearsonr = None
    SCIPY_OK = False


# ========================== CONFIG (можно переопределить из CLI) ==========================

SYMBOLS: List[str] = []                 # можно захардкодить здесь, но обычно используем файл
SYMBOLS_FILE: Optional[str] = "symbols.txt"

USE_FUTURES: bool = False               # False = spot; True = USDⓈ-M futures
TIMEFRAME: str = "1h"                   # "15m" или "1h" и т.д.
LOOKBACK_BARS: int = 500

# Минимальные требования по данным
MIN_OBS_PER_SYMBOL: int = 50            # минимум ненулевых наблюдений у символа
MIN_OBS_PER_PAIR: int = 30              # минимум наблюдений для расчёта корреляции пары

# RSI
RSI_PERIOD: int = 14

# Порог «слабой» корреляции (используем модуль — ближе к нулю считается слабой)
WEAK_CORR_ABS_THRESH: float = 0.30

# Вывод
OUTPUT_DIR: str = "./data"
JSON_SNAPSHOTS_FILE: str = os.path.join(OUTPUT_DIR, "correlation_snapshots.jsonl")
LATEST_JSON_FILE: str = os.path.join(OUTPUT_DIR, "latest_correlation.json")
LOG_FILE: str = os.path.join(OUTPUT_DIR, "correlation_bot.log")

# Алерты/сортировки
ALERT_NEGATIVE_THRESH: float = -0.30    # corr <= this → негативная корр
ALERT_NEAR_ZERO_ABS: float = 0.10       # |corr| <= this → почти нулевая корр
TOP_N_TO_LOG: int = 50                  # логировать топ-N
TOP_N_TO_PERSIST: Optional[int] = 200   # писать в снапшот первые N (None = все)

# Расписание
RUN_EVERY_MINUTES: int = 1              # если не указан hourly
HOURLY_AT: Optional[str] = None         # например ":01" → тогда игнорируем RUN_EVERY_MINUTES
RUN_ONCE: bool = False                  # один прогон и выход


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
            raise SystemExit("symbols.txt пуст. Заполни символы, по одному в строке (например, BTC/USDT).")
        return syms
    raise SystemExit("Не заданы символы. Укажи их в SYMBOLS или создай symbols.txt.")


def get_exchange():
    cfg = dict(enableRateLimit=True, options={"defaultType": "future" if USE_FUTURES else "spot"})
    ex = ccxt.binance(cfg)
    ex.load_markets()
    return ex


def fetch_closes(exchange, symbols: List[str], timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Возвращает DataFrame цен закрытия, индекс — UTC timestamp, колонки — символы.
    Отсутствующие/проблемные символы пропускаются.
    """
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
    """Считает RSI(period) для каждого столбца и возвращает последнее значение по каждому символу."""
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
            # Отбрасываем пары с нулевой дисперсией у любой серии
            if sub[s1].std(skipna=True) == 0 or sub[s2].std(skipna=True) == 0:
                continue

            r = sub[s1].corr(sub[s2])
            p_val: Optional[float] = None
            if use_scipy and SCIPY_OK:
                try:
                    r2, p_val = pearsonr(sub[s1].values, sub[s2].values)  # type: ignore
                    r = float(r2)  # выравниваем на SciPy-значение
                except Exception:
                    p_val = None
            pairs.append(PairStat(s1, s2, float(r), p_val, n))
    return pairs


def attach_rsi_to_pairs(pair_dicts: List[Dict[str, Any]], rsi_last: Dict[str, float]) -> None:
    """Добавляет symbol_1_rsi и symbol_2_rsi к каждому объекту пары (in-place)."""
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


def run_cycle(logger: logging.Logger) -> None:
    try:
        symbols = load_symbols(SYMBOLS_FILE)
        exchange = get_exchange()

        closes = fetch_closes(exchange, symbols, TIMEFRAME, LOOKBACK_BARS, logger)

        # ---- RSI по последним значениям (по всем символам) ----
        rsi_last_all = compute_rsi_last_map(closes, period=RSI_PERIOD)

        # ---- Расчёт доходностей (без fill_method по умолчанию) + чистка ----
        returns = closes.pct_change(fill_method=None)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(how="all")  # оставляем строки, где есть что-то полезное

        # ---- Фильтрация символов по количеству наблюдений и дисперсии ----
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
                "week_corr_pairs": [],
            }
            persist_snapshot(snapshot)
            return

        # Корр-матрица (для отладки/аналитики, с min_periods)
        _corr_matrix = returns.corr(min_periods=MIN_OBS_PER_PAIR)

        # Попарные корреляции
        pair_stats = compute_pairwise_correlations(returns, min_obs_pair=MIN_OBS_PER_PAIR, use_scipy=True)
        pair_stats_sorted = sorted(pair_stats, key=lambda x: x.corr)             # по возрастанию (более отрицательные сверху)
        pair_stats_by_abs = sorted(pair_stats, key=lambda x: abs(x.corr))        # ближе к нулю сверху

        ts = utc_now_iso()
        logger.info(
            "Cycle %s | timeframe=%s | lookback=%d | symbols=%d | pairs=%d",
            ts, TIMEFRAME, LOOKBACK_BARS, returns.shape[1], len(pair_stats_sorted)
        )

        # Списки словарей
        pairs_sorted_dicts = [p.as_dict() for p in pair_stats_sorted]
        pairs_abs_dicts    = [p.as_dict() for p in pair_stats_by_abs]

        # ---- week_corr_pairs (|corr| < WEAK_CORR_ABS_THRESH), сортируем по |corr| возрастанию ----
        week_pairs = [p.as_dict() for p in pair_stats if abs(p.corr) < WEAK_CORR_ABS_THRESH]
        week_pairs.sort(key=lambda d: abs(d["corr"]))

        # ---- Приклеиваем RSI ----
        attach_rsi_to_pairs(pairs_sorted_dicts, rsi_last_all)
        attach_rsi_to_pairs(pairs_abs_dicts,    rsi_last_all)
        attach_rsi_to_pairs(week_pairs,         rsi_last_all)

        # Алерты
        negative_alerts = [d for d in pairs_sorted_dicts if d["corr"] <= ALERT_NEGATIVE_THRESH][:TOP_N_TO_LOG]
        near_zero_alerts = [d for d in pairs_abs_dicts if abs(d["corr"]) <= ALERT_NEAR_ZERO_ABS][:TOP_N_TO_LOG]

        if negative_alerts:
            logger.info("Top negative correlation pairs (<= %.2f):", ALERT_NEGATIVE_THRESH)
            for p in negative_alerts[:TOP_N_TO_LOG]:
                logger.info("  %s | corr=%.4f | n=%d | p=%s | rsi=(%.2f, %.2f)",
                            p["pair"], p["corr"], p["n_obs"],
                            f"{p['p_value']:.4g}" if p["p_value"] is not None else "NA",
                            p.get("symbol_1_rsi") if p.get("symbol_1_rsi") is not None else float("nan"),
                            p.get("symbol_2_rsi") if p.get("symbol_2_rsi") is not None else float("nan"))

        if near_zero_alerts:
            logger.info("Top near-zero correlation pairs (|corr|<= %.2f):", ALERT_NEAR_ZERO_ABS)
            for p in near_zero_alerts[:TOP_N_TO_LOG]:
                logger.info("  %s | corr=%.4f | n=%d | p=%s | rsi=(%.2f, %.2f)",
                            p["pair"], p["corr"], p["n_obs"],
                            f"{p['p_value']:.4g}" if p["p_value"] is not None else "NA",
                            p.get("symbol_1_rsi") if p.get("symbol_1_rsi") is not None else float("nan"),
                            p.get("symbol_2_rsi") if p.get("symbol_2_rsi") is not None else float("nan"))

        # Срезы для снапшота (чтобы не раздувать файл)
        to_persist_sorted = (
            pairs_sorted_dicts[:TOP_N_TO_PERSIST]
            if TOP_N_TO_PERSIST else pairs_sorted_dicts
        )
        to_persist_abs = (
            pairs_abs_dicts[:TOP_N_TO_PERSIST]
            if TOP_N_TO_PERSIST else pairs_abs_dicts
        )
        # weak-пары пишем полностью (обычно их меньше), при желании можно тоже ограничивать
        to_persist_week = week_pairs

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
            "week_corr_pairs": to_persist_week,
        }
        persist_snapshot(snapshot)

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
    parser.add_argument("--weak-abs", type=float, default=WEAK_CORR_ABS_THRESH, help="abs(corr) < this is considered WEAK")
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

    # Первый прогон сразу
    run_cycle(logger)

    if RUN_ONCE:
        logger.info("Run-once mode: exiting.")
        return

    # Затем по расписанию
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
