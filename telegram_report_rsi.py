#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram High/Low RSI reporter (всегда .txt) + 4h в таблице
- Берём символы из snapshot["symbols"] и данные из snapshot["indicators_all"].
- Ранжируем по RSI на rank_tf (обычно 1h): TOP N и BOTTOM N.
- Всегда отправляем ОДИН .txt с коротким caption.
- Порядок отображения ТФ фиксирован: 15m | 1h | 4h | 1d (если какого-то ТФ нет в снапшоте — покажется '—').
"""

from __future__ import annotations

import os
import re
import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
import html as html_mod

# ----- helpers -----

def _load_latest_snapshot(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _fmt2(v: Any) -> str:
    if v is None:
        return "—"
    try:
        vf = float(v)
        if math.isnan(vf) or math.isinf(vf):
            return "—"
        return f"{vf:.2f}"
    except Exception:
        return "—"

def _macd_label(m: Optional[Dict[str, Any]], eps: float = 1e-6) -> str:
    if not m:
        return "—"
    h = m.get("hist")
    try:
        if h is not None:
            hf = float(h)
            if not math.isnan(hf) and not math.isinf(hf):
                return "BULL" if hf > eps else "BEAR"
    except Exception:
        pass
    try:
        macd = m.get("macd")
        if macd is None:
            return "—"
        mf = float(macd)
        if math.isnan(mf) or math.isinf(mf):
            return "—"
        return "BULL" if mf > eps else "BEAR"
    except Exception:
        return "—"

def _get_ind(ind_all: Dict[str, Any], symbol: str, tf: str) -> Dict[str, Any]:
    symmap = (ind_all.get(symbol) or {})
    return (symmap.get(tf) or {})

def _build_symbol_block(ind_all: Dict[str, Any], symbol: str, tfs_for_print: List[str]) -> str:
    rsi_line = " | ".join([f"{tf}: {_fmt2((_get_ind(ind_all, symbol, tf).get('rsi')))}" for tf in tfs_for_print])
    macd_line = " | ".join([f"{tf}: {_macd_label(_get_ind(ind_all, symbol, tf).get('macd'))}" for tf in tfs_for_print])
    return (
        f"<b>{symbol}</b>\n"
        f"      • RSI  {rsi_line}\n"
        f"      • MACD {macd_line}"
    )

def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit-1] + "…"

def _html_to_text(html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(text)

def _send_telegram_document(token: str, chat_id: str, filename: str, content_text: str, caption: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    files = {
        "document": (filename, content_text.encode("utf-8"), "text/plain"),
    }
    data = {
        "chat_id": chat_id,
        "caption": _truncate(caption, 950),
        "parse_mode": "HTML",
        "disable_content_type_detection": False,
    }
    r = requests.post(url, data=data, files=files, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error (doc): {r.status_code} {r.text}")

def _iso_trim(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z","+00:00"))
        return dt.replace(microsecond=0).isoformat()
    except Exception:
        return ts.split(".")[0] if "." in ts else ts

# ----- entry -----

def send_report_once(json_path: str, token: str, chat_id: str, top_n: int = 10, rank_tf: str = "1h") -> None:
    if not token or not chat_id:
        raise RuntimeError("Telegram token/chat_id not provided")

    data = _load_latest_snapshot(json_path)
    symbols: List[str] = list(data.get("symbols") or [])
    if not symbols:
        raise RuntimeError("No symbols in snapshot to rank by RSI")

    # ТФ, которые реально есть в снапшоте (рассчитывались основным ботом)
    tfs_snapshot: List[str] = data.get("indicator_tfs") or ["15m","1h","1d"]
    ind_all: Dict[str, Any] = data.get("indicators_all") or {}
    if not ind_all:
        raise RuntimeError("Snapshot has no indicators_all (enable it in the bot)")

    # Фиксируем порядок отображения: 15m | 1h | 4h | 1d
    preferred_order = ["15m", "1h", "4h", "1d"]
    # Берём только те, что есть в снапшоте, но сохраняем предпочтительный порядок
    tfs_for_print = [tf for tf in preferred_order if tf in tfs_snapshot]
    # Если в снапшоте есть доп. ТФ (редкий случай) — добавим их хвостом
    extras = [tf for tf in tfs_snapshot if tf not in tfs_for_print]
    tfs_for_print += extras

    timeframe = data.get("timeframe", "?")
    ts = _iso_trim(data.get("timestamp_utc") or datetime.now(timezone.utc).isoformat())

    # Готовим список (symbol, rsi на rank_tf)
    ranked: List[Tuple[str, float]] = []
    for s in symbols:
        val = (_get_ind(ind_all, s, rank_tf) or {}).get("rsi")
        try:
            if val is None:
                continue
            f = float(val)
            if not math.isnan(f) and not math.isinf(f):
                ranked.append((s, f))
        except Exception:
            continue

    if not ranked:
        raise RuntimeError(f"No RSI values found for tf={rank_tf}")

    ranked.sort(key=lambda x: x[1], reverse=True)
    high = ranked[:top_n]
    low  = list(reversed(ranked[-top_n:]))

    header = (
        f"<b>BINANCE Futures : High/Low RSI Report</b>\n"
        f"Pairs TF: <b>{rank_tf}</b>\n"
        f"As of: <b>{ts}</b>\n"
    )

    lines_high = ["\n<b>HIGH RSI:</b>"]
    for sym, _ in high:
        lines_high.append(_build_symbol_block(ind_all, sym, tfs_for_print))

    lines_low = ["\n<b>LOW RSI:</b>"]
    for sym, _ in low:
        lines_low.append(_build_symbol_block(ind_all, sym, tfs_for_print))

    full_html = header + "\n".join(lines_high) + "\n\n" + "\n".join(lines_low)
    text_content = _html_to_text(full_html)
    caption = f"<b>High/Low RSI</b> • TF: <b>{rank_tf}</b> • {ts}"
    _send_telegram_document(token, chat_id, "rsi_report.txt", text_content, caption)
