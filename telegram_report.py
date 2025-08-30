#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram weak_corr_pairs reporter (list view)
- Экспортирует send_report_once(json_path, token, chat_id, max_rows)
- Формат: удобный для мобильного список; числа округлены до 2 знаков.
"""

from __future__ import annotations

import os
import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Any

import requests

def _load_latest_snapshot(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _select_weak_pairs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    arr = data.get("weak_corr_pairs") or data.get("week_corr_pairs")
    return arr if isinstance(arr, list) else []

def _fmt2(v: Any) -> str:
    # Округление до 2 знаков (для float/int). Пусто для None/NaN/Inf.
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        return f"{float(v):.2f}"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

def _fmt_pval(v: Any) -> str:
    if v is None:
        return ""
    try:
        x = float(v)
    except Exception:
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return ""
    # Очень маленькие p показываем научной нотацией; иначе 2 знака.
    return f"{x:.2e}" if x < 0.01 else f"{x:.2f}"

def _chunk_text(text: str, max_len: int = 4000) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts, cur, cur_len = [], [], 0
    for line in text.splitlines(True):
        if cur_len + len(line) > max_len:
            parts.append("".join(cur)); cur, cur_len = [line], len(line)
        else:
            cur.append(line); cur_len += len(line)
    if cur:
        parts.append("".join(cur))
    return parts

def _send_telegram_html(token: str, chat_id: str, html: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for part in _chunk_text(html):
        r = requests.post(url, data={
            "chat_id": chat_id,
            "text": part,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram error: {r.status_code} {r.text}")

def send_report_once(json_path: str, token: str, chat_id: str, max_rows: int = 30) -> None:
    if not token or not chat_id:
        raise RuntimeError("Telegram token/chat_id not provided")

    data = _load_latest_snapshot(json_path)
    rows = _select_weak_pairs(data)[:max_rows]
    timeframe = data.get("timeframe", "?")
    ts = data.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()

    header = (
        f"<b>Weak Corr Report</b>\n"
        f"Timeframe: <b>{timeframe}</b>\n"
        f"As of: <b>{ts}</b>\n\n"
    )

    if not rows:
        _send_telegram_html(token, chat_id, header + "(weak_corr_pairs пуст)")
        return

    # Список: по одной паре в 2–3 строки, удобочитаемо на мобильном
    lines = []
    for i, r in enumerate(rows, start=1):
        pair = r.get("pair", "")
        s1   = r.get("symbol_1", "")
        s2   = r.get("symbol_2", "")
        corr = _fmt2(r.get("corr"))
        pval = _fmt_pval(r.get("p_value"))
        n    = r.get("n_obs", "")
        rsi1 = _fmt2(r.get("symbol_1_rsi"))
        rsi2 = _fmt2(r.get("symbol_2_rsi"))
        lines.append(
            f"{i}. <b>{pair}</b>\n"
            f"   corr: <b>{corr}</b> | p: {pval} | n: {n}\n"
            f"   RSI: {s1} <b>{rsi1}</b> / {s2} <b>{rsi2}</b>"
        )

    body = "\n\n".join(lines)
    _send_telegram_html(token, chat_id, header + body)
