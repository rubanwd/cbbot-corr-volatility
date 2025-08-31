#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram weak_corr_pairs reporter
- MACD = BULL/BEAR (по знаку histogram, fallback macd)
- Если отчёт > лимита Telegram (~4096), шлём ОДНО сообщение с .txt файлом.
- BB не показываем.
"""

from __future__ import annotations

import os
import re
import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
import html as html_mod

# ─────────── I/O helpers ───────────

def _load_latest_snapshot(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _select_weak_pairs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    arr = data.get("weak_corr_pairs") or data.get("week_corr_pairs")
    return arr if isinstance(arr, list) else []

# ─────────── formatting ───────────

def _fmt2(v: Any) -> str:
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

def _get_ind(data: Dict[str, Any], symbol: str, tf: str) -> Dict[str, Any]:
    inds = data.get("indicators") or {}
    symmap = inds.get(symbol) or {}
    return symmap.get(tf) or {}

def _fmt_rsi_line(rsi_map: Dict[str, Optional[float]], tfs: List[str]) -> str:
    parts = []
    for tf in tfs:
        v = rsi_map.get(tf)
        parts.append(f"{tf}: {_fmt2(v) if v is not None else '—'}")
    return " | ".join(parts)

def _macd_label(m: Optional[Dict[str, Any]], eps: float = 1e-6) -> str:
    """
    Возвращает 'BULL' если hist>eps (или macd>eps, если hist нет),
    иначе 'BEAR'. При отсутствии данных — '—'.
    """
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

def _fmt_macd_labels_line(macd_map: Dict[str, Optional[Dict[str, Any]]], tfs: List[str]) -> str:
    parts = []
    for tf in tfs:
        parts.append(f"{tf}: {_macd_label(macd_map.get(tf))}")
    return " | ".join(parts)

def _build_symbol_block(data: Dict[str, Any], symbol: str, tfs: List[str]) -> str:
    rsi_map: Dict[str, Optional[float]] = {}
    macd_map: Dict[str, Optional[Dict[str, Any]]] = {}
    for tf in tfs:
        ind = _get_ind(data, symbol, tf)
        rsi_map[tf] = ind.get("rsi")
        macd_map[tf] = ind.get("macd")
    rsi_line = _fmt_rsi_line(rsi_map, tfs)
    macd_line = _fmt_macd_labels_line(macd_map, tfs)
    return (
        f"<b>{symbol}</b>\n"
        f"      • RSI  {rsi_line}\n"
        f"      • MACD {macd_line}"
    )

# ─────────── Telegram senders ───────────

TG_TEXT_LIMIT = 4096
# оставим запас на всякий случай (HTML-теги иногда считаются по-разному)
TG_SAFE_LIMIT = 3900

def _send_telegram_html_single(token: str, chat_id: str, html: str) -> None:
    """Отправляет ОДНО HTML-сообщение (без чанков). Вызывай только если <= TG_SAFE_LIMIT."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={
        "chat_id": chat_id,
        "text": html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error: {r.status_code} {r.text}")

def _html_to_text(html: str) -> str:
    # самый простой «стрипер»: убираем теги и декодируем сущности
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_mod.unescape(text)
    return text

def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit-1] + "…"

def _send_telegram_document(token: str, chat_id: str, filename: str, content_text: str, caption: str) -> None:
    """Отправляет один .txt с короткой подписью (caption у Telegram ограничен ~1024)."""
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    files = {
        "document": (filename, content_text.encode("utf-8"), "text/plain"),
    }
    data = {
        "chat_id": chat_id,
        "caption": _truncate(caption, 950),
        "parse_mode": "HTML",  # caption можно жирнить
        "disable_content_type_detection": False,
    }
    r = requests.post(url, data=data, files=files, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram error (doc): {r.status_code} {r.text}")

# ─────────── Entry ───────────

def send_report_once(json_path: str, token: str, chat_id: str, max_rows: int = 30) -> None:
    if not token or not chat_id:
        raise RuntimeError("Telegram token/chat_id not provided")

    data = _load_latest_snapshot(json_path)
    rows = _select_weak_pairs(data)[:max_rows]
    timeframe = data.get("timeframe", "?")
    ts = data.get("timestamp_utc") or datetime.now(timezone.utc).isoformat()
    tfs: List[str] = data.get("indicator_tfs") or ["15m", "1h", "1d"]

    header = (
        f"<b>Weak Corr Report</b>\n"
        f"Pairs TF: <b>{timeframe}</b>\n"
        f"As of: <b>{ts}</b>\n\n"
    )

    if not rows:
        _send_telegram_html_single(token, chat_id, header + "(weak_corr_pairs пуст)")
        return

    lines = []
    for i, r in enumerate(rows, start=1):
        pair = r.get("pair", "")
        s1   = r.get("symbol_1", "")
        s2   = r.get("symbol_2", "")
        corr = _fmt2(r.get("corr"))
        block = (
            f"{i}. <b>{pair}</b>\n"
            f"   corr: <b>{corr}</b>\n"
            f"   {_build_symbol_block(data, s1, tfs)}\n"
            f"   {_build_symbol_block(data, s2, tfs)}"
        )
        lines.append(block)

    body_html = "\n\n".join(lines)
    full_html = header + body_html

    if len(full_html) <= TG_SAFE_LIMIT:
        # поместилось — одно сообщение
        _send_telegram_html_single(token, chat_id, full_html)
    else:
        # длинно — одно сообщение с .txt
        text_content = _html_to_text(full_html)
        caption = f"<b>Weak Corr Report</b> • TF: <b>{timeframe}</b> • {ts}\nПолный отчёт во вложении."
        _send_telegram_document(token, chat_id, "corr_report.txt", text_content, caption)
