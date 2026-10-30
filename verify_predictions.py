"""
verify_predictions.py — Automated Prediction Verification Worker.

Fetches pending observations whose forecast period has matured, retrieves actual
market prices from Binance Public API, computes direction accuracy, and updates records.
Can be executed as a background worker, via cron/systemd, or triggered from the UI.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# Base directory & DB manager
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import db_manager

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
SIDEWAYS_THRESHOLD_PCT = 0.08  # Price changes within +-0.08% are classified as sideways


def get_actual_price_at_timestamp(target_dt_utc: datetime) -> Optional[float]:
    """Retrieve the actual BTCUSDT close price for the 1h candle corresponding to target_dt_utc."""
    # Ensure timezone-aware UTC
    if target_dt_utc.tzinfo is None:
        target_dt_utc = target_dt_utc.replace(tzinfo=timezone.utc)
    else:
        target_dt_utc = target_dt_utc.astimezone(timezone.utc)

    # Convert to candle open timestamp
    candle_time = target_dt_utc.floor("h") if hasattr(target_dt_utc, "floor") else pd.Timestamp(target_dt_utc).floor("h")
    start_ms = int(candle_time.timestamp() * 1000)

    params = {
        "symbol": SYMBOL,
        "interval": "1h",
        "startTime": start_ms,
        "limit": 2,
    }

    try:
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
        resp.raise_for_status()
        candles = resp.json()
        if not candles or not isinstance(candles, list):
            return None

        # Return close price of the candle
        return float(candles[0][4])
    except Exception as exc:
        print(f"[Verification Error] Could not fetch candle at {target_dt_utc}: {exc}")
        return None


def evaluate_single_observation(obs: Dict[str, Any], now_utc: datetime) -> Optional[Dict[str, Any]]:
    """Check and evaluate an individual observation if matured."""
    obs_id = str(obs.get("_id"))
    target_str = obs.get("target_time_utc")
    if not target_str:
        return None

    try:
        target_dt = pd.to_datetime(target_str, utc=True).to_pydatetime()
    except Exception:
        return None

    # If horizon has not yet matured, skip
    if target_dt > now_utc:
        return None

    initial_price = float(obs.get("initial_price", 0.0))
    if initial_price <= 0:
        return None

    # Fetch actual market price
    actual_price = get_actual_price_at_timestamp(target_dt)
    if actual_price is None:
        return {
            "status": "ERROR",
            "error": "Binance API dan haqiqiy narxni olib bo'lmadi",
        }

    price_diff = actual_price - initial_price
    pct_diff = (price_diff / initial_price) * 100.0

    # Determine actual direction
    if pct_diff > SIDEWAYS_THRESHOLD_PCT:
        actual_dir = "UP"
    elif pct_diff < -SIDEWAYS_THRESHOLD_PCT:
        actual_dir = "DOWN"
    else:
        actual_dir = "SIDEWAYS"

    # Evaluate AI
    ai_raw = obs.get("ai_forecast", {}).get("direction", "").upper()
    if "UP" in ai_raw or "KO" in ai_raw:
        ai_dir = "UP"
    elif "DOWN" in ai_raw or "TUSH" in ai_raw:
        ai_dir = "DOWN"
    else:
        ai_dir = "EQUAL"

    ai_correct = (ai_dir == actual_dir)

    # Evaluate User
    user_raw = obs.get("user_prediction", {}).get("direction", "").upper()
    if "UP" in user_raw or "KO" in user_raw:
        user_dir = "UP"
    elif "DOWN" in user_raw or "TUSH" in user_raw:
        user_dir = "DOWN"
    elif "SIDE" in user_raw or "YON" in user_raw:
        user_dir = "SIDEWAYS"
    else:
        user_dir = user_raw

    user_correct = (user_dir == actual_dir)

    eval_data = {
        "actual_price": actual_price,
        "price_change": round(price_diff, 2),
        "price_change_pct": round(pct_diff, 4),
        "actual_direction": actual_dir,
        "ai_correct": ai_correct,
        "user_correct": user_correct,
        "evaluated_at_utc": now_utc.isoformat(),
        "error_message": None,
    }

    db_manager.update_observation_evaluation(obs_id, eval_data, new_status="COMPLETED")

    # Send automated Telegram alert if configured
    try:
        import telegram_bot
        ai_res_str = "✅ TO'G'RI" if ai_correct else "❌ XATO"
        user_res_str = "✅ TO'G'RI" if user_correct else "❌ XATO"
        change_sign = "+" if price_diff > 0 else ""
        notif_text = (
            f"🔔 <b>KUZATUV NATIJASI TAYYOR!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Boshlang'ich narx: <b>${initial_price:,.2f}</b>\n"
            f"🏁 Haqiqiy narx: <b>${actual_price:,.2f}</b> ({change_sign}{pct_diff:.2f}%)\n"
            f"📊 Bozor yo'nalishi: <b>{actual_dir}</b>\n\n"
            f"🤖 AI prognozi: <b>{ai_dir}</b> ({ai_res_str})\n"
            f"👤 Sizning taxminingiz: <b>{user_dir}</b> ({user_res_str})\n\n"
            f"⏱ <i>Kuzatuv ufqi: {obs.get('horizon_hours', 4)} soat</i>"
        )
        telegram_bot.send_telegram_message(notif_text)
    except Exception as exc:
        print(f"[Telegram Notification Error]: {exc}")

    return eval_data


def run_verification() -> Dict[str, Any]:
    """Scan all pending observations and evaluate those that have reached their target time."""
    pending_list = db_manager.get_pending_observations()
    now_utc = datetime.now(timezone.utc)

    evaluated_count = 0
    skipped_count = 0
    error_count = 0

    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}] Tekshiruv boshlandi. Kutilayotgan kuzatuvlar: {len(pending_list)}")

    for obs in pending_list:
        res = evaluate_single_observation(obs, now_utc)
        if res is None:
            skipped_count += 1
        elif res.get("status") == "ERROR":
            error_count += 1
        else:
            evaluated_count += 1

    summary = {
        "checked_at": now_utc.isoformat(),
        "total_pending": len(pending_list),
        "evaluated": evaluated_count,
        "still_waiting": skipped_count,
        "errors": error_count,
    }

    print(f"Tekshiruv yakunlandi: {evaluated_count} ta baholandi, {skipped_count} ta hali kutilmoqda, {error_count} ta xato.")
    return summary


if __name__ == "__main__":
    run_verification()
