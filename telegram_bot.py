"""
telegram_bot.py — Telegram Bot & Alert Notification System for Bitcoin AI.

Allows monitoring Bitcoin AI predictions, live prices, accuracy stats, and recording
market observations directly from your smartphone via Telegram without needing a laptop.
Zero external library dependencies (uses requests).
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests

# Set working directory to project root
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import app
import db_manager

# Load environment variables
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
ADMIN_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

API_BASE = f"https://api.telegram.org/bot{TOKEN}"
BINANCE_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/price"
KST = "Asia/Seoul"


# =========================================================
# TELEGRAM API HELPERS
# =========================================================
def send_telegram_message(
    text: str,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    reply_markup: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send a Telegram message to a specific chat_id or the default ADMIN_CHAT_ID."""
    target_chat = chat_id or ADMIN_CHAT_ID
    if not TOKEN or not target_chat:
        return False

    url = f"{API_BASE}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": target_chat,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    try:
        resp = requests.post(url, json=payload, timeout=12)
        return resp.status_code == 200
    except Exception as exc:
        print(f"[Telegram Send Error]: {exc}")
        return False


def get_main_keyboard() -> Dict[str, Any]:
    """Return persistent reply keyboard for quick mobile actions."""
    return {
        "keyboard": [
            [{"text": "📊 4h AI Prognozi"}, {"text": "💰 Jonli Narx"}],
            [{"text": "📈 Aniqlik Statistikasi"}, {"text": "📋 Oxirgi Kuzatuvlar"}],
            [{"text": "➕ Shaxsiy Taxmin Kiritish"}],
        ],
        "resize_keyboard": True,
    }


def get_prediction_inline_keyboard() -> Dict[str, Any]:
    """Return inline buttons for fast prediction recording."""
    return {
        "inline_keyboard": [
            [
                {"text": "🟢 4h Ko'tariladi (UP)", "callback_data": "pred_UP_4h"},
                {"text": "🔴 4h Tushadi (DOWN)", "callback_data": "pred_DOWN_4h"},
            ],
            [
                {"text": "⚪ 4h Yonlama (SIDEWAYS)", "callback_data": "pred_SIDEWAYS_4h"},
            ],
        ]
    }


# =========================================================
# DATA & AI ANALYSIS HELPERS
# =========================================================
def get_live_price_data() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return live_price_usd, closed_price_usd, fx_rate."""
    live_usd, _ = app.get_live_btc_price()
    fx_rate, _, _ = app.get_usd_krw()

    try:
        raw_df = app.load_data()
        closed_usd = float(raw_df["close"].iloc[-1])
    except Exception:
        closed_usd = live_usd or 0.0

    return live_usd, closed_usd, fx_rate


def get_ai_forecast_summary() -> Dict[str, Any]:
    """Compute the current 4-hour AI forecast using trained models."""
    raw_df = app.load_data()
    vol_bundle = app.load_volatility_model()
    dir_bundle = app.load_direction_model()

    featured = app.build_features(raw_df)
    featured = featured.replace([np.inf, -np.inf], np.nan)

    vol_features = vol_bundle["features"]
    vol_model = vol_bundle["model"]
    dir_features = dir_bundle["features"]
    dir_model = dir_bundle["model"]

    valid_data = featured.dropna(subset=vol_features + dir_features)
    latest_row = valid_data.iloc[-1]
    closed_usd = float(latest_row["close"])

    # Volatility prediction
    latest_feat = latest_row[vol_features].to_frame().T
    vol_pred_raw = max(float(vol_model.predict(latest_feat)[0]), 0.0)
    vol_pred_pct = vol_pred_raw * 100.0

    # Direction prediction
    latest_dir_feat = latest_row[dir_features].to_frame().T
    dir_probs = dir_model.predict_proba(latest_dir_feat)[0]
    classes = list(dir_model.classes_)
    prob_up = float(dir_probs[classes.index(1)]) * 100.0
    prob_down = float(dir_probs[classes.index(0)]) * 100.0

    if prob_up > prob_down:
        ai_direction = "UP"
    elif prob_down > prob_up:
        ai_direction = "DOWN"
    else:
        ai_direction = "EQUAL"

    # 4-hour approximate price range
    sigma_4h = vol_pred_raw * np.sqrt(4)
    lower_usd = closed_usd * np.exp(-1.96 * sigma_4h)
    upper_usd = closed_usd * np.exp(1.96 * sigma_4h)

    candle_close_time = pd.Timestamp(latest_row["timestamp"]) + pd.Timedelta(hours=1)

    return {
        "closed_price_usd": closed_usd,
        "direction": ai_direction,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "volatility_pct": vol_pred_pct,
        "lower_usd": lower_usd,
        "upper_usd": upper_usd,
        "candle_close_time": candle_close_time,
        "model_name": dir_bundle.get("model_name", "Random Forest"),
    }


# =========================================================
# MESSAGE FORMATTERS
# =========================================================
def format_forecast_message() -> str:
    live_usd, closed_usd, fx_rate = get_live_price_data()
    fc = get_ai_forecast_summary()

    cur_price = live_usd or closed_usd
    krw_text = f" (₩{cur_price * fx_rate:,.0f})" if fx_rate else ""

    if fc["direction"] == "UP":
        dir_badge = "🟢 <b>KO'TARILISH EHTIMOLI YUQORI (UP)</b>"
    elif fc["direction"] == "DOWN":
        dir_badge = "🔴 <b>TUSHISH EHTIMOLI YUQORI (DOWN)</b>"
    else:
        dir_badge = "⚪ <b>IKKALA YO'NALISH EHTIMOLI TENG</b>"

    close_kst_str = app.format_timezone(fc["candle_close_time"], KST)

    msg = (
        f"⚡️ <b>BITCOIN 4 SOATLIK AI PROGNOZI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 <b>Joriy Narx:</b> ${cur_price:,.2f}{krw_text}\n"
        f"🕒 <b>Oxirgi sham:</b> {close_kst_str} KST\n\n"
        f"🎯 <b>Prognoz yo'nalishi:</b>\n{dir_badge}\n"
        f"• Ko'tarilish ehtimoli: <b>{fc['prob_up']:.1f}%</b>\n"
        f"• Tushish ehtimoli: <b>{fc['prob_down']:.1f}%</b>\n\n"
        f"📉 <b>Kutilayotgan volatillik:</b> <code>{fc['volatility_pct']:.3f}%</code>\n"
        f"📊 <b>Taxminiy 4h diapazon:</b>\n"
        f"   🔻 Pastki: <b>${fc['lower_usd']:,.2f}</b>\n"
        f"   🔺 Yuqori: <b>${fc['upper_usd']:,.2f}</b>\n\n"
        f"ℹ️ <i>Model: {fc['model_name']} · Horizon: 4 soat</i>\n"
        f"⚠️ <i>Eslatma: Bu moliyaviy maslahat emas.</i>"
    )
    return msg


def format_price_message() -> str:
    live_usd, closed_usd, fx_rate = get_live_price_data()
    cur_price = live_usd or closed_usd
    krw_str = f"₩{cur_price * fx_rate:,.0f} KRW" if fx_rate else "Mavjud emas"
    fx_rate_str = f"₩{fx_rate:,.2f}" if fx_rate else "—"

    now_kst = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

    msg = (
        f"🪙 <b>BITCOIN BOZOR KOTIROVKASI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💵 <b>USD Narxi:</b> <code>${cur_price:,.2f}</code>\n"
        f"🇰🇷 <b>KRW Narxi:</b> <code>{krw_str}</code>\n"
        f"💱 <b>USD/KRW Kursi:</b> {fx_rate_str}\n"
        f"🕒 <b>Vaqt:</b> {now_kst}\n"
        f"📡 <b>Manba:</b> Binance BTCUSDT"
    )
    return msg


def format_stats_message() -> str:
    stats = db_manager.calculate_accuracy_stats()

    ai_acc = f"{stats['ai_accuracy']:.1f}%" if stats["ai_accuracy"] is not None else "—"
    user_acc = f"{stats['user_accuracy']:.1f}%" if stats["user_accuracy"] is not None else "—"

    msg = (
        f"📈 <b>AI VA INSON ANIKLIK STATISTIKASI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 <b>AI aniqligi:</b> <b>{ai_acc}</b>\n"
        f"👤 <b>Sizning aniqligingiz:</b> <b>{user_acc}</b>\n\n"
        f"✅ <b>Baholangan kuzatuvlar:</b> {stats['completed_count']} ta\n"
        f"⏳ <b>Hali kutilayotganlar:</b> {stats['pending_count']} ta\n"
        f"📁 <b>Jami qaydlar:</b> {stats['total_observations']} ta\n\n"
        f"🔬 <i>1h ufq: AI ({stats['ai_1h_accuracy'] or '—'}%) · Siz ({stats['user_1h_accuracy'] or '—'}%)\n"
        f"🔬 4h ufq: AI ({stats['ai_4h_accuracy'] or '—'}%) · Siz ({stats['user_4h_accuracy'] or '—'}%)</i>"
    )
    return msg


def format_history_message() -> str:
    obs_list = db_manager.get_all_observations(limit=5)
    if not obs_list:
        return "📋 Hozircha hech qanday kuzatuv qayd etilmagan."

    msg = "📋 <b>OXIRGI 5 TA KUZATUV NATIJASI:</b>\n━━━━━━━━━━━━━━━━━━━\n"
    for i, o in enumerate(obs_list, 1):
        status = o.get("status", "PENDING")
        time_str = o.get("created_at_utc", "")[:16].replace("T", " ")
        init_p = float(o.get("initial_price", 0))
        ai_d = o.get("ai_forecast", {}).get("direction", "—")
        user_d = o.get("user_prediction", {}).get("direction", "—")

        eval_data = o.get("evaluation", {})
        if status == "COMPLETED":
            ai_icon = "✅" if eval_data.get("ai_correct") else "❌"
            user_icon = "✅" if eval_data.get("user_correct") else "❌"
            actual_p = eval_data.get("actual_price", 0)
            status_text = f"Natija: ${actual_p:,.2f}"
        else:
            ai_icon = "⏳"
            user_icon = "⏳"
            status_text = "Kutilmoqda..."

        msg += (
            f"<b>{i}. {time_str} UTC</b> ({o.get('horizon_hours', 4)}h)\n"
            f"• Narx: ${init_p:,.2f} → {status_text}\n"
            f"• AI: {ai_d} {ai_icon} | Siz: {user_d} {user_icon}\n\n"
        )
    return msg


def record_telegram_observation(user_direction: str, horizon_hours: int = 4) -> str:
    """Save an observation initiated from Telegram."""
    live_usd, closed_usd, _ = get_live_price_data()
    cur_price = live_usd or closed_usd
    fc = get_ai_forecast_summary()

    now_utc = datetime.now(timezone.utc)
    target_utc = now_utc + pd.Timedelta(hours=horizon_hours)

    doc = {
        "created_at_utc": now_utc.isoformat(),
        "created_at_local": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "horizon_hours": horizon_hours,
        "target_time_utc": target_utc.isoformat(),
        "initial_price": float(cur_price),
        "price_source": "Binance BTCUSDT (Telegram)",
        "currency": "USD",
        "ai_forecast": {
            "direction": fc["direction"],
            "probability_up": fc["prob_up"],
            "probability_down": fc["prob_down"],
            "volatility_pct": fc["volatility_pct"],
            "approx_lower": fc["lower_usd"],
            "approx_upper": fc["upper_usd"],
            "model_version": fc["model_name"],
        },
        "user_prediction": {
            "direction": user_direction,
            "notes": "Telegram bot orqali kiritildi",
        },
        "status": "PENDING",
        "evaluation": {},
    }

    ok, msg, _ = db_manager.save_observation(doc)
    if ok:
        dir_icon = "🟢" if user_direction == "UP" else ("🔴" if user_direction == "DOWN" else "⚪")
        return (
            f"✅ <b>Kuzatuv muvaffaqiyatli saqlandi!</b>\n\n"
            f"💰 Boshlang'ich narx: <b>${cur_price:,.2f}</b>\n"
            f"👤 Sizning taxminingiz: {dir_icon} <b>{user_direction}</b>\n"
            f"🤖 AI bashorati: <b>{fc['direction']}</b>\n"
            f"⏱ Muddat: <b>{horizon_hours} soat</b>\n\n"
            f"<i>Muddati tugagach, natija avtomatik tekshirilib sizga xabar yuboriladi!</i>"
        )
    return f"❌ Xatolik yuz berdi: {msg}"


# =========================================================
# TELEGRAM POLLING LOOP
# =========================================================
def run_telegram_bot():
    """Run persistent Telegram long polling listener."""
    if not TOKEN:
        print("[XATOLIK] .env faylida TELEGRAM_BOT_TOKEN topilmadi!")
        print("BotFather orqali yangi bot ochib, tokenni .env fayliga kiriting.")
        return

    print("🤖 Telegram bot ishga tushdi...")
    offset = 0

    while True:
        try:
            url = f"{API_BASE}/getUpdates"
            params = {"offset": offset, "timeout": 25}
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                time.sleep(2)
                continue

            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1

                # Handle Inline Button Clicks
                if "callback_query" in update:
                    cq = update["callback_query"]
                    cb_data = cq.get("data", "")
                    sender_chat_id = cq["message"]["chat"]["id"]

                    if cb_data.startswith("pred_"):
                        # e.g., pred_UP_4h
                        parts = cb_data.split("_")
                        user_choice = parts[1]
                        result_text = record_telegram_observation(user_choice, horizon_hours=4)
                        send_telegram_message(result_text, chat_id=sender_chat_id, reply_markup=get_main_keyboard())

                    # Acknowledge callback
                    try:
                        requests.post(f"{API_BASE}/answerCallbackQuery", json={"callback_query_id": cq["id"]})
                    except Exception:
                        pass
                    continue

                # Handle Text Messages
                message = update.get("message", {})
                text = message.get("text", "").strip()
                sender_chat_id = message.get("chat", {}).get("id")

                if not text or not sender_chat_id:
                    continue

                if text == "/start" or text == "/help":
                    welcome_msg = (
                        f"Assalomu alaykum, <b>{message.get('from', {}).get('first_name', 'Foydalanuvchi')}</b>! 👋\n\n"
                        f"Men <b>Bitcoin AI Tahlil va Bashorat Boti</b>man.\n"
                        f"Quyidagi tugmalar orqali istalgan vaqtda bozor holatini va AI prognozlarini kuzatib borishingiz mumkin:\n\n"
                        f"• <b>📊 4h AI Prognozi:</b> Keyingi 4 soatlik yo'nalish va narx oraliqlari\n"
                        f"• <b>💰 Jonli Narx:</b> Ayni lahzadagi BTC/USDT va KRW narxlari\n"
                        f"• <b>📈 Aniqlik:</b> AI va sizning taxminlaringiz yutuq foizi\n"
                        f"• <b>➕ Shaxsiy Taxmin:</b> Bozor bo'yicha o'z taxminingizni saqlash"
                    )
                    send_telegram_message(welcome_msg, chat_id=sender_chat_id, reply_markup=get_main_keyboard())

                elif text in ["/forecast", "📊 4h AI Prognozi"]:
                    send_telegram_message(format_forecast_message(), chat_id=sender_chat_id, reply_markup=get_main_keyboard())

                elif text in ["/price", "💰 Jonli Narx"]:
                    send_telegram_message(format_price_message(), chat_id=sender_chat_id, reply_markup=get_main_keyboard())

                elif text in ["/stats", "📈 Aniqlik Statistikasi"]:
                    send_telegram_message(format_stats_message(), chat_id=sender_chat_id, reply_markup=get_main_keyboard())

                elif text in ["/history", "📋 Oxirgi Kuzatuvlar"]:
                    send_telegram_message(format_history_message(), chat_id=sender_chat_id, reply_markup=get_main_keyboard())

                elif text in ["/predict", "➕ Shaxsiy Taxmin Kiritish"]:
                    prompt_text = (
                        "🎯 <b>Keyingi 4 soatda Bitcoin narxi qaysi tomonga harakat qiladi deb o'ylaysiz?</b>\n\n"
                        "Quyidagi tugmalardan birini tanlang:"
                    )
                    send_telegram_message(prompt_text, chat_id=sender_chat_id, reply_markup=get_prediction_inline_keyboard())

                else:
                    # Echo / default help
                    send_telegram_message(
                        "Iltimos, pastdagi menyu tugmalaridan birini tanlang 👇",
                        chat_id=sender_chat_id,
                        reply_markup=get_main_keyboard(),
                    )

        except Exception as exc:
            print(f"[Bot Polling Exception]: {exc}")
            time.sleep(3)


if __name__ == "__main__":
    run_telegram_bot()
