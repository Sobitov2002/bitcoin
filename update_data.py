"""
update_data.py — Automated Binance Hourly Candle Fetcher.

Designed for server cron execution. Fetches the latest completed hourly candles
from Binance Public API and merges them into data/bitcoin_hourly.csv without duplicates.
"""

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bitcoin_hourly.csv"
BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"


def update_candles() -> int:
    """Fetch recent completed 1h candles from Binance and append to CSV."""
    now_utc = pd.Timestamp.now(tz="UTC")
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Sham ma'lumotlarini yangilash boshlandi...")

    try:
        resp = requests.get(
            BINANCE_URL,
            params={"symbol": SYMBOL, "interval": "1h", "limit": 100},
            timeout=25,
        )
        resp.raise_for_status()
        candles = resp.json()
    except Exception as exc:
        print(f"[XATOLIK] Binance API bilan ulanishda muammo: {exc}")
        return 0

    if not isinstance(candles, list) or not candles:
        print("[OGOHLANTIRISH] Binance API bo'sh javob qaytardi.")
        return 0

    rows = []
    for c in candles:
        open_time = pd.to_datetime(c[0], unit="ms", utc=True)
        # Faqat to'liq yopilgan shamlarni olamiz
        if open_time + pd.Timedelta(hours=1) > now_utc:
            continue

        rows.append({
            "timestamp": open_time,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })

    if not rows:
        print("Yangi to'liq yopilgan shamlar topilmadi.")
        return 0

    fresh_df = pd.DataFrame(rows)

    if DATA_PATH.exists():
        existing_df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
        combined = pd.concat([existing_df, fresh_df], ignore_index=True)
    else:
        combined = fresh_df

    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)

    print(f"Muvaffaqiyatli yakunlandi. Jami yozuvlar: {len(combined)}. Qo'shildi/yangilandi: {len(fresh_df)} ta sham.")
    return len(fresh_df)


if __name__ == "__main__":
    update_candles()
