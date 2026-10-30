
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
DAYS = 365

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

# Oxirgi to'liq soatlik shamgacha yuklaymiz.
now = pd.Timestamp.now(tz="UTC")
end_time = int(now.floor("h").timestamp() * 1000)
start_time = int(
    (now - pd.Timedelta(days=DAYS)).floor("h").timestamp() * 1000
)

session = requests.Session()
all_rows = []
cursor = start_time

while cursor < end_time:
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": cursor,
        "endTime": end_time,
        "limit": 1000,
    }

    response = session.get(URL, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json()

    if not rows:
        break

    all_rows.extend(rows)

    # Keyingi soatlik shamdan davom etamiz.
    next_cursor = int(rows[-1][0]) + 1
    if next_cursor <= cursor:
        break
    cursor = next_cursor

    print(f"Yuklangan shamlar: {len(all_rows)}")
    time.sleep(0.15)

df = pd.DataFrame(all_rows, columns=COLUMNS)

if df.empty:
    raise RuntimeError("API ma'lumot qaytarmadi.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"], unit="ms", utc=True
)

numeric = ["open", "high", "low", "close", "volume"]
df[numeric] = df[numeric].astype(float)

# Faqat to'liq yopilgan shamlar.
df = df[df["timestamp"] + pd.Timedelta(hours=1) <= now]

df = (
    df[["timestamp", "open", "high", "low", "close", "volume"]]
    .drop_duplicates(subset="timestamp")
    .sort_values("timestamp")
    .reset_index(drop=True)
)

BASE_DIR = Path(__file__).resolve().parent
out = BASE_DIR / "data" / "bitcoin_hourly.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print("\n--- NATIJA ---")
print("Jami yozuv:", len(df))
print("Birinchi sana:", df["timestamp"].min())
print("Oxirgi sana:", df["timestamp"].max())
print("Takroriy vaqtlar:", df["timestamp"].duplicated().sum())
print("Fayl:", out.resolve())
print(df.tail())
