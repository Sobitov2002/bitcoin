
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_PATH = Path("data/bitcoin_hourly.csv")
MODEL_PATH = Path("models/ridge_volatility.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model topilmadi. Avval train_volatility_model.py ni ishga tushiring."
    )

if not DATA_PATH.exists():
    raise FileNotFoundError("Bitcoin CSV ma'lumotlari topilmadi.")

# Saqlangan modelni yuklash
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

# Ma'lumotni yuklash
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = (
    df.sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

close = df["close"]
volume = df["volume"]

# Model o'qitilgandagi feature'lar bilan bir xil
df["log_return"] = np.log(close / close.shift(1))
df["vol_4h"] = df["log_return"].rolling(4).std()
df["vol_24h"] = df["log_return"].rolling(24).std()
df["vol_72h"] = df["log_return"].rolling(72).std()
df["return_1h"] = df["log_return"]
df["return_4h"] = np.log(close / close.shift(4))
df["return_24h"] = np.log(close / close.shift(24))
df["volume_change_24h"] = volume.pct_change(24)
df["range_pct"] = (df["high"] - df["low"]) / close

latest = df.dropna(subset=features).iloc[[-1]]
prediction = float(model.predict(latest[features])[0])
prediction = max(prediction, 0.0)

latest_price = float(latest["close"].iloc[0])
latest_time = latest["timestamp"].iloc[0]

# Log-return standart og'ishini foizga aylantiramiz.
volatility_percent = prediction * 100

print("\n===== BITCOIN 4 SOATLIK VOLATILLIK PROGNOZI =====")
print(f"Ma'lumot vaqti (UTC): {latest_time}")
print(f"So'nggi yopilish narxi: ${latest_price:,.2f}")
print(f"Prognoz qilingan volatillik: {volatility_percent:.4f}%")
print("Prognoz ufqi: keyingi 4 soat")
print("\nEslatma:")
print("- Bu narxning ko'tarilishi yoki tushishini bildirmaydi.")
print("- Bu ko'rsatkich kafolatlangan narx oralig'i emas.")
print("- Prognoz so'nggi mavjud soatlik ma'lumotga asoslangan.")
