
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/bitcoin_hourly.csv")
MODEL_PATH = Path("models/ridge_volatility.joblib")

FEATURES = [
    "vol_4h",
    "vol_24h",
    "vol_72h",
    "return_1h",
    "return_4h",
    "return_24h",
    "volume_change_24h",
    "range_pct",
]

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = (
    df.sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

close = df["close"]
volume = df["volume"]

df["log_return"] = np.log(close / close.shift(1))
future_returns = pd.concat(
    [
        df["log_return"].shift(-step)
        for step in range(1, 5)
    ],
    axis=1,
)

df["target_volatility"] = future_returns.std(
    axis=1, ddof=0
)

df["vol_4h"] = df["log_return"].rolling(4).std()
df["vol_24h"] = df["log_return"].rolling(24).std()
df["vol_72h"] = df["log_return"].rolling(72).std()
df["return_1h"] = df["log_return"]
df["return_4h"] = np.log(close / close.shift(4))
df["return_24h"] = np.log(close / close.shift(24))
df["volume_change_24h"] = volume.pct_change(24)
df["range_pct"] = (df["high"] - df["low"]) / close

# Targeti hali to'liq kuzatilgan qatorlardagina o'qitamiz.
train_data = df.dropna(
    subset=FEATURES + ["target_volatility"]
)

if len(train_data) < 1000:
    raise ValueError("O'qitish uchun ma'lumot yetarli emas.")

X = train_data[FEATURES]
y = train_data["target_volatility"]

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0),
)

model.fit(X, y)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "model_name": "Ridge Regression",
        "features": FEATURES,
        "horizon_hours": 4,
        "target": "std_of_next_4_hourly_log_returns",
        "trained_until": str(train_data["timestamp"].max()),
    },
    MODEL_PATH,
)

print("Ridge volatillik modeli saqlandi.")
print("O'qitish yozuvlari:", len(train_data))
print("Oxirgi targeti ma'lum vaqt:",
      train_data["timestamp"].max())
print("Model fayli:", MODEL_PATH.resolve())
