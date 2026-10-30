
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA_PATH = Path("data/bitcoin_hourly.csv")
OUTPUT_PATH = Path("data/volatility_results.csv")

HORIZON = 4
N_SPLITS = 5
TEST_SIZE = 24 * 14

def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))

# 1. Ma'lumotlarni yuklash va tekshirish
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = (
    df.sort_values("timestamp")
    .drop_duplicates("timestamp")
    .reset_index(drop=True)
)

if not df["timestamp"].is_monotonic_increasing:
    raise ValueError("Vaqt bo'yicha tartib xato.")

time_diff = df["timestamp"].diff()
gaps = int((time_diff > pd.Timedelta(hours=1)).sum())

# 2. Soatlik log-return
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

# Joriy soatdan KEYINGI 4 soatning realized volatility qiymati.
# Faqat kelajakdagi log-returnlar target ichida ishlatiladi.
future_returns = pd.concat(
    [
        df["log_return"].shift(-step)
        for step in range(1, HORIZON + 1)
    ],
    axis=1,
)

df["target_volatility"] = future_returns.std(
    axis=1, ddof=0
)

# 3. Faqat prognoz vaqtidagi yoki undan oldingi xususiyatlar
df["vol_4h"] = df["log_return"].rolling(4).std()
df["vol_24h"] = df["log_return"].rolling(24).std()
df["vol_72h"] = df["log_return"].rolling(72).std()
df["return_1h"] = df["log_return"]
df["return_4h"] = np.log(df["close"] / df["close"].shift(4))
df["return_24h"] = np.log(df["close"] / df["close"].shift(24))
df["volume_change_24h"] = df["volume"].pct_change(24)
df["range_pct"] = (df["high"] - df["low"]) / df["close"]

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

data = df.dropna(
    subset=FEATURES + ["target_volatility"]
).reset_index(drop=True)

if len(data) <= N_SPLITS * TEST_SIZE + HORIZON:
    raise ValueError("Walk-forward test uchun ma'lumot yetarli emas.")

# 4. Modellar
models = {
    "Volatility Baseline": None,
    "Ridge": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
first_test_start = len(data) - N_SPLITS * TEST_SIZE

# 5. Walk-forward validation
for split_no in range(N_SPLITS):
    test_start = first_test_start + split_no * TEST_SIZE
    test_end = test_start + TEST_SIZE
    train_end = test_start - HORIZON

    train = data.iloc[:train_end]
    test = data.iloc[test_start:test_end]

    X_train = train[FEATURES]
    y_train = train["target_volatility"]
    X_test = test[FEATURES]
    y_test = test["target_volatility"]

    print(
        f"\nSplit {split_no + 1}/{N_SPLITS}: "
        f"Train={len(train)}, Test={len(test)}"
    )

    for name, model in models.items():
        if model is None:
            # Oddiy baseline: oxirgi 24 soatdagi volatillik
            pred = test["vol_24h"].to_numpy()
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        # Volatillik manfiy bo'la olmaydi
        pred = np.maximum(pred, 0)

        results.append({
            "split": split_no + 1,
            "model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": rmse(y_test, pred),
            "actual_vol_mean": y_test.mean(),
            "predicted_vol_mean": pred.mean(),
            "test_start": test["timestamp"].iloc[0],
            "test_end": test["timestamp"].iloc[-1],
        })

results_df = pd.DataFrame(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- HAR BIR DAVR NATIJASI ---")
print(
    results_df[
        ["split", "model", "MAE", "RMSE",
         "actual_vol_mean", "predicted_vol_mean"]
    ].round(6).to_string(index=False)
)

summary = results_df.groupby("model").agg(
    MAE_mean=("MAE", "mean"),
    MAE_std=("MAE", "std"),
    RMSE_mean=("RMSE", "mean"),
    RMSE_std=("RMSE", "std"),
)

print("\n--- UMUMIY TAQQOSLASH ---")
print(summary.round(6).sort_values("MAE_mean").to_string())

print("\nSoatlik uzilishlar:", gaps)
print("Dataset yozuvlari:", len(df))
print("Tahlil uchun yozuvlar:", len(data))
print("Natijalar saqlandi:", OUTPUT_PATH.resolve())
