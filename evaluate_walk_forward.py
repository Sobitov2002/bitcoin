
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/bitcoin_hourly.csv")
OUTPUT_PATH = Path("data/return_model_results.csv")

HORIZON = 4
N_SPLITS = 5
TEST_SIZE = 24 * 14

FEATURES = [
    "return_1h",
    "return_4h",
    "return_24h",
    "volatility_24h",
    "volume_change_24h",
    "sma_distance_6",
    "sma_distance_24",
    "sma_distance_72",
    "range_pct",
]

def calculate_rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))

# 1. Yuklash va tekshirish
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").drop_duplicates("timestamp")
df = df.reset_index(drop=True)

# Soatlik vaqt oralig'ini tekshirish
gaps = int(
    (df["timestamp"].diff() > pd.Timedelta(hours=1)).sum()
)

close = df["close"]
volume = df["volume"]

# 2. Faqat joriy va o'tmish ma'lumotlaridan feature yaratamiz
df["return_1h"] = close.pct_change(1)
df["return_4h"] = close.pct_change(4)
df["return_24h"] = close.pct_change(24)

df["volatility_24h"] = (
    df["return_1h"].rolling(24).std()
)

df["volume_change_24h"] = volume.pct_change(24)

for window in [6, 24, 72]:
    moving_average = close.rolling(window).mean()
    df[f"sma_distance_{window}"] = (
        close / moving_average - 1
    )

df["range_pct"] = (
    (df["high"] - df["low"]) / close
)

# 4 soatdan keyingi narxning foizli o'zgarishi
df["target_return"] = close.shift(-HORIZON) / close - 1

data = df.dropna(
    subset=FEATURES + ["target_return"]
).reset_index(drop=True)

if len(data) <= N_SPLITS * TEST_SIZE + HORIZON:
    raise ValueError("Test uchun ma'lumot yetarli emas.")

models = {
    "Zero-Return Baseline": None,
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        loss="huber",
        random_state=42,
    ),
}

results = []
first_test_start = len(data) - N_SPLITS * TEST_SIZE

# 3. Walk-forward validation
for split_no in range(N_SPLITS):
    test_start = first_test_start + split_no * TEST_SIZE
    test_end = test_start + TEST_SIZE
    train_end = test_start - HORIZON

    train = data.iloc[:train_end]
    test = data.iloc[test_start:test_end]

    X_train = train[FEATURES]
    y_train = train["target_return"]
    X_test = test[FEATURES]
    y_test = test["target_return"]

    current_price = test["close"].to_numpy()
    actual_price = current_price * (1 + y_test.to_numpy())

    print(
        f"\nSplit {split_no + 1}/{N_SPLITS}: "
        f"{test['timestamp'].iloc[0]} -> "
        f"{test['timestamp'].iloc[-1]}"
    )

    for model_name, model in models.items():
        if model is None:
            predicted_return = np.zeros(len(test))
        else:
            model.fit(X_train, y_train)
            predicted_return = model.predict(X_test)

        # Foizli prognozni narxga aylantiramiz
        predicted_price = current_price * (
            1 + predicted_return
        )

        actual_direction = np.sign(
            actual_price - current_price
        )
        predicted_direction = np.sign(
            predicted_price - current_price
        )

        # Nolga teng yo'nalishlarni accuracy'dan chiqaramiz
        nonzero = (
            (actual_direction != 0)
            & (predicted_direction != 0)
        )

        direction_accuracy = (
            float(np.mean(
                actual_direction[nonzero]
                == predicted_direction[nonzero]
            ))
            if nonzero.any()
            else np.nan
        )

        results.append({
            "split": split_no + 1,
            "model": model_name,
            "MAE_USDT": mean_absolute_error(
                actual_price, predicted_price
            ),
            "RMSE_USDT": calculate_rmse(
                actual_price, predicted_price
            ),
            "direction_accuracy": direction_accuracy,
            "test_start": test["timestamp"].iloc[0],
            "test_end": test["timestamp"].iloc[-1],
        })

results_df = pd.DataFrame(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- HAR BIR DAVR ---")
print(
    results_df[
        ["split", "model", "MAE_USDT",
         "RMSE_USDT", "direction_accuracy"]
    ].round(4).to_string(index=False)
)

summary = results_df.groupby("model").agg(
    MAE_mean=("MAE_USDT", "mean"),
    MAE_std=("MAE_USDT", "std"),
    RMSE_mean=("RMSE_USDT", "mean"),
    RMSE_std=("RMSE_USDT", "std"),
    direction_mean=("direction_accuracy", "mean"),
)

print("\n--- UMUMIY NATIJA ---")
print(summary.round(4).sort_values("MAE_mean").to_string())

print("\nSoatlik uzilishlar:", gaps)
print("Natijalar saqlandi:", OUTPUT_PATH.resolve())
