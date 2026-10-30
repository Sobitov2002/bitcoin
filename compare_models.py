
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent

# 1. Ma'lumotlarni tayyorlash
df = pd.read_csv(
    BASE_DIR / "data" / "bitcoin_prepared.csv",
    parse_dates=["timestamp"]
).sort_values("timestamp").reset_index(drop=True)

for lag in range(1, 25):
    df[f"close_lag_{lag}"] = df["close"].shift(lag)

features = [
    "open", "high", "low", "close", "volume"
] + [f"close_lag_{lag}" for lag in range(1, 25)]

df = df.dropna(subset=features + ["target_4h"]).reset_index(drop=True)

X = df[features]
y = df["target_4h"]

# 2. Vaqt tartibini saqlagan holda train/test
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 3. Baseline
baseline = X_test["close"].to_numpy()

# 4. Random Forest modelini o'qitish
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 5. Modellarni baholash
def metrics(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name:20s} MAE={mae:.2f} | RMSE={rmse:.2f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse}

results = [
    metrics("Baseline", y_test, baseline),
    metrics("Random Forest", y_test, predictions)
]

# 6. Natijalar va modelni saqlash
(BASE_DIR / "models").mkdir(exist_ok=True)
joblib.dump(model, BASE_DIR / "models" / "random_forest.joblib")

pd.DataFrame(results).to_csv(
    BASE_DIR / "data" / "model_comparison.csv", index=False
)

pd.DataFrame({
    "timestamp": df["timestamp"].iloc[split:].to_numpy(),
    "actual_4h": y_test.to_numpy(),
    "baseline_prediction": baseline,
    "random_forest_prediction": predictions
}).to_csv(BASE_DIR / "data" / "random_forest_predictions.csv", index=False)

print("\nTrain yozuvlari:", len(X_train))
print("Test yozuvlari:", len(X_test))
print("Random Forest modeli saqlandi.")
