
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent

# 1. Ma'lumotlarni yuklash
df = pd.read_csv(
    BASE_DIR / "data" / "bitcoin_prepared.csv",
    parse_dates=["timestamp"]
)
df = df.sort_values("timestamp").reset_index(drop=True)

# 2. Vaqt oralig'i uzluksizligini tekshirish va to'g'rilash
gaps = df["timestamp"].diff().dropna()
gap_count = int((gaps > pd.Timedelta(hours=1)).sum())
if gap_count > 0:
    print(f"Ogohlantirish: {gap_count} ta soatlik uzilish aniqlandi, ffill bilan to'ldirilmoqda.")
    df = df.set_index("timestamp").asfreq("1h").ffill().reset_index()

# 3. Oldingi 24 soatdagi yopilish narxlarini feature qilish
for lag in range(1, 25):
    df[f"close_lag_{lag}"] = df["close"].shift(lag)

# Narx va savdo hajmi haqidagi joriy ma'lumotlar
features = [
    "open", "high", "low", "close", "volume"
] + [f"close_lag_{lag}" for lag in range(1, 25)]

df = df.dropna(subset=features + ["target_4h"]).reset_index(drop=True)

X = df[features]
y = df["target_4h"]

# 4. Vaqt bo'yicha 80% train, 20% test
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 5. Baseline: hozirgi close narxini bashorat deb olamiz
baseline_predictions = X_test["close"].to_numpy()

# 6. Linear Regression modelini o'qitish
model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 7. Natijalarni baholash
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f} USDT")
    print(f"RMSE: {rmse:.2f} USDT")

evaluate("BASELINE", y_test, baseline_predictions)
evaluate("LINEAR REGRESSION", y_test, predictions)

# 8. Model va test bashoratlarini saqlash
(BASE_DIR / "models").mkdir(exist_ok=True)
joblib.dump(model, BASE_DIR / "models" / "linear_regression.joblib")

results = pd.DataFrame({
    "timestamp": df["timestamp"].iloc[split:],
    "actual_4h": y_test.to_numpy(),
    "baseline_prediction": baseline_predictions,
    "model_prediction": predictions
})

results.to_csv(BASE_DIR / "data" / "test_predictions.csv", index=False)

print("\nTrain yozuvlari:", len(X_train))
print("Test yozuvlari:", len(X_test))
print("Model saqlandi: models/linear_regression.joblib")
print("Bashoratlar saqlandi: data/test_predictions.csv")
