
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE_DIR / "data" / "test_predictions.csv",
    parse_dates=["timestamp"]
)

plt.figure(figsize=(14, 6))

plt.plot(
    df["timestamp"],
    df["actual_4h"],
    label="Actual price (4h later)"
)

plt.plot(
    df["timestamp"],
    df["baseline_prediction"],
    label="Baseline prediction",
    alpha=0.8
)

plt.plot(
    df["timestamp"],
    df["model_prediction"],
    label="Linear Regression prediction",
    alpha=0.8
)

plt.title("Bitcoin 4-Hour Forecast: Actual vs Predictions")
plt.xlabel("Time (UTC)")
plt.ylabel("BTC Price (USDT)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_img = BASE_DIR / "data" / "forecast_comparison.png"
plt.savefig(out_img, dpi=150)
plt.close()

print(f"Grafik saqlandi: {out_img}")
