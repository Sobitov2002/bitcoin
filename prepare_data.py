
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. Datasetni yuklash
data_path = BASE_DIR / "data" / "bitcoin_hourly.csv"
df = pd.read_csv(data_path, parse_dates=["timestamp"])

# 2. Vaqt bo'yicha tartiblash va tekshirish
df = (
    df.sort_values("timestamp")
      .drop_duplicates(subset="timestamp")
      .reset_index(drop=True)
)

# 3. Muhim ustunlarni tekshirish
required_columns = [
    "timestamp", "open", "high", "low", "close", "volume"
]

if df[required_columns].isnull().any().any():
    raise ValueError("Datasetda bo'sh qiymatlar mavjud!")

# 4. 4 soat keyingi yopilish narxi
df["target_4h"] = df["close"].shift(-4)

# 5. Oxirgi 4 qatorda kelajakdagi target yo'q
df = df.dropna(subset=["target_4h"]).reset_index(drop=True)

# 6. Natijalarni tekshirish
print("Dataset hajmi:", df.shape)
print("Birinchi vaqt:", df["timestamp"].min())
print("Oxirgi target vaqti:",
      df["timestamp"].iloc[-1] + pd.Timedelta(hours=4))

print("\nTayyorlangan ma'lumotlar:")
print(df.tail())

# 7. Saqlash
output_path = BASE_DIR / "data" / "bitcoin_prepared.csv"
df.to_csv(output_path, index=False)

print("\nFayl saqlandi:", output_path.resolve())
