
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bitcoin_hourly.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "data" / "direction_results.csv"
MODEL_PATH = MODEL_DIR / "ridge_direction.joblib"

HORIZON = 4

FEATURES = [
    "log_return",
    "return_4h",
    "return_24h",
    "vol_4h",
    "vol_24h",
    "vol_72h",
    "volume_change_24h",
    "range_pct",
]


def make_features(df):
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    df["log_return"] = np.log(close / close.shift(1))
    df["return_4h"] = np.log(close / close.shift(4))
    df["return_24h"] = np.log(close / close.shift(24))
    df["vol_4h"] = df["log_return"].rolling(4).std()
    df["vol_24h"] = df["log_return"].rolling(24).std()
    df["vol_72h"] = df["log_return"].rolling(72).std()
    df["volume_change_24h"] = volume.pct_change(24)
    df["range_pct"] = (df["high"] - df["low"]) / close

    # 1 = narx 4 soatdan keyin yuqoriroq, 0 = pastroq yoki teng
    future_close = close.shift(-HORIZON)
    df["target_up"] = (future_close > close).astype(float)
    df.loc[future_close.isna(), "target_up"] = np.nan

    return df


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def score_model(y_true, pred):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision_up": precision_score(y_true, pred, zero_division=0),
        "recall_up": recall_score(y_true, pred, zero_division=0),
        "f1_up": f1_score(y_true, pred, zero_division=0),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df = make_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["target_up"]).reset_index(drop=True)

    X = df[FEATURES]
    y = df["target_up"].astype(int)

    if len(df) < 500:
        raise ValueError("At least 500 usable hourly observations are recommended.")

    # 5 ta vaqt bo'yicha test bo'lagi.
    # gap=4: train va test targetlarining kelajak oynalari kesishishini kamaytiradi.
    cv = TimeSeriesSplit(n_splits=5, gap=HORIZON)
    models = build_models()
    rows = []

    print(f"Foydalanilgan yozuvlar: {len(df)}")
    print(f"Ma'lumot davri: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("Maqsad: narx 4 soatdan keyin yuqoriroq bo'ladimi?")
    print("\nWALK-FORWARD TEST NATIJALARI")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Oddiy baseline: train davridagi eng ko'p uchragan sinfni taxmin qiladi.
        majority_class = int(y_train.mean() >= 0.5)
        baseline_pred = np.full(len(test_idx), majority_class)

        baseline_scores = score_model(y_test, baseline_pred)
        rows.append({
            "fold": fold,
            "model": "Majority Baseline",
            "test_start": df.iloc[test_idx[0]]["timestamp"],
            "test_end": df.iloc[test_idx[-1]]["timestamp"],
            **baseline_scores,
        })

        # Momentum baseline: oxirgi 4 soatdagi yo'nalish davom etadi deb taxmin qiladi.
        momentum_pred = (df.iloc[test_idx]["return_4h"].to_numpy() > 0).astype(int)
        momentum_scores = score_model(y_test, momentum_pred)
        rows.append({
            "fold": fold,
            "model": "4h Momentum Baseline",
            "test_start": df.iloc[test_idx[0]]["timestamp"],
            "test_end": df.iloc[test_idx[-1]]["timestamp"],
            **momentum_scores,
        })

        for name, prototype in models.items():
            model = clone(prototype)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            scores = score_model(y_test, pred)
            rows.append({
                "fold": fold,
                "model": name,
                "test_start": df.iloc[test_idx[0]]["timestamp"],
                "test_end": df.iloc[test_idx[-1]]["timestamp"],
                **scores,
            })

            print(
                f"Fold {fold} | {name}: "
                f"Accuracy={scores['accuracy']:.3f}, "
                f"Balanced Accuracy={scores['balanced_accuracy']:.3f}, "
                f"F1={scores['f1_up']:.3f}"
            )

        print(
            f"Fold {fold} | Baseline: "
            f"Accuracy={baseline_scores['accuracy']:.3f} | "
            f"Momentum: Accuracy={momentum_scores['accuracy']:.3f}"
        )

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("\nO'RTACHA TEST NATIJALARI")
    summary = (
        results.groupby("model")[
            ["accuracy", "balanced_accuracy", "precision_up", "recall_up", "f1_up"]
        ]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    print(summary.round(4).to_string())

    # Yakuniy model barcha mavjud, targeti ma'lum ma'lumotlarda o'qitiladi.
    # Uning haqiqiy bahosi yuqoridagi walk-forward test natijalari hisoblanadi.
    final_models = build_models()
    final_scores = summary.loc[
        ["Logistic Regression", "Random Forest"], "balanced_accuracy"
    ]
    best_name = final_scores.idxmax()
    best_model = final_models[best_name]
    best_model.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "features": FEATURES,
        "horizon_hours": HORIZON,
        "target": "close_t_plus_4 > close_t",
        "trained_until": str(df["timestamp"].iloc[-1]),
        "walk_forward_summary": summary.to_dict(),
    }
    joblib.dump(bundle, MODEL_PATH)
    # Also save with accurate canonical name
    canonical_path = MODEL_DIR / "direction_classifier.joblib"
    joblib.dump(bundle, canonical_path)

    print(f"\nTanlangan model: {best_name}")
    print(f"Model saqlandi: {MODEL_PATH} va {canonical_path}")
    print(f"Test natijalari saqlandi: {RESULTS_PATH}")
    print(
        "Eslatma: bu model tajribaviy. Test natijalari kelajakdagi "
        "aniqlikni kafolatlamaydi."
    )


if __name__ == "__main__":
    main()
