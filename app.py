"""
app.py — Bitcoin AI Dashboard & Observation Analysis Platform.

Features:
- Real-time BTCUSDT live ticker & historical hourly candle analysis.
- 4-Hour Volatility Prediction (Ridge Regression Pipeline).
- 4-Hour Directional Probability (Random Forest Classifier).
- Approximate 4-Hour Price Range estimation with risk disclaimers.
- MongoDB & Local Storage Integration for personal market observations.
- Interactive Dialog Modal for recording AI Forecast vs User Hypothesis.
- Historical Observation Tracker with automated accuracy evaluation.
- Responsive, clean Light UI/UX optimized for both Desktop and Mobile devices.
- Bilingual support: O'zbekcha / 한국어.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Internal modules
import db_manager
import verify_predictions

# =========================================================
# CONFIGURATION & PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "bitcoin_hourly.csv"

# Models (Check canonical path first, fallback to original)
VOLATILITY_MODEL_PATH = BASE_DIR / "models" / "ridge_volatility.joblib"
DIRECTION_MODEL_PATH_CANONICAL = BASE_DIR / "models" / "direction_classifier.joblib"
DIRECTION_MODEL_PATH_FALLBACK = BASE_DIR / "models" / "ridge_direction.joblib"

BINANCE_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/price"
FX_URL = "https://open.er-api.com/v6/latest/USD"
KST = "Asia/Seoul"
UZT = "Asia/Tashkent"

st.set_page_config(
    page_title="Bitcoin AI Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM RESPONSIVE CSS (Clean Light Theme with Forest Green)
# =========================================================
CUSTOM_CSS = """
<style>
/* Main typography and container spacing */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Background & card styling */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* Custom metric card container */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 0.75rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
}
.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.metric-sub {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.35rem;
}

/* Accent Badges */
.badge-bullish {
    background-color: #DCFCE7;
    color: #15803D;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    display: inline-block;
}
.badge-bearish {
    background-color: #FEE2E2;
    color: #B91C1C;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    display: inline-block;
}
.badge-neutral {
    background-color: #F1F5F9;
    color: #475569;
    font-weight: 600;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    display: inline-block;
}

/* Touch-friendly buttons on mobile */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1rem;
    min-height: 42px;
}

/* Responsive adjustment for mobile screens */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-top: 1rem;
    }
    .metric-value {
        font-size: 1.35rem;
    }
    .metric-card {
        padding: 0.9rem;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# TRANSLATIONS (O'zbekcha / 한국어)
# =========================================================
TEXT = {
    "uz": {
        "title": "₿ Bitcoin AI Dashboard",
        "subtitle": "Bitcoin uchun sun’iy intellekt tahlili, bozor kuzatuvlari va aniqlik monitoringi",
        "tab_live": "📊 Jonli Tahlil va AI Prognozi",
        "tab_history": "📋 Kuzatuvlar Tarixi va Taqqoslash",
        "tab_system": "ℹ️ Model va Tizim Haqida",
        "language": "Til / 언어",
        "currency": "Valyuta",
        "controls": "Boshqaruv Paneli",
        "refresh_data": "🔄 Sham ma’lumotlarini yangilash",
        "refresh_ok": "Bozor shamlari yangilandi.",
        "refresh_err": "Ma’lumotlarni yangilab bo‘lmadi",
        "price_live": "Jonli BTC Narxi (Ayni lahza)",
        "price_closed": "Oxirgi Yopilgan Sham Narxi",
        "candle_time": "Sham Vaqti",
        "volatility_4h": "4 Soatlik Kutilayotgan Volatillik",
        "direction_4h": "4 Soatlik Narx Yo‘nalishi Ehtimoli",
        "approx_range": "4 Soatlik Taxminiy Narx Oralig‘i",
        "horizon": "Bashorat Ufqi",
        "hours": "soat",
        "up_prob": "Ko‘tarilish ehtimoli",
        "down_prob": "Tushish ehtimoli",
        "more_likely_up": "🟢 Ko‘tarilish ehtimoli yuqoriroq",
        "more_likely_down": "🔴 Tushish ehtimoli yuqoriroq",
        "equal_prob": "⚪ Ikkala yo‘nalish ehtimoli teng",
        "prob_disclaimer": "Eslatma: ehtimolliklar model bahosi bo‘lib, kelajak kafolati emas. Tarixiy balanced accuracy ~51.7%.",
        "vol_disclaimer": "Volatillik narx qanchalik keng tebranishini bildiradi, yo‘nalishni ko‘rsatmaydi.",
        "range_disclaimer": "Ushbu diapazon ±1.96σ normal taqsimot taxminiga asoslangan matematik oraliq; kafolatlangan chegara emas.",
        "save_obs_btn": "➕ Shaxsiy bozor kuzatuvini saqlash",
        "verify_btn": "⚡ Kutilayotgan natijalarni tekshirish",
        "verifying": "Tekshirilmoqda...",
        "obs_history_title": "AI Bashoratlari va Sizning Taxminlaringiz Taqqoslovi",
        "stats_ai_acc": "AI Yo‘nalish Aniqligi",
        "stats_user_acc": "Sizning Aniqligingiz",
        "stats_completed": "Baholangan Kuzatuvlar",
        "stats_pending": "Kutilayotgan Kuzatuvlar",
        "stats_1h": "1 Soatlik Natijalar",
        "stats_4h": "4 Soatlik Natijalar",
        "status_pending": "⏳ Kutilmoqda",
        "status_completed": "✅ Baholandi",
        "status_error": "⚠️ Xato",
        "dialog_title": "Bozor Kuzatuvi va Taxminni Saqlash",
        "dialog_ai_forecast": "Joriy AI Bashorati",
        "dialog_price_now": "Boshlang‘ich Bozor Narxi",
        "dialog_user_choice": "Sizning Taxminingiz",
        "dialog_up": "🟢 Ko‘tariladi (UP)",
        "dialog_down": "🔴 Tushadi (DOWN)",
        "dialog_sideways": "⚪ Yonlama / O‘zgarmaydi (SIDEWAYS)",
        "dialog_horizon": "Kuzatuv Muddati",
        "dialog_notes": "Izoh (ixtiyoriy)",
        "dialog_notes_ph": "Masalan: Texnik indikator RSI 30 dan qaytdi, yangiliklar ta'siri...",
        "dialog_save": "💾 Saqlash",
        "dialog_cancel": "Bekor qilish",
        "dialog_success": "Kuzatuv muvaffaqiyatli saqlandi!",
        "fx_rate_label": "USD/KRW Valyuta Kursi",
        "source": "Manbalar: Binance Public API · Valyuta: ExchangeRate-API · Ma'lumotlar bazasi: MongoDB Atlas",
        "db_mode_label": "Ma'lumotlar Bazasi Holati",
        "stale_data_warn": "Ogohlantirish: Mahalliy sham ma’lumotlari 2 soatdan ko‘proq eskirgan. Yangilash tugmasini bosing.",
    },
    "ko": {
        "title": "₿ 비트코인 AI 대시보드",
        "subtitle": "인공지능 기반 비트코인 시장 분석, 관찰 기록 및 예측 검증 플랫폼",
        "tab_live": "📊 실시간 분석 및 AI 예측",
        "tab_history": "📋 관찰 기록 및 AI 비교",
        "tab_system": "ℹ️ 모델 및 시스템 정보",
        "language": "언어 / Til",
        "currency": "통화",
        "controls": "제어판",
        "refresh_data": "🔄 캔들 데이터 새로고침",
        "refresh_ok": "시장 캔들 데이터가 업데이트되었습니다.",
        "refresh_err": "데이터를 업데이트하지 못했습니다",
        "price_live": "실시간 BTC 가격 (현재 체결가)",
        "price_closed": "최근 마감 캔들 종가",
        "candle_time": "캔들 마감 시간",
        "volatility_4h": "향후 4시간 예상 변동성",
        "direction_4h": "향후 4시간 가격 방향 예측",
        "approx_range": "향후 4시간 예상 가격 범위",
        "horizon": "예측 기간",
        "hours": "시간",
        "up_prob": "상승 확률",
        "down_prob": "하락 확률",
        "more_likely_up": "🟢 상승 가능성이 더 높습니다",
        "more_likely_down": "🔴 하락 가능성이 더 높습니다",
        "equal_prob": "⚪ 두 방향의 가능성이 같습니다",
        "prob_disclaimer": "주의: 비율은 모델 추정치이며 미래 결과를 보장하지 않습니다. 테스트 balanced accuracy는 약 51.7%였습니다.",
        "vol_disclaimer": "변동성은 가격의 변동 폭을 나타내며 방향을 나타내지 않습니다.",
        "range_disclaimer": "예상 범위는 ±1.96σ 정규분포 가정에 따른 근사치이며 보장된 범위가 아닙니다.",
        "save_obs_btn": "➕ 시장 관찰 기록 저장",
        "verify_btn": "⚡ 대기 중인 예측 결과 검증",
        "verifying": "검증 중...",
        "obs_history_title": "AI 예측과 사용자 판단 비교 분석",
        "stats_ai_acc": "AI 방향 적중률",
        "stats_user_acc": "사용자 적중률",
        "stats_completed": "평가 완료된 관찰",
        "stats_pending": "대기 중인 관찰",
        "stats_1h": "1시간 예측 결과",
        "stats_4h": "4시간 예측 결과",
        "status_pending": "⏳ 대기 중",
        "status_completed": "✅ 평가 완료",
        "status_error": "⚠️ 오류",
        "dialog_title": "시장 관찰 및 예측 기록 저장",
        "dialog_ai_forecast": "현재 AI 예측 결과",
        "dialog_price_now": "기준 시장 가격",
        "dialog_user_choice": "사용자 판단",
        "dialog_up": "🟢 상승 (UP)",
        "dialog_down": "🔴 하락 (DOWN)",
        "dialog_sideways": "⚪ 횡보 / 보합 (SIDEWAYS)",
        "dialog_horizon": "관찰 기간",
        "dialog_notes": "메모 (선택사항)",
        "dialog_notes_ph": "예: 지지선 반등, 기술적 지표 RSI 과매도...",
        "dialog_save": "💾 저장",
        "dialog_cancel": "취소",
        "dialog_success": "관찰 기록이 성공적으로 저장되었습니다!",
        "fx_rate_label": "USD/KRW 환율",
        "source": "데이터 출처: Binance Public API · 환율: ExchangeRate-API · DB: MongoDB Atlas",
        "db_mode_label": "데이터베이스 상태",
        "stale_data_warn": "주의: 로컬 캔들 데이터가 2시간 이상 경과했습니다. 새로고침을 권장합니다.",
    },
}


# =========================================================
# DATA & MODEL LOADERS
# =========================================================
@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    """Load local hourly historical dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset topilmadi: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


@st.cache_resource
def load_volatility_model():
    """Load trained Ridge volatility pipeline."""
    if not VOLATILITY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Volatillik modeli topilmadi: {VOLATILITY_MODEL_PATH}")
    return joblib.load(VOLATILITY_MODEL_PATH)


def _patch_pipeline_compat(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Patch scikit-learn cross-version serialization differences (e.g. SimpleImputer._fill_dtype)."""
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "strategy") and not hasattr(step, "_fill_dtype"):
                    step._fill_dtype = np.dtype(float)
    return bundle


@st.cache_resource
def load_direction_model():
    """Load trained direction classification pipeline."""
    path = DIRECTION_MODEL_PATH_CANONICAL
    if not path.exists():
        path = DIRECTION_MODEL_PATH_FALLBACK
    if not path.exists():
        raise FileNotFoundError(f"Yo'nalish modeli topilmadi: {path}")
    bundle = joblib.load(path)
    return _patch_pipeline_compat(bundle)


@st.cache_data(ttl=10)
def get_live_btc_price() -> Tuple[Optional[float], Optional[str]]:
    """Retrieve the real-time ticker price from Binance. Resilient to timeouts."""
    try:
        resp = requests.get(BINANCE_TICKER_URL, params={"symbol": "BTCUSDT"}, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        return float(data["price"]), "OK"
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=900)
def get_usd_krw() -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Retrieve USD to KRW exchange rate with graceful fallback."""
    try:
        resp = requests.get(FX_URL, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            rate = float(data["rates"]["KRW"])
            time_updated = data.get("time_last_update_utc", "Unknown")
            return rate, time_updated, None
        return None, None, "API returned unsuccessful status"
    except Exception as exc:
        return None, None, str(exc)


def refresh_market_data() -> int:
    """Fetch completed hourly candles from Binance and append to local CSV.

    Standardizes timestamp to candle open_time to maintain 100% consistency with download_data.py.
    """
    resp = requests.get(
        BINANCE_URL,
        params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1000},
        timeout=20,
    )
    resp.raise_for_status()
    candles = resp.json()

    if not isinstance(candles, list) or not candles:
        raise ValueError("Binance API candle ma'lumotlarini qaytarmadi.")

    now_utc = pd.Timestamp.now(tz="UTC")
    rows = []

    for c in candles:
        open_time = pd.to_datetime(c[0], unit="ms", utc=True)
        # Skip candle that is currently still in progress
        if open_time + pd.Timedelta(hours=1) > now_utc:
            continue

        rows.append({
            "timestamp": open_time,  # Standard: candle open time
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })

    if not rows:
        raise ValueError("Yopilgan soatlik shamlar topilmadi.")

    fresh_df = pd.DataFrame(rows)

    if DATA_PATH.exists():
        existing = load_data().copy()
        combined = pd.concat([existing, fresh_df], ignore_index=True)
    else:
        combined = fresh_df

    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_PATH, index=False)

    load_data.clear()
    get_live_btc_price.clear()
    get_usd_krw.clear()
    return len(fresh_df)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build mathematical features matching the training pipeline."""
    df = df.copy()
    close = df["close"]
    volume = df["volume"]

    df["log_return"] = np.log(close / close.shift(1))
    df["vol_4h"] = df["log_return"].rolling(4).std()
    df["vol_24h"] = df["log_return"].rolling(24).std()
    df["vol_72h"] = df["log_return"].rolling(72).std()

    df["return_1h"] = df["log_return"]
    df["return_4h"] = np.log(close / close.shift(4))
    df["return_24h"] = np.log(close / close.shift(24))

    df["volume_change_24h"] = volume.pct_change(24)
    df["range_pct"] = (df["high"] - df["low"]) / close
    return df


def format_timezone(dt, tz_name: str) -> str:
    """Format UTC datetime into target timezone string."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz_name).strftime("%Y-%m-%d %H:%M")


# =========================================================
# MODAL DIALOG: RECORD USER OBSERVATION
# =========================================================
@st.dialog("Kuzatuvni saqlash / 관찰 기록 저장")
def open_save_observation_dialog(
    ai_prediction: Dict[str, Any],
    default_price: float,
    current_curr: str,
    fx_rate: Optional[float],
    lang_code: str,
):
    dt = TEXT[lang_code]
    st.write(f"### {dt['dialog_title']}")

    # 1. AI Forecast summary card
    dir_label = (
        "🟢 UP (Ko'tarilish)"
        if ai_prediction["direction"] == "UP"
        else "🔴 DOWN (Tushish)"
        if ai_prediction["direction"] == "DOWN"
        else "⚪ EQUAL"
    )
    st.info(
        f"**{dt['dialog_ai_forecast']}:** {dir_label} "
        f"(Up: {ai_prediction['prob_up']:.1f}% | Down: {ai_prediction['prob_down']:.1f}%) · "
        f"Volatillik: {ai_prediction['volatility_pct']:.3f}%"
    )

    # 2. Baseline Price
    c1, c2 = st.columns([2, 1])
    with c1:
        price_input = st.number_input(
            f"{dt['dialog_price_now']} (USD)",
            min_value=1000.0,
            max_value=1000000.0,
            value=float(default_price),
            step=10.0,
            format="%.2f",
        )
    with c2:
        horizon_choice = st.selectbox(
            dt["dialog_horizon"],
            options=[4, 1],
            format_func=lambda h: f"{h} {dt['hours']}",
            index=0,
        )

    if fx_rate and current_curr == "KRW":
        st.caption(f"≈ ₩{(price_input * fx_rate):,.0f} KRW (Kurs: {fx_rate:,.2f})")

    # 3. User hypothesis
    user_dir_choice = st.radio(
        dt["dialog_user_choice"],
        options=["UP", "DOWN", "SIDEWAYS"],
        format_func=lambda d: dt["dialog_up"] if d == "UP" else (dt["dialog_down"] if d == "DOWN" else dt["dialog_sideways"]),
        horizontal=True,
    )

    # 4. User notes
    user_notes = st.text_area(
        dt["dialog_notes"],
        placeholder=dt["dialog_notes_ph"],
        max_chars=300,
    )

    # 5. Action buttons
    col_save, col_cancel = st.columns(2)
    with col_save:
        if st.button(dt["dialog_save"], use_container_width=True, type="primary"):
            now_utc = datetime.now(timezone.utc)
            target_time_utc = now_utc + pd.Timedelta(hours=horizon_choice)

            doc = {
                "created_at_utc": now_utc.isoformat(),
                "created_at_local": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
                "horizon_hours": int(horizon_choice),
                "target_time_utc": target_time_utc.isoformat(),
                "initial_price": float(price_input),
                "price_source": "Binance BTCUSDT",
                "currency": "USD",
                "ai_forecast": {
                    "direction": ai_prediction["direction"],
                    "probability_up": ai_prediction["prob_up"],
                    "probability_down": ai_prediction["prob_down"],
                    "volatility_pct": ai_prediction["volatility_pct"],
                    "approx_lower": ai_prediction["lower_price"],
                    "approx_upper": ai_prediction["upper_price"],
                    "model_version": ai_prediction.get("model_name", "RandomForest_v1"),
                },
                "user_prediction": {
                    "direction": user_dir_choice,
                    "notes": user_notes.strip(),
                },
                "status": "PENDING",
                "evaluation": {},
            }

            ok, msg, _ = db_manager.save_observation(doc)
            if ok:
                st.success(f"{dt['dialog_success']} ({msg})")
                time.sleep(1.2)
                st.rerun()
            else:
                st.error(f"Xatolik yuz berdi: {msg}")

    with col_cancel:
        if st.button(dt["dialog_cancel"], use_container_width=True):
            st.rerun()


# =========================================================
# MAIN APP EXECUTION
# =========================================================
def main():
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Konfiguratsiya / 설정")
        language_choice = st.selectbox(
            "Til / 언어",
            options=["O‘zbekcha", "한국어"],
            index=0,
        )
        lang = "uz" if language_choice == "O‘zbekcha" else "ko"
        t = TEXT[lang]

        currency_choice = st.radio(
            t["currency"],
            options=["KRW (₩)", "USD ($)"],
            index=0,
            horizontal=True,
        )
        active_curr = "KRW" if "KRW" in currency_choice else "USD"

        st.divider()
        st.subheader(t["controls"])

        if st.button(t["refresh_data"], use_container_width=True, type="secondary"):
            try:
                with st.spinner("Binance dan yangi shamlar yuklanmoqda..."):
                    updated_count = refresh_market_data()
                st.success(f"{t['refresh_ok']} ({updated_count} ta sham)")
                time.sleep(1)
                st.rerun()
            except Exception as exc:
                st.error(f"{t['refresh_err']}: {exc}")

        # Diagnostic DB connection status badge
        db_stat = db_manager.check_connection()
        st.divider()
        st.subheader(t["db_mode_label"])
        if db_stat["connected"]:
            st.success(f"🟢 {db_stat['message']}")
        else:
            st.warning(f"🟡 Rejim: Lokal xotira (JSON)\n\n{db_stat['message']}")

        st.caption(t["source"])

    # --- Header Title ---
    st.title(t["title"])
    st.caption(t["subtitle"])

    # Load data and models safely
    try:
        raw_df = load_data()
        vol_bundle = load_volatility_model()
        dir_bundle = load_direction_model()
    except Exception as exc:
        st.error(f"Tizim fayllarini yuklashda xatolik: {exc}")
        st.stop()

    # Get live BTC price & FX rate safely with graceful fallback
    live_price_usd, live_price_err = get_live_btc_price()
    fx_rate, fx_updated, fx_err = get_usd_krw()

    # Features and Latest completed candle
    featured = build_features(raw_df)
    featured = featured.replace([np.inf, -np.inf], np.nan)

    vol_features = vol_bundle["features"]
    vol_model = vol_bundle["model"]
    dir_features = dir_bundle["features"]
    dir_model = dir_bundle["model"]

    valid_data = featured.dropna(subset=vol_features + dir_features)
    if valid_data.empty:
        st.error("Model uchun yetarli xususiyatlar hisoblab bo'lmadi.")
        st.stop()

    latest_row = valid_data.iloc[-1]
    closed_price_usd = float(latest_row["close"])

    # Fallback if live price is unavailable
    if live_price_usd is None:
        display_live_usd = closed_price_usd
        live_status_tag = "(Oxirgi sham)"
    else:
        display_live_usd = live_price_usd
        live_status_tag = "(Jonli)"

    # Currency conversions
    if active_curr == "KRW" and fx_rate:
        display_live_price = f"₩{display_live_usd * fx_rate:,.0f}"
        display_closed_price = f"₩{closed_price_usd * fx_rate:,.0f}"
    else:
        display_live_price = f"${display_live_usd:,.2f}"
        display_closed_price = f"${closed_price_usd:,.2f}"

    # Model Predictions
    latest_feat = latest_row[vol_features].to_frame().T
    vol_pred_raw = max(float(vol_model.predict(latest_feat)[0]), 0.0)
    vol_pred_pct = vol_pred_raw * 100.0

    latest_dir_feat = latest_row[dir_features].to_frame().T
    dir_probs = dir_model.predict_proba(latest_dir_feat)[0]
    classes = list(dir_model.classes_)
    prob_up = float(dir_probs[classes.index(1)]) * 100.0
    prob_down = float(dir_probs[classes.index(0)]) * 100.0

    if prob_up > prob_down:
        ai_direction = "UP"
        ai_dir_text = t["more_likely_up"]
    elif prob_down > prob_up:
        ai_direction = "DOWN"
        ai_dir_text = t["more_likely_down"]
    else:
        ai_direction = "EQUAL"
        ai_dir_text = t["equal_prob"]

    # Approximate 4-hour range
    sigma_4h = vol_pred_raw * np.sqrt(4)
    lower_price_usd = closed_price_usd * np.exp(-1.96 * sigma_4h)
    upper_price_usd = closed_price_usd * np.exp(1.96 * sigma_4h)

    if active_curr == "KRW" and fx_rate:
        lower_price_display = f"₩{lower_price_usd * fx_rate:,.0f}"
        upper_price_display = f"₩{upper_price_usd * fx_rate:,.0f}"
    else:
        lower_price_display = f"${lower_price_usd:,.2f}"
        upper_price_display = f"${upper_price_usd:,.2f}"

    # Timestamps
    open_time = pd.Timestamp(latest_row["timestamp"])
    close_time = open_time + pd.Timedelta(hours=1)
    now_utc = pd.Timestamp.now(tz="UTC")
    data_age_hours = max(0.0, (now_utc - close_time).total_seconds() / 3600.0)

    ai_prediction_bundle = {
        "direction": ai_direction,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "volatility_pct": vol_pred_pct,
        "lower_price": lower_price_usd,
        "upper_price": upper_price_usd,
        "model_name": dir_bundle.get("model_name", "RandomForest_v1"),
    }

    # =========================================================
    # TABS NAVIGATION
    # =========================================================
    tab1, tab2, tab3 = st.tabs([t["tab_live"], t["tab_history"], t["tab_system"]])

    # ---------------------------------------------------------
    # TAB 1: JONLI TAHLIL VA PROGNOZ
    # ---------------------------------------------------------
    with tab1:
        if data_age_hours > 2.0:
            st.warning(f"⚠️ {t['stale_data_warn']} (Oxirgi sham: {data_age_hours:.1f} soat oldin)")

        # Hero Price & Metric Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['price_live']} <span style="font-size:0.75rem; color:#10B981;">{live_status_tag}</span></div>
                <div class="metric-value">{display_live_price}</div>
                <div class="metric-sub">Binance BTCUSDT</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['price_closed']}</div>
                <div class="metric-value">{display_closed_price}</div>
                <div class="metric-sub">{t['candle_time']}: {format_timezone(close_time, KST)} KST</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['volatility_4h']}</div>
                <div class="metric-value" style="color: #0284C7;">{vol_pred_pct:.3f}%</div>
                <div class="metric-sub">Std. of next 4h log returns</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            fx_display = f"₩{fx_rate:,.2f}" if fx_rate else "Mavjud emas"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['fx_rate_label']}</div>
                <div class="metric-value" style="font-size:1.35rem;">{fx_display}</div>
                <div class="metric-sub">ExchangeRate API</div>
            </div>
            """, unsafe_allow_html=True)

        # AI Forecast Section
        st.subheader(t["direction_4h"])
        f_col1, f_col2 = st.columns([1, 1])

        with f_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <span style="font-size:1.15rem; font-weight:700;">{ai_dir_text}</span>
                    <span class="{'badge-bullish' if ai_direction == 'UP' else 'badge-bearish' if ai_direction == 'DOWN' else 'badge-neutral'}">
                        {ai_direction}
                    </span>
                </div>
                <div style="margin-bottom:0.4rem;">
                    <span style="font-weight:600; color:#15803D;">{t['up_prob']}:</span> <b>{prob_up:.1f}%</b>
                    <div style="background:#E2E8F0; border-radius:6px; height:10px; width:100%; margin-top:3px;">
                        <div style="background:#10B981; border-radius:6px; height:10px; width:{prob_up}%;"></div>
                    </div>
                </div>
                <div style="margin-top:0.75rem;">
                    <span style="font-weight:600; color:#B91C1C;">{t['down_prob']}:</span> <b>{prob_down:.1f}%</b>
                    <div style="background:#E2E8F0; border-radius:6px; height:10px; width:100%; margin-top:3px;">
                        <div style="background:#EF4444; border-radius:6px; height:10px; width:{prob_down}%;"></div>
                    </div>
                </div>
                <div style="margin-top:1rem; font-size:0.8rem; color:#64748B;">
                    {t['prob_disclaimer']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with f_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['approx_range']} (4h)</div>
                <div style="display:flex; justify-content:space-between; margin-top:0.5rem; margin-bottom:0.5rem;">
                    <div>
                        <span style="font-size:0.8rem; color:#64748B;">Taxminiy quyi:</span><br>
                        <b style="font-size:1.25rem; color:#EF4444;">{lower_price_display}</b>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:0.8rem; color:#64748B;">Taxminiy yuqori:</span><br>
                        <b style="font-size:1.25rem; color:#10B981;">{upper_price_display}</b>
                    </div>
                </div>
                <div style="font-size:0.85rem; color:#475569; margin-top:0.5rem; background:#F8FAFC; padding:0.6rem; border-radius:8px;">
                    {t['range_disclaimer']}
                </div>
                <div style="margin-top:0.8rem; font-size:0.8rem; color:#64748B;">
                    Prognoz davri: {format_timezone(close_time, KST)} KST → {format_timezone(close_time + pd.Timedelta(hours=4), KST)} KST
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Action: Save Observation Button
        st.markdown("<div style='margin-top:1.2rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
        if st.button(t["save_obs_btn"], use_container_width=True, type="primary"):
            open_save_observation_dialog(
                ai_prediction=ai_prediction_bundle,
                default_price=display_live_usd,
                current_curr=active_curr,
                fx_rate=fx_rate,
                lang_code=lang,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Interactive Price Chart
        st.divider()
        st.subheader("📈 Bitcoin Narx Grafigi / 가격 차트")

        chart_len_days = st.select_slider(
            "Vaqt oralig'i / 기간",
            options=[1, 3, 7, 14, 30],
            value=7,
            format_func=lambda d: f"{d} kun",
        )

        chart_df = raw_df.tail(24 * chart_len_days).copy()
        chart_df["time_kst"] = chart_df["timestamp"].dt.tz_convert(KST)

        fig = go.Figure()
        if active_curr == "KRW" and fx_rate:
            chart_price_y = chart_df["close"] * fx_rate
            hover_prefix = "₩"
        else:
            chart_price_y = chart_df["close"]
            hover_prefix = "$"

        fig.add_trace(go.Scatter(
            x=chart_df["time_kst"],
            y=chart_price_y,
            mode="lines",
            name="BTC/USD",
            line=dict(color="#059669", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.08)",
            hovertemplate="<b>Vaqt:</b> %{x|%Y-%m-%d %H:%M KST}<br><b>Narx:</b> " + hover_prefix + "%{y:,.2f}<extra></extra>",
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: KUZATUVLAR TARIXI VA AI VS INSON
    # ---------------------------------------------------------
    with tab2:
        st.subheader(t["obs_history_title"])

        # Top Action row
        col_act1, col_act2 = st.columns([3, 1])
        with col_act2:
            if st.button(t["verify_btn"], use_container_width=True, type="secondary"):
                with st.spinner(t["verifying"]):
                    res_summary = verify_predictions.run_verification()
                st.success(f"Tekshirildi: {res_summary['evaluated']} ta baholandi, {res_summary['still_waiting']} ta kutilmoqda.")
                time.sleep(1)
                st.rerun()

        # Compute accuracy statistics
        stats = db_manager.calculate_accuracy_stats()

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            ai_acc_text = f"{stats['ai_accuracy']:.1f}%" if stats["ai_accuracy"] is not None else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['stats_ai_acc']}</div>
                <div class="metric-value" style="color: #059669;">{ai_acc_text}</div>
                <div class="metric-sub">Haqiqiy bozor natijasida</div>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            user_acc_text = f"{stats['user_accuracy']:.1f}%" if stats["user_accuracy"] is not None else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['stats_user_acc']}</div>
                <div class="metric-value" style="color: #2563EB;">{user_acc_text}</div>
                <div class="metric-sub">Sizning taxminlaringiz</div>
            </div>
            """, unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['stats_completed']}</div>
                <div class="metric-value">{stats['completed_count']}</div>
                <div class="metric-sub">Muddati tugaganlar</div>
            </div>
            """, unsafe_allow_html=True)

        with s4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{t['stats_pending']}</div>
                <div class="metric-value" style="color: #D97706;">{stats['pending_count']}</div>
                <div class="metric-sub">Vaqti yetmaganlar</div>
            </div>
            """, unsafe_allow_html=True)

        # 1h vs 4h Breakdown
        if stats["completed_count"] > 0:
            st.caption(
                f"**Ufqlar bo'yicha tahlil:** 1 Soatlik (AI: {stats['ai_1h_accuracy'] or '—'}% | Siz: {stats['user_1h_accuracy'] or '—'}%) · "
                f"4 Soatlik (AI: {stats['ai_4h_accuracy'] or '—'}% | Siz: {stats['user_4h_accuracy'] or '—'}%)"
            )

        # Fetch and render all observations
        observations = db_manager.get_all_observations(limit=200)

        if not observations:
            st.info("Hozircha hech qanday kuzatuv saqlanmagan. 1-bo'limga o'tib, 'Kuzatuvni saqlash' tugmasini bosing.")
        else:
            table_rows = []
            for obs in observations:
                stat = obs.get("status", "PENDING")
                created = obs.get("created_at_utc", "")[:16].replace("T", " ")
                horizon = f"{obs.get('horizon_hours', 4)}h"
                initial = f"${float(obs.get('initial_price', 0)):,.2f}"

                ai_f = obs.get("ai_forecast", {})
                ai_dir = ai_f.get("direction", "—")
                user_p = obs.get("user_prediction", {})
                user_dir = user_p.get("direction", "—")

                eval_data = obs.get("evaluation", {})
                actual_p = eval_data.get("actual_price")
                actual_str = f"${float(actual_p):,.2f}" if actual_p else "—"

                ai_res = "✅" if eval_data.get("ai_correct") is True else ("❌" if eval_data.get("ai_correct") is False else "⏳")
                user_res = "✅" if eval_data.get("user_correct") is True else ("❌" if eval_data.get("user_correct") is False else "⏳")

                status_label = t["status_pending"] if stat == "PENDING" else (t["status_completed"] if stat == "COMPLETED" else t["status_error"])

                table_rows.append({
                    "Vaqt (UTC)": created,
                    "Ufq": horizon,
                    "Boshlang'ich Narx": initial,
                    "AI Prognozi": f"{ai_dir} ({ai_res})",
                    "Sizning Taxminingiz": f"{user_dir} ({user_res})",
                    "Haqiqiy Narx": actual_str,
                    "Holati": status_label,
                    "Izoh": user_p.get("notes", ""),
                })

            df_table = pd.DataFrame(table_rows)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 3: MODEL VA TIZIM METAMA'LUMOTLARI
    # ---------------------------------------------------------
    with tab3:
        st.subheader("ℹ️ Model Arxitekturasi va Tahlil Natijalari")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            ### 1. Volatillik Modeli (Volatility)
            - **Model turi:** `Ridge Regression (alpha=1.0)` + `StandardScaler`
            - **Target:** Keyingi 4 soatlik log-qaytimlarning standart og'ishi
            - **Xususiyatlar:** `vol_4h`, `vol_24h`, `vol_72h`, `return_1h`, `return_4h`, `return_24h`, `volume_change_24h`, `range_pct`
            - **Natija:** Oddiy 24h volatillik baseline (MAE: 0.0019) ga nisbatan Ridge xatolikni **0.0013** ga kamaytirgan.
            """)

        with col_m2:
            st.markdown("""
            ### 2. Narx Yo'nalishi Modeli (Direction)
            - **Model turi:** `Random Forest Classifier` (n_estimators=300, max_depth=6)
            - **Target:** Binar yo'nalish: $C_{t+4} > C_t$
            - **Validatsiya:** 5 bosqichli `TimeSeriesSplit(gap=4)` Walk-Forward testi
            - **Natija:** Balanced Accuracy o'rtacha **~51.7%**
            - **Xulosa:** Bozor tasodifiy harakatga yaqin bo'lgani uchun natijalar sof ehtimollik bahosi sifatida ko'rilishi lozim.
            """)

        st.divider()
        st.markdown("""
        ### ⚠️ Moliyaviy Ogohlantirish va Mas'uliyat Cheklovi
        Ushbu tizim ilmiy-tadqiqot va tajriba maqsadida yaratilgan bo'lib, **moliyaviy yoki investitsion maslahat hisoblanmaydi**. 
        Kriptovalyuta bozori o'ta yuqori xatarga ega. Mashinaviy o'rganish modellarining o'tmishdagi sinov natijalari kelajakdagi aniqlikni kafolatlamaydi.
        """)


if __name__ == "__main__":
    main()
