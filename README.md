# ₿ Bitcoin AI Dashboard & Bozor Kuzatuvlari Tizimi

Ushbu loyiha Binance Public API orqali Bitcoin (BTCUSDT) soatlik ma'lumotlarini tahlil qiluvchi, sun'iy intellekt (Machine Learning) yordamida 4 soatlik volatillik va narx yo'nalishini bashorat qiluvchi, shuningdek shaxsiy bozor kuzatuvlarini **MongoDB** da saqlab, AI va inson aniqligini avtomatik taqqoslovchi professional tahliliy platformadir.

---

## 🚀 Asosiy Imkoniyatlar

1. **Jonli Bozor Ma'lumotlari:**
   - Binance API orqali real vaqtdagi jonli narx (Live Ticker) va so'nggi yopilgan sham narxini aniq ajratib ko'rsatish.
   - ExchangeRate API orqali USD va Janubiy Koreya voni (KRW - ₩) konvertatsiyasi.
2. **Mashinaviy O'rganish Prognozlari:**
   - **4 Soatlik Volatillik:** Ridge Regression pipelini yordamida kutilayotgan tebranish darajasi (log-qaytimlar standart og'ishi).
   - **4 Soatlik Narx Yo'nalishi:** Random Forest Classifier yordamida narx oshishi yoki tushishi ehtimolligi (Up vs Down %).
   - **Taxminiy Narx Oralig'i:** Volatillik asosida $\pm 1.96\sigma\sqrt{4}$ yaqinlashuvi bilan hisoblangan taxminiy narx chegarasi.
3. **MongoDB va Shaxsiy Kuzatuvlar (AI vs Inson):**
   - Streamlit native dialog oynasi (`@st.dialog`) orqali AI bashorati va o'z taxminingizni (Ko'tariladi, Tushadi, Yonlama) bir marta bosish bilan saqlash.
   - 1 soatlik va 4 soatlik prognoz ufqlarini alohida qayd etish.
   - MongoDB vaqtincha mavjud bo'lmasa, ma'lumotlarni yo'qotmaslik uchun avtomatik **lokal JSON zaxira (fallback)** tizimi.
4. **Avtomatik Tekshiruv va Aniqlik Tahlili:**
   - Vaqti yetgan har bir kuzatuv uchun Binance API dan haqiqiy narx olinadi va AI hamda foydalanuvchi taxmini to'g'ri/noto'g'riligi baholanadi.
   - AI aniqligi %, Foydalanuvchi aniqligi %, 1h va 4h natijalari statistikasi.
5. **Professional & Responsive UI/UX:**
   - Kompyuter va mobil telefonlar ekraniga to'liq moslashgan toza, yorug' (Light/Clean) dizayn, chuqur to'q yashil aksentlar.
   - Ikki tilli interfeys: **O'zbekcha** va **한국어 (Koreyscha)**.

---

## 📂 Loyiha Fayllari Tuzilishi

```text
BitCoin_Model/
├── app.py                      # Asosiy Streamlit veb-interfeysi (Responsive UI)
├── db_manager.py               # MongoDB ulanishi, CRUD operatsiyalari va lokal zaxira
├── verify_predictions.py       # Bashoratlarni avtomatik tekshiruvchi fon jarayoni
├── download_data.py            # Binance API dan tarixiy soatlik shamlarni yuklash
├── prepare_data.py             # Xususiyatlar va targetlarni tayyorlash
├── train_volatility_model.py   # Ridge volatillik modelini o'qitish va saqlash
├── direction_model.py          # Random Forest yo'nalish klassifikatorini o'qitish
├── evaluate_walk_forward.py    # Walk-Forward validatsiyasi (Data leakage'siz)
├── .env.example                # Konfiguratsiya va MongoDB uchun shablon
├── .env                        # Maxfiy konfiguratsiyalar (Git'ga yuklanmaydi)
├── requirements.txt            # Python kutubxonalari ro'yxati
├── data/                       # Tarixiy ma'lumotlar va baholash CSV fayllari
│   ├── bitcoin_hourly.csv      # 1 soatlik BTCUSDT shamlar bazasi
│   └── observations_backup.json# MongoDB oflayn bo'lganda lokal zaxira
├── models/                     # O'qitilgan ML modellari (.joblib)
│   ├── ridge_volatility.joblib # 4h volatillik pipelini
│   └── direction_classifier.joblib # 4h yo'nalish pipelini
└── tests/
    └── test_pipeline.py        # Avtomatlashtirilgan unit testlar
```

---

## ⚙️ O'rnatish va Ishga Tushirish

### 1. Virtual muhitni faollashtirish
```bash
cd ~/Desktop/App/BitCoin_Model
source .venv/bin/activate
```

### 2. Kutubxonalarni o'rnatish
```bash
pip install -r requirements.txt
```

### 3. MongoDB konfiguratsiyasi (`.env`)
`.env` faylini oching va `<db_username>` o'rniga o'zingizning haqiqiy MongoDB Atlas foydalanuvchi nomingizni kiriting:
```env
MONGO_URI=mongodb+srv://<db_username>:Glr0lRlRXkRPUDmy@cluster0.sin6sgp.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=bitcoin_ai_db
MONGO_COLLECTION=observations
```
*(Eslatma: Agar `.env` da foydalanuvchi nomi kiritilmasa ham, dastur to'xtab qolmaydi, balki lokal xotirada to'liq ishlayveradi).*

### 4. Dashboardni ishga tushirish
```bash
streamlit run app.py
```
Brauzeringizda `http://localhost:8501` ochiladi.

---

## 🔄 Bashoratlarni Avtomatik Tekshirish (Server / Cron)

Kuzatuvlar muddati (1 soat yoki 4 soat) tugagach, ularning haqiqiy natijalarini avtomatik baholash uchun:

### Qo'lda tekshirish:
Dashboarddagi **"Kuzatuvlar Tarixi"** bo'limida **"⚡ Kutilayotgan natijalarni tekshirish"** tugmasini bosing yoki terminalda:
```bash
python verify_predictions.py
```

### Serverda avtomatlashtirish (Cron):
Serverda har soat boshida tekshiruv avtomatik ishlashi uchun `crontab -e` ga quyidagilarni kiriting:
```bash
0 * * * * cd /path/to/BitCoin_Model && .venv/bin/python verify_predictions.py >> /var/log/btc_verify.log 2>&1
```

---

## 🧪 Avtomatlashtirilgan Testlarni Ishga Tushirish
Barcha modullar, ML xususiyatlari va baza mantiqini tekshirish uchun:
```bash
python -m unittest tests/test_pipeline.py
```

---

## ⚠️ Muhim Ogohlantirish
Ushbu loyiha ilmiy tahlil va tajribaviy tadqiqot maqsadida yaratilgan. Kriptovalyuta bozori o'ta tebranuvchan bo'lib, ushbu modellar moliyaviy maslahat hisoblanmaydi.
# bitcoin_analiytc
# bitcoin_analiytc
