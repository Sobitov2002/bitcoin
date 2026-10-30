#!/usr/bin/env bash
# ==============================================================================
# setup_server.sh — Contabo VPS (Ubuntu / Debian) da Bitcoin AI Dashboardni sozlash
# ==============================================================================
set -e

echo "=== 1. Tizim paketlarini yangilash ==="
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv git curl ufw

echo "=== 2. Fayl huquqlari va yo'llarini tekshirish ==="
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"
echo "Loyiha manzili: $APP_DIR"

echo "=== 3. Virtual muhit (.venv) yaratish ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo ".venv virtual muhiti yaratildi."
fi

echo "=== 4. Python kutubxonalarini o'rnatish ==="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "=== 5. Fayl xavfsizligi (.env) ==="
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env fayli yaratildi. Iltimos, undagi MONGO_URI sozlamalarini tekshiring!"
fi

echo "=== 6. Firewall (UFW) portlarini ochish ==="
sudo ufw allow 22/tcp || true
sudo ufw allow 8501/tcp || true
sudo ufw allow 80/tcp || true
sudo ufw allow 443/tcp || true
echo "Port 8501 (Streamlit) ochildi."

echo "=== 7. Systemd servisini o'rnatish (24/7 ishlash uchun) ==="
SERVICE_FILE="/etc/systemd/system/bitcoin-dashboard.service"

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=Bitcoin AI Streamlit Dashboard Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$APP_DIR/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
StandardOutput=append:/var/log/bitcoin-dashboard.log
StandardError=append:/var/log/bitcoin-dashboard.error.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bitcoin-dashboard
sudo systemctl restart bitcoin-dashboard

echo "=== 8. Telegram Bot servisini o'rnatish ==="
BOT_SERVICE_FILE="/etc/systemd/system/bitcoin-bot.service"
sudo bash -c "cat > $BOT_SERVICE_FILE" <<EOF
[Unit]
Description=Bitcoin AI Telegram Bot Service
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$APP_DIR/.venv/bin/python telegram_bot.py
Restart=always
RestartSec=5
StandardOutput=append:/var/log/bitcoin-bot.log
StandardError=append:/var/log/bitcoin-bot.error.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bitcoin-bot || true
sudo systemctl restart bitcoin-bot || true

echo "=== 9. Cron vazifalarini rejalashtirish ==="
CRON_CMD="5 * * * * cd $APP_DIR && $APP_DIR/.venv/bin/python update_data.py >> /var/log/btc_update.log 2>&1
10 * * * * cd $APP_DIR && $APP_DIR/.venv/bin/python verify_predictions.py >> /var/log/btc_verify.log 2>&1"

(crontab -l 2>/dev/null | grep -v "update_data.py" | grep -v "verify_predictions.py"; echo "$CRON_CMD") | crontab -

echo ""
echo "=========================================================================="
echo "Tabriklaymiz! Bitcoin AI Dashboard serverda muvaffaqiyatli ishga tushdi!"
echo "Statusni tekshirish: sudo systemctl status bitcoin-dashboard"
echo "Loglarni kuzatish: tail -f /var/log/bitcoin-dashboard.log"
echo "Brauzerda ochish: http://$(curl -s ifconfig.me):8501"
echo "=========================================================================="
