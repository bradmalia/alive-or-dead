#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/ubuntu/alive-or-dead}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"

cd "$APP_DIR"

$PYTHON_BIN -m py_compile main.py

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

sudo install -m 644 deploy/alive-or-dead-apache.conf /etc/apache2/conf-available/alive-or-dead.conf
sudo install -m 644 deploy/alive-or-dead.service /etc/systemd/system/alive-or-dead.service
sudo install -m 644 deploy/root-index.html /var/www/html/index.html

sudo a2enmod proxy proxy_http headers rewrite ssl >/dev/null
sudo a2enconf alive-or-dead >/dev/null

sudo systemctl daemon-reload
sudo systemctl enable alive-or-dead >/dev/null
sudo systemctl restart alive-or-dead
sudo apache2ctl configtest
sudo systemctl reload apache2
