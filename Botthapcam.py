# -*- coding: utf-8 -*-
"""
Bot AI Kid Trader (Trading + Tài Xỉu)
Phiên bản: V17.6 - FINAL FIX: API ENUMS, DYNAMIC SR/ATR & PRECISION
"""

# =======================
# 0. AUTO INSTALLER
# =======================
import sys, subprocess, shutil, os
def _run(cmd, quiet=True):
    if quiet:
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, None
        except subprocess.CalledProcessError as e:
            try: subprocess.check_call(cmd)
            except Exception as e2: return False, e2
            return True, None
    else:
        try:
            subprocess.check_call(cmd)
            return True, None
        except Exception as e:
            return False, e

print("⚙️ Auto-setup: kiểm tra môi trường...")
try:
    py_exec = sys.executable
    print("→ Kiểm tra/ nâng cấp pip (nếu cần)...")
    _run([py_exec, "-m", "pip", "install", "--upgrade", "pip"], quiet=True)
except Exception: pass

# Cần thêm 'python-binance' và 'ccxt' cho Multi-API
modules = [
    "flask", "pyTelegramBotAPI", "pandas", "numpy", "requests", 
    "pyppeteer", "gspread", "oauth2client", "joblib", "scikit-learn", 
    "matplotlib", "Pillow", "mplfinance", "pytz", 
    "TA-Lib", "python-binance", "ccxt",
    "websocket-client" # <<< THÊM THƯ VIỆN WEBSOCKET
]
for m in modules:
    try:
        __import__(m)
        print(f"✓ {m} (đã có)")
    except ImportError:
        print(f"⚙️ Cài module: {m} ...")
        ok, err = _run([py_exec, "-m", "pip", "install", m], quiet=True)
        if ok: print(f"✓ {m} (cài thành công)")
        else: print(f"✖ Lỗi cài {m}: {err}")

# try:
#     print("⚙️ Cài đặt Pyppeteer (Chromium)...")
#     _run([sys.executable, "-m", "pip", "install", "pyppeteer"], quiet=True)
#     try: subprocess.check_call([sys.executable, "-m", "pyppeteer", "install"])
#     except: pass
#     print("✓ Pyppeteer (Chromium) đã cài đặt.")
# except Exception: pass
print("⚙️ Auto-setup hoàn tất.\n")

# =======================
# 1. IMPORTS & CẤU HÌNH
# =======================
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
import gspread
import joblib
import time
import numpy as np
import talib 
import re
import threading
import asyncio
import json
import uuid
import math
import statistics
import io
import random
from collections import deque, Counter
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pytz 
import os 
import atexit 
import warnings
# Tắt cảnh báo "UserWarning: X does not have valid feature names..."
warnings.filterwarnings('ignore', category=UserWarning, message='X does not have valid feature names')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from oauth2client.service_account import ServiceAccountCredentials
from pyppeteer import launch
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import websocket
import ssl
import queue

# <<< THÊM IMPORTS CHO API >>>
import ccxt
from binance.client import Client
from binance.enums import *
from binance.enums import (
    SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, 
    TIME_IN_FORCE_GTC 
    # FIX: Đã XÓA HẲN các hằng số Futures bị lỗi import
)
# ===============================================

# =======================
# 2. HẰNG SỐ & BIẾN TOÀN CỤC
# =======================
TOKEN = os.environ.get("YOUR_TELEGRAM_BOT_TOKEN", "8569714455:AAFwuCEJS9bthTEp4oJ6LFXIDtNRTpXtNrI")
if TOKEN == "8569714455:AAFwuCEJS9bthTEp4oJ6LFXIDtNRTpXtNrI":
    print("===================================================================")
    print("⚠️ CẢNH BÁO BẢO MẬT: Bạn đang dùng Token Bot hardcode trong file code.")
    print("⚠️ Vui lòng xóa Token và dùng Biến Môi Trường (Environment Variable).")
    print("===================================================================")

bot = telebot.TeleBot(TOKEN, parse_mode="HTML")

# Cấu hình Google Sheet
SPREADSHEET_IDS = {
    "68gamebai": "1iNQbV9vm5YvR5J2bHpuUUadWtynz-i1A5-Q4_j7JR4Q",
    "xocdia88": "13tftVaa5VkiQ7wvKN4NZ8VexEtRQ61xsdnFbUdNdDTk",
    "b52": "14ktzOt8T7k9wEbgnrDkx-CifgTl8kQS5jVPSHiC2Cl8",
    "hitclub": "1mCE5lHml2sqabu5DfgpuKvA56h1eWzzncDlfcsbYjTo",
    "zomclub": "1LJ8k2rnVFWBlJYn3KOxrahZNctVZf8g0B55PBbqyiUM",
    "user_data": "YOUR_USER_DATA_SHEET_ID_HERE" 
}
SHEET_CREDENTIALS = "client_secret.json"
MODEL_FILE = "model_taixiu.pkl"

# Cấu hình Trading
PREFERRED_EXCHANGES = ["BINANCE", "OKX", "BYBIT", "BINGX", "TOOBIT"] 
AI_MEMORY_FILE = "ai_memory.json"
MIN_REPORT_INTERVAL_HOURS = 4.0 
PNL_RESET_DAYS = 7 # Số ngày để reset PnL
WATCHLIST_FILE = "user_watchlist.json"
AI_SIGNALS_LOG = "ai_signals.log"
TOP_COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","AVAXUSDT","DOGEUSDT","DOTUSDT","LINKUSDT"]
TZ = pytz.timezone('Asia/Ho_Chi_Minh') # Múi giờ VN
USER_LIST_FILE = "bot_users.json"
USER_DATA_FILE = "user_data_persistence.json" 
MESSAGE_QUEUE = queue.Queue()

# Biến toàn cục
user_data = {}
version_counter = 0
user_game_name = {}
ai_memory = {}
_symbol_cache = None
_user_states = {} 
USER_THROTTLE_CACHE = {} 
THROTTLE_TIME_SECONDS = 1.5 
LAST_SIGNAL_TIME = {} # {symbol: {timeframe: timestamp}}
_exchange_info_cache = None # Cache cho Exchange Info
REALTIME_PRICE_CACHE = {}
REALTIME_PRICE_LOCK = threading.Lock()

# --- KIỂM TRA THƯ VIỆN TESTNET (Không lưu keys toàn cục) ---
TESTNET_CLIENT = None 
try:
    Client("dummy", "dummy", testnet=True).futures_ping()
    print("✅ Thư viện Binance Testnet đã sẵn sàng.")
except Exception as e:
    print(f"⚠️ Thư viện Binance Testnet chưa sẵn sàng để kiểm tra: {e}")
# ------------------------------------------------------------------------


# =======================
# 3. KHỞI TẠO MODEL
# =======================
try:
    model = joblib.load(MODEL_FILE)
    print("✅ Model Tài Xỉu đã được load thành công!")
except:
    print("⚠️ Không tìm thấy model Tài Xỉu, sử dụng model giả.")
    class DummyModel:
        def predict(self, x): return [0]
        def predict_proba(self, x): return [[0.5, 0.5]]
    model = DummyModel()

trading_model = None
MODEL_PATH = 'trading_model.pkl'
if os.path.exists(MODEL_PATH):
    try:
        trading_model = joblib.load(MODEL_PATH)
        print(f"✅ Đã tải model AI trading từ {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Lỗi khi tải model AI: {e}. Bot sẽ hoạt động theo luật cũ.")
else:
    print(f"⚠️ Không tìm thấy file model AI tại {MODEL_PATH}. Bot sẽ hoạt động theo luật cũ (rule-based).")

# =======================
# 4. HÀM HỖ TRỢ CHUNG
# =======================
from cryptography.fernet import Fernet

# Khóa giải mã được tạo từ biến TOKEN (hoặc biến môi trường khác)
# CHÚ Ý: ĐỪNG ĐỂ LỘ KHÓA NÀY!
ENCRYPTION_KEY = Fernet.generate_key() 
if os.path.exists("encryption_key.key"):
    with open("encryption_key.key", "rb") as key_file:
        ENCRYPTION_KEY = key_file.read()
else:
    with open("encryption_key.key", "wb") as key_file:
        key_file.write(ENCRYPTION_KEY)
        
cipher_suite = Fernet(ENCRYPTION_KEY)

def encrypt_key(data):
    """Mã hóa chuỗi dữ liệu (API Key/Secret)."""
    if data is None: return None
    try:
        encoded_data = data.encode()
        return cipher_suite.encrypt(encoded_data).decode()
    except:
        return data

def decrypt_key(encrypted_data):
    """Giải mã chuỗi dữ liệu."""
    if encrypted_data is None: return None
    try:
        if encrypted_data.startswith("gAAAA"): # Kiểm tra định dạng Fernet
             decoded_data = encrypted_data.encode()
             return cipher_suite.decrypt(decoded_data).decode()
        return encrypted_data # Trả về nếu không phải dạng mã hóa (tương thích ngược)
    except:
        return encrypted_data # Trả về keys gốc nếu giải mã thất bại

def connect_google_sheets(app_name):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(SHEET_CREDENTIALS, scope)
        client = gspread.authorize(creds)
        return client.open_by_key(SPREADSHEET_IDS[app_name]).sheet1
    except Exception as e:
        print(f"❌ Lỗi kết nối Google Sheet '{app_name}': {e}")
        return None

def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" +
        "\U0001F600-\U0001F64F" +
        "\U0001F300-\U0001F5FF" +
        "\U0001F680-\U0001F6FF" +
        "\U0001F1E0-\U0001F1FF" +
        "\U00002700-\U000027BF" +
        "\U000024C2-\U0001F251" +
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# FILE: Botthapcamnhucac.py
def ensure_user_data_structure(chat_id):
    # Đảm bảo cấu trúc cơ bản cho cả Trading và Tài Xỉu
    user = user_data.setdefault(str(chat_id), {}) 
    if "mode" not in user:
        user["mode"] = "taixiu"
    
    if "taixiu" not in user:
        user["taixiu"] = {
            "history": [],
            "win": 0, "lose": 0, "balance": 0,
            "bet": 1000, "base_bet": 1000,
            "outcome_history": [],
            "history_deque": deque(maxlen=50)
        }
    
    if "trading" not in user:
        user["trading"] = {
            "balance": 0, "trades": {}, "last_signals": {},
            "watchlist": [], 
            "auto_trade_intervals": [], 
            "signal_pref": "short", "signals": [], 
            "alerts": {}, "exchange": "BINANCE",
            "risk_per_trade": 1.0, 
            "total_capital": 1000.0,
            "leverage": 5.0,
            "api_key": None, 
            "secret_key": None,
            "passphrase": None, 
            "report_interval": 4.0, 
            "style": "SWING",
            "pnl_counts": Counter(),
            "auto_exit_on_reversal": True # <<< MỚI: Tự động đóng lệnh API khi Reversal >>>
        }
    
    # --- Tương thích ngược & FIX LỖI (Quan trọng) ---
    if "auto_trade_intervals" not in user["trading"]:
        user["trading"]["auto_trade_intervals"] = [] 

    if "auto_trade_interval" in user["trading"]:
        interval_val = user["trading"]["auto_trade_interval"]
        if interval_val is not None and isinstance(interval_val, (int, float)) and int(interval_val) not in user["trading"]["auto_trade_intervals"]:
             user["trading"]["auto_trade_intervals"].append(int(interval_val))
        del user["trading"]["auto_trade_interval"]
        
    if "last_pnl_reset" not in user["trading"]:
        user["trading"]["last_pnl_reset"] = datetime.now(TZ).isoformat()
        
    if "passphrase" not in user["trading"]:
        user["trading"]["passphrase"] = None

    if "report_interval" not in user["trading"]:
        user["trading"]["report_interval"] = 4.0
        
    if "style" not in user["trading"]:
        user["trading"]["style"] = "SWING"
        
    if "auto_exit_on_reversal" not in user["trading"]:
        user["trading"]["auto_exit_on_reversal"] = True

    if "pnl_counts" not in user["trading"] or not isinstance(user["trading"]["pnl_counts"], Counter):
         if isinstance(user["trading"].get("pnl_counts"), dict):
              user["trading"]["pnl_counts"] = Counter(user["trading"]["pnl_counts"])
         else:
              user["trading"]["pnl_counts"] = Counter()
        
    return user
    # -----------------------------------------------
        
    return user

def save_user_data():
    """Lưu user_data (bộ nhớ) vào file JSON và mã hóa keys."""
    global user_data
    print(f"💾 Đang lưu user_data vào {USER_DATA_FILE}...")
    try:
        data_to_save = {}
        for chat_id, data in user_data.items():
            data_copy = data.copy() 
            if "taixiu" in data_copy and "history_deque" in data_copy["taixiu"]:
                if isinstance(data_copy["taixiu"], dict):
                    data_copy["taixiu"]["history_deque"] = list(data_copy["taixiu"]["history_deque"])
                else:
                    data_copy["taixiu"] = {"history_deque": []}
            
            # Xử lý PnL Counts
            if "trading" in data_copy and "pnl_counts" in data_copy["trading"]:
                 data_copy["trading"]["pnl_counts"] = dict(data_copy["trading"]["pnl_counts"])

            # <<< NÂNG CẤP 1: MÃ HÓA KEYS TRƯỚC KHI LƯU >>>
            if "trading" in data_copy:
                 data_copy["trading"]["api_key"] = encrypt_key(data_copy["trading"].get("api_key"))
                 data_copy["trading"]["secret_key"] = encrypt_key(data_copy["trading"].get("secret_key"))
                 data_copy["trading"]["passphrase"] = encrypt_key(data_copy["trading"].get("passphrase"))
            # <<< KẾT THÚC MÃ HÓA KEYS >>>

            data_to_save[chat_id] = data_copy
            
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        print("✅ Lưu user_data thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi lưu user_data: {e}")

def load_user_data():
    """Tải user_data từ file JSON khi bot khởi động và dọn dẹp tín hiệu cũ."""
    global user_data
    
    current_dt = datetime.now(TZ)
    # Chỉ giữ lại tín hiệu đã đóng trong 30 ngày gần nhất
    thirty_days_ago = current_dt - timedelta(days=30) 
    
    if not os.path.exists(USER_DATA_FILE):
        print(f"⚠️ Không tìm thấy file {USER_DATA_FILE}. Bắt đầu với user_data trống.")
        return

    print(f"🔄 Đang tải user_data từ {USER_DATA_FILE}...")
    try:
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            
        for chat_id, data in loaded_data.items():
            ensure_user_data_structure(chat_id) 
            
            if "taixiu" in data and isinstance(data["taixiu"], dict) and "history_deque" in data["taixiu"]:
                data["taixiu"]["history_deque"] = deque(data["taixiu"]["history_deque"], maxlen=50)
            
            if "trading" in data and "pnl_counts" in data["trading"] and isinstance(data["trading"]["pnl_counts"], dict):
                 data["trading"]["pnl_counts"] = Counter(data["trading"]["pnl_counts"])
            
            user_data[chat_id].update(data)
            
            # <<< LOGIC DỌN DẸP TÍN HIỆU CŨ (Tối ưu hóa) >>>
            if "trading" in user_data[chat_id] and "signals" in user_data[chat_id]["trading"]:
                signals = user_data[chat_id]["trading"]["signals"]
                filtered_signals = []
                
                for sig in signals:
                    status = sig.get("status")
                    
                    if status == "open":
                        filtered_signals.append(sig)
                        continue # Luôn giữ lệnh đang mở
                    
                    # Giữ lại các lệnh lỗi (để debug)
                    if status in ["closed_legacy_error", "error_trailing"]:
                         filtered_signals.append(sig)
                         continue
                    
                    # Kiểm tra thời gian tạo lệnh (cho các lệnh đã đóng khác)
                    created_at_iso = sig.get("created_at")
                    if created_at_iso:
                        try:
                            # Chỉ giữ lại lệnh đã đóng trong 30 ngày gần nhất
                            created_dt = datetime.fromisoformat(created_at_iso).replace(tzinfo=TZ)
                            if created_dt > thirty_days_ago: 
                                filtered_signals.append(sig)
                        except:
                            filtered_signals.append(sig) 
                            
                user_data[chat_id]["trading"]["signals"] = filtered_signals
                print(f"  -> User {chat_id}: Đã dọn dẹp tín hiệu cũ. Giữ lại {len(filtered_signals)} tín hiệu.")
            # <<< KẾT THÚC LOGIC DỌN DẸP >>>


        print(f"✅ Tải và khôi phục {len(user_data)} user(s) thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải user_data: {e}. Bắt đầu với user_data trống.")
        print(f"   Lỗi chi tiết tại: {e}")
        user_data = {} 

# --- V15: HÀM KIỂM TRA CHO ANTI-SPAM ---
def check_throttle(chat_id):
    """Kiểm tra xem người dùng có đang spam không."""
    global USER_THROTTLE_CACHE
    now = time.time()
    last_call = USER_THROTTLE_CACHE.get(chat_id, 0)
    
    if now - last_call < THROTTLE_TIME_SECONDS:
        return False # Đang bị chặn (spam)
    
    # Cập nhật thời gian gọi cuối cùng
    USER_THROTTLE_CACHE[chat_id] = now
    return True # Cho phép xử lý

# =======================
# 5. HÀM HỖ TRỢ TÀI XỈU
# =======================
def detect_game_from_text(text):
    if "🦍" in text or "b52" in text.lower(): return "b52"
    if "💥" in text or "hit" in text.lower(): return "hitclub"
    if "xocdia" in text.lower(): return "xocdia88"
    if "68" in text.lower(): return "68gamebai"
    return None

def extract_valid_md5(text):
    match = re.search(r'[a-zA-Z0-9]{32}', text)
    return match.group(0) if match else None

def extract_features(md5_hash):
    return [ord(c) % 10 for c in md5_hash[:8]]

def predict_md5(raw_text):
    raw_text = remove_emojis(raw_text)
    md5_hash = extract_valid_md5(raw_text)
    if not md5_hash:
        return ("❌ Không tìm thấy chuỗi hợp lệ", 0, 0)
    features = np.array([extract_features(md5_hash)])
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(features)[0]
            prediction = model.predict(features)[0]
            return ("Tài" if prediction == 1 else "Xỉu", prob[1]*100, prob[0]*100)
    except Exception as e:
        print(f"Lỗi predict_md5: {e}")
    return ("Không thể dự đoán", 0, 0)

def save_result_async(md5, result, dice, app_name, outcome, chat_id=None):
    def save():
        try:
            final_app = app_name or user_data.get(str(chat_id), {}).get("app_name")
            if not final_app or final_app not in SPREADSHEET_IDS:
                print(f"⚠️ Không xác định được app hợp lệ để lưu: {final_app}")
                return

            sheet = connect_google_sheets(final_app)
            if sheet is None:
                print(f"⚠️ Không thể kết nối GSheet cho app: {final_app}")
                return
                
            existing = sheet.get_all_values()
            existing = [row for row in existing if row]
            if any(len(row) > 0 and md5 == row[0] for row in existing):
                print(f"⚠️ MD5 {md5} đã tồn tại, bỏ qua.")
                return

            sheet.append_row([md5, result, dice, time.strftime("%Y-%m-%d %H:%M:%S"), outcome])
            print(f"✅ Đã lưu: {md5} -> {outcome} vào {final_app}")
        except Exception as e:
            print(f"❌ Lỗi khi lưu async: {e}")
            
    threading.Thread(target=save).start()

def detect_trend(history):
    if len(history) < 5: return "📊 Chưa đủ dữ liệu."
    last6 = history[-6:]
    last8 = history[-8:]
    if len(set(history[-3:])) == 1: return f"⚠️ Cầu Bệt {history[-1]}! (>=3)"
    if last6 == ["Tài", "Xỉu"] * 3 or last6 == ["Xỉu", "Tài"] * 3: return "🔄 Cầu 1–1 (Ping-pong)!"
    if last8 == ["Tài", "Tài", "Xỉu", "Xỉu"] * 2 or last8 == ["Xỉu", "Xỉu", "Tài", "Tài"] * 2: return "⛓️ Cầu 2–2!"
    if last6 == ["Tài", "Tài", "Tài", "Xỉu", "Xỉu", "Xỉu"] or last6 == ["Xỉu", "Xỉu", "Xỉu", "Tài", "Tài", "Tài"]: return "⛓️ Cầu 3–3!"
    return "📉 Không có cầu mạnh."

def parse_result_string(result_string):
    match = re.search(r"\{(\d+)[-_](\d+)[-_](\d+)\}", result_string)
    if match:
        a, b, c = map(int, match.groups())
        return ("Xỉu" if a + b + c <= 10 else "Tài", f"{a}-{b}-{c}")
    return None, None

def get_bet_suggestion(user, outcome):
    bet_amount = user.get("bet", 1000)
    base_bet = user.get("base_bet", 1000)
    outcomes = user.get("outcome_history", [])

    if len(outcomes) >= 2 and outcomes[-2:] == ["Thua", "Thua"]:
        bet_amount *= 2
    elif len(outcomes) >= 3 and outcomes[-3:] == ["Thắng", "Thắng", "Thắng"]:
        bet_amount = int(bet_amount * 1.5)
    elif bet_amount > base_bet * 8:
        bet_amount = base_bet
    elif outcome == "Thắng":
        bet_amount = base_bet 

    user["bet"] = bet_amount
    return bet_amount

def record_taixiu_result(chat_id, dice):
    user = ensure_user_data_structure(chat_id)
    hist = user["taixiu"].setdefault("history_deque", deque(maxlen=50))
    total = sum(dice)
    outcome = "TÀI" if total >= 11 else "XỈU"
    hist.append({"dice": dice, "total": total, "outcome": outcome, "time": datetime.now(TZ).isoformat()})
    return outcome, total

def predict_taixiu(chat_id):
    user = ensure_user_data_structure(chat_id)
    hist = list(user["taixiu"].get("history_deque", []))
    if not hist:
        return {"prediction": random.choice(["TÀI","XỈU"]), "confidence": 50}
    outcomes = [h["outcome"] for h in hist]
    cnt = Counter(outcomes)
    most_common, most_count = cnt.most_common(1)[0]
    base_conf = int(most_count/len(outcomes)*100)
    last3 = outcomes[-3:]
    if len(last3) == 3 and len(set(last3)) == 1:
        base_conf = min(95, base_conf + 15)
    conf = max(30, min(95, base_conf))
    return {"prediction": most_common, "confidence": conf}

# =======================
# 6. HÀM HỖ TRỢ TRADING
# =======================
def get_market_price(symbol):
    # FIX: Đã chuyển sang FAPI (Futures) thay vì API (Spot)
    url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return float(data['price'])
    except requests.exceptions.RequestException as e:
        # Tách riêng lỗi 400 (Symbol không tồn tại trên FAPI)
        try:
            if response.status_code == 400:
                 print(f"❌ Lỗi 400 (FUTURES): Symbol {symbol} không tồn tại trên Binance Futures.")
            else:
                 print(f"❌ Lỗi khi lấy giá thị trường (FUTURES) cho {symbol}: {e}")
        except:
             print(f"❌ Lỗi khi lấy giá thị trường (FUTURES) cho {symbol}: {e}")
        return None
#====================================
def get_bybit_market_price(symbol):
    """
    FIX LỖI 404: Lấy giá thị trường (ticker) từ Bybit V5, buộc dùng URL LIVE và thử cả SYMBOLPERP.
    """
    
    # --- BUỘC DÙNG URL LIVE ---
    base_url = "https://api.bybit.com"
    url = f"{base_url}/v5/market/tickers"
    # --------------------------
    
    # Logic thử tên cặp
    symbol_perp = symbol
    if symbol.endswith("USDT") and not symbol.endswith("PERP"):
        symbol_perp = f"{symbol}PERP"
    
    symbol_attempts = [symbol_perp, symbol]
    symbol_attempts = list(dict.fromkeys(symbol_attempts))
    
    for attempt_symbol in symbol_attempts:
        params = {
            "category": "linear",
            "symbol": attempt_symbol
        }
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status() 
            data = response.json()
            
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                return float(data["result"]["list"][0]["lastPrice"])
            
            if data.get("retCode") != 0:
                print(f"DEBUG (Price): Thử {attempt_symbol}: Lỗi API {data['retCode']} - {data.get('retMsg', 'Unknown')}")
                continue
            
        except requests.exceptions.RequestException as e:
            print(f"DEBUG (Price): Thử {attempt_symbol}: Lỗi request {e}")
            continue
            
    print(f"❌ Không thể lấy giá thị trường (BYBIT) cho {symbol}. Thất bại sau khi thử cả PERP.")
    return None


def get_user_exchange_client(user, for_check=False): 
    
    # <<< NÂNG CẤP: GIẢI MÃ KEYS TRƯỚC KHI SỬ DỤNG >>>
    api_key_enc = user["trading"].get("api_key")
    secret_key_enc = user["trading"].get("secret_key")
    passphrase_enc = user["trading"].get("passphrase")
    
    api_key = decrypt_key(api_key_enc)
    secret_key = decrypt_key(secret_key_enc)
    passphrase = decrypt_key(passphrase_enc)
    # <<< KẾT THÚC GIẢ MÃ KEYS >>>

    exchange = user["trading"].get("exchange")
    
    if not api_key or not secret_key:
        return None, "Chưa cài đặt API Keys cá nhân."
        
    if exchange == "BINANCE":
        try:
            # FIX LỖI 494/TIMEOUT: Thêm tld='com' và tăng requests_params timeout lên 30s
            client = Client(api_key, secret_key, testnet=True, tld='com', requests_params={"timeout": 30}) 
            return client, None
        except Exception as e:
            return None, f"Binance Testnet Error: {e}"
            
    elif exchange == "OKX":
        if not passphrase:
             return None, "Thiếu Passphrase OKX."
        try:
            client = ccxt.okx({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase, 
                'options': {'defaultType': 'swap'}, 
                'enableRateLimit': True,
                # CCXT có cơ chế Timeout riêng, thường là 10s mặc định.
                'timeout': 30000 # Tăng timeout lên 30 giây cho OKX
            })
            client.load_markets()
            return client, None
        except Exception as e:
            return None, f"OKX API Error (CCXT): {e}"
            
    return None, f"Sàn {exchange} chưa được hỗ trợ giao dịch tự động."


# <<< THÊM HÀM PRECISION (Triệt để) >>>
# Cache cho thông tin Exchange Info (Độ chính xác)
_exchange_info_cache = None

def get_exchange_info():
    """Tải và cache toàn bộ thông tin về độ chính xác (Precision) từ Binance Futures."""
    global _exchange_info_cache
    if _exchange_info_cache is None:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            info = r.json()
            
            # Xây dựng cache chỉ chứa thông tin quan trọng
            cache = {}
            for s in info.get('symbols', []):
                filters = s.get('filters', [])
                
                # Tìm Quantity Precision (stepSize)
                step_size = 0.001 
                for f in filters:
                    if f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        break
                        
                # Tìm Price Precision (tickSize)
                tick_size = 0.0001
                for f in filters:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size = float(f['tickSize'])
                        break
                
                cache[s['symbol']] = {
                    'stepSize': step_size, 
                    'tickSize': tick_size,
                }
            _exchange_info_cache = cache
        except Exception as e:
            print(f"❌ Lỗi khi tải ExchangeInfo (Futures): {e}")
            _exchange_info_cache = {}
            
    return _exchange_info_cache

def round_by_step(value, step_size):
    """Hàm làm tròn giá trị theo bước (stepSize hoặc tickSize) của Binance."""
    if step_size == 0: 
        # Nếu step_size = 0 (lỗi), làm tròn về 8 số thập phân an toàn
        return round(value, 8) 
    
    # Tính số lượng bước gần nhất, rồi nhân lại với step_size
    return round(math.floor(value / step_size) * step_size, 8) 
    
def get_symbol_precision(symbol):
    """Lấy stepSize và tickSize cho một symbol cụ thể."""
    info = get_exchange_info()
    return info.get(symbol, {'stepSize': 0.001, 'tickSize': 0.0001})

def execute_trade_testnet(symbol, trend_type, entry, sl, tps, position_size_qty, order_type, user):
    
    client, error_msg = get_user_exchange_client(user) 
    
    if client is None:
        return False, error_msg, None, None 
        
    exchange_name = user["trading"].get("exchange")
    
    side = client.SIDE_BUY if "Tăng" in trend_type else client.SIDE_SELL
    
    if exchange_name == "BINANCE":
        
        if "Tăng" in trend_type:
            position_side = 'LONG' 
        else:
            position_side = 'SHORT'
        
        precision = get_symbol_precision(symbol)
        step_size = precision['stepSize'] 
        tick_size = precision['tickSize'] 

        position_size_qty = round_by_step(position_size_qty, step_size)
        
        if position_size_qty <= 0.0:
            return False, f"Binance: Khối lượng tính toán quá nhỏ (0.0).", None, None 

        order_type_binance = client.ORDER_TYPE_MARKET
        
        tp1_price = round_by_step(tps[0], tick_size)
        sl_price = round_by_step(sl, tick_size)

        price_round_precision = int(-math.log10(tick_size)) if tick_size > 0 else 8
        
        try:
            # Gửi lệnh MARKET
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type=order_type_binance,
                quantity=position_size_qty,
                positionSide=position_side 
            )
            
            close_side = client.SIDE_SELL if side == client.SIDE_BUY else client.SIDE_BUY
            
            batch_orders_def = [
                # TP1
                {'symbol': symbol, 'side': close_side, 'type': client.ORDER_TYPE_TAKE_PROFIT_MARKET, 'quantity': position_size_qty, 'stopPrice': f"{tp1_price:.{price_round_precision}f}", 'timeInForce': client.TIME_IN_FORCE_GTC, 'positionSide': position_side, 'reduceOnly': True},
                # SL
                {'symbol': symbol, 'side': close_side, 'type': client.ORDER_TYPE_STOP_MARKET, 'quantity': position_size_qty, 'stopPrice': f"{sl_price:.{price_round_precision}f}", 'timeInForce': client.TIME_IN_FORCE_GTC, 'positionSide': position_side, 'reduceOnly': True}
            ]
            
            oco_orders = client.futures_place_batch_orders(batchOrders=batch_orders_def)
            
            sl_order_id = None
            tp_order_id = None
            
            for o in oco_orders:
                if o['type'] == 'STOP_MARKET':
                    sl_order_id = o['orderId']
                elif o['type'] == 'TAKE_PROFIT_MARKET':
                    tp_order_id = o['orderId']
            
            if not sl_order_id or not tp_order_id:
                 try: client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                 except: pass
                 return False, "Binance API Error: Không thể lấy OCO Order ID.", None, None

            return True, f"Binance: MARKET Order & OCO SL/TP (ID: {sl_order_id}) gửi thành công.", sl_order_id, tp_order_id
        
        except Exception as e:
            return False, f"Binance API Error: {e}", None, None 
            
    elif exchange_name == "OKX":
        return False, "OKX Trailing SL chưa được hỗ trợ trong bản sửa lỗi này.", None, None

    return False, f"Sàn {exchange_name} chưa được hỗ trợ.", None, None


async def _capture_tradingview_chart_async(symbol, exchange="BINANCE", width=1400, height=900, timeout=20):
    symbol_full = f"{exchange}:{symbol}"
    chart_url = f"https://www.tradingview.com/chart/?symbol={symbol_full}&theme=dark"
    
    # Tạo thư mục người dùng tạm thời (RẤT QUAN TRỌNG ĐỂ TRÁNH LỖI COOKIE OVERFLOW)
    user_data_dir = os.path.join(os.getcwd(), 'chrome_session_temp')
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
        
    # Khởi tạo trình duyệt headless (Chromium)
    browser = await launch(headless=True,
                        args=['--no-sandbox', '--disable-setuid-sandbox','--disable-gpu','--single-process'],
                        ignoreHTTPSErrors=True,
                        handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False,
                        userDataDir=user_data_dir) # <<< SỬ DỤNG USERDATADIR TẠM THỜI >>>
    
    page = await browser.newPage()
    await page.setViewport({"width": width, "height": height})
    
    # FIX LỖI COOKIE: Xóa tất cả Cookies trước khi tải trang (Đảm bảo Header nhỏ)
    await page.deleteCookie()
    
    try:
        # Tải trang TradingView
        await page.goto(chart_url, {"waitUntil": "networkidle2", "timeout": timeout*1000})
        await asyncio.sleep(4)
        
        # Thử đóng pop-up cookie/GDPR (nếu có)
        try:
            await page.evaluate("""() => {
                const btn = document.querySelector('button[data-name="onetrust-accept-btn-handler"]');
                if(btn) btn.click();
            }""")
        except: pass
        
        # Chụp màn hình và lưu file
        path = f"/tmp/{symbol}_{int(time.time())}.png"
        await page.screenshot({'path': path, 'fullPage': True})
        
        await browser.close()
        
        # DỌN DẸP: Xóa thư mục tạm thời sau khi sử dụng (để session luôn sạch)
        shutil.rmtree(user_data_dir, ignore_errors=True)
        
        return path
    except Exception as e:
        # Đóng trình duyệt nếu có lỗi
        try: await browser.close()
        except: pass
        
        # DỌN DẸP LỖI
        shutil.rmtree(user_data_dir, ignore_errors=True)
        
        raise # Ném lỗi để thông báo cho người dùng
        
# -------------------------------------------------------------------

def capture_tradingview_chart(symbol, exchange="BINANCE", width=1400, height=900, timeout=20):
    """
    PHIÊN BẢN GIẢM TẢI RAM: Tạm thời vô hiệu hóa chức năng chụp chart 
    bằng Pyppeteer/Chromium để tránh lỗi 'zsh: killed' (out of memory).
    """
    print("⚠️ Chức năng chụp chart TradingView đang bị vô hiệu hóa để giảm RAM.")
    
    # Bạn có thể chọn gửi cảnh báo cho người dùng
    # try:
    #     bot.send_message(chat_id, "⚠️ Chức năng chụp chart đang tạm ngưng do lỗi tài nguyên.")
    # except Exception:
    #     pass
        
    return None # Trả về None để các hàm gọi (ví dụ: analyze_and_send) có thể bỏ qua việc gửi ảnh.
    
    # [DÒNG CODE GỐC (ĐÃ BỊ VÔ HIỆU HÓA) SẼ KHÔNG CẦN CHẠY:]
    # try:
    #     path = asyncio.run(_capture_tradingview_chart_async(symbol, exchange, width, height, timeout))
    #     return path
    # except RuntimeError as e:
    #     if "cannot run current event loop" in str(e):
    #         loop = asyncio.get_event_loop()
    #         path = loop.run_until_complete(_capture_tradingview_chart_async(symbol, exchange, width, height, timeout))
    #         return path
    #     else:
    #         print(f"Lỗi Runtime asyncio: {e}")
    #         raise e
    # except Exception as e:
    #     print(f"Lỗi không xác định trong capture_tradingview_chart: {e}")
    #     raise e



def fetch_binance_klines(symbol, interval, limit=250):
    # FIX: Đã chuyển sang FAPI (Futures) thay vì API (Spot)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json() 
        return data 
    except Exception as e:
        print(f"Lỗi fetch_binance_klines (FUTURES) {symbol}: {e}")
        return None

# (Hàm fetch_binance_klines của Binance kết thúc ở đây)

def fetch_bybit_klines(symbol, interval, limit=250):
    """
    FIX TRIỆT ĐỂ: Buộc gọi API Live và PERP. Nếu vẫn lỗi 404, vấn đề là do IP/Region.
    """
    
    # --- 1. SỬ DỤNG URL LIVE MẶC ĐỊNH (KHÔNG DÒ TÌM) ---
    # Nếu cần Testnet, người dùng phải tự sửa thủ công base_url này.
    base_url = "https://api.bybit.com" 
    url = f"{base_url}/v5/market/klines"
    print("HINT: Đang sử dụng BYBIT LIVE URL (Mặc định).")
    
    # 2. Map interval
    interval_map = {
        "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "4h": "240", "1d": "D"
    }
    bybit_interval = interval_map.get(interval)
    if not bybit_interval:
        print(f"Lỗi Bybit: Khung thời gian {interval} không được hỗ trợ.")
        return None

    # --- 3. LOGIC THỬ TÊN CẶP ---
    symbol_perp = symbol
    if symbol.endswith("USDT") and not symbol.endswith("PERP"):
        symbol_perp = f"{symbol}PERP"
    
    symbol_attempts = [symbol_perp, symbol]
    symbol_attempts = list(dict.fromkeys(symbol_attempts))
    final_data = None

    for attempt_symbol in symbol_attempts:
        params = {
            "category": "linear", 
            "symbol": attempt_symbol,
            "interval": bybit_interval,
            "limit": limit
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status() 
            data = resp.json()
            
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                final_data = data
                break 
            
            if data.get("retCode") != 0:
                print(f"DEBUG: Thử {attempt_symbol}: Lỗi API {data['retCode']} - {data.get('retMsg', 'Unknown')}")
                continue 

        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Thử {attempt_symbol}: Lỗi request {e}")
            continue

    # --- 4. Xử lý kết quả cuối cùng ---
    if final_data is None:
        print(f"⚠️ Không thể lấy Klines cho {symbol} (Sàn: BYBIT) khung {interval}. Tất cả các lần thử đều thất bại.")
        return None
    
    # 5. Chuẩn hóa dữ liệu (Giữ nguyên)
    klines_list = final_data["result"]["list"]
    formatted_klines = []
    for k in klines_list:
        close_time_ms = 0
        if bybit_interval.isdigit():
            close_time_ms = int(k[0]) + (int(bybit_interval) * 60000) - 1 
        else:
            close_time_ms = int(k[0]) + 86400000 - 1 
        
        formatted_klines.append([
            int(k[0]), k[1], k[2], k[3], k[4], k[5], 
            close_time_ms, k[6], 0, 0, 0, "0"
        ])
    
    return formatted_klines[::-1]


def calculate_atr(highs, lows, closes, period=14):
    """Tính toán Average True Range (ATR) thủ công."""
    if len(closes) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(closes)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    if not true_ranges:
        return None
        
    atr = sum(true_ranges[-period:]) / period
    return atr

def calculate_adx(highs, lows, closes, period=14):
    """Tính toán ADX (Average Directional Index)."""
    if len(closes) < period * 2:
        return None, None, None  

    # SỬ DỤNG TA-LIB CHO ADX
    highs_arr = np.array(highs, dtype=float)
    lows_arr = np.array(lows, dtype=float)
    closes_arr = np.array(closes, dtype=float)
    
    adx_arr = talib.ADX(highs_arr, lows_arr, closes_arr, timeperiod=period)
    plus_di_arr = talib.PLUS_DI(highs_arr, lows_arr, closes_arr, timeperiod=period)
    minus_di_arr = talib.MINUS_DI(highs_arr, lows_arr, closes_arr, timeperiod=period)

    if np.isnan(adx_arr[-1]):
        # Fallback về hàm thủ công nếu TA-Lib không tính được
        return calculate_adx_manual(highs, lows, closes, period)
    
    return adx_arr[-1], plus_di_arr[-1], minus_di_arr[-1]

# Hàm ADX thủ công (giữ lại phòng khi TA-Lib lỗi)
def calculate_adx_manual(highs, lows, closes, period=14):
    if len(closes) < period * 2: return None, None, None
    plus_dm, minus_dm, tr_values = [], [], []
    for i in range(1, len(closes)):
        tr1, tr2, tr3 = highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])
        tr = max(tr1, tr2, tr3)
        tr_values.append(tr)
        up_move, down_move = highs[i] - highs[i-1], lows[i-1] - lows[i]
        plus_dm.append(up_move) if up_move > down_move and up_move > 0 else plus_dm.append(0)
        minus_dm.append(down_move) if down_move > up_move and down_move > 0 else minus_dm.append(0)
    if not tr_values: return None, None, None
    tr_sum, plus_dm_sum, minus_dm_sum = sum(tr_values[:period]), sum(plus_dm[:period]), sum(minus_dm[:period])
    def smooth(values, initial_sum):
        smoothed = [initial_sum]
        for val in values[period:]: smoothed.append((smoothed[-1] - (smoothed[-1] / period)) + val)
        return smoothed
    tr_smoothed = smooth(tr_values, tr_sum)
    plus_dm_smoothed = smooth(plus_dm, plus_dm_sum)
    minus_dm_smoothed = smooth(minus_dm, minus_dm_sum)
    plus_di = [(100 * (p / t)) if t > 0 else 0 for p, t in zip(plus_dm_smoothed, tr_smoothed)]
    minus_di = [(100 * (m / t)) if t > 0 else 0 for m, t in zip(minus_dm_smoothed, tr_smoothed)]
    if not plus_di or not minus_di: return None, None, None
    dx_values = []
    for i in range(len(plus_di)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0: dx_values.append(100 * (abs(plus_di[i] - minus_di[i]) / di_sum))
        else: dx_values.append(0)
    if len(dx_values) < period: return None, None, None
    adx_initial = sum(dx_values[:period]) / period
    adx_smoothed = smooth(dx_values[period:], adx_initial)
    return adx_smoothed[-1], plus_di[-1], minus_di[-1]


# <<< MỚI V16: HÀM XÁC ĐỊNH SR ZONE >>>
# THAY THẾ TOÀN BỘ HÀM (Dòng 885 - 925)

def find_dynamic_sr_levels(highs, lows, current_price):
    """
    FIX LỖI MẤT TP2/TP3: Tìm nhiều mức Hỗ trợ (S) và Kháng cản (R)
    bằng cách tăng phạm vi quét và giảm ngưỡng nhóm (clustering threshold).
    """
    # FIX: Tăng phạm vi tìm kiếm lên 300 nến gần nhất
    highs, lows = highs[-300:], lows[-300:]
    
    potential_levels = []
    
    # 1. Tìm tất cả các đỉnh/đáy cục bộ
    for i in range(1, len(highs) - 1):
        # Đỉnh cục bộ (Potential Resistance)
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            potential_levels.append(highs[i])
        # Đáy cục bộ (Potential Support)
        elif lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            potential_levels.append(lows[i])

    if not potential_levels:
        return [], [] # Trả về 2 list rỗng

    # --- 2. Logic Clustering (Quan trọng) ---
    potential_levels.sort()
    clusters = []
    if not potential_levels: return [], []
    
    current_cluster = [potential_levels[0]]
    
    # FIX TRIỆT ĐỂ: Giảm ngưỡng nhóm (Cluster Threshold) từ 0.5% xuống 0.2%
    cluster_threshold = current_price * 0.002 

    for level in potential_levels[1:]:
        if level - current_cluster[-1] < cluster_threshold:
            # Nếu level này gần cluster cũ, thêm vào
            current_cluster.append(level)
        else:
            # Nếu xa, chốt cluster cũ (lấy giá trị trung bình)
            clusters.append(sum(current_cluster) / len(current_cluster))
            # Bắt đầu cluster mới
            current_cluster = [level]
    
    # Chốt cluster cuối cùng
    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))
    
    # 3. Phân loại S/R và trả về
    clustered_levels = sorted(list(set(clusters)))
    
    supports = [l for l in clustered_levels if l < current_price]
    resistances = [l for l in clustered_levels if l > current_price]
    
    # Sắp xếp: Hỗ trợ (từ cao xuống thấp), Kháng cự (từ thấp lên cao)
    return sorted(supports, reverse=True), sorted(resistances)


# <<< MỚI V12: HÀM TÍNH ĐIỂM XU HƯỚNG/SỨC MẠNH (REFACTORING) - Đã FIX NoneType >>>
def calculate_trend_score(results, market_state):
    """Tính toán điểm Bullish/Bearish dựa trên các chỉ báo và trạng thái thị trường."""
    bullish_score = 0
    bearish_score = 0
    
    for name, frame_data in results.items():
        ema20 = frame_data.get("ema20")
        ema50 = frame_data.get("ema50")
        macd = frame_data.get("macd")
        macd_signal = frame_data.get("macd_signal")
        rsi = frame_data.get("rsi")
        
        # 1. Kiểm tra Crosses (Phải đảm bảo EMA20 và EMA50 Tồn tại)
        ema_cross_up = ema20 is not None and ema50 is not None and ema20 > ema50
        ema_cross_down = ema20 is not None and ema50 is not None and ema20 < ema50
        
        # 2. Kiểm tra MACD (Phải đảm bảo cả MACD và Signal Tồn tại)
        macd_cross_up = macd is not None and macd_signal is not None and macd > macd_signal
        macd_cross_down = macd is not None and macd_signal is not None and macd < macd_signal
        
        volume_support = frame_data["volume_spike"]
        
        # Logic ADX/Market State
        if market_state == "🟢TRENDING":
            if ema_cross_up: bullish_score += 1
            if ema_cross_down: bearish_score += 1
            if macd_cross_up: bullish_score += 0.5
            if macd_cross_down: bearish_score += 0.5
            
            # Cộng điểm thưởng nếu có Volume hỗ trợ
            if (ema_cross_up or macd_cross_up) and volume_support:
                bullish_score += 1 
            if (ema_cross_down or macd_cross_down) and volume_support:
                bearish_score += 1
        
        # Logic SIDEWAYS (Momentum Filter)
        elif market_state == "🟡SIDEWAYS":
            # Chỉ vào lệnh ngược xu hướng (RSI quá mua/quá bán)
            if rsi is not None:
                if rsi < 30: # Quá bán -> Tín hiệu Mua
                    bullish_score += 1.5 # Điểm cao hơn khi Sideways
                if rsi > 70: # Quá mua -> Tín hiệu Bán
                    bearish_score += 1.5

    return bullish_score, bearish_score

def decide_levels(symbol, current_timeframe=None, exchange="BINANCE"): 
    """
    HÀM ĐÃ SỬA (STEP 3): Thêm 'exchange' để gọi đúng API
    """
    intervals_map = {
        "5m": "5M", "15m": "15M", "30m": "30M", 
        "1h": "1H", "4h": "4H", "1d": "D" 
    }
    
    results = {}
    klines_data = {}
    
    h4_highs, h4_lows, h4_closes = [], [], []
    d1_highs, d1_lows, d1_closes = [], [], [] 
    
    for api_interval, name in intervals_map.items():
        
        # <<< SỬA ĐỔI 1 (STEP 3.2): GỌI HÀM KLINE THEO SÀN >>>
        kl_data = None
        if exchange == "BYBIT":
            kl_data = fetch_bybit_klines(symbol, api_interval, limit=250)
        else: # Mặc định là BINANCE
            kl_data = fetch_binance_klines(symbol, api_interval, limit=250) 
        # <<< KẾT THÚC SỬA ĐỔI 1 >>>

        if not kl_data:
            print(f"⚠️ Không thể lấy Klines cho {symbol} (Sàn: {exchange}) khung {name}. Bỏ qua...")
            continue 
        
        klines_data[name] = kl_data 
        closes = [float(candle[4]) for candle in kl_data] 
        highs = [float(candle[2]) for candle in kl_data] 
        lows = [float(candle[3]) for candle in kl_data] 
        opens = [float(candle[1]) for candle in kl_data] 
        volumes = [float(candle[5]) for candle in kl_data] 
        closes_arr = np.array(closes, dtype=float)
        highs_arr = np.array(highs, dtype=float)
        lows_arr = np.array(lows, dtype=float)
        if name == "4H": h4_highs, h4_lows, h4_closes = highs, lows, closes
        if name == "D": d1_highs, d1_lows, d1_closes = highs, lows, closes
        ema20_arr = talib.EMA(closes_arr, timeperiod=20); ema20_val = ema20_arr[-1] if not np.isnan(ema20_arr[-1]) else None
        ema50_arr = talib.EMA(closes_arr, timeperiod=50); ema50_val = ema50_arr[-1] if not np.isnan(ema50_arr[-1]) else None
        rsi_arr = talib.RSI(closes_arr, timeperiod=14); rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else None
        macd_arr, signal_arr, _ = talib.MACD(closes_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        macd_val = macd_arr[-1] if macd_arr is not None and not np.isnan(macd_arr[-1]) else None
        signal_val = signal_arr[-1] if signal_arr is not None and not np.isnan(signal_arr[-1]) else None 
        volume_spike = False
        if len(volumes) > 20:
            avg_volume_20 = sum(volumes[-21:-1]) / 20 
            current_volume = volumes[-1] 
            if avg_volume_20 > 0 and current_volume > (avg_volume_20 * 1.8): 
                volume_spike = True
        is_reversal_signal = False
        if name == "1H" and len(closes) >= 2:
            current_open, current_close = opens[-1], closes[-1]
            prev_open, prev_close = opens[-2], closes[-2]
            is_bearish_engulfing = (current_close < current_open and current_close < prev_close and current_open > prev_open and abs(current_close - current_open) > 1.5 * abs(prev_close - prev_open))
            is_bullish_engulfing = (current_close > current_open and current_close > prev_close and current_open < prev_open and abs(current_close - current_open) > 1.5 * abs(prev_close - prev_open))
            if is_bearish_engulfing or is_bullish_engulfing:
                is_reversal_signal = True
        atr_val = calculate_atr(highs, lows, closes, period=14)
        results[name] = {
            "close": closes[-1], "ema20": ema20_val, "ema50": ema50_val,
            "rsi": rsi_val, "macd": macd_val, "macd_signal": signal_val,
            "volume_spike": volume_spike, "is_reversal": is_reversal_signal,
            "atr": atr_val 
        }

    # <<< PHẦN TÍCH HỢP AI BẮT ĐẦU TỪ ĐÂY >>>
    global trading_model # Gọi 'bộ não' AI toàn cục
    
    trend = "Sideways/Không rõ"
    confidence = 50
    adx_d1, plus_di_d1, minus_di_d1 = None, None, None # Khởi tạo
    market_state = "🟢TRENDING" # Mặc định

    if trading_model is not None:
        # --- LOGIC 1: DÙNG AI (NẾU MODEL TỒN TẠI) ---
        print(f"    -> AI Model (Frame {current_timeframe}): Đang dự đoán...")
        try:
            # 1. Lấy features từ khung 1H (vì model được train 1H)
            h1_data = results.get("1H")
            if h1_data:
                # 2. Chuẩn bị features (Phải ĐÚNG THỨ TỰ như lúc train)
                # Features lúc train: ['rsi', 'macd_diff', 'ema_cross']
                
                rsi = h1_data.get('rsi')
                
                macd = h1_data.get('macd')
                macd_signal = h1_data.get('macd_signal')
                macd_diff = (macd - macd_signal) if (macd is not None and macd_signal is not None) else None
                
                ema_fast = h1_data.get('ema20')
                ema_slow = h1_data.get('ema50')
                ema_cross = (ema_fast - ema_slow) if (ema_fast is not None and ema_slow is not None) else None
                
                # 3. Kiểm tra xem có đủ features không
                if all(v is not None for v in [rsi, macd_diff, ema_cross]):
                    # 4. Tạo input cho model
                    features_input = np.array([[rsi, macd_diff, ema_cross]])
                    
                    # 5. DỰ ĐOÁN (Predict)
                    prediction = trading_model.predict(features_input)[0]
                    prediction_proba = trading_model.predict_proba(features_input)[0]
                    
                    if prediction == 1:
                        trend = "Tăng (AI)"
                        confidence = prediction_proba[ (trading_model.classes_ == 1).argmax() ] * 100 
                    elif prediction == 2:
                        trend = "Giảm (AI)"
                        confidence = prediction_proba[ (trading_model.classes_ == 2).argmax() ] * 100 
                    else: # prediction == 0 (GIỮ)
                        trend = "Sideways (AI)"
                        confidence = 50
                    
                    print(f"    -> AI Model: Dự đoán = {trend} (Conf: {confidence:.2f}%)")

                else:
                    print("    -> AI Model: Thiếu features 1H, chuyển sang Rule-based.")
                    trend = "Chuyển sang Rule-based" # Đặt cờ
            
            else:
                 print("    -> AI Model: Thiếu dữ liệu 1H, chuyển sang Rule-based.")
                 trend = "Chuyển sang Rule-based" # Đặt cờ
                 
        except Exception as e:
            print(f"    -> AI Model: Lỗi dự đoán: {e}. Chuyển sang Rule-based.")
            trend = "Chuyển sang Rule-based" # Đặt cờ
    
    # --- LOGIC 2: DÙNG RULE-BASED (NẾU AI LỖI/KHÔNG CÓ/HOẶC NÓI SIDEWAYS) ---
    if trading_model is None or "Rule-based" in trend or "Sideways (AI)" in trend:
        if trading_model is not None: # Nếu AI nói Sideways, thì dùng Rule-based
             print("    -> AI nói Sideways. Chuyển sang Rule-based để kiểm tra...")
             
        if d1_highs and d1_lows and d1_closes: 
            adx_d1, plus_di_d1, minus_di_d1 = calculate_adx(d1_highs, d1_lows, d1_closes)
            if adx_d1 is not None and adx_d1 > 25: 
                 market_state = "🟢TRENDING" 
        
        bullish_score, bearish_score = calculate_trend_score(results, market_state)
        score_diff = abs(bullish_score - bearish_score)
        total_score = bullish_score + bearish_score
        
        if total_score == 0: trend = "Không rõ (Dữ liệu yếu)"; confidence = 50
        elif score_diff <= 0.5: trend = "Sideways/Rủi ro cao"; confidence = 50 
        elif bullish_score >= 2.0 and bullish_score > bearish_score: trend = "Tăng (Rule-based)"; confidence = min(95, 50 + int(score_diff * 10)) 
        elif bearish_score >= 2.0 and bearish_score > bullish_score: trend = "Giảm (Rule-based)"; confidence = min(95, 50 + int(score_diff * 10)) 
        else: trend = "Sideways/Yếu"; confidence = max(50, int(total_score * 10))
    
    if "D" in results: results["D"]["adx"] = adx_d1
    else: results["D"] = {"adx": adx_d1}
    
    # <<< KẾT THÚC PHẦN TÍCH HỢP AI >>>


    # --- PHẦN LẤY GIÁ (GIỮ NGUYÊN) ---
    current_market_price = None
    if exchange == "BYBIT":
        current_market_price = get_bybit_market_price(symbol)
    else: # Mặc định là BINANCE
        current_market_price = get_market_price(symbol)
    
    if current_market_price is None:
        print(f"❌ Không thể lấy giá thị trường cho {symbol} (Sàn: {exchange}).")
        return None

    best_tf_name = None
    if current_timeframe and current_timeframe in results and klines_data.get(current_timeframe):
        best_tf_name = current_timeframe
    if not best_tf_name:
         best_confidence = 0
         for tf_name in ["5M", "15M", "30M", "1H", "4H"]:
             conf_tf = results.get(tf_name, {}).get("confidence", 0) 
             if conf_tf > best_confidence and klines_data.get(tf_name):
                  best_confidence = conf_tf
                  best_tf_name = tf_name
    if not best_tf_name: 
        if not klines_data: return None 
        best_tf_name = "1H" 
    kl_data_sr = klines_data.get(best_tf_name)
    if not kl_data_sr: 
        print(f"Lỗi: Không tìm thấy Klines cho timeframe {best_tf_name} (best_tf_name).")
        return None 
    highs_sr = [float(k[2]) for k in kl_data_sr]
    lows_sr = [float(k[3]) for k in kl_data_sr]
    closes_sr = [float(k[4]) for k in kl_data_sr]
    atr_value = results.get(best_tf_name, {}).get("atr")
    if atr_value is None: 
         atr_value = results.get("4H", {}).get("atr", current_market_price * 0.015) 
    if atr_value is None:
         atr_value = current_market_price * 0.015
    if best_tf_name == "5M": sl_multiplier = 1.8  # SL chặt hơn cho 5M
    elif best_tf_name == "15M": sl_multiplier = 2.0 # SL xa hơn cho 15M
    elif best_tf_name == "30M": sl_multiplier = 2.2
    elif best_tf_name == "1H": sl_multiplier = 2.5
    else: sl_multiplier = 2.0

    entry = current_market_price
    tp1, tp2, tp3, sl = None, None, None, None
    limit = False
    stoploss_distance = None 
    
    MIN_RR_RATIO = 0.5 
    atr_noise = sl_multiplier * atr_value 
    SL_MAX_LIMIT = 5.0 * atr_value 
    
    # <<< SỬA ĐỔI 3: LẤY PRECISION THEO SÀN >>>
    # LƯU Ý: Hiện tại bot chỉ có precision của Binance.
    # Để Bybit chạy đúng (auto-trade), cần bổ sung hàm get_bybit_precision
    precision = get_symbol_precision(symbol) # Tạm thời vẫn dùng Binance
    tick_size = precision['tickSize'] 
    round_precision = int(-math.log10(tick_size)) if tick_size > 0 else 8
    MIN_PRICE_DIFF = tick_size * 2.0 
    # <<< KẾT THÚC SỬA ĐỔI 3 >>>

    if "Tăng" in trend: # LONG
        sl_atr_base = entry - atr_noise 
        sl = round_by_step(max(entry - SL_MAX_LIMIT, sl_atr_base), tick_size) 
        stoploss_distance = entry - sl
        
    elif "Giảm" in trend: # SHORT
        sl_atr_base = entry + atr_noise
        sl = round_by_step(min(entry + SL_MAX_LIMIT, sl_atr_base), tick_size) 
        stoploss_distance = sl - entry
        
    if stoploss_distance and stoploss_distance > MIN_PRICE_DIFF:
        supports, resistances = find_dynamic_sr_levels(highs_sr, lows_sr, current_market_price)
        
        if "Tăng" in trend: # LONG
            if not resistances: 
                print(f"❌ FAIL TỶ LỆ RR (LONG): Không tìm thấy mức Kháng cự nào cho {symbol}.")
                trend = "Sideways/Không có TP"
            else:
                tp1 = round_by_step(resistances[0], tick_size)
                tp2 = round_by_step(resistances[1], tick_size) if len(resistances) > 1 else None
                tp3 = round_by_step(resistances[2], tick_size) if len(resistances) > 2 else None
                
                rr_tp1 = (tp1 - entry) / stoploss_distance
                if rr_tp1 < MIN_RR_RATIO:
                    print(f"❌ FAIL TỶ LỆ RR (LONG): R:R quá thấp (1:{rr_tp1:.1f}). Cần ít nhất 1:{MIN_RR_RATIO}. Bỏ qua.")
                    trend = "Sideways/R:R thấp" 

        elif "Giảm" in trend: # SHORT
            if not supports: 
                print(f"❌ FAIL TỶ LỆ RR (SHORT): Không tìm thấy mức Hỗ trợ nào cho {symbol}.")
                trend = "Sideways/Không có TP"
            else:
                tp1 = round_by_step(supports[0], tick_size)
                tp2 = round_by_step(supports[1], tick_size) if len(supports) > 1 else None
                tp3 = round_by_step(supports[2], tick_size) if len(supports) > 2 else None
                
                rr_tp1 = (entry - tp1) / stoploss_distance
                if rr_tp1 < MIN_RR_RATIO:
                    print(f"❌ FAIL TỶ LỆ RR (SHORT): R:R quá thấp (1:{rr_tp1:.1f}). Cần ít nhất 1:{MIN_RR_RATIO}. Bỏ qua.")
                    trend = "Sideways/R:R thấp" 

    has_reversal_signal = results.get("1H", {}).get("is_reversal", False)
    
    return {
        "trend": trend, "entry": entry, "tps": [tp1, tp2, tp3], "sl": sl,
        "limit": limit, "leverage": None, 
        "frames": results, "confidence": confidence,
        "atr_debug": atr_value, 
        "market_state_debug": market_state,
        "sl_distance": stoploss_distance,
        "is_reversal_priority": has_reversal_signal 
    }
        

def generate_and_send_candlestick_chart(chat_id, symbol, res):
    try:
        kl_data = fetch_binance_klines(symbol, "1h", limit=250) 
        if not kl_data:
            bot.send_message(chat_id, "Không thể tải dữ liệu nến để vẽ chart.")
            return

        df = pd.DataFrame(kl_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df[['open', 'high', 'low', 'close']].astype(float)

        # Sử dụng TA-Lib cho chart
        df['ema20'] = talib.EMA(df['close'].values, timeperiod=20)
        df['ema50'] = talib.EMA(df['close'].values, timeperiod=50)
        df['ema200'] = talib.EMA(df['close'].values, timeperiod=200)

        apds = [
            mpf.make_addplot(df['ema20'], color='yellow', width=0.8, panel=0),
            mpf.make_addplot(df['ema50'], color='blue', width=0.8, panel=0),
            mpf.make_addplot(df['ema200'], color='white', width=0.8, panel=0)
        ]

        entry_price = res['entry']
        sl_price = res['sl']
        tp_prices = res['tps']

        hlines = []
        hlines_colors = []
        hlines_labels = []
        hlines_style = []
        
        # Xác định độ chính xác làm tròn dựa trên entry price
        round_precision = 6
        if entry_price > 1000: round_precision = 2
        elif entry_price > 10: round_precision = 4

        if entry_price:
            hlines.append(entry_price)
            hlines_colors.append('yellow')
            hlines_labels.append(f'Entry: {entry_price:,.{round_precision}f}')
            hlines_style.append('--')
        if sl_price:
            hlines.append(sl_price)
            hlines_colors.append('red')
            hlines_labels.append(f'SL: {sl_price:,.{round_precision}f}')
            hlines_style.append('--')
        for i, tp in enumerate(tp_prices):
            if tp:
                hlines.append(tp)
                hlines_colors.append('lime')
                hlines_labels.append(f'TP{i+1}: {tp:,.{round_precision}f}')
                hlines_style.append(':')
        
        s = mpf.make_mpf_style(
            base_mpf_style='yahoo', 
            marketcolors=mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='in', ohlc='black'),
            figcolor='#1a1a1a', 
            facecolor='#1a1a1a', 
            gridcolor='dimgray',
            gridstyle=':',
            rc={'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'text.color': 'white'},
            y_on_right=False
        )
        
        title_text = f"Phân Tích {symbol} (1H) - Hành Động: {res['trend']}"

        fig, axes = mpf.plot(
            df,
            type='candle',
            style=s,
            title=title_text,
            ylabel='Giá',
            addplot=apds,
            hlines=dict(hlines=hlines, colors=hlines_colors, linewidths=1.2, linestyle=hlines_style, alpha=0.7),
            figscale=1.5, 
            returnfig=True
        )

        ax = axes[0] 
        if len(hlines) > 0:
            transform = ax.get_yaxis_transform()
            for i, hline_val in enumerate(hlines):
                label = hlines_labels[i]
                color = hlines_colors[i]
                # Sử dụng Annotate để hiển thị labels
                ax.annotate(label, xy=(0.98, hline_val), xycoords=transform, 
                            xytext=(5, 0), textcoords='offset points', 
                            color='black', fontsize=9, fontweight='bold',
                            ha='left', va='center', 
                            bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='none', alpha=0.9)) 

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(fig) 
        
        bot.send_photo(chat_id, buf, caption=f"📷 Chart Nến (1H) - {symbol}")
        buf.close()

    except Exception as e:
        print(f"Lỗi generate_candlestick_chart: {e}")
        bot.send_message(chat_id, f"⚠️ Lỗi khi vẽ chart nến: {e}. Vui lòng kiểm tra lại dữ liệu và thư viện.")

def analyze_and_send(chat_id, symbol, precomputed_res=None, timeframe_origin="Manual Scan"):
    """
    HÀM ĐÃ SỬA (FIX V18.7): 
     1. (V18.6) Gợi ý Đòn bẩy Động.
     2. (MỚI) Di chuyển dòng "Đòn bẩy Gợi ý" xuống dưới SL.
    """
    try:
        user = ensure_user_data_structure(chat_id)
        
        exchange = user.get("trading", {}).get("exchange", "BINANCE")
        
        res = precomputed_res
        is_manual_scan = (precomputed_res is None) 
        
        if is_manual_scan:
            bot.send_message(chat_id, f"🔎 Đang phân tích **{symbol}** (Sàn: {exchange})... Vui lòng chờ (10-20s)", parse_mode="Markdown")
            res = decide_levels(symbol, exchange=exchange) 
            timeframe_origin = "Multi-TF Analysis" 
        
        if not res:
            bot.send_message(chat_id, f"⚠️ Lỗi: Không thể lấy dữ liệu Klines/Giá cho {symbol} từ {exchange}. Vui lòng thử lại sau.")
            return

        current_time_vn_dt = datetime.now(TZ)
        current_time_vn = current_time_vn_dt.strftime("%H:%M:%S %d/%m/%Y")
        
        risk_settings = user["trading"]
        total_capital = risk_settings.get("total_capital", 1000.0)
        risk_per_trade = risk_settings.get("risk_per_trade", 1.0)
        leverage = risk_settings.get("leverage", 5.0) 
        risk_amount_usd = total_capital * (risk_per_trade / 100.0)
        entry = res['entry']
        sl = res['sl']
        stoploss_distance = res.get('sl_distance')
        
        # <<< FIX V18.6: TÍNH TOÁN ĐÒN BẨY ĐỘNG (DYNAMIC LEVERAGE) >>>
        
        # 1. Tính Đòn bẩy TỐI ĐA (An toàn 1R) (Như V18.5)
        max_safe_leverage = 0.0
        leverage_warning = ""
        if stoploss_distance and entry > 0:
            sl_percent_move = (stoploss_distance / entry) * 100.0
            if sl_percent_move > 0:
                max_safe_leverage = (100.0 / sl_percent_move) * 0.95 
                
                user_leverage = risk_settings.get("leverage", 5.0) 
                if user_leverage > max_safe_leverage:
                    liq_percent = (100.0 / user_leverage) * 0.9
                    leverage_warning = f" (⚠️ Đòn bẩy {int(user_leverage)}x của bạn quá cao!)"
                    if liq_percent < sl_percent_move:
                         leverage_warning += f"\n\n🔥 <b>CẢNH BÁO CHÁY: Bạn sẽ bị ký quỹ tại ~{liq_percent:.2f}% TRƯỚC KHI chạm SL (1R) tại {sl_percent_move:.2f}%!</b>"

        # 2. Tính Đòn bẩy GỢI Ý (Dựa trên chất lượng)
        is_reversal_signal = res.get("is_reversal_priority", False)
        confidence = res.get("confidence", 50)
        market_state = res.get("market_state_debug", "TRENDING")

        suggested_leverage_dynamic = 10 # Bẩy cơ sở (X10)
        
        if market_state == "SIDEWAYS":
            suggested_leverage_dynamic = 5 # Sideways (ngược xu hướng) -> X5
        elif is_reversal_signal:
            suggested_leverage_dynamic = 25 # Đảo chiều mạnh -> X25
        elif confidence > 85:
            suggested_leverage_dynamic = 20 # Rất tin cậy -> X20
        elif confidence > 75:
            suggested_leverage_dynamic = 15 # Tin cậy -> X15
            
        # 3. Áp dụng Giới hạn (Cap)
        if max_safe_leverage > 0:
            suggested_leverage_final = math.floor(min(suggested_leverage_dynamic, max_safe_leverage))
        else:
            suggested_leverage_final = suggested_leverage_dynamic
            
        suggested_leverage_final = max(1, min(suggested_leverage_final, 125))
        max_safe_leverage_rounded = math.floor(min(max_safe_leverage, 125.0)) if max_safe_leverage > 0 else "N/A"
        # <<< KẾT THÚC FIX V18.6 >>>
        
        position_size = 0.0
        position_value_usd = 0.0
        required_margin = 0.0 
        
        precision = get_symbol_precision(symbol) 
        tick_size = precision['tickSize'] 
        round_precision = int(-math.log10(tick_size)) if tick_size > 0 else 8
        
        tp1_rr, tp2_rr, tp3_rr = 0.0, 0.0, 0.0
        
        sl_price_val = res.get('sl')
        tp1_price_val = res.get('tps', [None])[0]
        tp2_price_val = res.get('tps', [None, None])[1]
        tp3_price_val = res.get('tps', [None, None, None])[2]
        
        risk_msg = "\n--🛡️ <b>Quản lý Rủi Ro (Chỉ tính cho Tín hiệu Long/Short)</b>" 
        
        if stoploss_distance and stoploss_distance > 0.000001 and sl_price_val is not None: 
             position_size_raw = risk_amount_usd / stoploss_distance
             current_atr_value = res.get('atr_debug', entry * 0.01) 
             ATR_BENCHMARK_PERCENT = 0.5 
             
             atr_percent_of_entry = 0.01 
             if entry > 0:
                 atr_percent_of_entry = (current_atr_value / entry) * 100.0 
             
             atr_factor = 1.0
             if atr_percent_of_entry > 0:
                 atr_factor = min(2.0, max(0.5, ATR_BENCHMARK_PERCENT / atr_percent_of_entry))
                 
             position_size = round_by_step(position_size_raw * atr_factor, precision['stepSize'])
             position_value_usd = round(position_size * entry, 2)
             if leverage > 0: required_margin = round(position_value_usd / leverage, 2)
             
             if tp1_price_val: tp1_rr = round(abs(tp1_price_val - entry) / stoploss_distance, 1)
             if tp2_price_val: tp2_rr = round(abs(tp2_price_val - entry) / stoploss_distance, 1)
             if tp3_price_val: tp3_rr = round(abs(tp3_price_val - entry) / stoploss_distance, 1)

             risk_msg = (f"\n--🛡️ <b>Quản lý Rủi Ro (Vốn: {total_capital:,.0f} USD | Rủi ro: {risk_per_trade:.1f}%)</b>\n"
                         f" • Rủi ro tối đa/lệnh: <b>{risk_amount_usd:,.2f} USD</b> (1R)\n"
                         f" • Khoảng cách SL: <b>{stoploss_distance:,.{round_precision}f}</b>\n"
                         f" • R:R Ratio (1R): <b>1:{tp1_rr} | 1:{tp2_rr} | 1:{tp3_rr}</b>\n" 
                         f" • Đòn bẩy Khuyến nghị (API): <b>{leverage:.1f}x</b>\n" 
                         f" • Khối lượng Khuyến nghị: <b>{position_size:,.2f} {symbol[:-4]}</b>\n"
                         f" • Giá trị Lệnh (Size * Entry): <b>{position_value_usd:,.2f} USD</b>\n"
                         f" • Ký quỹ cần thiết: <b>{required_margin:,.2f} USD</b>")
        
        trade_status_msg = ""
        order_type = "LIMIT" if res['limit'] else "MARKET"
        should_record_signal = False 
        sl_order_id_to_record = None
        tp_order_id_to_record = None
        # is_reversal_signal (Đã lấy ở trên)
        new_trend = res['trend']
        
        if not is_manual_scan and ("Tăng" in new_trend or "Giảm" in new_trend):
            api_key = user["trading"].get("api_key")
            secret_key = user["trading"].get("secret_key")

            if not api_key or not secret_key:
                trade_status_msg = "\n(Lưu ý: Bạn chưa cài API Key, bot chỉ thông báo)"
            else:
                MIN_NOTIONAL = 5.0 
                if position_value_usd < MIN_NOTIONAL:
                    trade_status_msg = f"\n❌ <b>THỰC THI API BỎ QUA:</b> Giá trị lệnh quá nhỏ ({position_value_usd:,.2f} USD). Cần tối thiểu {MIN_NOTIONAL} USD."
                else:
                    user_signals = user["trading"].get("signals", []) 
                    open_signals = [s for s in user_signals if s['symbol'] == symbol and s['status'] == 'open']
                    
                    has_open_position = bool(open_signals)
                    
                    if has_open_position:
                        if not is_reversal_signal:
                            print(f"    [LOGIC FIX]: {symbol} đã có lệnh mở. Bỏ qua tín hiệu cùng/yếu.")
                            return 
                        current_sig = open_signals[0]
                        current_trend = "Tăng" if current_sig['entry'] < current_sig['tp1'] else "Giảm"
                        
                        if is_reversal_signal and new_trend != current_trend:
                            trade_status_msg = f"\n⚠️ <b>TÍN HIỆU ĐẢO CHIỀU MẠNH:</b> Đóng lệnh cũ ({current_trend}) để vào lệnh mới ({new_trend})."
                            if user["trading"].get("auto_exit_on_reversal", True):
                                cmd_exit_manual(type('obj', (object,), {'text': f'/exit {symbol}', 'chat_id': int(chat_id)}))
                                trade_status_msg += "\n✅ Đã tự động đóng vị thế cũ qua API."
                            else:
                                trade_status_msg += f"\n🔔 API ĐÓNG LỆNH ĐẢO CHIỀU ĐANG TẮT. Lệnh cũ KHÔNG ĐƯỢC ĐÓNG tự động."
                        else:
                            trade_status_msg = "\nℹ️ <b>THỰC THI API BỎ QUA:</b> Đã có vị thế đang mở cho coin này."
                            return
                    
                    if exchange == "BYBIT":
                        trade_status_msg = f"\n⚠️ <b>THỰC THI API BỎ QUA:</b> Auto-Trading (thực thi lệnh) cho {exchange} chưa được hỗ trợ trong phiên bản này."
                    else:
                        success, message_api, sl_id, tp_id = execute_trade_testnet(
                            symbol, res['trend'], entry, sl, res['tps'], position_size, order_type, 
                            user
                        )
                        
                        if success:
                            trade_status_msg += f"\n✅ <b>THỰC THI API:</b> Lệnh đã được gửi thành công!\n  -> {message_api}"
                            should_record_signal = True 
                            sl_order_id_to_record = sl_id 
                            tp_order_id_to_record = tp_id
                        else:
                            trade_status_msg += f"\n❌ <b>THỰC THI API:</b> Lỗi khi gửi lệnh!\n  -> {message_api}"
                            print(f"Tín hiệu {symbol} bị lỗi thực thi API cho {chat_id}. LỖI API: {message_api}.")

        # --- PHẦN TẠO TIN NHẮN TÓM TẮT & CHI TIẾT ---
        frame_h1 = res['frames'].get("1H", {})
        frame_h4 = res['frames'].get("4H", {})
        
        # (FIX V18.1)
        ema20_h1 = frame_h1.get('ema20')
        ema50_h1 = frame_h1.get('ema50')
        rsi_h1 = frame_h1.get('rsi')
        macd_h1 = frame_h1.get('macd')
        ema20_h1_str = f"{ema20_h1:,.{round_precision}f}" if ema20_h1 is not None else "N/A"
        ema50_h1_str = f"{ema50_h1:,.{round_precision}f}" if ema50_h1 is not None else "N/A"
        rsi_h1_str = f"{rsi_h1:.2f}" if rsi_h1 is not None else "N/A"
        macd_h1_str = f"{macd_h1:.2f}" if macd_h1 is not None else "N/A"
        ema20_h4 = frame_h4.get('ema20')
        ema50_h4 = frame_h4.get('ema50')
        rsi_h4 = frame_h4.get('rsi')
        macd_h4 = frame_h4.get('macd')
        ema20_h4_str = f"{ema20_h4:,.{round_precision}f}" if ema20_h4 is not None else "N/A"
        ema50_h4_str = f"{ema50_h4:,.{round_precision}f}" if ema50_h4 is not None else "N/A"
        rsi_h4_str = f"{rsi_h4:.2f}" if rsi_h4 is not None else "N/A"
        macd_h4_str = f"{macd_h4:.2f}" if macd_h4 is not None else "N/A"
        atr_val = res.get('atr_debug')
        atr_str = f"{atr_val:.4f}" if atr_val is not None else "N/A"

        
        detail_msg_content = f"<b>CHI TIẾT PHÂN TÍCH: {symbol} (Sàn: {exchange})</b>\n" 
        detail_msg_content += f" • Nguồn: {timeframe_origin} | Thời gian: {current_time_vn}\n"
        detail_msg_content += f" • Nhận định xu hướng: <b>{new_trend}</b>\n"
        detail_msg_content += f" • Giá hiện tại: <b>{entry:,.{round_precision}f}</b>\n"
        detail_msg_content += "---------------------------------\n"
        detail_msg_content += f"<b>Khung 1H:</b>\n"
        detail_msg_content += f" • EMA20: {ema20_h1_str} | EMA50: {ema50_h1_str}\n"
        detail_msg_content += f" • RSI: {rsi_h1_str} | MACD: {macd_h1_str}\n"
        detail_msg_content += f"<b>Khung 4H:</b>\n"
        detail_msg_content += f" • EMA20: {ema20_h4_str} | EMA50: {ema50_h4_str}\n"
        detail_msg_content += f" • RSI: {rsi_h4_str} | MACD: {macd_h4_str}\n"
        detail_msg_content += "---------------------------------\n"
        detail_msg_content += f" • Market State: {res.get('market_state_debug', 'N/A')}\n"
        detail_msg_content += f" • Confidence: {res.get('confidence', 0):.1f}%\n"
        detail_msg_content += f" • ATR (Debug): {atr_str}\n"
        
        if "Tăng" in new_trend or "Giảm" in new_trend:
             detail_msg_content += risk_msg 
        
        
        # --- LOGIC GỬI TIN NHẮN (ĐÃ SỬA ĐỔI) ---
        
        if is_manual_scan:
            # ------- QUÉT THỦ CÔNG (Lệnh /btc) -------
            
            try:
                if exchange == "BINANCE": 
                    pass
                else:
                    bot.send_message(chat_id, f"ℹ️ (Vẽ chart nến tự động hiện chỉ hỗ trợ Binance. Sàn của bạn là {exchange}.)")
            except Exception as e:
                print(f"Lỗi gửi chart (manual scan) cho {chat_id}: {e}")
                bot.send_message(chat_id, f"⚠️ Lỗi khi vẽ chart nến: {e}")
            
            try:
                bot.send_message(chat_id, detail_msg_content, parse_mode="HTML")
            except Exception as e:
                print(f"Lỗi gửi chi tiết (manual scan) cho {chat_id}: {e}")
        
        else:
            # ------- AUTO-TRADING (Tự động) -------
            
            if not ("Tăng" in new_trend or "Giảm" in new_trend):
                print(f"    [SKIP AUTO]: {symbol} ({timeframe_origin}) - Trend is Sideways. Không gửi thông báo.")
                return

            trend_emoji = "🟢" if "Tăng" in new_trend else "🔴"
            trend_text_action = "LONG COIN" if "Tăng" in new_trend else "SHORT COIN"
            order_type_text = "MARKET" if order_type == "MARKET" else f"LIMIT ({order_type})"
            
            tp1_text = f"✅ TP1 (R:R 1:{tp1_rr}): {tp1_price_val:,.{round_precision}f}\n" if tp1_price_val else ""
            tp2_text = f"✅ TP2 (R:R 1:{tp2_rr}): {tp2_price_val:,.{round_precision}f}\n" if tp2_price_val else ""
            tp3_text = f"✅ TP3 (R:R 1:{tp3_rr}): {tp3_price_val:,.{round_precision}f}\n\n" if tp3_price_val else "\n"
            
            # (FIX V18.2)
            timeframe_origin_display = timeframe_origin
            if is_reversal_signal: 
                timeframe_origin_display = f"⚠️ {timeframe_origin} ( Đảo Chiều Mạnh ! )"

            # (FIX V18.1)
            sl_text = f"❌ SL (1R): {sl_price_val:,.{round_precision}f}\n\n" if sl_price_val is not None else "❌ SL (1R): N/A (Lỗi tính toán SL)\n\n"

            # (FIX V18.6)
            leverage_text = f"📈 Đòn bẩy Gợi ý: <b>~{suggested_leverage_final}x</b> (Tối đa An toàn: {max_safe_leverage_rounded}x){leverage_warning}\n"

            # <<< FIX V18.7: THAY ĐỔI THỨ TỰ TIN NHẮN >>>
            msg_summary = (f"🤖 BOT AI Nghèo Phố Wall (Sàn: {exchange})\n\n" 
                           f"⏱️ QUÉT KHUNG : {timeframe_origin_display}\n" 
                           f"{trend_emoji} {trend_text_action}: {symbol}\n"
                           f"Hành động : {trend_emoji}\n\n"
                           f"🇻🇳 {order_type_text} vào lệnh: {entry:,.{round_precision}f}\n\n"
                           f"{tp1_text}"
                           f"{tp2_text}"
                           f"{tp3_text}"
                           f"{sl_text}" 
                           f"{leverage_text}" # <-- ĐÃ DI CHUYỂN XUỐNG DƯỚI SL
                           f"\n⏰ Thời gian báo lệnh: {current_time_vn}")
            # <<< KẾT THÚC FIX V18.7 >>>
            
            if trade_status_msg:
                msg_summary += f"\n\n---------------------------------\n{trade_status_msg}"
            
            temp_id = str(uuid.uuid4())[:6]
            user["temp_detail_message"] = {'id': temp_id, 'content': detail_msg_content}
            markup = InlineKeyboardMarkup()
            markup.add(InlineKeyboardButton("📈 Xem Chi tiết (Nhận định)", callback_data=f"show_details:{temp_id}"))

            try:
                bot.send_message(chat_id, msg_summary, reply_markup=markup, parse_mode="HTML")
            except Exception as e:
                print(f"Lỗi gửi tóm tắt (auto) cho {chat_id}: {e}")
            
            if should_record_signal:
                try:
                    record_signal(chat_id, symbol, entry, tp1_price_val, tp2_price_val, tp3_price_val, sl_price_val, 
                                  order_type, res.get('confidence', 0), position_size, 
                                  sl_order_id_to_record, tp_order_id_to_record)
                except Exception as e:
                    print(f"Lỗi nghiêm trọng khi record_signal: {e}")
                    bot.send_message(chat_id, f"⚠️ Lỗi lưu tín hiệu vào database: {e}")

    except Exception as e:
        print(f"Lỗi phân tích tổng thể: {e}") 
        if 'chat not found' in str(e) or 'user is deactivated' in str(e) or 'Forbidden: bot was kicked' in str(e):
             if str(chat_id) in user_data:
                 print(f"❌ Xử lý lỗi cuối cùng: Xóa User {chat_id}")
                 del user_data[str(chat_id)]
                 save_user_data()
        elif "message text is empty" not in str(e): 
             try:
                 bot.send_message(chat_id, f"⚠️ Lỗi phân tích tổng thể: {e}")
             except:
                 pass

# <<< KẾT THÚC THAY THẾ analyze_and_send >>>
# ================================================
# CÁC HÀM XỬ LÝ WATCHLIST (Giữ nguyên)
# ================================================
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE): return {}
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_watchlist(w):
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(w, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi save_watchlist: {e}")

def normalize_and_split_coins(text):
    if not text: return []
    for sep in [";", "|", "/", "\n", "\t"]: text = text.replace(sep, ",")
    parts = []
    for part in text.split(","):
        p = part.strip().upper()
        if not p: continue
        if not p.endswith("USDT"): p += "USDT"
        parts.append(p)
    seen = set()
    out = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
# THAY THẾ TOÀN BỘ HÀM (TỪ DÒNG 1753 đến 1808)
def exchange_has_symbol(symbol, exchange="BINANCE"):
    """
    FIX V18.4: Kiểm tra xem symbol có tồn tại trên SÀN ĐÃ CHỌN (Binance hoặc Bybit).
    """
    global _exchange_info_cache
    
    if exchange == "BINANCE":
        # 1. Đảm bảo cache Binance được tải
        if _exchange_info_cache is None:
            print("Đang tải cache ExchangeInfo (Binance)...")
            get_exchange_info() # Hàm này tải cache Binance Futures
        
        if symbol in _exchange_info_cache:
            return True
        
        # 3. FIX: Nếu không tìm thấy, tải lại cache 1 lần
        print(f"⚠️ Thử tải lại ExchangeInfo Futures (Binance). Symbol {symbol} chưa có trong cache.")
        _exchange_info_cache = None 
        futures_symbol_cache = get_exchange_info() 
        return symbol in futures_symbol_cache

    elif exchange == "BYBIT":
        # 2. Kiểm tra Bybit (Không dùng cache, gọi trực tiếp)
        print(f"Đang kiểm tra {symbol} trên Bybit...")
        try:
            # Thử lấy giá (cách nhanh nhất để biết coin tồn tại)
            price = get_bybit_market_price(symbol)
            if price is not None:
                return True
            else:
                return False
        except Exception as e:
            print(f"Lỗi kiểm tra Bybit: {e}")
            return False
            
    else:
        # Sàn khác (OKX, BingX...) tạm thời chấp nhận
        return True

def handle_addcoin_input(chat_id, text):
    
    # (FIX V18.4) Lấy sàn hiện tại của user để kiểm tra
    user = ensure_user_data_structure(chat_id)
    current_exchange = user["trading"].get("exchange", "BINANCE")
    
    watch = load_watchlist()
    user_list = watch.get(str(chat_id), [])
    requested = normalize_and_split_coins(text)
    added, already, invalid = [], [], []
    
    for sym in requested:
        # --- (FIX V18.4) GỌI HÀM KIỂM TRA ĐÚNG SÀN ---
        if not exchange_has_symbol(sym, current_exchange):
            invalid.append(sym)
            continue
        # ----------------------------------------
        
        if sym in user_list:
            already.append(sym)
        else:
            user_list.append(sym)
            added.append(sym)
            
    watch[str(chat_id)] = user_list
    save_watchlist(watch) 
    
    try:
        user["trading"]["watchlist"] = user_list 
    except Exception as e:
        print(f"Lỗi đồng bộ watchlist vào user_data: {e}")

    parts = []
    if added: parts.append(f"✅ Đã thêm (Sàn {current_exchange}): " + ", ".join(added))
    if already: parts.append("ℹ️ Đã có sẵn: " + ", ".join(already))
    
    if invalid: parts.append(f"⚠️ Không hợp lệ (Sàn {current_exchange}): " + ", ".join(invalid))
    
    if not parts: return "⚠️ Không tìm thấy symbol hợp lệ."
    return "\n".join(parts)

# =======================
# 7. BOT HANDLERS (MENU & LỆNH)
# =======================
def answer_ok(call, text="✅ Đã cập nhật"):
    """Thực hiện Throttling cho nút bấm và trả lời query."""
    chat_id = str(call.message.chat.id)
    
    # --- ÁP DỤNG THROTTLE CHO CALLBACK ---
    if not check_throttle(chat_id):
        # Nếu đang bị chặn, chỉ cần gửi thông báo im lặng
        try:
            bot.answer_callback_query(call.id, "⚠️ Chậm lại chút nhé!", show_alert=False)
        except: pass
        return False # Báo hiệu không xử lý
    
    try:
        # Nếu không bị chặn, trả lời bình thường
        bot.answer_callback_query(call.id, text, show_alert=False)
    except: pass
    return True 

def save_user_on_start(message):
    """Lưu chat_id và username khi người dùng /start lần đầu"""
    try:
        chat_id = message.chat.id
        username = message.from_user.username or message.from_user.first_name
        users = {}
        if os.path.exists(USER_LIST_FILE):
            try:
                with open(USER_LIST_FILE, 'r', encoding='utf-8') as f:
                    users = json.load(f)
            except json.JSONDecodeError:
                users = {} 
        
        if str(chat_id) not in users:
            users[str(chat_id)] = {"username": username, "first_seen": datetime.now(TZ).isoformat()}
            with open(USER_LIST_FILE, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2, ensure_ascii=False)
            print(f"Người dùng mới: {username} (ID: {chat_id})")
    except Exception as e:
        print(f"Lỗi save_user_on_start: {e}")

def send_main_keyboard(chat_id, text="Bot đã sẵn sàng! Bấm 'MENU 🎛️' hoặc 'Auto-Trading ⏱️' ở dưới để xem các tùy chọn."):
    """Gửi menu bàn phím (ở dưới ô chat)"""
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False, row_width=2)
    markup.add(KeyboardButton("MENU 🎛️"), KeyboardButton("Auto-Trading ⏱️")) # Nút ở dưới
    
    if user_data.get(str(chat_id), {}).get("monitor_started", False) == False: 
         bot.send_message(chat_id, text, reply_markup=markup)

# --- Menu Chính (V8/V9 UI) ---
def build_main_menu_markup():
    m = InlineKeyboardMarkup(row_width=1)
    m.add(
        InlineKeyboardButton("💹 Trading", callback_data="menu_trading"),
        InlineKeyboardButton("🎲 Tài Xỉu", callback_data="menu_taixiu"),
        InlineKeyboardButton("📘 Hướng dẫn sử dụng", callback_data="menu_guide")
    )
    return m

@bot.message_handler(commands=["start", "menu"])
def cmd_show_main(message):
    chat_id = message.chat.id
    
    save_user_on_start(message)
    user = ensure_user_data_structure(chat_id) 
    
    user["monitor_started"] = True 
        
    bot.send_message(chat_id, "⬇️​ MENU CHÍNH:", reply_markup=build_main_menu_markup())
    send_main_keyboard(chat_id) 

@bot.callback_query_handler(func=lambda c: c.data == "back_main")
def cb_back_main(call):
    chat_id = call.message.chat.id
    try:
        bot.edit_message_text("⬇️​ MENU CHÍNH:", chat_id, call.message.message_id, reply_markup=build_main_menu_markup())
    except Exception:
        bot.send_message(chat_id, "⬇️​ MENU CHÍNH:", reply_markup=build_main_menu_markup())
    answer_ok(call, "⬅️ Về Menu Chính")

# --- Menu Trading (V8/V9 UI) ---
@bot.callback_query_handler(func=lambda c: c.data == "menu_trading")
def cb_menu_trading_select_exchange(call):
    chat_id = call.message.chat.id
    markup = InlineKeyboardMarkup(row_width=2)
    buttons = [InlineKeyboardButton(ex, callback_data=f"set_exchange:{ex}") for ex in PREFERRED_EXCHANGES]
    markup.add(*buttons)
    markup.add(InlineKeyboardButton("⬅️ Quay lại", callback_data="back_main"))
    text = "Sàn của bạn là gì? (Chart sẽ được chụp từ sàn này)"
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "Chọn sàn giao dịch")

@bot.callback_query_handler(func=lambda c: c.data.startswith("set_exchange:"))
def cb_set_exchange_and_show_menu(call):
    chat_id = call.message.chat.id
    exchange = call.data.split(":", 1)[1]
    
    user = ensure_user_data_structure(chat_id)
    user["trading"]["exchange"] = exchange
    user["mode"] = "trading" 
    
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("💎 Coin Mặc Định", callback_data="trading_defaults"),
        InlineKeyboardButton("➕ Thêm coin", callback_data="trading_add"),
        InlineKeyboardButton("📋 Watchlist", callback_data="trading_watch"),
        InlineKeyboardButton("⏱️ Auto-Trading", callback_data="trading_auto_menu"), 
        InlineKeyboardButton("📊 Phân tích (Gõ /SYMBOL)", callback_data="trading_help_analyze"),
        InlineKeyboardButton("📘 Hướng dẫn Giao dịch", callback_data="trading_help"),
        InlineKeyboardButton("⬅️ Quay lại", callback_data="back_main")
    )
    text = f"💹 TRADING MENU (Sàn: {exchange} ) Nạp 100$ nhận ngay 10$ free (Inbox: @taikhongdoixiu888))"
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, f"Đã chọn sàn {exchange}")

@bot.callback_query_handler(func=lambda c: c.data == "trading_add")
def cb_trading_add(call):
    chat_id = call.message.chat.id
    text = "💬 Hãy nhập mã coin (hoặc nhiều coin, phân cách bởi dấu phẩy) — ví dụ: SOL, INJ, BTC"
    _user_states[str(chat_id)] = "awaiting_addcoin"
    bot.send_message(chat_id, text)
    answer_ok(call, "➕ Nhập coin để thêm")

@bot.callback_query_handler(func=lambda c: c.data == "trading_watch")
def cb_trading_watch(call):
    chat_id = call.message.chat.id
    wl = user_data.get(str(chat_id), {}).get("trading", {}).get("watchlist", [])
    text = "📋 Watchlist của bạn:\n" + ("\n".join(wl) if wl else "— Trống —")
    markup = InlineKeyboardMarkup().add(InlineKeyboardButton("⬅️ Quay lại", callback_data="menu_trading"))
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "📋 Watchlist")

# <<< THÊM MỚI (FIX V16): HÀM TẠO MARKUP CHO AUTO-TRADING >>>
def build_autotrading_markup(chat_id):
    markup = InlineKeyboardMarkup(row_width=3)
    
    user = ensure_user_data_structure(chat_id)
    current_intervals = user["trading"].get("auto_trade_intervals", [])
    auto_exit_status = user["trading"].get("auto_exit_on_reversal", True)
    
    intervals_map = {
        300: "5m", 900: "15m", 1800: "30m", 
        3600: "1h", 14400: "4h"
    }
    
    buttons = []
    
    for sec, name in intervals_map.items():
        emoji = "✅" if sec in current_intervals else "❌"
        buttons.append(InlineKeyboardButton(f"{emoji} {name}", callback_data=f"auto_toggle:{sec}:{name}"))
        
    markup.add(*buttons)
    
    # <<< THÊM NÚT TOGGLE API EXIT >>>
    exit_emoji = "🟢" if auto_exit_status else "🔴"
    exit_text = f"⚙️ API Exit {exit_emoji}"
    markup.add(InlineKeyboardButton(exit_text, callback_data="toggle_auto_exit_api"))
    # <<< KẾT THÚC THÊM NÚT >>>
    
    markup.add(InlineKeyboardButton("⛔ TẮT AUTO-TRADING", callback_data="auto_off"))
    markup.add(InlineKeyboardButton("⬅️ Quay lại", callback_data="back_main"))
    
    return markup
# <<< KẾT THÚC HÀM TẠO MARKUP >>>

@bot.callback_query_handler(func=lambda c: c.data == "trading_auto_menu")
def cb_trading_auto_menu(call):
    handle_autotrading_command(call.message)
    answer_ok(call, "⏱️ Mở menu Auto-Trading")

# <<< SỬA ĐỔI (FIX V16): HÀM XỬ LÝ LỆNH AUTO TRADING >>>
@bot.message_handler(commands=['autotrading'])
def handle_autotrading_command(message):
    chat_id = message.chat.id
    
    markup = build_autotrading_markup(chat_id)
    
    text = "⏱️ Chọn chu kỳ quét Auto-Trading (Đa khung được bật):"
    bot.send_message(chat_id, text, reply_markup=markup)
# <<< KẾT THÚC SỬA ĐỔI >>>


@bot.callback_query_handler(func=lambda c: c.data == "trading_help" or c.data == "guide_trading")
def cb_trading_help(call):
    chat_id = call.message.chat.id
    text = ("📘 HƯỚNG DẪN TRADING\n\n"
            "1) Dùng /start và chọn 💹 Trading.\n"
            "2) Dùng 🔑 /setbinancekeys (Testnet) hoặc /setokxkeys (Live) để thiết lập Key cá nhân.\n"
            "3) Bấm [➕ Thêm coin] và gõ tên coin (VD: BTC, ETH, SOL) để thêm vào Watchlist.\n"
            "4) Bấm [⏱️ Auto-Trading] để Bật/Tắt bot tự động quét và thực thi lệnh demo.\n"
            "5) Gõ lệnh /SYMBOL (ví dụ /BTCUSDT) để yêu cầu phân tích ngay lập tắt.\n"
            "6) Gõ /setcap [vốn], /setrisk [risk%], và /setleverage [đòn bẩy] để quản lý rủi ro.\n"
            "7) Gõ /pnl để xem thống kê Lãi/Lỗ.") 
    markup = InlineKeyboardMarkup().add(InlineKeyboardButton("⬅️ Quay lại", callback_data="menu_trading"))
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "📘 Hướng dẫn Trading")

@bot.callback_query_handler(func=lambda c: c.data == "trading_help_analyze")
def cb_trading_help_analyze(call):
    bot.send_message(call.message.chat.id, "Gõ lệnh /SYMBOL (ví dụ: /BTCUSDT) để yêu cầu phân tích ngay.")
    answer_ok(call, "Gõ /SYMBOL")

# --- Menu Tài Xỉu (V8/V9 UI) ---
@bot.callback_query_handler(func=lambda c: c.data == "menu_taixiu")
def cb_menu_taixiu(call):
    chat_id = call.message.chat.id
    ensure_user_data_structure(chat_id)
    user_data[str(chat_id)]["mode"] = "taixiu" 
    
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton("🎯 Chơi Tài Xỉu (Thử)", callback_data="tx_play"),
        InlineKeyboardButton("📊 Chiến thuật & Nhận diện cầu", callback_data="tx_strategy"),
        InlineKeyboardButton("📘 Hướng dẫn Tài Xỉu", callback_data="tx_help"),
        InlineKeyboardButton("⬅️ Quay lại", callback_data="back_main")
    )
    text = "🎲 TÀI XỈU MENU\n(Chế độ Tài Xỉu đã BẬT)"
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "🎲 Đang mở Tài Xỉu Menu")

@bot.callback_query_handler(func=lambda c: c.data == "tx_play")
def cb_tx_play(call):
    chat_id = call.message.chat.id
    pred = predict_taixiu(chat_id)
    prediction, conf = pred.get("prediction"), pred.get("confidence")
    dice = [random.randint(1,6) for _ in range(3)]
    outcome, total = record_taixiu_result(chat_id, dice)
    
    text = f"🎲 Kết quả: {dice[0]} + {dice[1]} + {dice[2]} = {total} → <b>{outcome}</b>\n\n"
    text += f"🔮 Dự đoán (AI): <b>{prediction}</b> (Conf: {conf}%)\n"
    text += "✅ Dự đoán đúng!" if prediction == outcome else "❌ Dự đoán sai."
    
    bot.send_message(chat_id, text, parse_mode="HTML")
    answer_ok(call, "Đã tung xúc xắc!")

@bot.callback_query_handler(func=lambda c: c.data == "tx_strategy")
def cb_tx_strategy(call):
    chat_id = call.message.chat.id
    user = ensure_user_data_structure(chat_id)
    history = [h["outcome"] for h in user["taixiu"].get("history_deque", [])]
    trend = detect_trend(history)
    text = f"📊 CHIẾN THUẬT & NHẬN DIỆN CẦU\n\nCầu hiện tại: {trend}"
    markup = InlineKeyboardMarkup().add(InlineKeyboardButton("⬅️ Quay lại", callback_data="menu_taixiu"))
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "📊 Chiến thuật Tài Xỉu")

@bot.callback_query_handler(func=lambda c: c.data == "tx_help" or c.data == "guide_taixiu")
def cb_tx_help(call):
    chat_id = call.message.chat.id
    text = ("📘 HƯỚNG DẪN TÀI XỈU\n\n"
            "1) Dùng /start và chọn 🎲 Tài Xỉu.\n"
            "2) Gửi MD5 (32 ký tự) để bot dự đoán.\n"
            "3) Gửi kết quả dạng {a-b-c} (ví dụ {1-2-3}) để bot cập nhật lịch sử & vốn.\n"
            "4) Dùng /setbase [số tiền] để đặt mức cược gốc.")
    markup = InlineKeyboardMarkup().add(InlineKeyboardButton("⬅️ Quay lại", callback_data="menu_taixiu"))
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "📘 Hướng dẫn Tài Xỉu")

# --- Menu Hướng Dẫn (Chung) ---
@bot.callback_query_handler(func=lambda c: c.data == "menu_guide")
def cb_menu_guide(call):
    chat_id = call.message.chat.id
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton("💹 Hướng dẫn Trading", callback_data="guide_trading"),
        InlineKeyboardButton("🎲 Hướng dẫn Tài Xỉu", callback_data="guide_taixiu"),
        InlineKeyboardButton("⬅️ Quay lại", callback_data="back_main")
    )
    text = "📘 HƯỚNG DẪN SỬ DỤNG"
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "📘 Mở Hướng dẫn")

# --- Lệnh (Tài Xỉu) ---
@bot.message_handler(commands=['setbase'])
def set_base_bet(message):
    chat_id = message.chat.id
    parts = message.text.split()
    if len(parts) == 2 and parts[1].isdigit():
        base = int(parts[1])
        user = ensure_user_data_structure(chat_id)
        user["taixiu"]['base_bet'] = base
        user["taixiu"]['bet'] = base 
        bot.send_message(chat_id, f"✅ Đã đặt mức cược gốc: {base:,} VND")
    else:
        bot.send_message(chat_id, "⚠️ Dùng lệnh: /setbase 2000")

# --- MỚI V14: LỆNH TẮT TAIIXU ---
@bot.message_handler(commands=['taixiu'])
def cmd_taixiu_shortcut(message):
    chat_id = message.chat.id
    user = ensure_user_data_structure(chat_id)
    user["mode"] = "taixiu" 
    
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton("🎯 Chơi Tài Xỉu (Thử)", callback_data="tx_play"),
        InlineKeyboardButton("📊 Chiến thuật & Nhận diện cầu", callback_data="tx_strategy"),
        InlineKeyboardButton("📘 Hướng dẫn Tài Xỉu", callback_data="tx_help"),
        InlineKeyboardButton("⬅️ Quay lại", callback_data="back_main")
    )
    bot.send_message(chat_id, "🎲 Đang mở Tài Xỉu Menu\n(Chế độ Tài Xỉu đã BẬT)", reply_markup=markup)
# ----------------------------------


# --- LỆNH RISK MANAGEMENT ---
@bot.message_handler(commands=['setcap'])
def cmd_set_capital(message):
    chat_id = message.chat.id
    parts = message.text.split()
    if len(parts) == 2 and parts[1].replace('.', '').replace(',', '').isdigit():
        try:
            capital = float(parts[1].replace(',', ''))
            user = ensure_user_data_structure(chat_id)
            user["trading"]["total_capital"] = capital
            bot.send_message(chat_id, f"✅ Đã đặt tổng vốn giao dịch: <b>{capital:,.0f} USD</b>", parse_mode="HTML")
        except:
            bot.send_message(chat_id, "⚠️ Dùng lệnh: /setcap 5000 (Vốn tính bằng USD)")
    else:
        bot.send_message(chat_id, "⚠️ Dùng lệnh: /setcap 5000 (Vốn tính bằng USD)")

@bot.message_handler(commands=['setrisk'])
def cmd_set_risk(message):
    chat_id = message.chat.id
    parts = message.text.split()
    if len(parts) == 2:
        try:
            risk = float(parts[1])
            if 0.1 <= risk <= 25.0:
                user = ensure_user_data_structure(chat_id)
                user["trading"]["risk_per_trade"] = risk
                bot.send_message(chat_id, f"✅ Đã đặt rủi ro tối đa/lệnh: <b>{risk:.1f}%</b>", parse_mode="HTML")
            else:
                bot.send_message(chat_id, "⚠️ Rủi ro/lệnh phải nằm trong khoảng 0.1% đến 15.0%")
        except:
            bot.send_message(chat_id, "⚠️ Dùng lệnh: /setrisk 1.0 (Phần trăm rủi ro trên tổng vốn)")
    else:
        bot.send_message(chat_id, "⚠️ Dùng lệnh: /setrisk 1.0 (Phần trăm rủi ro trên tổng vốn)")

@bot.message_handler(commands=['setleverage'])
def cmd_set_leverage(message):
    chat_id = message.chat.id
    parts = message.text.split()
    if len(parts) == 2 and parts[1].isdigit():
        try:
            leverage = float(parts[1])
            if 1 <= leverage <= 100:
                user = ensure_user_data_structure(chat_id)
                user["trading"]["leverage"] = leverage
                bot.send_message(chat_id, f"✅ Đã đặt đòn bẩy khuyến nghị: <b>{leverage:.0f}x</b>", parse_mode="HTML")
            else:
                bot.send_message(chat_id, "⚠️ Đòn bẩy phải nằm trong khoảng 1x đến 100x.")
        except:
            bot.send_message(chat_id, "⚠️ Dùng lệnh: /setleverage 5 (Đòn bẩy khuyến nghị, ví dụ: 5x)")
    else:
        bot.send_message(chat_id, "⚠️ Dùng lệnh: /setleverage 5 (Đòn bẩy khuyến nghị, ví dụ: 5x)")
# ---------------------------------------------
@bot.message_handler(commands=['setexitapi'])
def cmd_set_auto_exit(message):
    chat_id = message.chat.id
    user = ensure_user_data_structure(chat_id)
    parts = message.text.split()

    if len(parts) == 2 and parts[1].lower() in ['on', 'off']:
        is_on = parts[1].lower() == 'on'
        user["trading"]["auto_exit_on_reversal"] = is_on
        save_user_data()
        
        status = "BẬT (ON) - Bot SẼ TỰ ĐỘNG ĐÓNG lệnh cũ và MỞ lệnh mới khi có tín hiệu đảo chiều mạnh." if is_on else "TẮT (OFF) - Bot CHỈ BÁO tín hiệu đảo chiều mạnh và BỎ QUA lệnh đóng/mở tự động."
        bot.send_message(chat_id, f"⚙️ Chế độ API Đóng lệnh Đảo chiều đã được cập nhật:\n\n**Trạng thái:** {status}", parse_mode="Markdown")
    else:
        current_status = "ON" if user["trading"].get("auto_exit_on_reversal", True) else "OFF"
        bot.send_message(chat_id, 
                         f"⚠️ Cú pháp sai. Trạng thái hiện tại: **{current_status}**\n"
                         f"Dùng lệnh:\n"
                         f"  `/setexitapi on` (Bật)\n"
                         f"  `/setexitapi off` (Tắt)", parse_mode="Markdown")


@bot.message_handler(commands=['setbinancekeys'])
def set_binance_keys(message):
    chat_id = message.chat.id
    user = ensure_user_data_structure(chat_id)
    
    parts = message.text.split(maxsplit=1)
    
    if len(parts) != 2 or not parts[1].strip():
        bot.send_message(chat_id, 
                         "🔑 Vui lòng nhập API Key và Secret Key BINANCE TESTNET theo cú pháp:\n"
                         "<code>/setbinancekeys &lt;API_KEY&gt; , &lt;SECRET_KEY&gt;</code>\n"
                         "Ví dụ: <code>/setbinancekeys XAuso...ffP , iDhLNt...j2NT</code>",
                         parse_mode="HTML")
        return

    # SỬA LOGIC: Chấp nhận dấu phẩy hoặc dấu cách làm phân cách
    key_input = parts[1].strip()
    
    if ',' in key_input:
        keys_list = [k.strip() for k in key_input.split(',', 1)]
    else:
        # Nếu không có dấu phẩy, dùng maxsplit=2 cho dấu cách
        keys_list = [k.strip() for k in key_input.split(maxsplit=2)]
        
    if len(keys_list) != 2:
        bot.send_message(chat_id, 
                         "❌ Lỗi cú pháp: Vui lòng nhập đủ <b>API Key</b> và <b>Secret Key</b>.",
                         parse_mode="HTML")
        return

    api_key = keys_list[0]
    secret_key = keys_list[1]
    
    # Lưu tạm thời (chưa mã hóa) để kiểm tra
    user["trading"]["api_key"] = api_key
    user["trading"]["secret_key"] = secret_key
    user["trading"]["passphrase"] = None 

    try:
        # GỌI HÀM get_user_exchange_client VỚI CỜ for_check=True 
        client, error_msg = get_user_exchange_client(user, for_check=True)
        
        # Kiểm tra kết nối Futures (Ping)
        if client and not error_msg:
             client.futures_ping()
             
             # Lưu vĩnh viễn (sẽ được mã hóa trong save_user_data)
             save_user_data() 
             
             bot.send_message(chat_id, 
                             "✅ **Thiết lập API Binance Testnet thành công!**", 
                             parse_mode="HTML")
        else:
             # Nếu client là None hoặc có error_msg
             raise Exception(error_msg or "Kết nối API thất bại.")
             
    except Exception as e:
        # <<< FIX: XÓA KEYS VÀ BÁO LỖI NẾU LỖI XẢY RA >>>
        # Xóa Keys khỏi bộ nhớ (sẽ được lưu lại là None trong save_user_data)
        user["trading"]["api_key"] = None
        user["trading"]["secret_key"] = None
        user["trading"]["passphrase"] = None
        save_user_data()
        
        error_display = str(e)
        if "Binance Testnet Error:" in error_display:
            error_display = error_display.replace("Binance Testnet Error: ", "")
            
        bot.send_message(chat_id, 
                         f"❌ **Lỗi kết nối API Binance Testnet!** Vui lòng kiểm tra lại Key/Secret và quyền TRADE. Lỗi: {error_display}", 
                         parse_mode="HTML")
# ------------------------------------------------------------------------

# --- HANDLER: OKX API KEYS CÁ NHÂN ---
@bot.message_handler(commands=['setokxkeys'])
def set_okx_keys(message):
    chat_id = message.chat.id
    user = ensure_user_data_structure(chat_id)
    
    parts = message.text.split()
    if len(parts) != 4:
        bot.send_message(chat_id, 
                         "🔑 Vui lòng nhập API Key, Secret Key và Passphrase OKX theo cú pháp:\n"
                         "<code>/setokxkeys &lt;API_KEY&gt; &lt;SECRET_KEY&gt; &lt;PASSPHRASE&gt;</code>",
                         parse_mode="HTML")
        return

    api_key = parts[1]
    secret_key = parts[2]
    passphrase = parts[3]
    
    # Lưu vào dữ liệu người dùng
    user["trading"]["api_key"] = api_key
    user["trading"]["secret_key"] = secret_key
    user["trading"]["passphrase"] = passphrase
    save_user_data()

    try:
        # Kiểm tra kết nối OKX (dùng hàm chung)
        client, error_msg = get_user_exchange_client(user)
        
        if client and not error_msg:
             bot.send_message(chat_id, 
                             "✅ **Thiết lập API OKX thành công!**\n"
                             "Lưu ý: OKX không có Testnet riêng, bot sẽ chạy trên môi trường LIVE.", 
                             parse_mode="HTML")
        else:
            raise Exception(error_msg)
    except Exception as e:
        # Xóa Keys nếu kiểm tra thất bại
        user["trading"]["api_key"] = None
        user["trading"]["secret_key"] = None
        user["trading"]["passphrase"] = None
        save_user_data()
        bot.send_message(chat_id, 
                         f"❌ **Lỗi kết nối API OKX!** Vui lòng kiểm tra lại Key/Secret/Passphrase. Lỗi: {e}", 
                         parse_mode="HTML")
# ------------------------------------------------------------------------
@bot.message_handler(commands=['setbybitkeys'])
def set_bybit_keys(message):
    chat_id = message.chat.id
    user = ensure_user_data_structure(chat_id)
    
    parts = message.text.split(maxsplit=3)
    
    if len(parts) < 3:
        bot.send_message(chat_id, 
                         "🔑 Vui lòng nhập API Key và Secret Key BYBIT theo cú pháp:\n"
                         "<code>/setbybitkeys &lt;API_KEY&gt; &lt;SECRET_KEY&gt; [Testnet/Live]</code>\n"
                         "Ví dụ: <code>/setbybitkeys XAuso...ffP iDhLNt...j2NT Testnet</code>",
                         parse_mode="HTML")
        return

    api_key = parts[1]
    secret_key = parts[2]
    
    # Mặc định là Live nếu không chỉ định rõ
    is_testnet = (len(parts) == 4 and parts[3].lower() == 'testnet')
    
    # Lưu vào dữ liệu người dùng
    user["trading"]["api_key"] = api_key
    user["trading"]["secret_key"] = secret_key
    user["trading"]["passphrase"] = None 
    user["trading"]["bybit_testnet"] = is_testnet # Thêm flag riêng cho Bybit
    save_user_data()

    try:
        # TẠO CLIENT SỬ DỤNG CCXT ĐỂ KIỂM TRA
        client = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret_key,
            'options': {'defaultType': 'swap'}, 
            'enableRateLimit': True,
            'timeout': 30000 
        })
        
        # Thiết lập URL Testnet/Live
        if is_testnet:
             client.set_urls({'api': 'https://api-testnet.bybit.com'})
             
        # Kiểm tra kết nối
        client.fetch_time() 

        env_status = "TESTNET (Demo)" if is_testnet else "LIVE (Thật)"
        bot.send_message(chat_id, 
                         f"✅ **Thiết lập API Bybit thành công!**\n"
                         f"Môi trường: **{env_status}**\n"
                         f"Lưu ý: Chức năng thực thi lệnh tự động (Auto-Trade) trên Bybit vẫn chưa được hỗ trợ hoàn toàn.", 
                         parse_mode="HTML")
                         
    except Exception as e:
        # Xóa Keys nếu kiểm tra thất bại
        user["trading"]["api_key"] = None
        user["trading"]["secret_key"] = None
        user["trading"]["passphrase"] = None
        user["trading"]["bybit_testnet"] = False
        save_user_data()

        
        error_display = str(e)
        if "API-key format invalid" in error_display:
            error_display = "API-key format invalid. Vui lòng kiểm tra lại Key/Secret."
            
        bot.send_message(chat_id, 
                         f"❌ **Lỗi kết nối API Bybit!** Vui lòng kiểm tra lại Key/Secret. Lỗi: {error_display}", 
                         parse_mode="HTML")
# ------------------------------------------------------------------------

# --- LỆNH /addcoin (ĐÃ SỬA ĐỂ XỬ LÝ ĐẦU VÀO TRỰC TIẾP) ---
@bot.message_handler(commands=['addcoin'])
def cmd_addcoin_shortcut(message):
    chat_id = message.chat.id
    parts = message.text.split(maxsplit=1)
    
    # 1. KIỂM TRA: Nếu có coin được nhập ngay sau lệnh
    if len(parts) >= 2 and parts[1].strip():
        text = parts[1]
        feedback = handle_addcoin_input(chat_id, text)
        bot.send_message(chat_id, feedback)
        # Không cần đặt trạng thái, đã xử lý xong.
        return
        
    # 2. KHÔNG CÓ COIN: Chuyển sang trạng thái chờ nhập (như trước)
    text = "💬 Hãy nhập mã coin (hoặc nhiều coin, phân cách bởi dấu phẩy) — ví dụ: SOL, INJ, BTC"
    _user_states[str(chat_id)] = "awaiting_addcoin"
    bot.send_message(chat_id, text)

# Hàm add_coin_cmd (dùng lệnh /add_coin) có thể giữ nguyên, nhưng khuyến nghị dùng /addcoin
@bot.message_handler(commands=['add_coin'])
def add_coin_cmd(message):
    chat_id = message.chat.id
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        # Nếu không có coin, gọi lại hàm xử lý chính
        return cmd_addcoin_shortcut(message) 
    
    text = parts[1]
    feedback = handle_addcoin_input(chat_id, text)
    bot.send_message(chat_id, feedback)

# <<< MỚI V15.2: HANDLER CHO LỆNH XÓA COIN >>>
@bot.message_handler(commands=['delcoin', 'xoacoin'])
def cmd_delete_coin(message):
    """Xóa 1 hoặc nhiều coin khỏi Watchlist."""
    chat_id = message.chat.id
    parts = message.text.split(maxsplit=1)
    
    if len(parts) < 2 or not parts[1].strip():
        bot.send_message(chat_id, "⚠️ Vui lòng nhập cú pháp đúng:\n`/delcoin BTC` hoặc `/xoacoin SOL, ETH`", parse_mode="Markdown")
        return
        
    coins_to_remove = normalize_and_split_coins(parts[1])
    user = ensure_user_data_structure(chat_id)
    user_list = user["trading"]["watchlist"]
    
    removed = []
    not_found = []
    
    for sym in coins_to_remove:
        if sym in user_list:
            user_list.remove(sym)
            removed.append(sym)
        else:
            not_found.append(sym)
            
    # Đồng bộ với file
    save_watchlist({str(chat_id): user_list}) 

    feedback = []
    if removed:
        feedback.append("✅ Đã xóa khỏi Watchlist: " + ", ".join(removed))
    if not_found:
        feedback.append("ℹ️ Không tìm thấy trong Watchlist: " + ", ".join(not_found))
    if not removed and not not_found:
         bot.send_message(chat_id, "⚠️ Lỗi xử lý. Vui lòng kiểm tra lại cú pháp.")
         return

    bot.send_message(chat_id, "\n".join(feedback))

# ---------------------------------------------
    
@bot.message_handler(commands=['watchlist'])
def cmd_watchlist(message):
    chat_id = message.chat.id 
    wl = user_data.get(str(chat_id), {}).get("trading", {}).get("watchlist", [])
    text = "📋 Watchlist của bạn:\n" + ("\n".join(wl) if wl else "— Trống —")
    bot.send_message(chat_id, text)

@bot.message_handler(commands=['stopautotrading'])
def cmd_stopautotrading_shortcut(message):
    """Lệnh tắt cho TẮT AUTO-TRADING"""
    chat_id = message.chat.id
    user = ensure_user_data_structure(chat_id)
    user["trading"]["auto_trade_intervals"] = [] 
    bot.send_message(chat_id, "🔕 Auto-Trading OFF")
# ---------------------------------------------

# --- Auto-Trading (V10 - Tái cấu trúc) ---

# FILE: Botthapcamnhucac.py
# THAY THẾ HÀM record_signal (khoảng dòng 1599)

def record_signal(chat_id, symbol, entry, tp1, tp2, tp3, sl, order_type, conf, position_size, sl_order_id, tp_order_id):
    """(Cập nhật) Lưu tín hiệu mới với ID, PnL, Khối lượng và Order IDs."""
    user = ensure_user_data_structure(chat_id)
    signals = user["trading"].setdefault("signals", [])
    sig_id = str(uuid.uuid4())[:8] # Thêm ID cho tín hiệu
    sig = {
        "id": sig_id,
        "symbol": symbol, "entry": float(entry), "tp1": float(tp1),
        "tp2": float(tp2), "tp3": float(tp3), "sl": float(sl),
        "order_type": order_type, "confidence": conf, "status": "open",
        "created_at": datetime.now(TZ).isoformat(), 
        "last_checked": None, 
        "events": [],
        "pnl_percent": None,
        "high_price": entry, 
        "low_price": entry,
        
        # <<< MỚI: Các trường bắt buộc cho Trailing SL >>>
        "position_size": float(position_size),
        "sl_order_id": sl_order_id,
        "tp_order_id": tp_order_id,
        "trailing_level": 0 # 0=Gốc, 1=Về Entry, 2=Về TP1, 3=Về TP2
        # <<< KẾT THÚC MỚI >>>
    }
    signals.append(sig)
    print(f"Tín hiệu mới [{chat_id}]: {symbol} (ID: {sig_id}) | Size: {position_size} | SL ID: {sl_order_id}")

# <<< SỬA ĐỔI (FIX V16): HANDLER CHO CALLBACK AUTO-TRADING >>>
@bot.callback_query_handler(func=lambda call: call.data.startswith("auto_toggle:"))
def cb_auto_set_time(call):
    """Xử lý việc Bật/Tắt các khung giờ Auto-Trading."""
    chat_id = str(call.message.chat.id)
    parts = call.data.split(':')
    interval_seconds = int(parts[1])
    interval_name = parts[2]
    
    if not answer_ok(call, f"Đang cập nhật {interval_name}"): return # Anti-Spam
    
    user = ensure_user_data_structure(chat_id)
    intervals_list = user["trading"].setdefault("auto_trade_intervals", [])

    if interval_seconds in intervals_list:
        intervals_list.remove(interval_seconds)
        status = "TẮT"
    else:
        intervals_list.append(interval_seconds)
        status = "BẬT"
        
    # Cập nhật lại menu để hiển thị trạng thái mới
    text = "⏱️ Chọn chu kỳ quét Auto-Trading (Đa khung được bật):"
    try:
        # Sử dụng build_autotrading_markup để lấy markup mới nhất
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=build_autotrading_markup(chat_id))
    except Exception as e:
        print(f"Lỗi cập nhật menu Auto-Trade: {e}")
        
    print(f"Auto-Trade [{chat_id}]: {interval_name} đã {status}")
# <<< KẾT THÚC SỬA ĐỔI >>>

@bot.callback_query_handler(func=lambda call: call.data == "auto_off")
def cb_auto_off(call):
    """Tắt hoàn toàn Auto-Trading và cập nhật menu."""
    chat_id = str(call.message.chat.id)
    user = ensure_user_data_structure(chat_id)
    user["trading"]["auto_trade_intervals"] = [] 
    
    text = "🔕 Auto-Trading OFF.\n\n⏱️ Chọn chu kỳ quét Auto-Trading (Đa khung được bật):"
    
    # Cập nhật menu Auto-Trading về trạng thái OFF (tất cả là ❌)
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=build_autotrading_markup(chat_id))
    except Exception as e:
        # Fallback nếu edit lỗi (ví dụ tin nhắn quá cũ)
        bot.send_message(chat_id, text, reply_markup=build_autotrading_markup(chat_id))

    answer_ok(call, "Đã tắt Auto-Trading")
    print(f"Auto-Trade [{chat_id}]: Đã TẮT")

@bot.callback_query_handler(func=lambda call: call.data == "toggle_auto_exit_api")
def cb_auto_set_exit_api(call):
    chat_id = str(call.message.chat.id)
    if not answer_ok(call, "Đang cập nhật chế độ API..."): return

    user = ensure_user_data_structure(chat_id)
    current_status = user["trading"].get("auto_exit_on_reversal", True)
    
    # Đảo trạng thái
    new_status = not current_status
    user["trading"]["auto_exit_on_reversal"] = new_status
    save_user_data()
    
    # Cập nhật menu và thông báo
    text = "⏱️ Chọn chu kỳ quét Auto-Trading (Đa khung được bật):"
    
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=build_autotrading_markup(chat_id))
    except Exception:
        bot.send_message(chat_id, text, reply_markup=build_autotrading_markup(chat_id))
        
    status_text = "BẬT (ON)" if new_status else "TẮT (OFF)"
    bot.send_message(chat_id, f"⚙️ **Chế độ API Đóng lệnh Đảo chiều** đã chuyển sang **{status_text}**.", parse_mode="Markdown")

# --- Handlers cho Coin Mặc Định (Giữ nguyên) ---
@bot.callback_query_handler(func=lambda c: c.data == "trading_defaults")
def cb_show_default_coins(call):
    chat_id = call.message.chat.id
    markup = InlineKeyboardMarkup(row_width=4) 
    buttons = []
    coins = ["BTC","ETH","SOL","BNB","XRP","DOGE","LINK","TON","NEAR","AVAX"]
    for c in coins:
        sym = f"{c}USDT"
        buttons.append(InlineKeyboardButton(c, callback_data=f"coin_{sym}"))
    
    markup.add(*buttons) 
    markup.add(InlineKeyboardButton("⬅️ Về Trading", callback_data="menu_trading"))
    
    text = "💎 Chọn Coin Mặc Định để phân tích:"
    try:
        bot.edit_message_text(text, chat_id, call.message.message_id, reply_markup=markup)
    except Exception:
        bot.send_message(chat_id, text, reply_markup=markup)
    answer_ok(call, "💎 Coin Mặc Định")

@bot.callback_query_handler(func=lambda c: c.data.startswith("coin_"))
def cb_handle_coin_select(call):
    chat_id = call.message.chat.id
    symbol = call.data.split("_",1)[1]
    analyze_and_send(chat_id, symbol, precomputed_res=None)
    answer_ok(call, f"Đang phân tích {symbol}...")

# <<< BẮT ĐẦU CHÈN HÀM MỚI TẠI ĐÂY >>>
@bot.callback_query_handler(func=lambda c: c.data.startswith("show_details:"))
def cb_show_details(call):
    chat_id = str(call.message.chat.id)
    temp_id = call.data.split(":")[1]
    
    # Kiểm tra và trả lời callback query
    if not answer_ok(call, "Đang mở chi tiết..."): 
        return 
    
    # Truy cập dữ liệu chi tiết tạm thời
    detail_data = user_data.get(chat_id, {}).get("temp_detail_message")
    
    if detail_data and detail_data.get('id') == temp_id:
        # Gửi tin nhắn chi tiết
        bot.send_message(chat_id, detail_data['content'], parse_mode="HTML")
        
        # Xóa nội dung chi tiết cũ khỏi user_data sau khi sử dụng (để dọn dẹp bộ nhớ)
        try:
            del user_data[chat_id]['temp_detail_message']
            save_user_data()
        except:
            pass

    else:
        # Nếu không tìm thấy dữ liệu
        bot.send_message(chat_id, "⚠️ Dữ liệu chi tiết đã hết hạn (chỉ lưu trữ trong thời gian ngắn) hoặc bot đã được khởi động lại. Vui lòng chạy lại lệnh /SYMBOL.")
# <<< KẾT THÚC CHÈN HÀM MỚI >>>

@bot.message_handler(commands=['pnl'])
def cmd_pnl_stats(message):
    """Hiển thị thống kê PnL (Lãi/Lỗ) từ các tín hiệu đã đóng, bao gồm lợi nhuận USD và PnL ước tính lệnh mở."""
    chat_id = str(message.chat.id)
    user = ensure_user_data_structure(chat_id)
    
    trading_data = user.get("trading", {})
    signals = trading_data.get("signals", [])
    
    total_capital = trading_data.get("total_capital", 0.0)
    leverage = trading_data.get("leverage", 1.0) 
    
    last_reset_iso = trading_data.get("last_pnl_reset", datetime.now(TZ).isoformat())
    last_reset_dt = datetime.fromisoformat(last_reset_iso).replace(tzinfo=TZ)
    current_dt = datetime.now(TZ)
    
    # 1. KIỂM TRA VÀ THỰC HIỆN RESET HÀNG TUẦN (Giữ nguyên)
    if (current_dt - last_reset_dt).days >= PNL_RESET_DAYS:
        # ... (Logic reset giữ nguyên) ...
        open_signals_data = [s for s in signals if s.get("status") == "open"]
        user["trading"]["signals"] = open_signals_data
        user["trading"]["pnl_counts"] = Counter() 
        user["trading"]["last_pnl_reset"] = current_dt.isoformat()
        save_user_data()
        bot.send_message(chat_id, "🔔 **PNL ĐÃ ĐƯỢC RESET** 🔔\n\nHiệu suất tuần trước đã được xóa. Chỉ các lệnh đang mở được giữ lại.", parse_mode="Markdown")
        signals = open_signals_data
        
    closed_signals = [s for s in signals if s.get("status") not in ["open", "closed_legacy_error", "error_trailing"] and "pnl_percent" in s]
    
    # 2. XỬ LÝ VỊ THẾ ĐANG MỞ (NÂNG CẤP 2)
    open_signals = [s for s in signals if s.get("status") == "open"]
    total_open_pnl_percent = 0.0
    open_pnl_details = []
    
    open_count = len(open_signals)
    if open_count > 0:
        for sig in open_signals:
            symbol = sig['symbol']
            entry = sig['entry']
            pos_size = sig.get('position_size', 0)
            is_long = entry < sig.get('tp1', entry + 1)
            
            price = get_market_price(symbol)
            if price is not None and pos_size > 0:
                if is_long:
                    pnl_percent = ((price - entry) / entry) * 100
                else:
                    pnl_percent = ((entry - price) / entry) * 100
                    
                total_open_pnl_percent += pnl_percent
                
                # Tính PnL USD (ước tính)
                pnl_usd = 0.0
                if total_capital > 0 and leverage > 0:
                     pnl_usd = (total_capital * (pnl_percent / 100)) * leverage * (pos_size * entry / (total_capital * leverage)) 
                
                open_pnl_details.append({
                    "symbol": symbol,
                    "pnl_percent": pnl_percent,
                    "pnl_usd": pnl_usd
                })

    # 3. HIỂN THỊ THỐNG KÊ LỆNH ĐÓNG (Giữ nguyên)
    total_closed = len(closed_signals)
    total_closed_pnl_percent = sum(s.get("pnl_percent", 0) for s in closed_signals)
    exit_counts = Counter(s.get("status", "UNKNOWN").lower() for s in closed_signals)
    
    wins = exit_counts['tp1'] + exit_counts['tp2'] + exit_counts['tp3'] + exit_counts['sl_profit']
    losses = exit_counts['sl']
    breakeven_or_exit = exit_counts['sl_breakeven'] + exit_counts['exit_signal'] + exit_counts['exit_manual']
    
    win_rate = (wins / total_closed) * 100 if total_closed else 0
    avg_pnl_percent = total_closed_pnl_percent / total_closed if total_closed else 0
    
    estimated_closed_profit_usd = (total_capital * (total_closed_pnl_percent / 100)) * leverage
    
    capital_text = f"<b>{total_capital:,.0f} USD</b>" if total_capital > 0 else "N/A (Dùng /setcap)"
    leverage_text = f"<b>{leverage:.1f}x</b>" if leverage > 1 else "1x"
    
    # <<< ĐÂY LÀ PHẦN BỊ THIẾU (ĐÃ SỬA) >>>
    # Sắp xếp các tín hiệu đã đóng để lấy Top 5
    sorted_closed = sorted(closed_signals, key=lambda s: s.get('pnl_percent', 0), reverse=True)
    
    # Lấy 5 lệnh thắng tốt nhất (pnl > 0)
    top_5_best = [s for s in sorted_closed if s.get('pnl_percent', 0) > 0][:5]
    
    # Lấy 5 lệnh thua tệ nhất (pnl < 0)
    # Sắp xếp ngược lại (từ tệ nhất đến ít tệ nhất)
    top_5_worst = sorted([s for s in closed_signals if s.get('pnl_percent', 0) < 0], key=lambda s: s.get('pnl_percent', 0))[:5]
    # <<< KẾT THÚC PHẦN SỬA LỖI >>>

    msg = f"<b>📊 Thống Kê PnL (Từ {last_reset_dt.strftime('%d/%m/%Y %H:%M')})</b>\n\n"
    msg += f" • Vốn cài đặt (/setcap): {capital_text}\n"
    msg += f" • Đòn bẩy ước tính (/setleverage): {leverage_text}\n"
    msg += "--- \n"
    
    # <<< HIỂN THỊ PNL ĐANG MỞ (NÂNG CẤP 2) >>>
    if open_count > 0:
        msg += f"🔥 <b>LỆNH ĐANG MỞ: {open_count}</b>\n"
        for detail in open_pnl_details:
             pnl_color = "🟢" if detail['pnl_percent'] >= 0 else "🔴"
             msg += f" • {pnl_color} {detail['symbol']}: {detail['pnl_percent']:,.2f}% ({detail['pnl_usd']:,.2f} USD)\n"
        msg += "---\n"

    # HIỂN THỊ LỆNH ĐÃ ĐÓNG
    msg += f"📉 <b>LỆNH ĐÃ ĐÓNG: {total_closed}</b>\n"
    msg += f" • Tỷ lệ thắng (Winrate): <b>{win_rate:.2f}%</b>\n\n"
    
    # BÁO CÁO CHI TIẾT SL/TP
    msg += "--- Kết quả đóng lệnh chi tiết ---\n"
    msg += f" • <b>❌ Dính SL (Lỗ):</b> {losses} lệnh\n"
    msg += f" • <b>🟡 Hòa vốn/Thoát sớm:</b> {breakeven_or_exit} lệnh\n"
    msg += f" • <b>🟢 SL Lời:</b> {exit_counts['sl_profit']} lệnh\n"
    msg += f" • <b>🟢 Cán TP:</b> {exit_counts['tp1']+exit_counts['tp2']+exit_counts['tp3']} lệnh\n"
    msg += "---------------------------------\n"
    
    msg += f" • Tổng PnL Đóng (%): <b>{total_closed_pnl_percent:.2f}%</b>\n"
    
    if total_capital > 0:
        msg += f" • Lợi nhuận Đóng ước tính (USD): <b>{estimated_closed_profit_usd:,.2f} USD</b>\n\n"
    else:
        msg += f" • Lợi nhuận Đóng ước tính (USD): ⚠️ Vui lòng cài đặt vốn bằng /setcap để xem USD.\n\n"


    if top_5_best:
        msg += "<b>🏆 5 Lệnh Thắng Tốt Nhất:</b>\n"
        for s in top_5_best:
            msg += f"  • {s['symbol']}: <b>+{s['pnl_percent']:.2f}%</b>\n"
            
    if top_5_worst:
        msg += "\n<b>📉 5 Lệnh Thua Tệ Nhất:</b>\n"
        for s in top_5_worst:
            msg += f"  • {s['symbol']}: <b>{s['pnl_percent']:.2f}%</b>\n"

    bot.send_message(chat_id, msg, parse_mode="HTML")
# <<< KẾT THÚC THAY THẾ cmd_pnl_stats >>>

@bot.message_handler(commands=['balance', 'taikhoan'])
def cmd_check_balance(message):
    chat_id = str(message.chat.id)
    bot.send_message(chat_id, "Đang kiểm tra số dư Testnet...")
    
    try:
        user = ensure_user_data_structure(chat_id)
        client, error_msg = get_user_exchange_client(user)
        
        if client is None:
            bot.send_message(chat_id, f"❌ Lỗi: Bạn chưa kết nối API. Vui lòng dùng lệnh /setbinancekeys.\n{error_msg}")
            return

        # Gọi API lấy số dư tài khoản Futures
        balance_info = client.futures_account_balance()
        
        usdt_balance = None
        for asset in balance_info:
            if asset.get('asset') == 'USDT':
                usdt_balance = asset
                break
        
        if usdt_balance:
            total_balance = float(usdt_balance.get('balance', 0))
            available_balance = float(usdt_balance.get('availableBalance', 0))
            
            msg = (f"✅ **Kết nối API Testnet thành công!**\n\n"
                   f"<b>Tài khoản Futures (Demo):</b>\n"
                   f" • 💵 Tổng số dư (Wallet): <b>{total_balance:,.2f} USDT</b>\n"
                   f" • 💰 Khả dụng (Available): <b>{available_balance:,.2f} USDT</b>")
            bot.send_message(chat_id, msg, parse_mode="HTML")
        else:
            bot.send_message(chat_id, "❌ Lỗi: Đã kết nối nhưng không tìm thấy số dư USDT trong tài khoản Futures.")
            
    except Exception as e:
        bot.send_message(chat_id, f"❌ Lỗi API nghiêm trọng khi kiểm tra số dư:\n<code>{e}</code>", parse_mode="HTML")

def run_backtest_strategy(symbol):
    """
    (Chức năng Beta - Đang phát triển)
    Chạy chiến lược trên dữ liệu quá khứ.
    """
    klines = fetch_binance_klines(symbol, '1h', limit=1000)
    if not klines or len(klines) < 250:
        return "Không đủ dữ liệu lịch sử (cần > 250 nến) để backtest."

    print(f"Backtest: Bắt đầu {symbol} với {len(klines)} nến...")
    trades = []
    open_trade = None
    pnl_percent = 100.0 

    for i in range(250, len(klines)):
        past_klines_data = klines[:i]
        
        closes = [float(k[4]) for k in past_klines_data]
        current_price = closes[-1]
        
        closes_arr = np.array(closes, dtype=float)
        rsi_arr = talib.RSI(closes_arr, timeperiod=14)
        rsi_val = rsi_arr[-1] if not np.isnan(rsi_arr[-1]) else None

        if rsi_val is None: continue

        if open_trade is None:
            if rsi_val < 30: 
                open_trade = {"type": "LONG", "entry": current_price, "sl": current_price * 0.95, "tp": current_price * 1.10}
                trades.append(f"LONG @ {current_price} (RSI: {rsi_val:.2f})")
            elif rsi_val > 70: 
                open_trade = {"type": "SHORT", "entry": current_price, "sl": current_price * 1.05, "tp": current_price * 0.90}
                trades.append(f"SHORT @ {current_price} (RSI: {rsi_val:.2f})")
        
        elif open_trade:
            if open_trade["type"] == "LONG":
                if current_price >= open_trade["tp"]:
                    pnl_percent *= (open_trade["tp"] / open_trade["entry"])
                    trades.append(f"CLOSE (TP) @ {open_trade['tp']}. Vốn: {pnl_percent:.2f}%")
                    open_trade = None
                elif current_price <= open_trade["sl"]:
                    pnl_percent *= (open_trade["sl"] / open_trade["entry"])
                    trades.append(f"CLOSE (SL) @ {open_trade['sl']}. Vốn: {pnl_percent:.2f}%")
                    open_trade = None
            elif open_trade["type"] == "SHORT":
                if current_price <= open_trade["tp"]:
                    pnl_percent *= (open_trade["entry"] / open_trade["tp"])
                    trades.append(f"CLOSE (TP) @ {open_trade['tp']}. Vốn: {pnl_percent:.2f}%")
                    open_trade = None
                elif current_price >= open_trade["sl"]:
                    pnl_percent *= (open_trade["entry"] / open_trade["sl"])
                    trades.append(f"CLOSE (SL) @ {open_trade['sl']}. Vốn: {pnl_percent:.2f}%")
                    open_trade = None

    total_trades = len([t for t in trades if "LONG @" in t or "SHORT @" in t])
    return f"<b>Backtest (Beta - RSI 70/30) - {symbol} (1H)</b>\n" \
           f" • Tổng số lệnh: {total_trades}\n" \
           f" • Kết quả cuối cùng (100% vốn ban đầu): <b>{pnl_percent:.2f}%</b>"

@bot.message_handler(commands=['backtest'])
def cmd_backtest_strategy(message):
    chat_id = message.chat.id
    parts = message.text.split()
    if len(parts) != 2:
        bot.send_message(chat_id, "⚠️ Cú pháp: /backtest [SYMBOL]\nVí dụ: /backtest BTCUSDT")
        return
        
    symbol = parts[1].upper()
    if not symbol.endswith("USDT"): symbol += "USDT"

    bot.send_message(chat_id, f"⏳ Đang chạy backtest (beta) cho {symbol} trên 1000 nến 1H... Vui lòng chờ.")
    try:
        result = run_backtest_strategy(symbol)
        bot.send_message(chat_id, result, parse_mode="HTML")
    except Exception as e:
        print(f"Lỗi Backtest: {e}")
        bot.send_message(chat_id, f"❌ Lỗi khi chạy backtest: {e}")

@bot.message_handler(commands=['exit'])
def cmd_exit_manual(message):
    chat_id = str(message.chat.id)
    parts = message.text.split()
    
    if len(parts) != 2:
        bot.send_message(chat_id, "⚠️ Cú pháp: /exit [SYMBOL]\nVí dụ: /exit BTCUSDT. Lệnh này sẽ đóng tất cả vị thế đang mở và hủy OCO.")
        return
        
    symbol = parts[1].upper()
    if not symbol.endswith("USDT"): symbol += "USDT"

    user = ensure_user_data_structure(chat_id)
    
    open_signals = [s for s in user["trading"]["signals"] if s.get("status") == "open" and s["symbol"] == symbol]
    
    if not open_signals:
        bot.send_message(chat_id, f"ℹ️ Không tìm thấy lệnh {symbol} đang mở trong hệ thống theo dõi.")
        return

    sig = open_signals[0]
    pos_size = sig.get("position_size", 0)
    
    client, error_msg = get_user_exchange_client(user)
    if client is None:
        bot.send_message(chat_id, f"❌ Lỗi kết nối API: {error_msg}")
        return

    is_long = sig["entry"] < sig["tp1"]
    
    try:
        client.futures_cancel_all_open_orders(symbol=symbol)
        
        if pos_size <= 0:
            bot.send_message(chat_id, "ℹ️ Vị thế đã đóng hoặc khối lượng bằng 0. Đã hủy lệnh cũ.")
            sig["status"] = "exit_manual"
            save_user_data()
            return

        close_side = Client.SIDE_SELL if is_long else Client.SIDE_BUY
        
        precision = get_symbol_precision(symbol)
        pos_size_rounded = round_by_step(pos_size, precision['stepSize'])

        if pos_size_rounded > 0:
             client.futures_create_order(
                 symbol=symbol, side=close_side, type=Client.ORDER_TYPE_MARKET,
                 quantity=pos_size_rounded, reduceOnly=True
             )
        
        sig["status"] = "exit_manual"
        
        market_price = get_market_price(symbol)
        if market_price:
            round_precision = 4 
            if market_price > 1000: round_precision = 2
            elif market_price > 10: round_precision = 4

            if is_long:
                pnl_percent = ((market_price - sig['entry']) / sig['entry']) * 100
            else:
                pnl_percent = ((sig['entry'] - market_price) / sig['entry']) * 100
            
            sig["pnl_percent"] = pnl_percent
            sig["events"].append({"type": "EXIT_MANUAL", "price": market_price, "pnl": pnl_percent, "time": datetime.now(TZ).isoformat()})
            user["trading"]["pnl_counts"]["exit_manual"] = user["trading"]["pnl_counts"].get("exit_manual", 0) + 1 
            
            bot.send_message(chat_id, f"✅ **ĐÓNG LỆNH THÀNH CÔNG:** {symbol} đã được đóng tại giá thị trường ({market_price:,.{round_precision}f})\n💰 PnL ước tính: **{pnl_percent:,.2f}%**", parse_mode="HTML")
            
        else:
            bot.send_message(chat_id, f"✅ Đóng lệnh {symbol} thành công. Lỗi lấy giá thị trường để tính PnL.")
            
        save_user_data()

    except Exception as e:
        bot.send_message(chat_id, f"❌ LỖI API khi đóng lệnh {symbol}: {e}")

# ================================
# 8. HANDLER CHÍNH (BẮT TIN NHẮN)
# ================================

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    chat_id = str(message.chat.id) 
    text = (message.text or "").strip()
    
    # --- ÁP DỤNG THROTTLE CHO TIN NHẮN ---
    if not check_throttle(chat_id):
        return # Bỏ qua tin nhắn bị spam
    # ------------------------------------
    
    user = ensure_user_data_structure(chat_id)
    mode = user.get("mode", "taixiu")

    # 1. TRẠNG THÁI CHỜ (AWAITING_ADDCOIN)
    if _user_states.get(chat_id) == "awaiting_addcoin":
        feedback = handle_addcoin_input(chat_id, text)
        bot.send_message(chat_id, feedback)
        _user_states.pop(chat_id, None) 
        return

    is_command = text.startswith('/')
    
    # Danh sách các lệnh đã được định nghĩa handler
    # BỔ SUNG: Kiểm tra các nút bàn phím ảo (MENU 🎛️, Auto-Trading ⏱️)
    defined_commands = ['/start', '/menu', '/pnl', '/backtest', '/autotrading', '/add_coin', 
                        '/watchlist', '/setbase', '/setcap', '/setrisk', '/setleverage', 
                        '/taixiu', '/stopautotrading', '/delcoin', '/xoacoin', '/getdata', 
                        '/setbinancekeys', '/setokxkeys', '/exit', '/setexitapi', 
                        'MENU 🎛️', 'Auto-Trading ⏱️'] 

    # 2. XỬ LÝ CHẾ ĐỘ TRADING (LỆNH /SYMBOL)
    if mode == "trading":
        
        cmd_root = text.lower().split('@')[0]
        # Lệnh /SYMBOL: Bắt đầu bằng / và không phải các lệnh đã có handler
        is_symbol_command = is_command and len(text) > 2 and ' ' not in text and cmd_root not in [c.lower() for c in defined_commands] 
        
        if is_symbol_command:
            symbol = text[1:].upper()
            
            if '@' in symbol:
                symbol = symbol.split('@')[0] 
                
            if not symbol.endswith("USDT"): symbol += "USDT"
            
            analyze_and_send(chat_id, symbol, precomputed_res=None) 
            return # Thoát hàm sau khi xử lý lệnh /SYMBOL

    # 3. XỬ LÝ CHẾ ĐỘ TÀI XỈU (VỐN, MD5, KẾT QUẢ)
    if mode == "taixiu":
        global version_counter
        user_tx = user["taixiu"] 

        # Xử lý Vốn
        if user_tx.get("awaiting_balance", False):
            cleaned_text = text.replace(".", "").replace(",", "").replace(" ", "")
            if cleaned_text.isdigit() and len(cleaned_text) < 15:
                user_tx["balance"] = int(cleaned_text)
                if user_tx["balance"] >= 10_000_000:
                    user_tx["base_bet"] = random.choice([20000, 30000, 40000, 50000])
                else:
                    user_tx["base_bet"] = max(1000, user_tx["balance"] // 100)
                user_tx["bet"] = user_tx["base_bet"]
                user_tx["awaiting_balance"] = False
                bot.send_message(chat_id, f"💰 Vốn ban đầu: {user_tx['balance']:,} VND\n👉 Cược gốc: {user_tx['base_bet']:,} VND")
                return
            else:
                bot.send_message(chat_id, "⚠️ Vui lòng nhập số hợp lệ (VD: 50000)")
                return

        # Xử lý MD5
        if len(text) == 32 and all(c.isalnum() for c in text):
            # ... (Logic predict_md5 giữ nguyên) ...
            prediction, prob_tai, prob_xiu = predict_md5(text)
            ket_qua_ket = ""
            if max(prob_tai, prob_xiu) < 55:
                user_tx["last_prediction"] = np.random.choice(["Tài", "Xỉu"])
                ket_qua_ket = f"🔄 AI không chắc chắn ➔ Random: {user_tx['last_prediction']}"
            elif prob_tai > 80: user_tx["last_prediction"] = "Tài"; ket_qua_ket = "💪 KẾT TÀI!"
            elif prob_xiu > 80: user_tx["last_prediction"] = "Xỉu"; ket_qua_ket = "💪 KẾT XỈU!"
            else: user_tx["last_prediction"] = prediction; ket_qua_ket = "❌ Không kết!"
            
            user_tx["last_md5"] = text
            msg = (f"📢 MD5: {text}\n"
                   f"🔮 Xác suất: (Tài: {prob_tai:.2f}% | Xỉu: {prob_xiu:.2f}%)\n"
                   f"<b># ✅ Dự đoán: {user_tx['last_prediction']}</b>\n{ket_qua_ket}")
            bot.send_message(chat_id, msg, parse_mode="HTML")
            return # Thoát hàm sau khi xử lý MD5

        # Xử lý kết quả {a-b-c}
        if "{" in text and "}" in text and text.count('-') == 2:
            # ... (Logic parse_result_string và update history/balance giữ nguyên) ...
            actual_result, dice_numbers = parse_result_string(text)
            if not actual_result:
                bot.send_message(chat_id, "⚠️ Sai định dạng. Gửi: {a-b-c}")
                return

            user_tx.setdefault("history", []).append(actual_result)
            if len(user_tx["history"]) > 20: user_tx["history"].pop(0)
            
            dice_list = [int(d) for d in dice_numbers.split('-')]
            record_taixiu_result(chat_id, dice_list)
            
            trend = detect_trend(user_tx["history"])
            if any(k in trend for k in ["Cầu Bệt", "1-1", "2–2", "3–3"]):
                bot.send_message(chat_id, f"🚨 ALERT: {trend}")

            bet_suggestion = "⚪ Chưa có gợi ý cược"
            if user_tx.get("last_md5"):
                outcome = "Thắng" if user_tx["last_prediction"] == actual_result else "Thua"
                user_tx["win"] = user_tx.get("win", 0) + (1 if outcome == "Thắng" else 0)
                user_tx["lose"] = user_tx.get("lose", 0) + (1 if outcome == "Thua" else 0)

                bet_amount = user_tx.get("bet", 1000)
                if outcome == "Thắng":
                    user_tx["balance"] = user_tx.get("balance", 0) + int(bet_amount * 0.98)
                else:
                    user_tx["balance"] = user_tx.get("balance", 0) - bet_amount

                if user_tx["balance"] <= 0:
                    user_tx["balance"] = 0
                    bot.send_message(chat_id, "⚠️ Bạn đã thua hết vốn! Dùng /setbase [số tiền] để đặt lại vốn.")
                else:
                    outcomes = user_tx.get("outcome_history", [])
                    outcomes.append(outcome)
                    if len(outcomes) > 20: outcomes.pop(0)
                    user_tx["outcome_history"] = outcomes
                    next_bet = get_bet_suggestion(user_tx, outcome)
                    bet_suggestion = f"💡 Gợi ý cược tiếp: {next_bet:,} VND"

                save_result_async(user_tx.get("last_md5"), actual_result, dice_numbers, user.get("app_name", ""), outcome, chat_id)
                user_tx["last_md5"] = None
                
            message_text = (
                f"🎲 Kết quả: {actual_result} ({dice_numbers})\n"
                f"🔵 Thắng: {user_tx.get('win',0)} | 🔴 Thua: {user_tx.get('lose',0)}\n"
                f"💰 Vốn: {user_tx.get('balance',0):,} VND\n\n"
                f"📈 Cầu: {trend}\n\n"
                f"{bet_suggestion}"
            )
            bot.send_message(chat_id, message_text, parse_mode="HTML")
            return # Thoát hàm sau khi xử lý kết quả Tài Xỉu
        
    # 4. CATCH-ALL TRIỆT ĐỂ (Ngăn lỗi "...")
    # Nếu tin nhắn không phải lệnh và không khớp với bất kỳ luồng xử lý nào ở trên
    if not is_command:
        if mode == "trading":
            # Luôn gửi một tin nhắn phản hồi, ngay cả khi nó không hợp lệ.
            bot.send_message(chat_id, "ℹ️ Vui lòng nhập lệnh `/SYMBOL` (ví dụ: `/BTCUSDT`) hoặc bấm nút **MENU 🎛️** để chọn chức năng khác.", parse_mode="Markdown")
            return
            
        if mode == "taixiu":
            # Luôn gửi một tin nhắn phản hồi.
            bot.send_message(chat_id, "ℹ️ Vui lòng nhập **MD5** (32 ký tự) hoặc kết quả `{a-b-c}`. Bấm **MENU 🎛️** để chọn chế độ Trading.", parse_mode="Markdown")
            return
        
        # Nếu không ở chế độ nào, hoặc đã bị lỗi logic trước đó:
        bot.send_message(chat_id, "ℹ️ Bot không hiểu lệnh này. Vui lòng chọn chức năng từ bàn phím hoặc gõ `/start`.")
        return
    
    # Nếu là lệnh (is_command) nhưng không được handle ở các @bot.message_handler khác (ví dụ: /start, /pnl)
    # thì nó sẽ được xử lý bởi handler tương ứng. Ta chỉ cần return để kết thúc handler chung này.
    return

@bot.message_handler(func=lambda message: message.text == "MENU 🎛️")
def handle_menu_button(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "⬇️​ MENU CHÍNH:", reply_markup=build_main_menu_markup())

@bot.message_handler(func=lambda message: message.text == "Auto-Trading ⏱️")
def handle_auto_trading_button(message):
    chat_id = message.chat.id
    handle_autotrading_command(message)

# =======================
# 9. CÁC LUỒNG TOÀN CỤC (GLOBAL THREADS) 
# =======================

def auto_save_thread(interval=300):
    """Luồng 1: Tự động lưu user_data mỗi 5 phút."""
    print("🚀 Luồng Auto-Save (Lưu dữ liệu) đã khởi chạy...")
    while True:
        time.sleep(interval)
        save_user_data()

def websocket_price_monitor_thread():
    """
    Luồng 4: (FIX V18) Chạy WebSocket để nhận Mark Price
    VÀ TÍNH TOÁN High/Low real-time cho Luồng Monitor.
    """
    url = "wss://fstream.binance.com/ws/!markPrice@arr@1s"

    def on_message(ws, message):
        try:
            data = json.loads(message)
            
            global REALTIME_PRICE_CACHE, REALTIME_PRICE_LOCK
            with REALTIME_PRICE_LOCK:
                for ticker in data:
                    symbol = ticker['s']
                    price = float(ticker['p'])
                    
                    if symbol not in REALTIME_PRICE_CACHE:
                        # Lần đầu tiên thấy Symbol này, khởi tạo
                        REALTIME_PRICE_CACHE[symbol] = {"high": price, "low": price, "close": price}
                    else:
                        # (FIX V18) Cập nhật High/Low liên tục
                        REALTIME_PRICE_CACHE[symbol]["high"] = max(REALTIME_PRICE_CACHE[symbol].get("high", price), price)
                        REALTIME_PRICE_CACHE[symbol]["low"] = min(REALTIME_PRICE_CACHE[symbol].get("low", price), price)
                        REALTIME_PRICE_CACHE[symbol]["close"] = price # Luôn cập nhật giá Close
                    
        except Exception as e:
            print(f"Lỗi xử lý WebSocket message: {e}")

    def on_error(ws, error):
        print(f"Lỗi WebSocket (Mark Price): {error}")

    def on_close(ws, close_status_code, close_msg):
        print("--- WebSocket (Mark Price) đã đóng ---")

    def on_open(ws):
        print("🚀 Luồng WebSocket (Giá Real-time) đã kết nối...")

    # Khởi chạy WebSocket vĩnh viễn
    while True:
        try:
            ws = websocket.WebSocketApp(url,
                                      on_message=on_message,
                                      on_error=on_error,
                                      on_close=on_close,
                                      on_open=on_open)
            # FIX LỖI SSL (TỪ LẦN TRƯỚC)
            ssl_opts = {"cert_reqs": ssl.CERT_NONE}
            ws.run_forever(ping_interval=60, ping_timeout=10, sslopt=ssl_opts)
        except Exception as e:
            print(f"Lỗi WebSocket run_forever: {e}. Đang thử kết nối lại sau 10s...")
            time.sleep(10)

def global_signal_monitor_thread(check_interval=5): # (Sửa check_interval=5)
    """
    Luồng 2: (FIX CỦA BẠN)
     1. Xóa bỏ logic API, BẮT BUỘC TẤT CẢ USER dùng chung Price Watcher (WebSocket).
     2. Giữ nguyên logic "Gửi thông báo TRƯỚC, cập nhật status SAU".
    """
    print("🚀 Luồng Global Signal Monitor (FIX: ALL USERS USE PRICE WATCHER) đã khởi chạy...")
    
    # Cache client không còn cần thiết nữa vì chúng ta không check API
    # client_cache = {} 
    
    global REALTIME_PRICE_CACHE, REALTIME_PRICE_LOCK
    
    while True:
        try:
            all_open_signals = {} 
            all_users_data = dict(user_data) 
            
            # 1. Thu thập TẤT CẢ tín hiệu đang mở
            symbols_to_check = set()
            
            for chat_id, data in all_users_data.items():
                signals = data.get("trading", {}).get("signals", [])
                for sig in signals:
                    if sig.get("status") == "open":
                        symbol = sig["symbol"]
                        # (FIX) Chỉ quét sàn BINANCE (vì WebSocket của chúng ta là Binance)
                        exchange = data.get("trading", {}).get("exchange", "BINANCE")
                        if exchange == "BINANCE":
                             all_open_signals.setdefault(symbol, []).append((chat_id, sig.get("id", "")))
                             symbols_to_check.add(symbol)
                        
            if not all_open_signals:
                time.sleep(check_interval)
                continue

            # 2. Lấy giá High/Low/Close từ CACHE WEBSOCKET (và RESET H/L)
            prices_cache = {}
            with REALTIME_PRICE_LOCK:
                for symbol in symbols_to_check:
                    if symbol in REALTIME_PRICE_CACHE:
                        prices_cache[symbol] = REALTIME_PRICE_CACHE[symbol].copy()
                        current_close = REALTIME_PRICE_CACHE[symbol]["close"]
                        REALTIME_PRICE_CACHE[symbol]["high"] = current_close
                        REALTIME_PRICE_CACHE[symbol]["low"] = current_close

            # 3. Xử lý logic SL/TP/Exit
            for symbol, signals_list in all_open_signals.items():
                
                candle_data = prices_cache.get(symbol)
                if candle_data is None: 
                    price = get_market_price(symbol)
                    if price is None: continue 
                    current_high = price
                    current_low = price
                else:
                    price = candle_data["close"]
                    current_high = candle_data["high"] 
                    current_low = candle_data["low"]   
                
                precision = get_symbol_precision(symbol)
                tick_size = precision['tickSize'] 
                price_round_precision = int(-math.log10(tick_size)) if tick_size > 0 else 8
                
                for chat_id, sig_id in signals_list:
                    if not sig_id: continue 
                    
                    closed_by_order_id = None # Khởi tạo lại
                    closed_order_type = None  # Khởi tạo lại
                    pnl_percent = 0.0         # Khởi tạo PNL
                    
                    try:
                        # <<< FIX LỖI CONCURRENCY: Luôn lấy sig TƯƠI TỪ user_data >>>
                        sig = next((s for s in user_data[chat_id]["trading"]["signals"] if s.get("id") == sig_id), None)
                        if sig is None or sig.get("status") != "open":
                             continue 
                        
                        # =============================================================
                        # <<< FIX CỦA BẠN: LOGIC CHUNG CHO TẤT CẢ USER (PRICE WATCHER) >>>
                        # (Toàn bộ khối logic check API đã bị xóa bỏ)
                        
                        entry = sig["entry"]
                        sl_price = sig["sl"]
                        tp1 = sig.get('tp1')
                        tp2 = sig.get('tp2')
                        tp3 = sig.get('tp3')
                        
                        # Cập nhật High/Low của LỆNH (không phải của 5s)
                        sig["high_price"] = max(sig.get("high_price", entry), current_high) 
                        sig["low_price"] = min(sig.get("low_price", entry), current_low)   
                        
                        tracked_high = sig["high_price"] # High/Low từ lúc mở lệnh
                        tracked_low = sig["low_price"]
                        
                        is_long = entry < sig.get('tp1', entry + 1)
                        
                        if is_long:
                            if tracked_low <= sl_price: closed_order_type = 'sl' 
                            elif tp3 and tracked_high >= tp3: closed_order_type = 'tp3'
                            elif tp2 and tracked_high >= tp2: closed_order_type = 'tp2'
                            elif tp1 and tracked_high >= tp1: closed_order_type = 'tp1'
                        else: # Short
                            if tracked_high >= sl_price: closed_order_type = 'sl' 
                            elif tp3 and tracked_low <= tp3: closed_order_type = 'tp3'
                            elif tp2 and tracked_low <= tp2: closed_order_type = 'tp2'
                            elif tp1 and tracked_low <= tp1: closed_order_type = 'tp1'

                        if closed_order_type is None:
                            # Gửi PNL Ước tính (Logic này giữ nguyên)
                            if is_long: pnl_percent_estimate = ((price - entry) / entry) * 100
                            else: pnl_percent_estimate = ((entry - price) / entry) * 100
                            last_reported_pnl = sig.setdefault("last_reported_pnl", 0.0)
                            if abs(pnl_percent_estimate - last_reported_pnl) >= 1.0: # (Chỉ báo mỗi 1%)
                                sig["last_reported_pnl"] = pnl_percent_estimate
                                try:
                                    bot.send_message(chat_id, f"📈 Cập nhật {symbol} (ID: {sig_id}):\n"
                                                            f"**PnL Ước tính:** **{pnl_percent_estimate:,.2f}%**\n"
                                                            f"Giá hiện tại: **{price:,.{price_round_precision}f}**", parse_mode="Markdown")
                                except Exception: pass
                        else:
                            # Đã đóng bằng Price Watcher
                            closed_by_order_id = "Price_Watcher" 
                        
                        # <<< KẾT THÚC FIX >>>
                        # =============================================================
                        
                    except (KeyError, StopIteration, TypeError) as e:
                        print(f"Lỗi khi xử lý tín hiệu {sig_id} cho user {chat_id}: {e}")
                        continue 
                    
                    
                    # --- D. XỬ LÝ LỆNH ĐÃ ĐÓNG (GIỮ NGUYÊN FIX "KHÔNG BÁO") ---
                    
                    if closed_by_order_id is not None:
                        try:
                            close_price = price
                            
                            # (FIX) Kiểm tra xem user có API không, chỉ để HỦY LỆNH DƯ
                            user = all_users_data[chat_id]
                            has_api_keys = user["trading"].get("api_key") and user["trading"].get("secret_key")
                            is_binance = user["trading"]["exchange"] == "BINANCE"
                            
                            if has_api_keys and is_binance:
                                # Chỉ dùng API để HỦY LỆNH DƯ, không dùng để check
                                try:
                                    client, _ = get_user_exchange_client(user)
                                    sl_order_id = sig.get("sl_order_id")
                                    tp_order_id = sig.get("tp_order_id")
                                    
                                    if closed_order_type == 'sl' and tp_order_id:
                                        client.futures_cancel_order(symbol=symbol, orderId=tp_order_id)
                                    elif closed_order_type.startswith('tp') and sl_order_id:
                                        client.futures_cancel_order(symbol=symbol, orderId=sl_order_id)
                                except Exception as e:
                                    print(f"DEBUG: Lỗi hủy lệnh còn lại (API User): {e}")

                            # 2. Tính PnL
                            entry = sig["entry"]
                            is_long = entry < sig.get('tp1', entry + 1)
                            if is_long:
                                pnl_percent = ((close_price - entry) / entry) * 100
                            else: 
                                pnl_percent = ((entry - close_price) / entry) * 100
                            
                            # 3. Xác định loại sự kiện và Emoji
                            ev_type = closed_order_type.upper()
                            emoji = "🟢" if "TP" in ev_type else "❌"
                            
                            # (FIX) Logic Trailing SL (dựa trên API) đã bị xóa, 
                            # nên ta không cần check "sl_profit" ở đây nữa.
                            
                            order_side = "LONG" if is_long else "SHORT"
                            
                            # 4. GỬI TIN NHẮN (QUAN TRỌNG)
                            bot.send_message(chat_id, f"{emoji} {symbol} (ID: {sig_id}) đã đóng lệnh \"{order_side}\" tại {close_price:,.{price_round_precision}f} ({ev_type})\n"
                                                     f"💰 PnL: <b>{pnl_percent:.2f}%</b>", parse_mode="HTML")
                        
                            # 5. CẬP NHẬT TRẠNG THÁI (CHỈ CHẠY NẾU GỬI THÀNH CÔNG)
                            sig["status"] = closed_order_type
                            sig["pnl_percent"] = pnl_percent
                            
                            pnl_counter = user_data[chat_id]["trading"]["pnl_counts"]
                            pnl_counter[closed_order_type] = pnl_counter.get(closed_order_type, 0) + 1 
                            
                            ev = {"type": ev_type, "price": close_price, "pnl": pnl_percent, "time": datetime.now(TZ).isoformat()}
                            sig["events"].append(ev)

                        except Exception as e:
                            print(f"Lỗi gửi thông báo đóng lệnh cho {chat_id}: {e}. Sẽ thử lại...")
                            if 'chat not found' in str(e) or 'user is deactivated' in str(e) or 'Forbidden: bot was kicked' in str(e):
                                print(f"-> User {chat_id} đã chặn bot. Đóng vĩnh viễn tín hiệu {sig_id}.")
                                sig["status"] = "closed_user_blocked"
                                # (Logic tính PNL lỗi giữ nguyên)
            
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong Global Signal Monitor: {e}")
            
        save_user_data() # (Lưu ý: Luồng Auto-Save vẫn đang chạy)
        time.sleep(check_interval)

# =======================
# 10. KHỞI CHẠY BOT
# =======================

# <<< HÀM BỊ THIẾU (LUỒNG 3) >>>
def global_auto_trader_thread(check_interval=60):
    """
    Luồng 3: (FIX CỦA BẠN - Producer)
    Chỉ quét tín hiệu và "NÉM" job vào MESSAGE_QUEUE.
    Không gọi analyze_and_send() và không sleep(1) nữa.
    """
    print("🚀 Luồng Auto-Trader (Producer - V18.3 Queue) đã khởi chạy...")
    
    last_scan_times = {}
    
    while True:
        try:
            current_time = time.time()
            all_users_data = dict(user_data) 
            analysis_cache = {} # Cache phân tích (giữ nguyên)

            for chat_id_str, data in all_users_data.items():
                chat_id = int(chat_id_str)
                user = ensure_user_data_structure(chat_id)
                
                intervals_to_scan = user["trading"].get("auto_trade_intervals", [])
                watchlist = user["trading"].get("watchlist", [])
                
                if not intervals_to_scan or not watchlist:
                    continue 

                if chat_id not in last_scan_times:
                    last_scan_times[chat_id] = {}

                for interval_sec in intervals_to_scan:
                    
                    if interval_sec not in last_scan_times[chat_id]:
                        last_scan_times[chat_id][interval_sec] = {}
                        
                    timeframe_cache = last_scan_times[chat_id][interval_sec]
                    last_global_scan = timeframe_cache.get("GLOBAL_TIMER", 0)
                    if current_time - last_global_scan < interval_sec:
                        continue 
                        
                    timeframe_cache["GLOBAL_TIMER"] = current_time
                    
                    intervals_map = {300: "5M", 900: "15M", 1800: "30M", 3600: "1H", 14400: "4H"}
                    timeframe_origin_name = intervals_map.get(interval_sec, f"{int(interval_sec/60)}M")

                    print(f"--- Auto-Trader: Đang quét {timeframe_origin_name} cho User {chat_id}...")

                    for symbol in watchlist:
                        try:
                            res = analysis_cache.get(symbol)
                            if res is None:
                                print(f"    [Cache]: {symbol} chưa có, đang phân tích...")
                                exchange = user.get("trading", {}).get("exchange", "BINANCE")
                                res = decide_levels(symbol, exchange=exchange)
                                analysis_cache[symbol] = res 
                            
                            if not res:
                                print(f"    [SKIP AUTO]: {symbol} ({timeframe_origin_name}) - Lỗi lấy dữ liệu Klines/Giá.")
                                continue

                            new_trend = res['trend']
                            if not ("Tăng" in new_trend or "Giảm" in new_trend):
                                # (Bỏ qua log Sideways cho đỡ rối)
                                continue
                                
                            last_signal_time = timeframe_cache.get(symbol, 0)
                            report_interval_hours = user["trading"].get("report_interval", MIN_REPORT_INTERVAL_HOURS)
                            
                            if (current_time - last_signal_time) < (report_interval_hours * 3600):
                                print(f"    [FILTERED]: {symbol} ({timeframe_origin_name}) đã báo gần đây. Bỏ qua.")
                                continue
                                
                            # =============================================================
                            # <<< FIX CỦA BẠN: NÉM VÀO HÀNG ĐỢI (QUEUE) >>>
                            print(f"    ✅ TÍN HIỆU MỚI: {symbol} ({timeframe_origin_name}) -> {new_trend}. Đang ném vào Queue...")
                            
                            timeframe_cache[symbol] = current_time
                            
                            # Tạo "job"
                            job = (chat_id, symbol, res, timeframe_origin_name)
                            # Ném vào hàng đợi
                            MESSAGE_QUEUE.put(job)
                            
                            # (ĐÃ XÓA time.sleep(1))
                            # =============================================================

                        except Exception as e:
                            print(f"    ❌ Lỗi Auto-Trader (Symbol: {symbol}): {e}")
                            
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong Global Auto-Trader: {e}")
            
        time.sleep(check_interval)
# <<< KẾT THÚC HÀM BỊ THIẾU >>>

def message_sending_worker():
    """
    Luồng 5: (FIX CỦA BẠN - Consumer)
    Nhân viên gửi tin nhắn. Lấy job từ MESSAGE_QUEUE và thực thi analyze_and_send.
    """
    while True:
        try:
            # Lấy job từ hàng đợi (sẽ block cho đến khi có job)
            job = MESSAGE_QUEUE.get() 
            
            chat_id, symbol, res, timeframe_name = job
            
            print(f"    [Worker]: Đang xử lý {symbol} cho {chat_id}...")
            
            # Gọi hàm gửi tin (hàm này đã bao gồm logic anti-spam của Telegram)
            analyze_and_send(chat_id, symbol, precomputed_res=res, timeframe_origin=timeframe_name)
            
            # Báo cho Queue biết là đã xử lý xong job này
            MESSAGE_QUEUE.task_done()
            
            # Thêm 1 khoảng nghỉ nhỏ (0.2s) để tránh bị Rate Limit quá nhanh
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong Message Worker: {e}")
            # Nếu lỗi, cũng báo là đã xong (để không bị kẹt queue)
            try: MESSAGE_QUEUE.task_done()
            except: pass

def start_message_workers(num_workers=10):
    """Khởi chạy 10 nhân viên (threads) để gửi tin song song."""
    print(f"🚀 Khởi chạy {num_workers} Message Workers (Luồng 5)...")
    for i in range(num_workers):
        threading.Thread(target=message_sending_worker, daemon=True).start()


# =======================
# 10. KHỞI CHẠY BOT
# =======================
if __name__ == "__main__":
    print("🤖 Bot AI (Trading + Tài Xỉu) V17.1 - Hoàn chỉnh...")
    
    # 1. Tải dữ liệu cũ
    load_user_data()
    
    # 2. Đăng ký lưu dữ liệu khi tắt bot
    atexit.register(save_user_data)
    
    # 3. Khởi chạy các luồng toàn cục
    # Luồng 1: Auto-Save
    threading.Thread(target=auto_save_thread, args=(300,), daemon=True).start()

    # Luồng 4: WebSocket (Giá)
    threading.Thread(target=websocket_price_monitor_thread, daemon=True).start()

    # Luồng 2: Quét SL/TP (Đã fix "như nhau" & chạy 5s/lần)
    threading.Thread(target=global_signal_monitor_thread, args=(5,), daemon=True).start()

    # Luồng 3: Quét tín hiệu (Producer - Đã fix "queue")
    threading.Thread(target=global_auto_trader_thread, args=(60,), daemon=True).start()

    # (FIX MỚI) Luồng 5: Khởi chạy 10 Nhân viên Gửi tin (Consumers)
    start_message_workers(num_workers=10)

    # 4. Khởi chạy Bot Polling (Luồng chính)
    print("🚀 Bot Polling (Nhận tin nhắn) đã khởi chạy...")
    bot.infinity_polling(timeout=60, long_polling_timeout=90)