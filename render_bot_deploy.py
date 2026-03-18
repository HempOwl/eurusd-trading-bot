import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import aiohttp
import time
import pytz
from typing import List, Dict, Optional, Set, Any
import threading
import sqlite3
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

# ========== НАСТРОЙКА ЛОГИРОВАНИЯ ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ==========
app = Flask(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
TWELVE_API_KEY = os.environ.get('TWELVE_API_KEY')
ADMIN_CHAT_ID = os.environ.get('ADMIN_CHAT_ID')

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не задан!")
if not TWELVE_API_KEY:
    logger.error("❌ TWELVE_API_KEY не задан!")
if not ADMIN_CHAT_ID:
    logger.warning("⚠️ ADMIN_CHAT_ID не задан. Уведомления админу отключены.")

# ========== ОСНОВНЫЕ ВАЛЮТНЫЕ ПАРЫ (5 шт) ==========
SYMBOLS = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']

# ========== ПУТЬ К БАЗЕ ДАННЫХ ==========
DB_PATH = os.path.join(os.path.dirname(__file__), "bot.db")

# ========== ГЛОБАЛЬНЫЙ СЧЁТЧИК ОШИБОК API ==========
api_errors = {}
API_ERROR_THRESHOLD = 5
API_ERROR_WINDOW = 60 * 60  # 1 час

# ========== ФУНКЦИЯ УВЕДОМЛЕНИЯ АДМИНИСТРАТОРА ==========
async def notify_admin(bot: Bot, message: str):
    if ADMIN_CHAT_ID:
        try:
            await bot.send_message(ADMIN_CHAT_ID, f"⚠️ Админ-уведомление\n{message}")
            logger.info("📨 Уведомление администратору отправлено")
        except Exception as e:
            logger.error(f"❌ Не удалось отправить уведомление админу: {e}")

# ========== СИНХРОННАЯ ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ (С НОВЫМИ ТАБЛИЦАМИ) ==========
def init_db_sync():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS subscribers (
            user_id INTEGER PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            direction TEXT NOT NULL,
            tp REAL,
            sl REAL,
            spread REAL,
            result TEXT,
            exit_price REAL,
            exit_time INTEGER,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, datetime)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS indicators_cache (
            symbol TEXT PRIMARY KEY,
            timestamp INTEGER NOT NULL,
            indicators TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS ml_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            features TEXT NOT NULL,
            result INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result) WHERE result IS NULL')
    conn.commit()
    conn.close()
    logger.info("✅ База данных инициализирована (с новыми таблицами)")

# ========== СИНХРОННЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С БД ==========
def get_subscribers_sync() -> Set[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT user_id FROM subscribers')
    rows = c.fetchall()
    conn.close()
    return {row[0] for row in rows}

def add_subscriber_sync(user_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO subscribers (user_id) VALUES (?)', (user_id,))
    added = c.rowcount > 0
    conn.commit()
    conn.close()
    return added

def remove_subscriber_sync(user_id: int) -> bool:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM subscribers WHERE user_id = ?', (user_id,))
    removed = c.rowcount > 0
    conn.commit()
    conn.close()
    return removed

def add_signal_sync(signal: Dict, features: Optional[List] = None, spread: Optional[float] = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO signals (timestamp, symbol, price, direction, tp, sl, spread, result, exit_price, exit_time, features)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        signal['timestamp'],
        signal['symbol'],
        signal['price'],
        signal['direction'],
        signal.get('tp'),
        signal.get('sl'),
        spread,
        signal.get('result'),
        signal.get('exit_price'),
        signal.get('exit_time'),
        json.dumps(features) if features else None
    ))
    conn.commit()
    conn.close()

def update_signal_results_sync(symbol: str, current_price: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, timestamp, price, direction, tp, sl, features
        FROM signals
        WHERE symbol = ? AND result IS NULL
    ''', (symbol,))
    rows = c.fetchall()
    now = int(time.time())
    updated = False
    for row in rows:
        signal_id, ts, entry, direction, tp, sl, features_json = row
        if now - ts > 3600:
            c.execute('''
                UPDATE signals SET result = 'timeout', exit_price = ?, exit_time = ?
                WHERE id = ?
            ''', (current_price, now, signal_id))
            updated = True
            continue
        if direction == 'buy':
            if tp and current_price >= tp:
                c.execute('''
                    UPDATE signals SET result = 'profit', exit_price = ?, exit_time = ?
                    WHERE id = ?
                ''', (tp, now, signal_id))
                updated = True
                result = 1
            elif sl and current_price <= sl:
                c.execute('''
                    UPDATE signals SET result = 'loss', exit_price = ?, exit_time = ?
                    WHERE id = ?
                ''', (sl, now, signal_id))
                updated = True
                result = 0
            else:
                continue
        else:
            if tp and current_price <= tp:
                c.execute('''
                    UPDATE signals SET result = 'profit', exit_price = ?, exit_time = ?
                    WHERE id = ?
                ''', (tp, now, signal_id))
                updated = True
                result = 1
            elif sl and current_price >= sl:
                c.execute('''
                    UPDATE signals SET result = 'loss', exit_price = ?, exit_time = ?
                    WHERE id = ?
                ''', (sl, now, signal_id))
                updated = True
                result = 0
            else:
                continue
        # Если сделка закрылась, сохраняем в ml_training_data
        if features_json:
            try:
                features = json.loads(features_json)
                c.execute('INSERT INTO ml_training_data (features, result) VALUES (?, ?)',
                          (json.dumps(features), result))
            except:
                pass
    if updated:
        conn.commit()
    conn.close()

def get_summary_sync(symbol: Optional[str] = None) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if symbol:
        c.execute('SELECT COUNT(*) FROM signals WHERE symbol = ?', (symbol,))
    else:
        c.execute('SELECT COUNT(*) FROM signals')
    total = c.fetchone()[0]
    if total == 0:
        conn.close()
        return {
            'total': 0, 'profit': 0, 'loss': 0, 'timeout': 0, 'unknown': 0,
            'win_rate': 0, 'avg_profit': 0, 'avg_loss': 0,
            'total_profit_pips': 0, 'total_loss_pips': 0
        }
    if symbol:
        c.execute('''
            SELECT 
                COUNT(*) FILTER (WHERE result = 'profit'),
                COUNT(*) FILTER (WHERE result = 'loss'),
                COUNT(*) FILTER (WHERE result = 'timeout'),
                COUNT(*) FILTER (WHERE result IS NULL)
            FROM signals WHERE symbol = ?
        ''', (symbol,))
    else:
        c.execute('''
            SELECT 
                COUNT(*) FILTER (WHERE result = 'profit'),
                COUNT(*) FILTER (WHERE result = 'loss'),
                COUNT(*) FILTER (WHERE result = 'timeout'),
                COUNT(*) FILTER (WHERE result IS NULL)
            FROM signals
        ''')
    profit, loss, timeout, unknown = c.fetchone()
    if symbol:
        c.execute('''
            SELECT 
                COALESCE(SUM(
                    CASE 
                        WHEN result = 'profit' AND direction = 'buy' THEN (exit_price - price) * 10000
                        WHEN result = 'profit' AND direction = 'sell' THEN (price - exit_price) * 10000
                        ELSE 0
                    END
                ), 0),
                COALESCE(SUM(
                    CASE 
                        WHEN result = 'loss' AND direction = 'buy' THEN ABS((exit_price - price) * 10000)
                        WHEN result = 'loss' AND direction = 'sell' THEN ABS((price - exit_price) * 10000)
                        ELSE 0
                    END
                ), 0)
            FROM signals
            WHERE result IN ('profit', 'loss') AND symbol = ?
        ''', (symbol,))
    else:
        c.execute('''
            SELECT 
                COALESCE(SUM(
                    CASE 
                        WHEN result = 'profit' AND direction = 'buy' THEN (exit_price - price) * 10000
                        WHEN result = 'profit' AND direction = 'sell' THEN (price - exit_price) * 10000
                        ELSE 0
                    END
                ), 0),
                COALESCE(SUM(
                    CASE 
                        WHEN result = 'loss' AND direction = 'buy' THEN ABS((exit_price - price) * 10000)
                        WHEN result = 'loss' AND direction = 'sell' THEN ABS((price - exit_price) * 10000)
                        ELSE 0
                    END
                ), 0)
            FROM signals
            WHERE result IN ('profit', 'loss')
        ''')
    total_profit_pips, total_loss_pips = c.fetchone()
    conn.close()
    win_rate = (profit / (profit + loss) * 100) if (profit + loss) > 0 else 0
    avg_profit = total_profit_pips / profit if profit > 0 else 0
    avg_loss = total_loss_pips / loss if loss > 0 else 0
    return {
        'total': total,
        'profit': profit,
        'loss': loss,
        'timeout': timeout,
        'unknown': unknown,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'total_profit_pips': total_profit_pips,
        'total_loss_pips': total_loss_pips
    }

def get_cached_candles_sync(symbol: str, needed_bars: int = 200) -> Optional[List[Dict]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT datetime, open, high, low, close, volume
        FROM candles
        WHERE symbol = ?
        ORDER BY datetime DESC
        LIMIT ?
    ''', (symbol, needed_bars))
    rows = c.fetchall()
    conn.close()
    if len(rows) < needed_bars:
        return None
    candles = []
    for row in reversed(rows):
        dt, o, h, l, c_val, v = row
        candles.append({
            'datetime': dt,
            'open': o,
            'high': h,
            'low': l,
            'close': c_val,
            'volume': v
        })
    return candles

def save_candles_to_cache_sync(symbol: str, candles: List[Dict]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for candle in candles:
        c.execute('''
            INSERT OR REPLACE INTO candles (symbol, datetime, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            candle['datetime'],
            candle['open'],
            candle['high'],
            candle['low'],
            candle['close'],
            candle.get('volume', 0)
        ))
    conn.commit()
    conn.close()

# ========== НОВЫЕ ФУНКЦИИ ДЛЯ КЭШИРОВАНИЯ ИНДИКАТОРОВ ==========
def save_indicators_cache_sync(symbol: str, indicators: Dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO indicators_cache (symbol, timestamp, indicators)
        VALUES (?, ?, ?)
    ''', (symbol, int(time.time()), json.dumps(indicators)))
    conn.commit()
    conn.close()

def load_indicators_cache_sync(symbol: str, max_age: int = 60) -> Optional[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT timestamp, indicators FROM indicators_cache WHERE symbol = ?', (symbol,))
    row = c.fetchone()
    conn.close()
    if row:
        ts, ind_json = row
        if time.time() - ts < max_age:
            return json.loads(ind_json)
    return None

# ========== ФУНКЦИЯ ДЛЯ ПОЛУЧЕНИЯ СПРЕДА ==========
async def fetch_spread(symbol: str, api_key: str) -> Optional[float]:
    url = "https://api.twelvedata.com/spread"
    params = {'symbol': symbol, 'apikey': api_key}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        return float(data[0].get('spread', 0))
                    elif isinstance(data, dict):
                        return float(data.get('spread', 0))
                return None
    except Exception as e:
        logger.error(f"Error fetching spread for {symbol}: {e}")
        return None

# ========== ФУНКЦИЯ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ ==========
async def train_model_periodically():
    """Запускается в отдельном фоновом потоке, раз в сутки обучает модель."""
    while True:
        await asyncio.sleep(86400)  # 24 часа
        await train_model()

async def train_model():
    def _train():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT features, result FROM ml_training_data')
        rows = c.fetchall()
        conn.close()
        if len(rows) < 100:
            logger.info("Not enough training data yet")
            return
        X = [json.loads(row[0]) for row in rows]
        y = [row[1] for row in rows]
        ml_gen.train(X, y)
    await asyncio.to_thread(_train)

# ========== ГЛОБАЛЬНОЕ МНОЖЕСТВО ПОДПИСЧИКОВ И ЗАМОК ==========
subscribers = set()
subscribers_lock = threading.Lock()

def load_subscribers_from_db():
    global subscribers
    db_subs = get_subscribers_sync()
    with subscribers_lock:
        subscribers.clear()
        subscribers.update(db_subs)
    logger.info(f"👥 Загружено {len(subscribers)} подписчиков из БД")

def is_subscriber(user_id: int) -> bool:
    with subscribers_lock:
        return user_id in subscribers

def add_subscriber_mem(user_id: int) -> bool:
    added = add_subscriber_sync(user_id)
    if added:
        with subscribers_lock:
            subscribers.add(user_id)
    return added

def remove_subscriber_mem(user_id: int) -> bool:
    removed = remove_subscriber_sync(user_id)
    if removed:
        with subscribers_lock:
            subscribers.discard(user_id)
    return removed

# ========== КЛАСС ДЛЯ МАШИННОГО ОБУЧЕНИЯ ==========
class MLSignalGenerator:
    def __init__(self, model_path='model.pkl'):
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"ML model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            logger.info("New ML model created")

    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)

    def prepare_features(self, ind: Dict) -> List[float]:
        features = [
            ind.get('rsi', 50),
            ind.get('macd', 0),
            (ind.get('price', 1) - ind.get('bb_lower', 0)) / (ind.get('bb_upper', 2) - ind.get('bb_lower', 1))
            if (ind.get('bb_upper', 2) - ind.get('bb_lower', 1)) != 0 else 0.5,
            ind.get('sma', {}).get(5, 0),
            ind.get('sma', {}).get(10, 0),
            ind.get('sma', {}).get(20, 0),
            ind.get('ema', {}).get(5, 0),
            ind.get('ema', {}).get(10, 0),
            ind.get('ema', {}).get(20, 0),
            ind.get('atr', 0),
            ind.get('obv', 0),
            ind.get('stoch_k', 50),
            ind.get('adx', 0),
            ind.get('plus_di', 0),
            ind.get('minus_di', 0),
        ]
        return features

    def predict(self, ind: Dict) -> float:
        if self.model is None:
            return 0.5
        try:
            if not hasattr(self.model, 'estimators_'):
                return 0.5
            features = self.prepare_features(ind)
            X = np.array(features).reshape(1, -1)
            proba = self.model.predict_proba(X)[0]
            return proba[1] if len(proba) > 1 else proba[0]
        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return 0.5

    def train(self, X: List[List[float]], y: List[int]):
        self.model.fit(X, y)
        self.save_model()
        logger.info("ML model trained and saved")

ml_gen = MLSignalGenerator()

# ========== ЭКОНОМИЧЕСКИЙ КАЛЕНДАРЬ ==========
class EconomicCalendar:
    def __init__(self, cache_minutes: int = 60):
        self.cache_minutes = cache_minutes
        self.cache = {}
        self.last_update = None
        self.base_url = "https://api.finnworlds.com/v1/macroeconomic-calendar"
        self.symbol_to_country = {
            'EUR/USD': ['DE', 'FR', 'IT', 'ES', 'EU'],
            'GBP/USD': ['GB'],
            'USD/JPY': ['JP'],
            'AUD/USD': ['AU'],
            'USD/CAD': ['CA'],
        }
        self.important_levels = ['High', 'Medium']

    async def _fetch_events(self, date_from: str, date_to: str) -> List[Dict]:
        params = {"from": date_from, "to": date_to}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = []
                        for item in data:
                            event = {
                                'date': item.get('date'),
                                'time': item.get('time'),
                                'country': item.get('country_code'),
                                'event': item.get('event_name'),
                                'impact': item.get('impact'),
                                'currency': item.get('currency')
                            }
                            events.append(event)
                        return events
                    else:
                        logger.error(f"Ошибка календаря: статус {resp.status}")
                        return []
        except asyncio.TimeoutError:
            logger.error("Таймаут при запросе к календарю")
            return []
        except Exception as e:
            logger.error(f"Ошибка получения календаря: {e}")
            return []

    def _should_update(self) -> bool:
        if not self.last_update:
            return True
        return (datetime.now() - self.last_update) > timedelta(minutes=self.cache_minutes)

    async def update_cache(self):
        if not self._should_update():
            return
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        events = await self._fetch_events(today, tomorrow)
        self.cache.clear()
        for event in events:
            if not event.get('date') or not event.get('time'):
                continue
            try:
                dt_str = f"{event['date']} {event['time']}"
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                dt = pytz.UTC.localize(dt)
                key = dt.strftime('%Y-%m-%d %H:%M')
                self.cache[key] = event
            except:
                continue
        self.last_update = datetime.now()
        logger.info(f"✅ Календарь Finnworlds обновлён: {len(self.cache)} событий")

    async def get_upcoming_events(self, minutes_ahead: int = 30) -> List[Dict]:
        await self.update_cache()
        now = datetime.now(pytz.UTC)
        time_limit = now + timedelta(minutes=minutes_ahead)
        upcoming = []
        for dt_str, event in self.cache.items():
            try:
                event_dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M')
                event_dt = pytz.UTC.localize(event_dt)
                if now <= event_dt <= time_limit:
                    upcoming.append(event)
            except:
                continue
        return upcoming

    async def check_symbol_risk(self, symbol: str, minutes_window: int = 5) -> tuple:
        countries = self.symbol_to_country.get(symbol, [])
        if not countries:
            return False, []
        upcoming = await self.get_upcoming_events(minutes_ahead=minutes_window * 2)
        risk_events = []
        for event in upcoming:
            if event['country'] in countries and event.get('impact') in self.important_levels:
                risk_events.append(event)
        return len(risk_events) > 0, risk_events

economic_calendar = EconomicCalendar()

# ========== ХРАНИЛИЩЕ ДАННЫХ ДЛЯ КАЖДОЙ ПАРЫ (ДОБАВЛЕНО ПОЛЕ current_spread) ==========
class PriceStorage:
    def __init__(self, maxlen=200):
        self.maxlen = maxlen
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.volumes = []
        self.last_signal = None
        self.cached_indicators = None
        self.cache_time = 0
        self.cache_ttl = 60
        self.m5_opens = []
        self.m5_highs = []
        self.m5_lows = []
        self.m5_closes = []
        self.m5_timestamps = []
        self._current_m5 = None
        self.current_spread = None  # для хранения последнего спреда

    def is_cache_valid(self):
        return self.cached_indicators is not None and (time.time() - self.cache_time) < self.cache_ttl

    def add_candle(self, candle):
        self.opens.append(float(candle['open']))
        self.highs.append(float(candle['high']))
        self.lows.append(float(candle['low']))
        self.closes.append(float(candle['close']))
        self.volumes.append(float(candle.get('volume', 0)))
        if len(self.opens) > self.maxlen:
            self.opens.pop(0); self.highs.pop(0); self.lows.pop(0); self.closes.pop(0); self.volumes.pop(0)
        self.cached_indicators = None
        dt = datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S')
        minute = (dt.minute // 5) * 5
        m5_key = dt.replace(minute=minute, second=0, microsecond=0)
        if self._current_m5 is None:
            self._current_m5 = {'open': float(candle['open']), 'high': float(candle['high']),
                                'low': float(candle['low']), 'close': float(candle['close']), 'time': m5_key}
        elif m5_key != self._current_m5['time']:
            self.m5_opens.append(self._current_m5['open'])
            self.m5_highs.append(self._current_m5['high'])
            self.m5_lows.append(self._current_m5['low'])
            self.m5_closes.append(self._current_m5['close'])
            self.m5_timestamps.append(self._current_m5['time'])
            if len(self.m5_opens) > self.maxlen:
                self.m5_opens.pop(0); self.m5_highs.pop(0); self.m5_lows.pop(0); self.m5_closes.pop(0); self.m5_timestamps.pop(0)
            self._current_m5 = {'open': float(candle['open']), 'high': float(candle['high']),
                                'low': float(candle['low']), 'close': float(candle['close']), 'time': m5_key}
        else:
            self._current_m5['high'] = max(self._current_m5['high'], float(candle['high']))
            self._current_m5['low'] = min(self._current_m5['low'], float(candle['low']))
            self._current_m5['close'] = float(candle['close'])

    def clear(self):
        self.opens.clear(); self.highs.clear(); self.lows.clear(); self.closes.clear(); self.volumes.clear()
        self.m5_opens.clear(); self.m5_highs.clear(); self.m5_lows.clear(); self.m5_closes.clear(); self.m5_timestamps.clear()
        self._current_m5 = None; self.cached_indicators = None; self.current_spread = None

price_storages = {sym: PriceStorage() for sym in SYMBOLS}

# ========== ФУНКЦИИ ИНДИКАТОРОВ (БЕЗ ИЗМЕНЕНИЙ) ==========
# (здесь все функции sma, ema, rsi, bbands, macd, calculate_atr, calculate_obv, obv_trend,
#  detect_false_breakout, find_support_resistance, adx, stochastic, pivot_points,
#  get_5min_trend, calculate_normalized_score – они уже есть, я их не дублирую для краткости,
#  в реальном коде они остаются на месте. В финальном файле они будут присутствовать.)

# ========== ЗАГРУЗКА ДАННЫХ ЧЕРЕЗ TWELVE DATA ==========
async def fetch_candles(symbol: str, api_key: str, bars: int = 50) -> Optional[List[Dict]]:
    global api_errors
    cached = await asyncio.to_thread(get_cached_candles_sync, symbol, bars)
    if cached:
        return cached
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': symbol, 'interval': '1min', 'outputsize': bars, 'apikey': api_key}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get('values')
                    if values is None:
                        logger.error(f"Twelve Data error: {data}")
                        return None
                    await asyncio.to_thread(save_candles_to_cache_sync, symbol, values)
                    return values
                else:
                    error_text = await resp.text()
                    logger.error(f"Twelve Data error for {symbol}: {resp.status} - {error_text}")
                    now = time.time()
                    key = f"{symbol}_fetch"
                    if key not in api_errors:
                        api_errors[key] = {'count': 0, 'first': now}
                    api_errors[key]['count'] += 1
                    if api_errors[key]['count'] >= API_ERROR_THRESHOLD:
                        if now - api_errors[key]['first'] < API_ERROR_WINDOW:
                            bot = Bot(token=BOT_TOKEN)
                            await notify_admin(bot, f"⚠️ Множественные ошибки Twelve Data для {symbol}: {error_text}")
                            api_errors[key]['count'] = 0
                    return None
    except Exception as e:
        logger.error(f"fetch error for {symbol}: {e}")
        return None

async def fetch_last_candle(symbol: str, api_key: str) -> Optional[Dict]:
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': symbol, 'interval': '1min', 'outputsize': 1, 'apikey': api_key}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get('values')
                    if values and len(values) > 0:
                        return values[0]
                    return None
                else:
                    error_text = await resp.text()
                    logger.error(f"Twelve Data error for {symbol}: {resp.status} - {error_text}")
                    now = time.time()
                    key = f"{symbol}_last"
                    if key not in api_errors:
                        api_errors[key] = {'count': 0, 'first': now}
                    api_errors[key]['count'] += 1
                    if api_errors[key]['count'] >= API_ERROR_THRESHOLD:
                        if now - api_errors[key]['first'] < API_ERROR_WINDOW:
                            bot = Bot(token=BOT_TOKEN)
                            await notify_admin(bot, f"⚠️ Множественные ошибки получения последней свечи {symbol}: {error_text}")
                            api_errors[key]['count'] = 0
                    return None
    except Exception as e:
        logger.error(f"fetch_last_candle error for {symbol}: {e}")
        return None

async def update_prices(symbol: str) -> bool:
    candles = await fetch_candles(symbol, TWELVE_API_KEY, 200)
    if candles:
        price_storages[symbol].clear()
        for c in candles[::-1]:
            price_storages[symbol].add_candle(c)
        return True
    return False

async def get_indicators(symbol: str) -> Optional[Dict]:
    storage = price_storages.get(symbol)
    if not storage:
        logger.error(f"Нет хранилища для {symbol}")
        return None

    # Сначала пробуем загрузить из кэша БД
    cached = await asyncio.to_thread(load_indicators_cache_sync, symbol, storage.cache_ttl)
    if cached:
        storage.cached_indicators = cached
        storage.cache_time = time.time()
        return cached

    if storage.is_cache_valid():
        return storage.cached_indicators
    try:
        if len(storage.closes) < 20:
            if not await update_prices(symbol):
                return None
        c = storage.closes
        h = storage.highs
        l = storage.lows
        cur = c[-1]
        ind = {'price': cur}
        if len(c) >= 3:
            price_3min_ago = c[-3]
            change_3min = cur - price_3min_ago
            ind['change_3min'] = change_3min
            ind['change_3min_pct'] = (change_3min / price_3min_ago) * 100
        else:
            ind['change_3min'] = 0; ind['change_3min_pct'] = 0
        ind['rsi'] = rsi(c, 14)
        if ind['rsi'] > 70:
            ind['rsi_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (сигнал к продаже)'
        elif ind['rsi'] < 30:
            ind['rsi_signal'] = 'ПЕРЕПРОДАННОСТЬ (сигнал к покупке)'
        elif ind['rsi'] > 50:
            ind['rsi_signal'] = 'ВОСХОДЯЩИЙ ТРЕНД'
        else:
            ind['rsi_signal'] = 'НИСХОДЯЩИЙ ТРЕНД'
        macd_line = macd(c, 12, 26)
        ind['macd'] = macd_line
        ind['macd_trend'] = 'БЫЧИЙ СИГНАЛ' if macd_line > 0 else 'МЕДВЕЖИЙ СИГНАЛ' if macd_line < 0 else 'НЕЙТРАЛЬНО'
        upper, mid, lower = bbands(c, 20, 2)
        ind['bb_upper'] = upper; ind['bb_middle'] = mid; ind['bb_lower'] = lower
        ind['bb_width'] = ((upper - lower) / mid) * 100
        if cur >= upper:
            ind['bb_position'] = 'ВЫШЕ ВЕРХНЕЙ ПОЛОСЫ'
            ind['bb_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (возможен откат вниз)'
        elif cur <= lower:
            ind['bb_position'] = 'НИЖЕ НИЖНЕЙ ПОЛОСЫ'
            ind['bb_signal'] = 'ПЕРЕПРОДАННОСТЬ (возможен отскок вверх)'
        elif cur > mid:
            ind['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И ВЕРХНЕЙ'
            ind['bb_signal'] = 'НЕЙТРАЛЬНО'
        else:
            ind['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И НИЖНЕЙ'
            ind['bb_signal'] = 'НЕЙТРАЛЬНО'
        ind['sma'] = {}
        for p in [5, 10, 20, 50]:
            val = sma(c, p)
            ind['sma'][p] = val
            if cur > val:
                ind[f'sma_{p}_signal'] = '⬆️ ВЫШЕ'
            elif cur < val:
                ind[f'sma_{p}_signal'] = '⬇️ НИЖЕ'
            else:
                ind[f'sma_{p}_signal'] = '⏺️ ОКОЛО'
        ind['ema'] = {}
        for p in [5, 10, 20]:
            val = ema(c, p)
            ind['ema'][p] = val
            if cur > val:
                ind[f'ema_{p}_signal'] = '⬆️ ВЫШЕ'
            elif cur < val:
                ind[f'ema_{p}_signal'] = '⬇️ НИЖЕ'
            else:
                ind[f'ema_{p}_signal'] = '⏺️ ОКОЛО'
        if storage.volumes:
            obv_values = calculate_obv(storage.closes, storage.volumes)
            ind['obv'] = obv_values[-1] if obv_values else 0
            ind['obv_trend'] = obv_trend(obv_values, 14)
        else:
            ind['obv'] = 0; ind['obv_trend'] = 'neutral'
        atr, atr_pct = calculate_atr(storage.highs, storage.lows, storage.closes, 14)
        ind['atr'] = atr; ind['atr_percent'] = atr_pct
        sup, res, ns, nr = find_support_resistance(h, l, c)
        ind['support_levels'] = sup; ind['resistance_levels'] = res
        ind['nearest_support'] = ns; ind['nearest_resistance'] = nr
        ind['distance_to_support'] = (cur - ns) * 10000 if ns else 0
        ind['distance_to_resistance'] = (nr - cur) * 10000 if nr else 0
        ind['breakout'] = detect_false_breakout(storage.highs, storage.lows, storage.closes)
        adx_val, plus_di, minus_di = adx(storage.highs, storage.lows, storage.closes, 14)
        ind['adx'] = adx_val; ind['plus_di'] = plus_di; ind['minus_di'] = minus_di
        stoch_k, stoch_d = stochastic(storage.highs, storage.lows, storage.closes, 14, 3)
        ind['stoch_k'] = stoch_k; ind['stoch_d'] = stoch_d
        if len(storage.highs) >= 1440:
            day_high = max(storage.highs[-1440:]); day_low = min(storage.lows[-1440:]); day_close = storage.closes[-1440]
            pivot, r1, r2, r3, s1, s2, s3 = pivot_points(day_high, day_low, day_close)
            ind['pivot'] = pivot; ind['r1'] = r1; ind['r2'] = r2; ind['r3'] = r3
            ind['s1'] = s1; ind['s2'] = s2; ind['s3'] = s3
        ind['ml_score'] = calculate_normalized_score(ind)
        if ind['ml_score'] >= 0:
            ind['prob_up'] = 50 + ind['ml_score'] / 2
            ind['prob_down'] = 50 - ind['ml_score'] / 2
        else:
            ind['prob_up'] = 50 + ind['ml_score'] / 2
            ind['prob_down'] = 50 - ind['ml_score'] / 2
        ind['confidence'] = abs(ind['ml_score'])
        # Получаем предсказание ML и корректируем уверенность
        ml_prob = ml_gen.predict(ind)
        ind['ml_prob_up'] = ml_prob
        # Комбинируем уверенность с ML (вес 70% от текущей, 30% от ML)
        ind['confidence'] = ind['confidence'] * 0.7 + ml_prob * 30

        trend_5min = get_5min_trend(storage)
        ind['trend_5min'] = trend_5min
        up = ind['prob_up']; down = ind['prob_down']
        if (up > down and trend_5min == 'down') or (down > up and trend_5min == 'up'):
            ind['confidence'] = ind['confidence'] * 0.7
            logger.info(f"📉 {symbol}: сигнал ослаблен из-за противоречия тренду 5мин ({trend_5min})")
        ind['price'] = cur
        ind['timestamp'] = datetime.now().strftime('%H:%M:%S')
        storage.last_signal = ind
        storage.cached_indicators = ind
        storage.cache_time = time.time()
        # Сохраняем в кэш БД
        await asyncio.to_thread(save_indicators_cache_sync, symbol, ind)
        return ind
    except Exception as e:
        logger.error(f"Error in get_indicators for {symbol}: {e}")
        return None

# ========== ГЕНЕРАЦИЯ СООБЩЕНИЯ ==========
def generate_message(ind, symbol, warning=None):
    try:
        up = ind['prob_up']; down = ind['prob_down']; conf = ind['confidence']
        if up > down + 10 and conf > 50:
            rec = "📈 СИЛЬНАЯ ПОКУПКА"; emoji = "🟢"
        elif up > down:
            rec = "📈 ПОКУПКА"; emoji = "🟢"
        elif down > up + 10 and conf > 50:
            rec = "📉 СИЛЬНАЯ ПРОДАЖА"; emoji = "🔴"
        elif down > up:
            rec = "📉 ПРОДАЖА"; emoji = "🔴"
        else:
            rec = "⏸️ ОЖИДАНИЕ"; emoji = "⚪"
        msg = f"""{emoji} *АНАЛИЗ {symbol}* {emoji}

📊 *ОБЩАЯ ВЕРОЯТНОСТЬ*
┌─ ⬆️ ВВЕРХ: {up:.1f}%
└─ ⬇️ ВНИЗ: {down:.1f}%
🎯 Уверенность: {conf:.1f}%
💡 Рекомендация: {rec}
"""
        if 'atr_percent' in ind:
            msg += f"\n📊 *ATR*: {ind['atr_percent']:.3f}%"
        if 'ml_prob_up' in ind:
            msg += f"\n🤖 *ML вероятность*: {ind['ml_prob_up'] * 100:.1f}%"
        if 'adx' in ind:
            msg += f"\n📈 *ADX*: {ind['adx']:.1f} (DI+:{ind['plus_di']:.1f} DI-:{ind['minus_di']:.1f})"
        if 'stoch_k' in ind:
            msg += f"\n📊 *Stochastic*: %K={ind['stoch_k']:.1f} %D={ind['stoch_d']:.1f}"
        if 'change_3min_pct' in ind:
            arrow = "⬆️" if ind['change_3min'] > 0 else "⬇️" if ind['change_3min'] < 0 else "➡️"
            msg += f"\n📊 *3 мин изменение*: {arrow} {ind['change_3min_pct']:.3f}%"
        if 'trend_5min' in ind:
            trend_arrow = "⬆️" if ind['trend_5min'] == 'up' else "⬇️" if ind['trend_5min'] == 'down' else "➡️"
            msg += f"\n📈 *Тренд 5 мин*: {trend_arrow}"
        if warning:
            msg += f"\n\n⚠️ *ВНИМАНИЕ:* {warning}"
        hashtag = symbol.replace('/', '').upper()
        msg += f"\n\n#{hashtag}"
        return msg
    except KeyError as e:
        logger.error(f"KeyError in generate_message: {e}")
        return f"❌ Ошибка формирования сигнала: отсутствует ключ {e}"
    except Exception as e:
        logger.error(f"Unexpected error in generate_message: {e}")
        return "❌ Внутренняя ошибка при формировании сигнала"

# ========== КЛАВИАТУРЫ ==========
def main_menu():
    kb = [
        [InlineKeyboardButton("📈 Статус", callback_data='status'),
         InlineKeyboardButton("📊 Статистика", callback_data='stats')],
        [InlineKeyboardButton("🔔 Автосигнал", callback_data='auto_on'),
         InlineKeyboardButton("⏹️ Стоп", callback_data='auto_off')],
    ]
    return InlineKeyboardMarkup(kb)

# ========== ОБРАБОТЧИКИ ==========
@app.before_request
def before_request():
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

@app.route('/')
def index():
    return "<h1>🤖Currency pair</h1><p>Running</p>"

@app.route('/health')
def health():
    with subscribers_lock:
        count = len(subscribers)
    return jsonify({'status': 'ok', 'subscribers': count})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        logger.info("📨 Webhook received")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if 'callback_query' in data:
            chat_id = data['callback_query']['from']['id']
            cb = data['callback_query']['data']
            cb_id = data['callback_query']['id']
            loop.run_until_complete(handle_callback(chat_id, cb, cb_id))
        elif 'message' in data and 'text' in data['message']:
            chat_id = data['message']['chat']['id']
            text = data['message']['text']
            loop.run_until_complete(handle_message(chat_id, text))
        loop.close()
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        if ADMIN_CHAT_ID:
            try:
                bot = Bot(token=BOT_TOKEN)
                asyncio.run_coroutine_threadsafe(
                    notify_admin(bot, f"❌ Критическая ошибка в webhook: {e}"),
                    asyncio.get_event_loop()
                )
            except:
                pass
        return jsonify({'ok': False}), 500

async def handle_callback(chat_id, cb, cb_id):
    logger.info(f"🔥 Callback received: {cb} from {chat_id}")
    bot = Bot(token=BOT_TOKEN)
    try:
        try:
            await bot.answer_callback_query(cb_id)
        except Exception as e:
            logger.warning(f"Не удалось ответить на callback: {e}")

        if cb == 'status':
            await send_status(bot, chat_id)
        elif cb == 'stats':
            await send_stats(bot, chat_id)
        elif cb == 'auto_on':
            added = await asyncio.to_thread(add_subscriber_mem, chat_id)
            if added:
                await bot.send_message(chat_id, "✅ Автосигналы включены")
                logger.info(f"✅ Подписчик {chat_id} добавлен")
            else:
                await bot.send_message(chat_id, "ℹ️ Автосигналы уже были включены")
        elif cb == 'auto_off':
            removed = await asyncio.to_thread(remove_subscriber_mem, chat_id)
            if removed:
                await bot.send_message(chat_id, "⏹️ Автосигналы остановлены")
                logger.info(f"⏹️ Подписчик {chat_id} удалён")
            else:
                await bot.send_message(chat_id, "ℹ️ Автосигналы не были включены")
        elif cb == 'back':
            await bot.send_message(chat_id, "Главное меню", reply_markup=main_menu())
        else:
            await bot.send_message(chat_id, "❓ Неизвестная команда")
            logger.warning(f"Unknown callback: {cb} from {chat_id}")
    except Exception as e:
        logger.error(f"Error in handle_callback: {e}")
        try:
            await bot.send_message(chat_id, "⚠️ Внутренняя ошибка")
            await notify_admin(bot, f"Ошибка в handle_callback: {e}")
        except:
            pass

async def handle_message(chat_id, text):
    bot = Bot(token=BOT_TOKEN)
    if text == '/start':
        await bot.send_message(chat_id, "🤖Currency pair", reply_markup=main_menu())
    elif text == '/status':
        await send_status(bot, chat_id)
    elif text == '/stats':
        await send_stats(bot, chat_id)
    elif text == '/stop':
        removed = await asyncio.to_thread(remove_subscriber_mem, chat_id)
        if removed:
            await bot.send_message(chat_id, "⏹️ Автосигналы остановлены")
        else:
            await bot.send_message(chat_id, "ℹ️ Автосигналы не были включены")
    else:
        await bot.send_message(chat_id, "❌ Неизвестная команда")

async def send_status(bot, chat_id):
    is_sub = await asyncio.to_thread(is_subscriber, chat_id)
    auto = "вкл" if is_sub else "выкл"
    await bot.send_message(chat_id, f"📊 Статус:\nАвтосигналы: {auto}\nОтслеживаемые пары: {', '.join(SYMBOLS)}")

async def send_stats(bot, chat_id):
    summary = await asyncio.to_thread(get_summary_sync)
    logger.info(f"📊 Статистика запрошена: всего сигналов = {summary['total']}")
    if summary['total'] == 0:
        await bot.send_message(chat_id, "📊 Статистика пока пуста.")
        return
    text = f"""📊 *Статистика сигналов*

Всего сигналов: {summary['total']}
✅ Прибыльных: {summary['profit']}
❌ Убыточных: {summary['loss']}
⏳ Истекло: {summary['timeout']}
❓ Неизвестно: {summary['unknown']}

📈 Винрейт: {summary['win_rate']:.1f}%
💰 Средняя прибыль: {summary['avg_profit']:.1f} пипсов
📉 Средний убыток: {summary['avg_loss']:.1f} пипсов
💵 Общая прибыль: {summary['total_profit_pips']:.1f} пипсов
💸 Общий убыток: {summary['total_loss_pips']:.1f} пипсов
📊 Чистый результат: {summary['total_profit_pips'] - summary['total_loss_pips']:.1f} пипсов
"""
    await bot.send_message(chat_id, text, parse_mode='Markdown')

# ========== ФОНОВЫЙ ПОТОК (АВТОРАССЫЛКА) ==========
async def auto_worker():
    logger.info("🚀 Автосигналы запущены")
    if ADMIN_CHAT_ID:
        try:
            bot = Bot(token=BOT_TOKEN)
            await notify_admin(bot, "✅ Бот успешно запущен")
        except:
            pass
    while True:
        try:
            await asyncio.sleep(600)  # 10 минут
            subs = await asyncio.to_thread(get_subscribers_sync)
            if not subs:
                logger.info("😴 Нет подписчиков, пропускаем рассылку")
                continue
            for symbol in SYMBOLS:
                try:
                    # Обновляем спред для пары
                    spread = await fetch_spread(symbol, TWELVE_API_KEY)
                    if spread is not None:
                        price_storages[symbol].current_spread = spread
                    # Получаем свечу
                    new_candle = await fetch_last_candle(symbol, TWELVE_API_KEY)
                    if new_candle:
                        price_storages[symbol].add_candle(new_candle)
                        current_price = float(new_candle['close'])
                        logger.info(f"✅ {symbol}: новая свеча {new_candle.get('datetime')} = {current_price}")
                        await asyncio.to_thread(update_signal_results_sync, symbol, current_price)
                    else:
                        logger.warning(f"⚠️ {symbol}: не удалось получить свечу")

                    for uid in subs:
                        try:
                            bot = Bot(token=BOT_TOKEN)
                            ind = await get_indicators(symbol)
                            if ind and ind.get('confidence', 0) >= 49.9:
                                has_risk, events = await economic_calendar.check_symbol_risk(symbol)
                                warning = None
                                if has_risk:
                                    event_names = [e['event'] for e in events[:1]]
                                    event_time = events[0]['time']
                                    warning = f"Через {event_time} важная новость: {event_names[0]}"
                                    logger.info(f"📰 {symbol}: предупреждение о новости")
                                up = ind['prob_up']; down = ind['prob_down']
                                direction = 'buy' if up > down else 'sell'
                                entry = ind['price']
                                atr = ind.get('atr', 0.001)
                                tp_distance = atr * 1.5
                                sl_distance = atr * 0.75
                                spread = price_storages[symbol].current_spread or 0.0002  # запасное значение
                                if direction == 'buy':
                                    tp = entry + tp_distance - spread/2
                                    sl = entry - sl_distance + spread/2
                                else:
                                    tp = entry - tp_distance + spread/2
                                    sl = entry + sl_distance - spread/2
                                signal_record = {
                                    'timestamp': time.time(),
                                    'symbol': symbol,
                                    'price': entry,
                                    'direction': direction,
                                    'tp': tp,
                                    'sl': sl,
                                    'result': None,
                                    'exit_price': None,
                                    'exit_time': None
                                }
                                features = ml_gen.prepare_features(ind)
                                await asyncio.to_thread(add_signal_sync, signal_record, features, spread)
                                await bot.send_message(uid, generate_message(ind, symbol, warning), parse_mode='Markdown')
                                logger.info(f"✅ {symbol} сигнал отправлен {uid}")
                            else:
                                conf = ind.get('confidence', 0) if ind else 0
                                logger.info(f"📉 {symbol} сигнал для {uid} пропущен (уверенность {conf:.1f}% < 49.9)")
                        except Exception as e:
                            logger.error(f"❌ Ошибка отправки для {uid} по {symbol}: {e}")
                            try:
                                bot = Bot(token=BOT_TOKEN)
                                await notify_admin(bot, f"❌ Ошибка отправки для {uid} по {symbol}: {e}")
                            except:
                                pass
                except Exception as e:
                    logger.error(f"❌ Ошибка при обработке пары {symbol}: {e}")
                    try:
                        bot = Bot(token=BOT_TOKEN)
                        await notify_admin(bot, f"❌ Ошибка при обработке пары {symbol}: {e}")
                    except:
                        pass
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в auto_worker: {e}")
            try:
                bot = Bot(token=BOT_TOKEN)
                await notify_admin(bot, f"❌ Критическая ошибка в auto_worker: {e}")
            except:
                pass
            await asyncio.sleep(10)

def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(auto_worker())

def start_training():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(train_model_periodically())

# ========== ИНИЦИАЛИЗАЦИЯ ==========
init_db_sync()
load_subscribers_from_db()

# ========== ЗАПУСК ФОНОВЫХ ПОТОКОВ ==========
threading.Thread(target=start_worker, daemon=True).start()
threading.Thread(target=start_training, daemon=True).start()
logger.info("✅ Фоновые потоки запущены (автосигналы + обучение модели)")

# ========== ЗАПУСК FLASK ==========
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)