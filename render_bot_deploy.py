import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import aiohttp
import aiofiles
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import time
import pytz
from typing import List, Dict, Optional, Set
import threading

# Для машинного обучения (пока заготовка)
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

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

# ========== РАСШИРЕННЫЙ СПИСОК ВАЛЮТНЫХ ПАР ==========
SYMBOLS = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',
    'USD/CHF', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY'
]

# ========== ФАЙЛЫ ПОДПИСЧИКОВ ==========
SUBSCRIBERS_FILE = "subscribers.json"
STATS_FILE = "stats.json"

# ========== АСИНХРОННЫЕ ФУНКЦИИ ДЛЯ РАБОТЫ С ФАЙЛАМИ ==========
async def async_load_subscribers() -> Set[int]:
    try:
        async with aiofiles.open(SUBSCRIBERS_FILE, 'r') as f:
            data = await f.read()
            subs = set(json.loads(data))
            logger.info(f"📂 Загружено подписчиков: {len(subs)}")
            return subs
    except FileNotFoundError:
        logger.warning(f"📂 Файл {SUBSCRIBERS_FILE} не существует")
        return set()
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки подписчиков: {e}")
        return set()

async def async_save_subscribers(subs: Set[int]):
    try:
        async with aiofiles.open(SUBSCRIBERS_FILE, 'w') as f:
            await f.write(json.dumps(list(subs), indent=2))
        logger.info(f"💾 Сохранено подписчиков: {len(subs)}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения подписчиков: {e}")

async def async_load_stats() -> List[Dict]:
    try:
        async with aiofiles.open(STATS_FILE, 'r') as f:
            data = await f.read()
            signals = json.loads(data)
            logger.info(f"📊 Загружено записей статистики: {len(signals)}")
            return signals
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.error(f"Ошибка загрузки статистики: {e}")
        return []

async def async_save_stats(signals: List[Dict]):
    try:
        async with aiofiles.open(STATS_FILE, 'w') as f:
            await f.write(json.dumps(signals, indent=2))
        logger.info(f"💾 Статистика сохранена: {len(signals)} записей")
    except Exception as e:
        logger.error(f"Ошибка сохранения статистики: {e}")

# ========== КЛАСС ДЛЯ УПРАВЛЕНИЯ СТАТИСТИКОЙ ==========
class StatsManager:
    def __init__(self):
        self.signals = []
        self.lock = asyncio.Lock()

    async def load(self):
        async with self.lock:
            self.signals = await async_load_stats()

    async def save(self):
        async with self.lock:
            await async_save_stats(self.signals)

    async def add_signal(self, signal: Dict):
        async with self.lock:
            self.signals.append(signal)
            await async_save_stats(self.signals)

    async def update_results(self, symbol: str, current_price: float):
        updated = False
        async with self.lock:
            for sig in self.signals:
                if sig.get('result') is not None or sig.get('symbol') != symbol:
                    continue
                if time.time() - sig['timestamp'] > 3600:
                    sig['result'] = 'timeout'
                    sig['exit_price'] = current_price
                    sig['exit_time'] = time.time()
                    updated = True
                    continue

                direction = sig['direction']
                entry = sig['price']
                tp = sig.get('tp')
                sl = sig.get('sl')

                if direction == 'buy':
                    if tp and current_price >= tp:
                        sig['result'] = 'profit'
                        sig['exit_price'] = tp
                        sig['exit_time'] = time.time()
                        updated = True
                    elif sl and current_price <= sl:
                        sig['result'] = 'loss'
                        sig['exit_price'] = sl
                        sig['exit_time'] = time.time()
                        updated = True
                elif direction == 'sell':
                    if tp and current_price <= tp:
                        sig['result'] = 'profit'
                        sig['exit_price'] = tp
                        sig['exit_time'] = time.time()
                        updated = True
                    elif sl and current_price >= sl:
                        sig['result'] = 'loss'
                        sig['exit_price'] = sl
                        sig['exit_time'] = time.time()
                        updated = True

            if updated:
                await async_save_stats(self.signals)

    def get_summary(self):
        total = len(self.signals)
        if total == 0:
            return {
                'total': 0, 'profit': 0, 'loss': 0, 'timeout': 0, 'unknown': 0,
                'win_rate': 0, 'avg_profit': 0, 'avg_loss': 0,
                'total_profit_pips': 0, 'total_loss_pips': 0
            }
        profit = sum(1 for s in self.signals if s.get('result') == 'profit')
        loss = sum(1 for s in self.signals if s.get('result') == 'loss')
        timeout = sum(1 for s in self.signals if s.get('result') == 'timeout')
        unknown = total - profit - loss - timeout

        total_profit_pips = 0
        total_loss_pips = 0
        for s in self.signals:
            if s.get('result') == 'profit' and s.get('exit_price') and s.get('price'):
                if s['direction'] == 'buy':
                    pips = (s['exit_price'] - s['price']) * 10000
                else:
                    pips = (s['price'] - s['exit_price']) * 10000
                total_profit_pips += pips
            elif s.get('result') == 'loss' and s.get('exit_price') and s.get('price'):
                if s['direction'] == 'buy':
                    pips = (s['exit_price'] - s['price']) * 10000
                else:
                    pips = (s['price'] - s['exit_price']) * 10000
                total_loss_pips += abs(pips)

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

stats_manager = StatsManager()

# ========== ФУНКЦИЯ УВЕДОМЛЕНИЯ АДМИНИСТРАТОРА ==========
async def notify_admin(bot: Bot, message: str):
    if ADMIN_CHAT_ID:
        try:
            await bot.send_message(ADMIN_CHAT_ID, f"⚠️ *Админ-уведомление*\n{message}", parse_mode='Markdown')
            logger.info("📨 Уведомление администратору отправлено")
        except Exception as e:
            logger.error(f"❌ Не удалось отправить уведомление админу: {e}")

# ========== АСИНХРОННЫЙ ЭКОНОМИЧЕСКИЙ КАЛЕНДАРЬ ==========
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
            'USD/CHF': ['CH'],
            'NZD/USD': ['NZ'],
            'EUR/GBP': ['EU', 'GB'],
            'EUR/JPY': ['EU', 'JP'],
            'GBP/JPY': ['GB', 'JP']
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
                self.model = None
        else:
            logger.info("No model file found, ML disabled")
            self.model = None

    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)

    def prepare_features(self, ind):
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
        ]
        return features

    def predict(self, ind):
        if self.model is None:
            return 0.5
        try:
            if not hasattr(self.model, 'estimators_'):
                return 0.5
            features = self.prepare_features(ind)
            X = np.array(features).reshape(1, -1)
            proba = self.model.predict_proba(X)[0]
            if len(proba) > 1:
                return proba[1]
            else:
                return proba[0]
        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return 0.5

ml_gen = MLSignalGenerator()

# ========== ХРАНИЛИЩЕ ДАННЫХ ДЛЯ КАЖДОЙ ПАРЫ ==========
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

    def is_cache_valid(self):
        return self.cached_indicators is not None and (time.time() - self.cache_time) < self.cache_ttl

    def add_candle(self, candle):
        self.opens.append(float(candle['open']))
        self.highs.append(float(candle['high']))
        self.lows.append(float(candle['low']))
        self.closes.append(float(candle['close']))
        self.volumes.append(float(candle.get('volume', 0)))
        if len(self.opens) > self.maxlen:
            self.opens.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.volumes.pop(0)

        self.cached_indicators = None

        dt = datetime.strptime(candle['datetime'], '%Y-%m-%d %H:%M:%S')
        minute = (dt.minute // 5) * 5
        m5_key = dt.replace(minute=minute, second=0, microsecond=0)

        if self._current_m5 is None:
            self._current_m5 = {
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'time': m5_key
            }
        elif m5_key != self._current_m5['time']:
            self.m5_opens.append(self._current_m5['open'])
            self.m5_highs.append(self._current_m5['high'])
            self.m5_lows.append(self._current_m5['low'])
            self.m5_closes.append(self._current_m5['close'])
            self.m5_timestamps.append(self._current_m5['time'])
            if len(self.m5_opens) > self.maxlen:
                self.m5_opens.pop(0)
                self.m5_highs.pop(0)
                self.m5_lows.pop(0)
                self.m5_closes.pop(0)
                self.m5_timestamps.pop(0)
            self._current_m5 = {
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'time': m5_key
            }
        else:
            self._current_m5['high'] = max(self._current_m5['high'], float(candle['high']))
            self._current_m5['low'] = min(self._current_m5['low'], float(candle['low']))
            self._current_m5['close'] = float(candle['close'])

    def clear(self):
        self.opens.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.volumes.clear()
        self.m5_opens.clear()
        self.m5_highs.clear()
        self.m5_lows.clear()
        self.m5_closes.clear()
        self.m5_timestamps.clear()
        self._current_m5 = None
        self.cached_indicators = None

price_storages = {sym: PriceStorage() for sym in SYMBOLS}

# ========== ФУНКЦИИ ИНДИКАТОРОВ ==========
def sma(data, period):
    if len(data) < period:
        return data[-1]
    return sum(data[-period:]) / period

def ema(data, period):
    if len(data) < period:
        return data[-1]
    multiplier = 2 / (period + 1)
    ema_val = sum(data[-period:]) / period
    for price in data[-period + 1:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val

def rsi(data, period=14):
    if len(data) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        change = data[-i] - data[-i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bbands(data, period=20, std=2):
    if len(data) < period:
        m = data[-1]
        return m * 1.02, m, m * 0.98
    m = sum(data[-period:]) / period
    variance = sum((x - m) ** 2 for x in data[-period:]) / period
    s = variance ** 0.5
    return m + std * s, m, m - std * s

def macd(data, fast=12, slow=26):
    if len(data) < slow:
        return 0.0
    return ema(data, fast) - ema(data, slow)

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0.0, 0.0
    tr = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hc, lc))
    atr = sum(tr[-period:]) / period
    atr_percentage = (atr / closes[-1]) * 100
    return atr, atr_percentage

def calculate_obv(closes, volumes):
    if len(closes) < 2 or len(volumes) < 2:
        return [0]
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv

def obv_trend(obv_values, period=14):
    if len(obv_values) < period:
        return "neutral"
    obv_ema = ema(obv_values, period)
    return "bullish" if obv_values[-1] > obv_ema else "bearish"

def detect_false_breakout(highs, lows, closes, lookback=5):
    if len(closes) < lookback + 2:
        return "no_breakout"
    recent_high = max(highs[-lookback - 1:-1])
    recent_low = min(lows[-lookback - 1:-1])
    current_close = closes[-1]
    if current_close > recent_high:
        if closes[-2] < recent_high:
            return "false_breakout_up"
        else:
            return "valid_breakout_up"
    if current_close < recent_low:
        if closes[-2] > recent_low:
            return "false_breakout_down"
        else:
            return "valid_breakout_down"
    return "no_breakout"

def find_support_resistance(high, low, close, window=5):
    supports, resistances = [], []
    n = len(close)
    for i in range(window, n - window):
        if all(low[i] <= low[i - j] for j in range(1, window + 1)) and \
                all(low[i] <= low[i + j] for j in range(1, window + 1)):
            supports.append(low[i])
        if all(high[i] >= high[i - j] for j in range(1, window + 1)) and \
                all(high[i] >= high[i + j] for j in range(1, window + 1)):
            resistances.append(high[i])

    def cluster(levels, thr=0.0005):
        if not levels:
            return []
        levels.sort()
        cl = [levels[0]]
        res = []
        for lev in levels[1:]:
            if abs(lev - sum(cl) / len(cl)) < thr:
                cl.append(lev)
            else:
                res.append(sum(cl) / len(cl))
                cl = [lev]
        res.append(sum(cl) / len(cl))
        return res

    supports = cluster(supports)
    resistances = cluster(resistances)
    cur = close[-1]
    ns = None
    nr = None
    for s in supports:
        if s < cur:
            ns = s
    for r in resistances:
        if r > cur:
            nr = r
            break
    return supports[-3:], resistances[-3:], ns, nr

def adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return 0, 0, 0
    tr = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr.append(max(hl, hc, lc))
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    atr = sum(tr[-period:]) / period
    plus_di = 100 * (sum(plus_dm[-period:]) / period) / atr if atr != 0 else 0
    minus_di = 100 * (sum(minus_dm[-period:]) / period) / atr if atr != 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
    adx_val = sum([dx] * period) / period
    return adx_val, plus_di, minus_di

def stochastic(highs, lows, closes, k_period=14, d_period=3):
    if len(closes) < k_period + d_period:
        return 50, 50
    highest_high = max(highs[-k_period:])
    lowest_low = min(lows[-k_period:])
    k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low) if (highest_high - lowest_low) != 0 else 50
    d = sum([k] * d_period) / d_period
    return k, d

def pivot_points(high, low, close):
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def get_5min_trend(storage):
    if len(storage.m5_closes) < 2:
        return 'neutral'
    if storage.m5_closes[-1] > storage.m5_closes[-2]:
        return 'up'
    elif storage.m5_closes[-1] < storage.m5_closes[-2]:
        return 'down'
    else:
        return 'neutral'

def calculate_normalized_score(ind):
    score = 0
    max_score = 0

    weight_rsi = 2
    weight_macd = 2
    weight_bb = 2
    weight_ema = 1
    weight_sr = 2
    weight_3min = 1

    adx_value = ind.get('adx', 0)
    if adx_value < 20:
        weight_adx = 1
        weight_stoch = 3
    elif adx_value > 25:
        weight_adx = 3
        weight_stoch = 1
    else:
        weight_adx = 2
        weight_stoch = 2

    if ind['rsi'] < 30:
        score += weight_rsi
    elif ind['rsi'] > 70:
        score -= weight_rsi
    max_score += weight_rsi

    if ind['macd'] > 0:
        score += weight_macd
    else:
        score -= weight_macd
    max_score += weight_macd

    price = ind['price']
    if price <= ind['bb_lower']:
        score += weight_bb
    elif price >= ind['bb_upper']:
        score -= weight_bb
    max_score += weight_bb

    if ind['ema'][5] > ind['ema'][20]:
        score += weight_ema
    else:
        score -= weight_ema
    max_score += weight_ema

    if ind.get('adx', 0) > 25:
        if ind['plus_di'] > ind['minus_di']:
            score += weight_adx
        else:
            score -= weight_adx
    max_score += weight_adx

    stoch_k = ind.get('stoch_k', 50)
    if stoch_k < 20:
        score += weight_stoch
    elif stoch_k > 80:
        score -= weight_stoch
    max_score += weight_stoch

    dist_to_sup = ind.get('distance_to_support', 1000)
    dist_to_res = ind.get('distance_to_resistance', 1000)
    if dist_to_sup < 10 and dist_to_sup < dist_to_res:
        score += weight_sr
    elif dist_to_res < 10 and dist_to_res < dist_to_sup:
        score -= weight_sr
    max_score += weight_sr

    change = ind.get('change_3min', 0)
    if change > 0.0001:
        score += weight_3min
    elif change < -0.0001:
        score -= weight_3min
    max_score += weight_3min

    normalized = (score / max_score) * 100 if max_score > 0 else 0
    return normalized

# ========== ЗАГРУЗКА ДАННЫХ ==========
async def fetch_candles(symbol, api_key, bars=50):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': '1min',
        'outputsize': bars,
        'apikey': api_key
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('values')
    except Exception as e:
        logger.error(f"fetch error for {symbol}: {e}")
    return None

async def fetch_last_candle(symbol, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': '1min',
        'outputsize': 1,
        'apikey': api_key
    }
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    values = data.get('values')
                    if values:
                        return values[0]
    except Exception as e:
        logger.error(f"fetch_last_candle error for {symbol}: {e}")
    return None

async def update_prices(symbol):
    candles = await fetch_candles(symbol, TWELVE_API_KEY, 200)
    if candles:
        price_storages[symbol].clear()
        for c in candles[::-1]:
            price_storages[symbol].add_candle(c)
        return True
    return False

async def get_indicators(symbol):
    storage = price_storages.get(symbol)
    if not storage:
        logger.error(f"Нет хранилища для {symbol}")
        return None

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
        ind = {}
        ind['price'] = cur

        if len(c) >= 3:
            price_3min_ago = c[-3]
            change_3min = cur - price_3min_ago
            ind['change_3min'] = change_3min
            ind['change_3min_pct'] = (change_3min / price_3min_ago) * 100
        else:
            ind['change_3min'] = 0
            ind['change_3min_pct'] = 0

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
        ind['bb_upper'] = upper
        ind['bb_middle'] = mid
        ind['bb_lower'] = lower
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
            ind['obv'] = 0
            ind['obv_trend'] = 'neutral'

        atr, atr_pct = calculate_atr(storage.highs, storage.lows, storage.closes, 14)
        ind['atr'] = atr
        ind['atr_percent'] = atr_pct

        sup, res, ns, nr = find_support_resistance(h, l, c)
        ind['support_levels'] = sup
        ind['resistance_levels'] = res
        ind['nearest_support'] = ns
        ind['nearest_resistance'] = nr
        ind['distance_to_support'] = (cur - ns) * 10000 if ns else 0
        ind['distance_to_resistance'] = (nr - cur) * 10000 if nr else 0

        ind['breakout'] = detect_false_breakout(storage.highs, storage.lows, storage.closes)

        adx_val, plus_di, minus_di = adx(storage.highs, storage.lows, storage.closes, 14)
        ind['adx'] = adx_val
        ind['plus_di'] = plus_di
        ind['minus_di'] = minus_di

        stoch_k, stoch_d = stochastic(storage.highs, storage.lows, storage.closes, 14, 3)
        ind['stoch_k'] = stoch_k
        ind['stoch_d'] = stoch_d

        if len(storage.highs) >= 1440:
            day_high = max(storage.highs[-1440:])
            day_low = min(storage.lows[-1440:])
            day_close = storage.closes[-1440]
            pivot, r1, r2, r3, s1, s2, s3 = pivot_points(day_high, day_low, day_close)
            ind['pivot'] = pivot
            ind['r1'] = r1
            ind['r2'] = r2
            ind['r3'] = r3
            ind['s1'] = s1
            ind['s2'] = s2
            ind['s3'] = s3

        ind['ml_score'] = calculate_normalized_score(ind)

        if ind['ml_score'] >= 0:
            ind['prob_up'] = 50 + ind['ml_score'] / 2
            ind['prob_down'] = 50 - ind['ml_score'] / 2
        else:
            ind['prob_up'] = 50 + ind['ml_score'] / 2
            ind['prob_down'] = 50 - ind['ml_score'] / 2
        ind['confidence'] = abs(ind['ml_score'])

        ind['ml_prob_up'] = ml_gen.predict(ind)

        trend_5min = get_5min_trend(storage)
        ind['trend_5min'] = trend_5min

        up = ind['prob_up']
        down = ind['prob_down']
        if (up > down and trend_5min == 'down') or (down > up and trend_5min == 'up'):
            ind['confidence'] = ind['confidence'] * 0.7
            logger.info(f"📉 {symbol}: сигнал ослаблен из-за противоречия тренду 5мин ({trend_5min})")

        ind['price'] = cur
        ind['timestamp'] = datetime.now().strftime('%H:%M:%S')
        storage.last_signal = ind

        storage.cached_indicators = ind
        storage.cache_time = time.time()
        return ind
    except Exception as e:
        logger.error(f"Error in get_indicators for {symbol}: {e}")
        return None

# ========== ГЕНЕРАЦИЯ СООБЩЕНИЯ ==========
def generate_message(ind, symbol, warning=None):
    try:
        up = ind['prob_up']
        down = ind['prob_down']
        conf = ind['confidence']
        if up > down + 10 and conf > 50:
            rec = "📈 СИЛЬНАЯ ПОКУПКА"
            emoji = "🟢"
        elif up > down:
            rec = "📈 ПОКУПКА"
            emoji = "🟢"
        elif down > up + 10 and conf > 50:
            rec = "📉 СИЛЬНАЯ ПРОДАЖА"
            emoji = "🔴"
        elif down > up:
            rec = "📉 ПРОДАЖА"
            emoji = "🔴"
        else:
            rec = "⏸️ ОЖИДАНИЕ"
            emoji = "⚪"

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
        [InlineKeyboardButton("🔔 Автосигнал (3 мин)", callback_data='auto_on'),
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
    return jsonify({'status': 'ok', 'subscribers': len(subscribers)})

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

subscribers = set()
subscribers_lock = asyncio.Lock()

async def handle_callback(chat_id, cb, cb_id):
    logger.info(f"🔥 Callback received: {cb} from {chat_id}")
    bot = Bot(token=BOT_TOKEN)
    try:
        await bot.answer_callback_query(cb_id)
        if cb == 'status':
            await send_status(bot, chat_id)
        elif cb == 'stats':
            await send_stats(bot, chat_id)
        elif cb == 'auto_on':
            async with subscribers_lock:
                subscribers.add(chat_id)
                await async_save_subscribers(subscribers)
            await bot.send_message(chat_id, "✅ Автосигналы включены (каждые 3 мин)")
            logger.info(f"✅ Подписчик {chat_id} добавлен")
        elif cb == 'auto_off':
            async with subscribers_lock:
                if chat_id in subscribers:
                    subscribers.remove(chat_id)
                    await async_save_subscribers(subscribers)
                    await bot.send_message(chat_id, "⏹️ Автосигналы остановлены")
                    logger.info(f"⏹️ Подписчик {chat_id} удалён")
                else:
                    await bot.send_message(chat_id, "❌ Автосигналы не были включены")
        elif cb == 'back':
            await bot.send_message(chat_id, "Главное меню", reply_markup=main_menu())
        else:
            await bot.send_message(chat_id, "❓ Неизвестная команда или устаревшая кнопка")
            logger.warning(f"Unknown callback: {cb} from {chat_id}")
    except Exception as e:
        logger.error(f"Error in handle_callback: {e}")
        try:
            await bot.send_message(chat_id, "⚠️ Внутренняя ошибка при обработке запроса")
            await notify_admin(bot, f"❌ Ошибка в handle_callback для {chat_id}: {e}")
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
        async with subscribers_lock:
            if chat_id in subscribers:
                subscribers.remove(chat_id)
                await async_save_subscribers(subscribers)
                await bot.send_message(chat_id, "⏹️ Автосигналы остановлены")
            else:
                await bot.send_message(chat_id, "❌ Автосигналы не были включены")
    else:
        await bot.send_message(chat_id, "❌ Неизвестная команда")

async def send_status(bot, chat_id):
    async with subscribers_lock:
        auto = "вкл" if chat_id in subscribers else "выкл"
    await bot.send_message(chat_id, f"📊 Статус:\nАвтосигналы: {auto}\nОтслеживаемые пары: {', '.join(SYMBOLS)}")

async def send_stats(bot, chat_id):
    summary = stats_manager.get_summary()
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

# ========== ФОНОВЫЙ ПОТОК ==========
async def auto_worker():
    logger.info("🚀 Автосигналы запущены (интервал 3 мин)")
    if ADMIN_CHAT_ID:
        try:
            bot = Bot(token=BOT_TOKEN)
            await notify_admin(bot, "✅ Бот успешно запущен и начал авто-рассылку.")
        except:
            pass

    while True:
        try:
            await asyncio.sleep(180)

            file_subs = await async_load_subscribers()
            async with subscribers_lock:
                if file_subs != subscribers:
                    logger.warning(f"🔄 Подписчики в памяти ({len(subscribers)}) отличаются от файла ({len(file_subs)}). Синхронизируем.")
                    subscribers.clear()
                    subscribers.update(file_subs)
                subs = list(subscribers)

            if not subs:
                logger.info("😴 Нет подписчиков, пропускаем рассылку")
                continue

            for symbol in SYMBOLS:
                try:
                    new_candle = await fetch_last_candle(symbol, TWELVE_API_KEY)
                    if new_candle:
                        price_storages[symbol].add_candle(new_candle)
                        current_price = float(new_candle['close'])
                        logger.info(f"✅ {symbol}: новая свеча {new_candle.get('datetime')} = {current_price}")
                        await stats_manager.update_results(symbol, current_price)
                    else:
                        logger.warning(f"⚠️ {symbol}: не удалось получить свечу")

                    for uid in subs:
                        try:
                            bot = Bot(token=BOT_TOKEN)
                            ind = await get_indicators(symbol)
                            if ind and ind.get('confidence', 0) >= 65:
                                has_risk, events = await economic_calendar.check_symbol_risk(symbol)
                                warning = None
                                if has_risk:
                                    event_names = [e['event'] for e in events[:1]]
                                    event_time = events[0]['time']
                                    warning = f"Через {event_time} важная новость: {event_names[0]}"
                                    logger.info(f"📰 {symbol}: предупреждение о новости")

                                up = ind['prob_up']
                                down = ind['prob_down']
                                direction = 'buy' if up > down else 'sell'
                                entry = ind['price']
                                atr = ind.get('atr', 0.001)
                                tp_distance = atr * 1.5
                                sl_distance = atr * 0.75
                                if direction == 'buy':
                                    tp = entry + tp_distance
                                    sl = entry - sl_distance
                                else:
                                    tp = entry - tp_distance
                                    sl = entry + sl_distance

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
                                await stats_manager.add_signal(signal_record)

                                await bot.send_message(uid, generate_message(ind, symbol, warning),
                                                       parse_mode='Markdown')
                                logger.info(f"✅ {symbol} сигнал отправлен {uid}")
                            else:
                                conf = ind.get('confidence', 0) if ind else 0
                                logger.info(f"📉 {symbol} сигнал для {uid} пропущен (уверенность {conf:.1f}% < 65)")
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

async def init():
    await stats_manager.load()
    global subscribers
    subscribers = await async_load_subscribers()
    logger.info(f"👥 Загружено {len(subscribers)} подписчиков из файла")

asyncio.run(init())

threading.Thread(target=start_worker, daemon=True).start()
logger.info("✅ Фоновый поток запущен")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)