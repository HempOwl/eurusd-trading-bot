import asyncio
import logging
import os
import threading
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import time

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

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не задан!")
if not TWELVE_API_KEY:
    logger.error("❌ TWELVE_API_KEY не задан!")

# Файл для хранения подписчиков
SUBSCRIBERS_FILE = "subscribers.json"
full_path = os.path.abspath(SUBSCRIBERS_FILE)
logger.info(f"📁 Путь к файлу подписчиков: {full_path}")


def load_subscribers():
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, 'r') as f:
                data = json.load(f)
                subs = set(data)
                logger.info(f"📂 Загружено подписчиков: {len(subs)}")
                return subs
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки подписчиков: {e}")
    else:
        logger.warning(f"📂 Файл {SUBSCRIBERS_FILE} не существует")
    return set()


def save_subscribers(subs):
    try:
        with open(SUBSCRIBERS_FILE, 'w') as f:
            json.dump(list(subs), f)
        logger.info(f"💾 Сохранено подписчиков: {len(subs)}")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения подписчиков: {e}")


subscribers = load_subscribers()
subscribers_lock = threading.Lock()

# ======================================================================
# Класс для управления статистикой сигналов
# ======================================================================
class StatsManager:
    def __init__(self, stats_file='stats.json'):
        self.stats_file = stats_file
        self.signals = []
        self.load_stats()

    def load_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.signals = json.load(f)
                logger.info(f"📊 Загружено записей статистики: {len(self.signals)}")
            except Exception as e:
                logger.error(f"Ошибка загрузки статистики: {e}")
                self.signals = []
        else:
            self.signals = []

    def save_stats(self):
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.signals, f, indent=2)
            logger.info(f"💾 Статистика сохранена: {len(self.signals)} записей")
        except Exception as e:
            logger.error(f"Ошибка сохранения статистики: {e}")

    def add_signal(self, signal):
        self.signals.append(signal)
        self.save_stats()

    def update_results(self, current_price):
        updated = False
        for sig in self.signals:
            if sig.get('result') is not None:
                continue
            # Тайм-аут через 1 час
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
            self.save_stats()

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

# ========== КЛАСС ДЛЯ МАШИННОГО ОБУЧЕНИЯ (ЗАГОТОВКА) ==========
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
            # Защита от случая, когда predict_proba возвращает один класс
            if len(proba) > 1:
                return proba[1]
            else:
                return proba[0]
        except Exception as e:
            logger.warning(f"ML prediction error: {e}")
            return 0.5


# Глобальный объект ML (создаётся один раз)
ml_gen = MLSignalGenerator()


# ========== ХРАНИЛИЩЕ ДАННЫХ ==========
class PriceStorage:
    def __init__(self, maxlen=200):
        self.maxlen = maxlen
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.volumes = []  # для OBV
        self.last_signal = None

    def add_candle(self, candle):
        self.opens.append(float(candle['open']))
        self.highs.append(float(candle['high']))
        self.lows.append(float(candle['low']))
        self.closes.append(float(candle['close']))
        self.volumes.append(float(candle.get('volume', 0)))

        # Ограничиваем длину
        if len(self.opens) > self.maxlen:
            self.opens.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.volumes.pop(0)

    def clear(self):
        self.opens.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.volumes.clear()


price_storage = PriceStorage()


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
    """Average True Range - измеряет волатильность"""
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
    recent_high = max(highs[-lookback - 1:-1])  # исключаем последнюю свечу
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
    """Возвращает ADX, DI+, DI-"""
    if len(closes) < period + 1:
        return 0, 0, 0
    tr = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr.append(max(hl, hc, lc))
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    atr = sum(tr[-period:]) / period
    plus_di = 100 * (sum(plus_dm[-period:]) / period) / atr if atr != 0 else 0
    minus_di = 100 * (sum(minus_dm[-period:]) / period) / atr if atr != 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) != 0 else 0
    adx_val = sum([dx] * period) / period  # упрощённо, на самом деле сглаживают
    return adx_val, plus_di, minus_di


def stochastic(highs, lows, closes, k_period=14, d_period=3):
    """Возвращает %K и %D"""
    if len(closes) < k_period + d_period:
        return 50, 50
    highest_high = max(highs[-k_period:])
    lowest_low = min(lows[-k_period:])
    k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low) if (highest_high - lowest_low) != 0 else 50
    # Простейшее сглаживание %D как SMA от %K за 3 периода
    d = sum([k] * d_period) / d_period
    return k, d


def pivot_points(high, low, close):
    """Классические Pivot Points на основе вчерашнего дня"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3


def calculate_normalized_score(ind):
    """
    Нормализованная оценка от -100 (макс. продажа) до +100 (макс. покупка).
    """
    score = 0
    max_score = 0

    # RSI (вес 2)
    if ind['rsi'] < 30:
        score += 2
    elif ind['rsi'] > 70:
        score -= 2
    max_score += 2

    # MACD (вес 2)
    if ind['macd'] > 0:
        score += 2
    else:
        score -= 2
    max_score += 2

    # Bollinger Bands (вес 2)
    price = ind['price']
    if price <= ind['bb_lower']:
        score += 2
    elif price >= ind['bb_upper']:
        score -= 2
    max_score += 2

    # EMA тренд (вес 1)
    if ind['ema'][5] > ind['ema'][20]:
        score += 1
    else:
        score -= 1
    max_score += 1

    # ADX + направление (вес 3, если тренд сильный)
    if ind.get('adx', 0) > 25:
        if ind['plus_di'] > ind['minus_di']:
            score += 3
        else:
            score -= 3
        max_score += 3

    # Stochastic (вес 2)
    if ind.get('stoch_k', 50) < 20:
        score += 2
    elif ind.get('stoch_k', 50) > 80:
        score -= 2
    max_score += 2

    # Поддержка/сопротивление (близость к уровням) (вес 2)
    dist_to_sup = ind.get('distance_to_support', 1000)
    dist_to_res = ind.get('distance_to_resistance', 1000)
    if dist_to_sup < 10 and dist_to_sup < dist_to_res:
        score += 2  # у поддержки, возможен отскок
    elif dist_to_res < 10 and dist_to_res < dist_to_sup:
        score -= 2  # у сопротивления, возможен откат
    max_score += 2

    # Изменение за 3 минуты (вес 1)
    change = ind.get('change_3min', 0)
    if change > 0.0001:  # примерно 1 пипс
        score += 1
    elif change < -0.0001:
        score -= 1
    max_score += 1

    # Нормализация в проценты от -100% до +100%
    normalized = (score / max_score) * 100 if max_score > 0 else 0
    return normalized


def generate_message(ind):
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

        msg = f"""{emoji} *АНАЛИЗ EUR/USD* {emoji}

📊 *ОБЩАЯ ВЕРОЯТНОСТЬ*
┌─ ⬆️ ВВЕРХ: {up:.1f}%
└─ ⬇️ ВНИЗ: {down:.1f}%
🎯 Уверенность: {conf:.1f}%
💡 Рекомендация: {rec}
"""

        # ATR
        if 'atr_percent' in ind:
            msg += f"\n📊 *ATR*: {ind['atr_percent']:.3f}%"

        # ML вероятность
        if 'ml_prob_up' in ind:
            msg += f"\n🤖 *ML вероятность*: {ind['ml_prob_up'] * 100:.1f}%"

        # ADX
        if 'adx' in ind:
            msg += f"\n📈 *ADX*: {ind['adx']:.1f} (DI+:{ind['plus_di']:.1f} DI-:{ind['minus_di']:.1f})"

        # Stochastic
        if 'stoch_k' in ind:
            msg += f"\n📊 *Stochastic*: %K={ind['stoch_k']:.1f} %D={ind['stoch_d']:.1f}"

        # 3-минутное изменение
        if 'change_3min_pct' in ind:
            arrow = "⬆️" if ind['change_3min'] > 0 else "⬇️" if ind['change_3min'] < 0 else "➡️"
            msg += f"\n📊 *3 мин изменение*: {arrow} {ind['change_3min_pct']:.3f}%"

        return msg
    except KeyError as e:
        logger.error(f"KeyError in generate_message: {e}")
        return f"❌ Ошибка формирования сигнала: отсутствует ключ {e}"
    except Exception as e:
        logger.error(f"Unexpected error in generate_message: {e}")
        return "❌ Внутренняя ошибка при формировании сигнала"

# ========== ЗАГРУЗКА ДАННЫХ ==========
async def fetch_candles(api_key, bars=50):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': 'EUR/USD',
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
        logger.error(f"fetch error: {e}")
    return None


async def fetch_last_candle(api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': 'EUR/USD',
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
        logger.error(f"fetch_last_candle error: {e}")
    return None


async def update_prices():
    candles = await fetch_candles(TWELVE_API_KEY, 200)
    if candles:
        price_storage.clear()
        for c in candles[::-1]:
            price_storage.add_candle(c)
        return True
    return False


async def get_indicators():
    try:
        if len(price_storage.closes) < 20:
            if not await update_prices():
                return None

        c = price_storage.closes
        h = price_storage.highs
        l = price_storage.lows
        cur = c[-1]
        ind = {}
        ind['price'] = cur

        # Изменение за 3 минуты (если есть данные)
        if len(c) >= 3:
            price_3min_ago = c[-3]
            change_3min = cur - price_3min_ago
            ind['change_3min'] = change_3min
            ind['change_3min_pct'] = (change_3min / price_3min_ago) * 100
        else:
            ind['change_3min'] = 0
            ind['change_3min_pct'] = 0

        # RSI
        ind['rsi'] = rsi(c, 14)
        if ind['rsi'] > 70:
            ind['rsi_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (сигнал к продаже)'
        elif ind['rsi'] < 30:
            ind['rsi_signal'] = 'ПЕРЕПРОДАННОСТЬ (сигнал к покупке)'
        elif ind['rsi'] > 50:
            ind['rsi_signal'] = 'ВОСХОДЯЩИЙ ТРЕНД'
        else:
            ind['rsi_signal'] = 'НИСХОДЯЩИЙ ТРЕНД'

        # MACD
        macd_line = macd(c, 12, 26)
        ind['macd'] = macd_line
        ind['macd_trend'] = 'БЫЧИЙ СИГНАЛ' if macd_line > 0 else 'МЕДВЕЖИЙ СИГНАЛ' if macd_line < 0 else 'НЕЙТРАЛЬНО'

        # Bollinger Bands
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

        # SMA
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

        # EMA
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

        # OBV
        if hasattr(price_storage, 'volumes') and price_storage.volumes:
            obv_values = calculate_obv(price_storage.closes, price_storage.volumes)
            ind['obv'] = obv_values[-1] if obv_values else 0
            ind['obv_trend'] = obv_trend(obv_values, 14)
        else:
            ind['obv'] = 0
            ind['obv_trend'] = 'neutral'

        # ATR
        atr, atr_pct = calculate_atr(price_storage.highs, price_storage.lows, price_storage.closes, 14)
        ind['atr'] = atr
        ind['atr_percent'] = atr_pct

        # Support / Resistance
        sup, res, ns, nr = find_support_resistance(h, l, c)
        ind['support_levels'] = sup
        ind['resistance_levels'] = res
        ind['nearest_support'] = ns
        ind['nearest_resistance'] = nr
        ind['distance_to_support'] = (cur - ns) * 10000 if ns else 0
        ind['distance_to_resistance'] = (nr - cur) * 10000 if nr else 0

        # False breakout
        ind['breakout'] = detect_false_breakout(price_storage.highs, price_storage.lows, price_storage.closes)

        # ADX
        adx_val, plus_di, minus_di = adx(price_storage.highs, price_storage.lows, price_storage.closes, 14)
        ind['adx'] = adx_val
        ind['plus_di'] = plus_di
        ind['minus_di'] = minus_di

        # Stochastic
        stoch_k, stoch_d = stochastic(price_storage.highs, price_storage.lows, price_storage.closes, 14, 3)
        ind['stoch_k'] = stoch_k
        ind['stoch_d'] = stoch_d

        # Pivot Points (если достаточно данных)
        if len(price_storage.highs) >= 1440:  # 24 часа по 1 минуте
            day_high = max(price_storage.highs[-1440:])
            day_low = min(price_storage.lows[-1440:])
            day_close = price_storage.closes[-1440]  # закрытие 24 часа назад
            pivot, r1, r2, r3, s1, s2, s3 = pivot_points(day_high, day_low, day_close)
            ind['pivot'] = pivot
            ind['r1'] = r1
            ind['r2'] = r2
            ind['r3'] = r3
            ind['s1'] = s1
            ind['s2'] = s2
            ind['s3'] = s3

        # Нормализованная оценка
        ind['ml_score'] = calculate_normalized_score(ind)

        # Преобразуем ml_score в вероятности для совместимости с generate_message
        if ind['ml_score'] >= 0:
            ind['prob_up'] = 50 + ind['ml_score'] / 2
            ind['prob_down'] = 50 - ind['ml_score'] / 2
        else:
            ind['prob_up'] = 50 + ind['ml_score'] / 2  # ml_score отрицательный, prob_up < 50
            ind['prob_down'] = 50 - ind['ml_score'] / 2
        ind['confidence'] = abs(ind['ml_score'])

        # ML модель (если есть)
        ind['ml_prob_up'] = ml_gen.predict(ind)

        ind['price'] = cur
        ind['timestamp'] = datetime.now().strftime('%H:%M:%S')
        price_storage.last_signal = ind
        return ind
    except Exception as e:
        logger.error(f"Error in get_indicators: {e}")
        return None


# ========== КЛАВИАТУРЫ ==========
def main_menu():
    kb = [
        [InlineKeyboardButton("📊 Получить сигнал", callback_data='signal'),
         InlineKeyboardButton("📈 Статус", callback_data='status')],
        [InlineKeyboardButton("📊 Статистика", callback_data='stats'),
         InlineKeyboardButton("🔔 Автосигнал (3 мин)", callback_data='auto_on')],
        [InlineKeyboardButton("⏹️ Стоп", callback_data='auto_off')],
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
    return "<h1>EUR/USD Pro Bot</h1><p>Running</p>"


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
            cb_id = data['callback_query']['id']          # ID callback'а для answer_callback_query
            loop.run_until_complete(handle_callback(chat_id, cb, cb_id))
        elif 'message' in data and 'text' in data['message']:
            chat_id = data['message']['chat']['id']
            text = data['message']['text']
            loop.run_until_complete(handle_message(chat_id, text))
        loop.close()
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'ok': False}), 500


async def handle_callback(chat_id, cb, cb_id):
    logger.info(f"🔥 Callback received: {cb} from {chat_id}")
    bot = Bot(token=BOT_TOKEN)
    try:
        await bot.answer_callback_query(cb_id)

        if cb == 'signal':
            await send_signal(bot, chat_id)
        elif cb == 'status':
            await send_status(bot, chat_id)
        elif cb == 'stats':
            await send_stats(bot, chat_id)
        elif cb == 'auto_on':
            with subscribers_lock:
                subscribers.add(chat_id)
                save_subscribers(subscribers)
                logger.info(f"✅ Подписчик {chat_id} добавлен, теперь всего {len(subscribers)}")
            await bot.send_message(chat_id, "✅ Автосигналы включены (каждые 3 мин)")
        elif cb == 'auto_off':
            with subscribers_lock:
                if chat_id in subscribers:
                    subscribers.remove(chat_id)
                    save_subscribers(subscribers)
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
        except:
            pass


async def handle_message(chat_id, text):
    bot = Bot(token=BOT_TOKEN)
    if text == '/start':
        await bot.send_message(chat_id, "🤖 EUR/USD", reply_markup=main_menu())
    elif text == '/signal':
        await send_signal(bot, chat_id)
    elif text == '/status':
        await send_status(bot, chat_id)
    elif text == '/stats':
        await send_stats(bot, chat_id)
    elif text == '/stop':
        with subscribers_lock:
            if chat_id in subscribers:
                subscribers.remove(chat_id)
                save_subscribers(subscribers)
                await bot.send_message(chat_id, "⏹️ Автосигналы остановлены")
            else:
                await bot.send_message(chat_id, "❌ Автосигналы не были включены")
    else:
        await bot.send_message(chat_id, "❌ Неизвестная команда")


async def send_signal(bot, chat_id):
    await bot.send_message(chat_id, "🔄 Анализирую...")
    ind = await get_indicators()
    if not ind:
        await bot.send_message(chat_id, "❌ Ошибка получения данных")
        return

    up = ind['prob_up']
    down = ind['prob_down']
    direction = 'buy' if up > down else 'sell'
    entry = ind['price']
    atr = ind.get('atr', 0.001)  # запасное значение
    tp_distance = atr * 1.5
    sl_distance = atr * 0.75

    if direction == 'buy':
        tp = entry + tp_distance
        sl = entry - sl_distance
    else:
        tp = entry - tp_distance
        sl = entry + sl_distance

    # Сохраняем сигнал в статистику
    signal_record = {
        'timestamp': time.time(),
        'price': entry,
        'direction': direction,
        'tp': tp,
        'sl': sl,
        'result': None,
        'exit_price': None,
        'exit_time': None
    }
    stats_manager.add_signal(signal_record)

    msg = generate_message(ind)
    await bot.send_message(chat_id, msg, parse_mode='Markdown')


async def send_status(bot, chat_id):
    with subscribers_lock:
        auto = "вкл" if chat_id in subscribers else "выкл"
    await bot.send_message(chat_id, f"📊 Статус:\nАвтосигналы: {auto}\nСвечей: {len(price_storage.closes)}")

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
    while True:
        try:
            await asyncio.sleep(180)  # 3 минуты

            # Синхронизация подписчиков с файлом
            file_subs = load_subscribers()
            with subscribers_lock:
                if file_subs != subscribers:
                    logger.warning(
                        f"🔄 Подписчики в памяти ({len(subscribers)}) отличаются от файла ({len(file_subs)}). Синхронизируем.")
                    subscribers.clear()
                    subscribers.update(file_subs)
                subs = list(subscribers)
                logger.info(f"📋 Подписчиков в памяти: {len(subs)}")

            # Обновляем последнюю свечу
            new_candle = await fetch_last_candle(TWELVE_API_KEY)
            if new_candle:
                price_storage.add_candle(new_candle)
                logger.info(f"✅ Добавлена новая свеча: {new_candle.get('datetime')} = {new_candle.get('close')}")
            else:
                logger.warning("⚠️ Не удалось получить новую свечу, сигнал будет на старых данных")

            # Обновляем результаты старых сигналов
            if new_candle:
                current_price = float(new_candle['close'])
                stats_manager.update_results(current_price)

            # Рассылаем сигналы подписчикам
            if not subs:
                logger.info("😴 Нет подписчиков, пропускаем рассылку")
                continue

            logger.info(f"🔄 Рассылка для {len(subs)} подписчиков")
            for uid in subs:
                try:
                    bot = Bot(token=BOT_TOKEN)
                    ind = await get_indicators()
                    if ind:
                        await bot.send_message(uid, generate_message(ind), parse_mode='Markdown')
                        logger.info(f"✅ Сигнал отправлен {uid}")
                    else:
                        await bot.send_message(uid, "❌ Ошибка получения сигнала")
                        logger.error(f"❌ Нет индикаторов для {uid}")
                except Exception as e:
                    logger.error(f"❌ Ошибка отправки для {uid}: {e}")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка в auto_worker: {e}")
            await asyncio.sleep(10)


def start_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(auto_worker())


threading.Thread(target=start_worker, daemon=True).start()
logger.info("✅ Фоновый поток запущен")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)