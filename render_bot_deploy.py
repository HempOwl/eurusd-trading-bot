import asyncio
import logging
import os
import threading
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

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


# ========== КЛАСС ДЛЯ МАШИННОГО ОБУЧЕНИЯ (ЗАГОТОВКА) ==========
class MLSignalGenerator:
    def __init__(self, model_path='model.pkl'):
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=10)

    def save_model(self):
        joblib.dump(self.model, self.model_path)

    def prepare_features(self, ind):
        # Вектор признаков для модели
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
            return 0.5  # нейтрально
        features = self.prepare_features(ind)
        X = np.array(features).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        # Предполагаем класс 1 = вверх, 0 = вниз
        return proba[1]


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
    obv_ema = ema(obv_values, period)  # используем функцию ema, определённую выше
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


def calculate_normalized_score(ind):
    """
    Нормализованная оценка от -100 (макс. продажа) до +100 (макс. покупка).
    """
    score = 0
    max_score = 0

    # RSI
    if ind['rsi'] < 30:
        score += 1
    elif ind['rsi'] > 70:
        score -= 1
    max_score += 1

    # MACD
    if ind['macd'] > 0:
        score += 1
    else:
        score -= 1
    max_score += 1

    # Bollinger Bands
    price = ind['price']
    if price <= ind['bb_lower']:
        score += 1
    elif price >= ind['bb_upper']:
        score -= 1
    max_score += 1

    # EMA тренд (EMA(5) > EMA(20) – бычий)
    if ind['ema'][5] > ind['ema'][20]:
        score += 1
    else:
        score -= 1
    max_score += 1

    # Нормализация в проценты от -100% до +100%
    normalized = (score / max_score) * 100
    return normalized


def generate_message(ind):
    if not ind:
        return "❌ Нет данных"
    price = ind['price']
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

📈 *RSI*
┌─ Значение: `{ind['rsi']:.1f}`
└─ Сигнал: {ind['rsi_signal']}

📊 *MACD*
┌─ MACD: `{ind['macd']:.5f}`
├─ Сигнал: `0.00000`
├─ Гистограмма: `0.00000`
└─ Тренд: {ind['macd_trend']}

📉 *Полосы Боллинджера*
┌─ Верхняя: `{ind['bb_upper']:.5f}`
├─ Средняя: `{ind['bb_middle']:.5f}`
├─ Нижняя: `{ind['bb_lower']:.5f}`
├─ Ширина: `{ind['bb_width']:.2f}%`
├─ Позиция: {ind['bb_position']}
└─ Сигнал: {ind['bb_signal']}

📏 *Скользящие средние (SMA)*\n"""
    for p in [5, 10, 20, 50]:
        if p in ind['sma']:
            sig = ind.get(f'sma_{p}_signal', '')
            msg += f"├─ SMA({p}): `{ind['sma'][p]:.5f}` {sig}\n"
    msg += "\n📊 *Экспоненциальные средние (EMA)*\n"
    for p in [5, 10, 20]:
        if p in ind['ema']:
            sig = ind.get(f'ema_{p}_signal', '')
            msg += f"├─ EMA({p}): `{ind['ema'][p]:.5f}` {sig}\n"
    sup_str = f"`{ind['nearest_support']:.5f}`" if ind['nearest_support'] else "`не определен`"
    res_str = f"`{ind['nearest_resistance']:.5f}`" if ind['nearest_resistance'] else "`не определен`"
    d_sup = f"{ind['distance_to_support']:.0f}" if ind['distance_to_support'] else "?"
    d_res = f"{ind['distance_to_resistance']:.0f}" if ind['distance_to_resistance'] else "?"
    msg += f"""
📊 *Уровни поддержки/сопротивления*
┌─ Ближайшая поддержка: {sup_str} (дист: {d_sup} пипсов)
└─ Ближайшее сопротивление: {res_str} (дист: {d_res} пипсов)
"""
    if ind['support_levels']:
        msg += f"├─ Уровни поддержки: {', '.join([f'{x:.5f}' for x in ind['support_levels']])}\n"
    if ind['resistance_levels']:
        msg += f"└─ Уровни сопротивления: {', '.join([f'{x:.5f}' for x in ind['resistance_levels']])}\n"

    # Добавим новые показатели (если есть)
    if 'atr_percent' in ind:
        msg += f"\n📊 *ATR*: {ind['atr_percent']:.3f}%"
    if 'breakout' in ind and ind['breakout'] != 'no_breakout':
        msg += f"\n🚩 *Пробой*: {ind['breakout']}"
    if 'ml_prob_up' in ind:
        msg += f"\n🤖 *ML вероятность*: {ind['ml_prob_up'] * 100:.1f}%"

    msg += f"\n\n#{'BUY' if up > down else 'SELL'} #EURUSD"
    return msg


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
    if len(price_storage.closes) < 20:
        if not await update_prices():
            return None

    c = price_storage.closes
    h = price_storage.highs
    l = price_storage.lows
    cur = c[-1]
    ind = {}

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


# ========== КЛАВИАТУРЫ ==========
def main_menu():
    kb = [
        [InlineKeyboardButton("📊 Получить сигнал", callback_data='signal'),
         InlineKeyboardButton("📈 Статус", callback_data='status')],
        [InlineKeyboardButton("🔔 Автосигнал (5 мин)", callback_data='auto_on'),
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
            loop.run_until_complete(handle_callback(chat_id, cb))
        elif 'message' in data and 'text' in data['message']:
            chat_id = data['message']['chat']['id']
            text = data['message']['text']
            loop.run_until_complete(handle_message(chat_id, text))
        loop.close()
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'ok': False}), 500


async def handle_callback(chat_id, cb):
    logger.info(f"🔥 Callback received: {cb} from {chat_id}")
    bot = Bot(token=BOT_TOKEN)
    if cb == 'signal':
        await send_signal(bot, chat_id)
    elif cb == 'status':
        await send_status(bot, chat_id)
    elif cb == 'auto_on':
        with subscribers_lock:
            subscribers.add(chat_id)
            save_subscribers(subscribers)
            logger.info(f"✅ Подписчик {chat_id} добавлен, теперь всего {len(subscribers)}")
        await bot.send_message(chat_id, "✅ Автосигналы включены (каждые 5 мин)")
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
        await bot.send_message(chat_id, "❓ Неизвестная команда")


async def handle_message(chat_id, text):
    bot = Bot(token=BOT_TOKEN)
    if text == '/start':
        await bot.send_message(chat_id, "🤖 EUR/USD", reply_markup=main_menu())
    elif text == '/signal':
        await send_signal(bot, chat_id)
    elif text == '/status':
        await send_status(bot, chat_id)
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
    await bot.send_message(chat_id, generate_message(ind), parse_mode='Markdown')


async def send_status(bot, chat_id):
    with subscribers_lock:
        auto = "вкл" if chat_id in subscribers else "выкл"
    await bot.send_message(chat_id, f"📊 Статус:\nАвтосигналы: {auto}\nСвечей: {len(price_storage.closes)}")


# ========== ФОНОВЫЙ ПОТОК ==========
async def auto_worker():
    logger.info("🚀 Автосигналы запущены (интервал 5 мин)")
    while True:
        try:
            await asyncio.sleep(290)  # 5 минут

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