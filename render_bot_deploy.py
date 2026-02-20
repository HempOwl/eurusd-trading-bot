import asyncio
import logging
import json
import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import psycopg2
from psycopg2.extras import Json

# ========== НАСТРОЙКА ЛОГИРОВАНИЯ ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ ==========
DATABASE_URL = os.environ.get('DATABASE_URL')
conn = None
if DATABASE_URL:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        # Создаём таблицу, если её нет
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    chat_id BIGINT PRIMARY KEY
                )
            """)
            conn.commit()
        logger.info("✅ Подключение к БД установлено")
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к БД: {e}")
else:
    logger.error("❌ DATABASE_URL не задана, подписчики не будут сохраняться!")

# ========== ФУНКЦИИ ДЛЯ РАБОТЫ С ПОДПИСЧИКАМИ ==========
def load_subscribers():
    """Загружает подписчиков из БД"""
    if not conn:
        logger.error("❌ load_subscribers: нет подключения к БД")
        return set()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id FROM subscribers")
            rows = cur.fetchall()
            subs = set(row[0] for row in rows)
            logger.info(f"✅ Загружено {len(subs)} подписчиков из БД")
            return subs
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки подписчиков: {e}")
        return set()

def save_subscribers(subs):
    """Сохраняет подписчиков в БД (полная перезапись)"""
    if not conn:
        logger.error("❌ save_subscribers: нет подключения к БД")
        return
    try:
        with conn.cursor() as cur:
            # Очищаем таблицу
            cur.execute("DELETE FROM subscribers")
            logger.debug("Очищена таблица subscribers")
            # Вставляем всех текущих подписчиков
            for chat_id in subs:
                cur.execute("INSERT INTO subscribers (chat_id) VALUES (%s)", (chat_id,))
                logger.debug(f"Добавлен подписчик {chat_id}")
            conn.commit()
            logger.info(f"✅ Сохранено {len(subs)} подписчиков в БД")
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения подписчиков: {e}")
        # Не пытаемся переподключиться здесь, чтобы избежать проблем с global

# ========== ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ==========
app = Flask(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
TWELVE_API_KEY = os.environ.get('TWELVE_API_KEY')

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не задан!")
if not TWELVE_API_KEY:
    logger.error("❌ TWELVE_API_KEY не задан!")

# Множество подписчиков (загружается из БД)
subscribers_lock = threading.Lock()
subscribers = load_subscribers()

# ========== ХРАНИЛИЩЕ ДАННЫХ ==========
class PriceStorage:
    def __init__(self, maxlen=50):
        self.maxlen = maxlen
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.last_signal = None

    def add_candle(self, candle):
        self.opens.append(float(candle['open']))
        self.highs.append(float(candle['high']))
        self.lows.append(float(candle['low']))
        self.closes.append(float(candle['close']))
        if len(self.opens) > self.maxlen:
            self.opens.pop(0); self.highs.pop(0); self.lows.pop(0); self.closes.pop(0)

    def clear(self):
        self.opens.clear(); self.highs.clear(); self.lows.clear(); self.closes.clear()

price_storage = PriceStorage()

# ========== ФУНКЦИИ ИНДИКАТОРОВ ==========
def sma(data, period):
    if len(data) < period: return data[-1]
    return sum(data[-period:]) / period

def ema(data, period):
    if len(data) < period: return data[-1]
    multiplier = 2 / (period + 1)
    ema = sum(data[-period:]) / period
    for price in data[-period+1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def rsi(data, period=14):
    if len(data) < period + 1: return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        change = data[-i] - data[-i-1]
        if change > 0:
            gains.append(change); losses.append(0.0)
        else:
            gains.append(0.0); losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0: return 100.0
    return 100 - (100 / (1 + avg_gain/avg_loss))

def bbands(data, period=20, std=2):
    if len(data) < period:
        m = data[-1]; return m*1.02, m, m*0.98
    m = sum(data[-period:]) / period
    variance = sum((x - m)**2 for x in data[-period:]) / period
    s = variance ** 0.5
    return m + std*s, m, m - std*s

def macd(data, fast=12, slow=26):
    if len(data) < slow: return 0.0
    return ema(data, fast) - ema(data, slow)

def find_support_resistance(high, low, close, window=5):
    supports, resistances = [], []
    n = len(close)
    for i in range(window, n - window):
        if all(low[i] <= low[i-j] for j in range(1, window+1)) and all(low[i] <= low[i+j] for j in range(1, window+1)):
            supports.append(low[i])
        if all(high[i] >= high[i-j] for j in range(1, window+1)) and all(high[i] >= high[i+j] for j in range(1, window+1)):
            resistances.append(high[i])
    def cluster(levels, thr=0.0005):
        if not levels: return []
        levels.sort()
        cl = [levels[0]]
        res = []
        for lev in levels[1:]:
            if abs(lev - sum(cl)/len(cl)) < thr:
                cl.append(lev)
            else:
                res.append(sum(cl)/len(cl))
                cl = [lev]
        res.append(sum(cl)/len(cl))
        return res
    supports = cluster(supports)
    resistances = cluster(resistances)
    cur = close[-1]
    ns = None; nr = None
    for s in supports:
        if s < cur: ns = s
    for r in resistances:
        if r > cur:
            nr = r
            break
    return supports[-3:], resistances[-3:], ns, nr

def calculate_probability(ind):
    up = down = 0
    if ind.get('rsi', 50) < 30: up += 3
    elif ind.get('rsi', 50) > 70: down += 3
    elif ind.get('rsi', 50) > 50: up += 1
    else: down += 1
    if 'macd_trend' in ind:
        if 'БЫЧИЙ' in ind['macd_trend']: up += 2
        elif 'МЕДВЕЖИЙ' in ind['macd_trend']: down += 2
    if 'bb_signal' in ind:
        if 'ПЕРЕПРОДАННОСТЬ' in ind['bb_signal']: up += 2
        elif 'ПЕРЕКУПЛЕННОСТЬ' in ind['bb_signal']: down += 2
    for p in [5,10,20,50]:
        sig = ind.get(f'sma_{p}_signal', '')
        if 'ВЫШЕ' in sig: up += 1
        elif 'НИЖЕ' in sig: down += 1
    for p in [5,10,20]:
        sig = ind.get(f'ema_{p}_signal', '')
        if 'ВЫШЕ' in sig: up += 1
        elif 'НИЖЕ' in sig: down += 1
    if ind.get('nearest_support') and ind.get('nearest_resistance'):
        if ind['distance_to_support'] < ind['distance_to_resistance']: up += 1
        else: down += 1
    total = up + down
    if total:
        prob_up = round((up / total) * 100, 1)
        prob_down = round((down / total) * 100, 1)
        conf = round(abs(up - down) / total * 100, 1)
    else:
        prob_up = prob_down = 50.0; conf = 0.0
    return prob_up, prob_down, conf

def generate_message(ind):
    if not ind: return "❌ Нет данных"
    price = ind['price']; up = ind['prob_up']; down = ind['prob_down']; conf = ind['confidence']
    if up > down + 10 and conf > 50: rec = "📈 СИЛЬНАЯ ПОКУПКА"; emoji = "🟢"
    elif up > down: rec = "📈 ПОКУПКА"; emoji = "🟢"
    elif down > up + 10 and conf > 50: rec = "📉 СИЛЬНАЯ ПРОДАЖА"; emoji = "🔴"
    elif down > up: rec = "📉 ПРОДАЖА"; emoji = "🔴"
    else: rec = "⏸️ ОЖИДАНИЕ"; emoji = "⚪"
    msg = f"""{emoji} *ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ EUR/USD* {emoji}
⏰ {ind['timestamp']}
💰 *Цена:* `{price:.5f}`

📊 *ОБЩАЯ ВЕРОЯТНОСТЬ*
┌─ ⬆️ ВВЕРХ: {up}%
└─ ⬇️ ВНИЗ: {down}%
🎯 Уверенность: {conf}%
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
    for p in [5,10,20,50]:
        if p in ind['sma']:
            sig = ind.get(f'sma_{p}_signal', '')
            msg += f"├─ SMA({p}): `{ind['sma'][p]:.5f}` {sig}\n"
    msg += "\n📊 *Экспоненциальные средние (EMA)*\n"
    for p in [5,10,20]:
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
    msg += f"\n#{'BUY' if up > down else 'SELL'} #EURUSD"
    return msg

# ========== ЗАГРУЗКА ДАННЫХ ==========
async def fetch_candles(api_key, bars=50):
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': 'EUR/USD', 'interval': '1min', 'outputsize': bars, 'apikey': api_key}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('values')
    except Exception as e:
        logger.error(f"fetch error: {e}")
    return None

async def update_prices():
    candles = await fetch_candles(TWELVE_API_KEY, 50)
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
    c = price_storage.closes; h = price_storage.highs; l = price_storage.lows
    cur = c[-1]
    ind = {}
    # RSI
    ind['rsi'] = rsi(c, 14)
    if ind['rsi'] > 70: ind['rsi_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (сигнал к продаже)'
    elif ind['rsi'] < 30: ind['rsi_signal'] = 'ПЕРЕПРОДАННОСТЬ (сигнал к покупке)'
    elif ind['rsi'] > 50: ind['rsi_signal'] = 'ВОСХОДЯЩИЙ ТРЕНД'
    else: ind['rsi_signal'] = 'НИСХОДЯЩИЙ ТРЕНД'
    # MACD
    macd_line = macd(c, 12, 26)
    ind['macd'] = macd_line
    ind['macd_trend'] = 'БЫЧИЙ СИГНАЛ' if macd_line > 0 else 'МЕДВЕЖИЙ СИГНАЛ' if macd_line < 0 else 'НЕЙТРАЛЬНО'
    # BB
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
    # SMA
    ind['sma'] = {}
    for p in [5,10,20,50]:
        val = sma(c, p)
        ind['sma'][p] = val
        if cur > val: ind[f'sma_{p}_signal'] = '⬆️ ВЫШЕ'
        elif cur < val: ind[f'sma_{p}_signal'] = '⬇️ НИЖЕ'
        else: ind[f'sma_{p}_signal'] = '⏺️ ОКОЛО'
    # EMA
    ind['ema'] = {}
    for p in [5,10,20]:
        val = ema(c, p)
        ind['ema'][p] = val
        if cur > val: ind[f'ema_{p}_signal'] = '⬆️ ВЫШЕ'
        elif cur < val: ind[f'ema_{p}_signal'] = '⬇️ НИЖЕ'
        else: ind[f'ema_{p}_signal'] = '⏺️ ОКОЛО'
    # S/R
    sup, res, ns, nr = find_support_resistance(h, l, c)
    ind['support_levels'] = sup; ind['resistance_levels'] = res
    ind['nearest_support'] = ns; ind['nearest_resistance'] = nr
    ind['distance_to_support'] = (cur - ns)*10000 if ns else 0
    ind['distance_to_resistance'] = (nr - cur)*10000 if nr else 0
    # Вероятность
    ind['prob_up'], ind['prob_down'], ind['confidence'] = calculate_probability(ind)
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
        [InlineKeyboardButton("ℹ️ Помощь", callback_data='help'),
         InlineKeyboardButton("⚙️ Настройки", callback_data='settings')]
    ]
    return InlineKeyboardMarkup(kb)

def help_menu():
    kb = [
        [InlineKeyboardButton("📊 Индикаторы", callback_data='help_indicators'),
         InlineKeyboardButton("💰 Торговля", callback_data='help_trading')],
        [InlineKeyboardButton("❓ FAQ", callback_data='help_faq'),
         InlineKeyboardButton("◀️ Назад", callback_data='back')]
    ]
    return InlineKeyboardMarkup(kb)

def settings_menu():
    kb = [
        [InlineKeyboardButton("📏 RSI период", callback_data='set_rsi'),
         InlineKeyboardButton("📊 MACD", callback_data='set_macd')],
        [InlineKeyboardButton("📉 Bollinger Bands", callback_data='set_bb'),
         InlineKeyboardButton("◀️ Назад", callback_data='back')]
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
    bot = Bot(token=BOT_TOKEN)
    if cb == 'signal':
        await send_signal(bot, chat_id)
    elif cb == 'status':
        await send_status(bot, chat_id)
    elif cb == 'auto_on':
        with subscribers_lock:
            subscribers.add(chat_id)
            save_subscribers(subscribers)
        await bot.send_message(chat_id, "✅ Автосигналы включены (каждые 5 мин)")
    elif cb == 'auto_off':
        with subscribers_lock:
            if chat_id in subscribers:
                subscribers.remove(chat_id)
                save_subscribers(subscribers)
                await bot.send_message(chat_id, "⏹️ Автосигналы остановлены")
            else:
                await bot.send_message(chat_id, "❌ Автосигналы не были включены")
    elif cb == 'help':
        await bot.send_message(chat_id, "📖 Раздел помощи", reply_markup=help_menu())
    elif cb == 'settings':
        await bot.send_message(chat_id, "⚙️ Настройки", reply_markup=settings_menu())
    elif cb == 'back':
        await bot.send_message(chat_id, "Главное меню", reply_markup=main_menu())
    elif cb == 'help_indicators':
        await bot.send_message(chat_id, "📊 Индикаторы: RSI(14), MACD(12/26), BB(20,2), SMA(5,10,20,50), EMA(5,10,20)")
    elif cb == 'help_trading':
        await bot.send_message(chat_id, "💰 Риск ≤3%, экспирация 3 мин, уверенность >60%")
    elif cb == 'help_faq':
        await bot.send_message(chat_id, "❓ Бот работает 24/7, сигналы по запросу и авто каждые 5 мин")
    elif cb == 'set_rsi':
        await bot.send_message(chat_id, "📏 RSI период = 14 (изменение через код)")
    elif cb == 'set_macd':
        await bot.send_message(chat_id, "📊 MACD: 12/26/9")
    elif cb == 'set_bb':
        await bot.send_message(chat_id, "📉 BB: period=20, std=2")
    else:
        await bot.send_message(chat_id, "❓ Неизвестная команда")

async def handle_message(chat_id, text):
    bot = Bot(token=BOT_TOKEN)
    if text == '/start':
        await bot.send_message(chat_id, "🤖 EUR/USD Pro Bot", reply_markup=main_menu())
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
            logger.debug("⏳ Цикл автосигналов: ожидание 5 минут...")
            await asyncio.sleep(300)  # 5 минут
            logger.debug("⏰ Цикл проснулся, проверяем подписчиков")
            with subscribers_lock:
                subs = list(subscribers)
                logger.info(f"📋 Текущие подписчики: {subs}")
            if not subs:
                logger.info("😴 Нет подписчиков, пропускаем рассылку")
                continue
            logger.info(f"🔄 Начинаю рассылку для {len(subs)} подписчиков")
            for uid in subs:
                try:
                    logger.debug(f"📤 Отправка сигнала для {uid}")
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