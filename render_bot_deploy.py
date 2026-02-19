import asyncio
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
import os
import threading

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== ФАЙЛ ДЛЯ ХРАНЕНИЯ ПОДПИСЧИКОВ ==========
SUBSCRIBERS_FILE = "subscribers.json"


def load_subscribers():
    """Загружает список подписчиков из файла"""
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, 'r') as f:
                data = json.load(f)
                return set(data)
        except Exception as e:
            logger.error(f"Ошибка загрузки подписчиков: {e}")
            return set()
    return set()


def save_subscribers(subscribers_set):
    """Сохраняет список подписчиков в файл"""
    try:
        with open(SUBSCRIBERS_FILE, 'w') as f:
            json.dump(list(subscribers_set), f)
    except Exception as e:
        logger.error(f"Ошибка сохранения подписчиков: {e}")


# ========== ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ==========
app = Flask(__name__)

BOT_TOKEN = os.environ.get('BOT_TOKEN')
TWELVE_API_KEY = os.environ.get('TWELVE_API_KEY')

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не задан!")
if not TWELVE_API_KEY:
    logger.error("❌ TWELVE_API_KEY не задан!")

# Множество подписчиков
subscribers = load_subscribers()
subscribers_lock = threading.Lock()


# ========== ХРАНИЛИЩЕ ЦЕН (без pandas) ==========
class PriceStorage:
    def __init__(self, maxlen=50):
        self.maxlen = maxlen
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.timestamps = []
        self.last_signal = None

    def add_candle(self, candle):
        self.opens.append(float(candle['open']))
        self.highs.append(float(candle['high']))
        self.lows.append(float(candle['low']))
        self.closes.append(float(candle['close']))
        self.timestamps.append(candle['datetime'])
        if len(self.opens) > self.maxlen:
            self.opens.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            self.closes.pop(0)
            self.timestamps.pop(0)

    def clear(self):
        self.opens.clear()
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.timestamps.clear()


price_storage = PriceStorage(maxlen=50)


# ========== ФУНКЦИИ ДЛЯ РАСЧЁТА ИНДИКАТОРОВ ==========
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


def bbands(data, period=20, std_dev=2):
    if len(data) < period:
        middle = data[-1]
        upper = middle * 1.02
        lower = middle * 0.98
        return upper, middle, lower
    middle = sum(data[-period:]) / period
    variance = sum((x - middle) ** 2 for x in data[-period:]) / period
    std = variance ** 0.5
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def macd(data, fast=12, slow=26, signal=9):
    if len(data) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    return macd_line, 0.0, 0.0


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

    # Группировка уровней
    def cluster(levels, threshold=0.0005):
        if not levels:
            return []
        levels.sort()
        clustered = []
        current = [levels[0]]
        for lev in levels[1:]:
            if abs(lev - sum(current) / len(current)) < threshold:
                current.append(lev)
            else:
                clustered.append(sum(current) / len(current))
                current = [lev]
        clustered.append(sum(current) / len(current))
        return clustered

    supports = cluster(supports)
    resistances = cluster(resistances)
    current_price = close[-1]
    nearest_support = None
    nearest_resistance = None
    for s in supports:
        if s < current_price:
            nearest_support = s
    for r in resistances:
        if r > current_price:
            nearest_resistance = r
            break
    return {
        'supports': supports[-3:],
        'resistances': resistances[-3:],
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance
    }


def calculate_probability(indicators):
    votes_up = 0
    votes_down = 0

    # RSI
    rsi_val = indicators.get('rsi', 50)
    if rsi_val < 30:
        votes_up += 3
    elif rsi_val > 70:
        votes_down += 3
    elif rsi_val > 50:
        votes_up += 1
    else:
        votes_down += 1

    # MACD
    macd_trend = indicators.get('macd_trend', '')
    if 'БЫЧИЙ' in macd_trend:
        votes_up += 2
    elif 'МЕДВЕЖИЙ' in macd_trend:
        votes_down += 2

    # Bollinger Bands
    bb_signal = indicators.get('bb_signal', '')
    if 'ПЕРЕПРОДАННОСТЬ' in bb_signal:
        votes_up += 2
    elif 'ПЕРЕКУПЛЕННОСТЬ' in bb_signal:
        votes_down += 2

    # SMA
    for period in [5, 10, 20, 50]:
        sig = indicators.get(f'sma_{period}_signal', '')
        if 'ВЫШЕ' in sig:
            votes_up += 1
        elif 'НИЖЕ' in sig:
            votes_down += 1

    # EMA
    for period in [5, 10, 20]:
        sig = indicators.get(f'ema_{period}_signal', '')
        if 'ВЫШЕ' in sig:
            votes_up += 1
        elif 'НИЖЕ' in sig:
            votes_down += 1

    # Support/Resistance
    dist_s = indicators.get('distance_to_support', 100)
    dist_r = indicators.get('distance_to_resistance', 100)
    if dist_s < dist_r:
        votes_up += 1
    else:
        votes_down += 1

    total = votes_up + votes_down
    if total:
        prob_up = round((votes_up / total) * 100, 1)
        prob_down = round((votes_down / total) * 100, 1)
        conf = round(abs(votes_up - votes_down) / total * 100, 1)
    else:
        prob_up = prob_down = 50.0
        conf = 0.0
    return prob_up, prob_down, conf


def generate_message(indicators):
    if not indicators:
        return "❌ Нет данных"

    price = indicators['price']
    prob_up = indicators['prob_up']
    prob_down = indicators['prob_down']
    confidence = indicators['confidence']

    if prob_up > prob_down + 10 and confidence > 50:
        rec = "📈 СИЛЬНАЯ ПОКУПКА"
        emoji = "🟢"
    elif prob_up > prob_down:
        rec = "📈 ПОКУПКА"
        emoji = "🟢"
    elif prob_down > prob_up + 10 and confidence > 50:
        rec = "📉 СИЛЬНАЯ ПРОДАЖА"
        emoji = "🔴"
    elif prob_down > prob_up:
        rec = "📉 ПРОДАЖА"
        emoji = "🔴"
    else:
        rec = "⏸️ ОЖИДАНИЕ"
        emoji = "⚪"

    msg = f"""
{emoji} *ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ EUR/USD* {emoji}
⏰ {indicators['timestamp']}
💰 *Цена:* `{price:.5f}`

📊 *ОБЩАЯ ВЕРОЯТНОСТЬ*
┌─ ⬆️ ВВЕРХ: {prob_up}%
└─ ⬇️ ВНИЗ: {prob_down}%
🎯 Уверенность: {confidence}%
💡 Рекомендация: {rec}

📈 *RSI (индекс относительной силы)*
┌─ Значение: `{indicators['rsi']:.1f}`
└─ Сигнал: {indicators['rsi_signal']}

📊 *MACD*
┌─ MACD: `{indicators['macd']:.5f}`
├─ Сигнал: `{indicators['macd_signal']:.5f}`
├─ Гистограмма: `{indicators['macd_hist']:.5f}`
└─ Тренд: {indicators['macd_trend']}

📉 *Полосы Боллинджера*
┌─ Верхняя: `{indicators['bb_upper']:.5f}`
├─ Средняя: `{indicators['bb_middle']:.5f}`
├─ Нижняя: `{indicators['bb_lower']:.5f}`
├─ Ширина: `{indicators['bb_width']:.2f}%`
├─ Позиция: {indicators['bb_position']}
└─ Сигнал: {indicators['bb_signal']}

📏 *Скользящие средние (SMA)*
"""
    for p in [5, 10, 20, 50]:
        if p in indicators['sma']:
            sig = indicators.get(f'sma_{p}_signal', '')
            msg += f"├─ SMA({p}): `{indicators['sma'][p]:.5f}` {sig}\n"

    msg += "\n📊 *Экспоненциальные средние (EMA)*\n"
    for p in [5, 10, 20]:
        if p in indicators['ema']:
            sig = indicators.get(f'ema_{p}_signal', '')
            msg += f"├─ EMA({p}): `{indicators['ema'][p]:.5f}` {sig}\n"

    # Уровни поддержки/сопротивления
    support_str = f"`{indicators['nearest_support']:.5f}`" if indicators['nearest_support'] else "`не определен`"
    resistance_str = f"`{indicators['nearest_resistance']:.5f}`" if indicators[
        'nearest_resistance'] else "`не определен`"
    dist_s = f"{indicators['distance_to_support']:.0f}" if indicators['distance_to_support'] else "?"
    dist_r = f"{indicators['distance_to_resistance']:.0f}" if indicators['distance_to_resistance'] else "?"

    msg += f"""
📊 *Уровни поддержки/сопротивления*
┌─ Ближайшая поддержка: {support_str} (дист: {dist_s} пипсов)
└─ Ближайшее сопротивление: {resistance_str} (дист: {dist_r} пипсов)
"""
    if indicators['support_levels']:
        supps = [f"{x:.5f}" for x in indicators['support_levels'] if x]
        msg += f"├─ Уровни поддержки: {', '.join(supps)}\n"
    if indicators['resistance_levels']:
        ress = [f"{x:.5f}" for x in indicators['resistance_levels'] if x]
        msg += f"└─ Уровни сопротивления: {', '.join(ress)}\n"

    msg += f"\n#{'BUY' if prob_up > prob_down else 'SELL'} #EURUSD #TECHNICAL #ANALYSIS"
    return msg


async def fetch_candles(api_key, bars=50):
    """Получение свечей с Twelve Data"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': 'EUR/USD',
        'interval': '1min',
        'outputsize': bars,
        'apikey': api_key
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('values')
    except Exception as e:
        logger.error(f"Ошибка fetch: {e}")
    return None


async def update_price_storage():
    """Обновление хранилища цен"""
    candles = await fetch_candles(TWELVE_API_KEY, 50)
    if candles:
        price_storage.clear()
        for c in candles[::-1]:  # от старых к новым
            price_storage.add_candle(c)
        return True
    return False


async def get_indicators():
    """Получение всех индикаторов"""
    if len(price_storage.closes) < 20:
        if not await update_price_storage():
            return None

    closes = price_storage.closes
    highs = price_storage.highs
    lows = price_storage.lows
    current_price = closes[-1]
    ind = {}

    # RSI
    ind['rsi'] = rsi(closes, 14)
    if ind['rsi'] > 70:
        ind['rsi_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (сигнал к продаже)'
    elif ind['rsi'] < 30:
        ind['rsi_signal'] = 'ПЕРЕПРОДАННОСТЬ (сигнал к покупке)'
    elif ind['rsi'] > 50:
        ind['rsi_signal'] = 'ВОСХОДЯЩИЙ ТРЕНД'
    else:
        ind['rsi_signal'] = 'НИСХОДЯЩИЙ ТРЕНД'

    # MACD
    macd_line, _, _ = macd(closes, 12, 26, 9)
    ind['macd'] = macd_line
    ind['macd_signal'] = 0.0
    ind['macd_hist'] = 0.0
    ind['macd_trend'] = 'БЫЧИЙ СИГНАЛ' if macd_line > 0 else 'МЕДВЕЖИЙ СИГНАЛ' if macd_line < 0 else 'НЕЙТРАЛЬНО'

    # Bollinger Bands
    upper, middle, lower = bbands(closes, 20, 2)
    ind['bb_upper'] = upper
    ind['bb_middle'] = middle
    ind['bb_lower'] = lower
    ind['bb_width'] = ((upper - lower) / middle) * 100

    if current_price >= upper:
        ind['bb_position'] = 'ВЫШЕ ВЕРХНЕЙ ПОЛОСЫ'
        ind['bb_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (возможен откат вниз)'
    elif current_price <= lower:
        ind['bb_position'] = 'НИЖЕ НИЖНЕЙ ПОЛОСЫ'
        ind['bb_signal'] = 'ПЕРЕПРОДАННОСТЬ (возможен отскок вверх)'
    elif current_price > middle:
        ind['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И ВЕРХНЕЙ'
        ind['bb_signal'] = 'НЕЙТРАЛЬНО'
    else:
        ind['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И НИЖНЕЙ'
        ind['bb_signal'] = 'НЕЙТРАЛЬНО'

    # SMA
    ind['sma'] = {}
    for p in [5, 10, 20, 50]:
        val = sma(closes, p)
        ind['sma'][p] = val
        if current_price > val:
            ind[f'sma_{p}_signal'] = '⬆️ ВЫШЕ'
        elif current_price < val:
            ind[f'sma_{p}_signal'] = '⬇️ НИЖЕ'
        else:
            ind[f'sma_{p}_signal'] = '⏺️ ОКОЛО'

    # EMA
    ind['ema'] = {}
    for p in [5, 10, 20]:
        val = ema(closes, p)
        ind['ema'][p] = val
        if current_price > val:
            ind[f'ema_{p}_signal'] = '⬆️ ВЫШЕ'
        elif current_price < val:
            ind[f'ema_{p}_signal'] = '⬇️ НИЖЕ'
        else:
            ind[f'ema_{p}_signal'] = '⏺️ ОКОЛО'

    # Support/Resistance
    sr = find_support_resistance(highs, lows, closes)
    ind['support_levels'] = sr['supports']
    ind['resistance_levels'] = sr['resistances']
    ind['nearest_support'] = sr['nearest_support']
    ind['nearest_resistance'] = sr['nearest_resistance']
    ind['distance_to_support'] = (current_price - sr['nearest_support']) * 10000 if sr['nearest_support'] else 0
    ind['distance_to_resistance'] = (sr['nearest_resistance'] - current_price) * 10000 if sr[
        'nearest_resistance'] else 0

    # Вероятность
    ind['prob_up'], ind['prob_down'], ind['confidence'] = calculate_probability(ind)
    ind['price'] = current_price
    ind['timestamp'] = datetime.now().strftime('%H:%M:%S')

    price_storage.last_signal = ind
    return ind


# ========== МЕНЮ ==========
def get_main_menu():
    kb = [
        [InlineKeyboardButton("📊 Получить сигнал", callback_data='signal'),
         InlineKeyboardButton("📈 Статус", callback_data='status')],
        [InlineKeyboardButton("🔔 Автосигнал", callback_data='auto_on'),
         InlineKeyboardButton("⏹️ Стоп", callback_data='auto_off')],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data='help'),
         InlineKeyboardButton("⚙️ Настройки", callback_data='settings')]
    ]
    return InlineKeyboardMarkup(kb)


def get_help_menu():
    kb = [
        [InlineKeyboardButton("📊 Индикаторы", callback_data='help_indicators'),
         InlineKeyboardButton("💰 Торговля", callback_data='help_trading')],
        [InlineKeyboardButton("❓ FAQ", callback_data='help_faq'),
         InlineKeyboardButton("◀️ Назад", callback_data='back_to_main')]
    ]
    return InlineKeyboardMarkup(kb)


def get_settings_menu():
    kb = [
        [InlineKeyboardButton("📏 RSI период", callback_data='set_rsi'),
         InlineKeyboardButton("📊 MACD", callback_data='set_macd')],
        [InlineKeyboardButton("📉 Bollinger Bands", callback_data='set_bb'),
         InlineKeyboardButton("◀️ Назад", callback_data='back_to_main')]
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
    return """
    <h1>🤖 EUR/USD Pro Trading Bot</h1>
    <p>Бот работает 24/7!</p>
    <p>Команды в Telegram: /start</p>
    """


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'subscribers': len(subscribers),
        'candles': len(price_storage.closes)
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        logger.info("📨 Получено обновление")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if 'callback_query' in data:
            chat_id = data['callback_query']['from']['id']
            cb_data = data['callback_query']['data']
            loop.run_until_complete(handle_callback(chat_id, cb_data))
        elif 'message' in data and 'text' in data['message']:
            chat_id = data['message']['chat']['id']
            text = data['message']['text']
            loop.run_until_complete(handle_message(chat_id, text))

        loop.close()
        return jsonify({'ok': True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'ok': False}), 500


async def handle_callback(chat_id, cb_data):
    """Обработка нажатий на кнопки"""
    bot = Bot(token=BOT_TOKEN)

    if cb_data == 'signal':
        await send_signal(bot, chat_id)
    elif cb_data == 'status':
        await send_status(bot, chat_id)
    elif cb_data == 'auto_on':
        with subscribers_lock:
            subscribers.add(chat_id)
            save_subscribers(subscribers)
        await bot.send_message(
            chat_id,
            "✅ Автоматические сигналы включены! Теперь вы будете получать сигнал каждые 5 минут.\n"
            "Для остановки нажмите кнопку ⏹️ Стоп или отправьте /stop"
        )
        logger.info(f"✅ Пользователь {chat_id} включил автосигналы")
    elif cb_data == 'auto_off':
        with subscribers_lock:
            if chat_id in subscribers:
                subscribers.remove(chat_id)
                save_subscribers(subscribers)
                await bot.send_message(chat_id, "⏹️ Автоматические сигналы остановлены.")
                logger.info(f"⏹️ Пользователь {chat_id} отключил автосигналы")
            else:
                await bot.send_message(chat_id, "❌ Автоматические сигналы не были включены.")
    elif cb_data == 'help':
        await bot.send_message(
            chat_id,
            "📖 *Раздел помощи*\n\nВыберите тему:",
            reply_markup=get_help_menu(),
            parse_mode='Markdown'
        )
    elif cb_data == 'settings':
        await show_settings(bot, chat_id)
    elif cb_data == 'back_to_main':
        await bot.send_message(
            chat_id,
            "🤖 *Главное меню*",
            reply_markup=get_main_menu(),
            parse_mode='Markdown'
        )
    elif cb_data == 'help_indicators':
        await send_help_indicators(bot, chat_id)
    elif cb_data == 'help_trading':
        await send_help_trading(bot, chat_id)
    elif cb_data == 'help_faq':
        await send_help_faq(bot, chat_id)
    elif cb_data == 'set_rsi':
        await bot.send_message(
            chat_id,
            "📏 RSI период: 14\n\n"
            "Для изменения отредактируйте файл render_bot_deploy.py"
        )
    elif cb_data == 'set_macd':
        await bot.send_message(
            chat_id,
            "📊 MACD: 12/26/9\n\n"
            "Для изменения отредактируйте файл render_bot_deploy.py"
        )
    elif cb_data == 'set_bb':
        await bot.send_message(
            chat_id,
            "📉 Bollinger Bands: период 20, отклонение 2\n\n"
            "Для изменения отредактируйте файл render_bot_deploy.py"
        )
    else:
        await bot.send_message(chat_id, "❓ Неизвестная команда")


async def handle_message(chat_id, text):
    """Обработка текстовых сообщений"""
    bot = Bot(token=BOT_TOKEN)

    if text == '/start':
        await bot.send_message(
            chat_id,
            "🤖 *EUR/USD ПРОФЕССИОНАЛЬНЫЙ БОТ*\n\n"
            "Я анализирую валютную пару EUR/USD с использованием профессиональных индикаторов.\n\n"
            "📊 *Доступные функции:*\n"
            "• Получение сигналов с анализом\n"
            "• Статус и настройки бота\n"
            "• Подробная помощь по индикаторам\n"
            "• 🔔 Автоматические сигналы каждые 5 минут\n\n"
            "*Бот работает 24/7 на Render.com!*",
            reply_markup=get_main_menu(),
            parse_mode='Markdown'
        )
        logger.info(f"✅ Приветствие отправлено пользователю {chat_id}")
    elif text == '/signal':
        await send_signal(bot, chat_id)
    elif text == '/status':
        await send_status(bot, chat_id)
    elif text == '/stop':
        with subscribers_lock:
            if chat_id in subscribers:
                subscribers.remove(chat_id)
                save_subscribers(subscribers)
                await bot.send_message(chat_id, "⏹️ Автоматические сигналы остановлены.")
                logger.info(f"⏹️ Пользователь {chat_id} отключил автосигналы")
            else:
                await bot.send_message(chat_id, "❌ Автоматические сигналы не были включены.")
    else:
        await bot.send_message(
            chat_id,
            "❌ Неизвестная команда. Используйте кнопки меню или /help"
        )


async def send_signal(bot, chat_id):
    """Отправка сигнала"""
    try:
        await bot.send_message(chat_id, "🔄 Анализирую рынок...")
        indicators = await get_indicators()
        if not indicators:
            await bot.send_message(
                chat_id,
                "❌ Не удалось получить данные для анализа. Проверьте API ключ Twelve Data."
            )
            return
        msg = generate_message(indicators)
        await bot.send_message(chat_id, msg, parse_mode='Markdown')
        logger.info(f"✅ Сигнал отправлен пользователю {chat_id}")
    except Exception as e:
        logger.error(f"❌ Ошибка при отправке сигнала: {e}")
        await bot.send_message(chat_id, "❌ Произошла ошибка при генерации сигнала")


async def send_status(bot, chat_id):
    """Отправка статуса"""
    with subscribers_lock:
        auto_status = "включены" if chat_id in subscribers else "отключены"
    status_text = f"""
📊 *СТАТУС БОТА*

✅ Бот работает 24/7
📈 Индикаторы активны
💹 Последний сигнал: {'есть' if price_storage.last_signal else 'нет'}
🔔 Автосигналы: {auto_status}
⏰ Время сервера: {datetime.now().strftime('%H:%M:%S')}

*Настройки индикаторов:*
• RSI период: 14
• MACD: 12/26/9
• Полосы Боллинджера: 20/2
• SMA периоды: 5,10,20,50
• EMA периоды: 5,10,20
"""
    await bot.send_message(chat_id, status_text, parse_mode='Markdown')


async def show_settings(bot, chat_id):
    """Показать меню настроек"""
    settings_text = f"""
⚙️ *ТЕКУЩИЕ НАСТРОЙКИ*

📏 RSI период: 14
📊 MACD: 12/26/9
📉 Bollinger Bands: период 20, отклонение 2
📏 SMA периоды: 5,10,20,50
📈 EMA периоды: 5,10,20

Выберите параметр для изменения:
"""
    await bot.send_message(
        chat_id,
        settings_text,
        reply_markup=get_settings_menu(),
        parse_mode='Markdown'
    )


async def send_help_indicators(bot, chat_id):
    """Помощь по индикаторам"""
    text = """
📊 *ИНДИКАТОРЫ*

📈 **RSI (Relative Strength Index)**
• Значение >70: перекупленность (сигнал к продаже)
• Значение <30: перепроданность (сигнал к покупке)
• Период: 14 свечей

📊 **MACD**
• Бычий сигнал: MACD выше сигнальной линии
• Медвежий сигнал: MACD ниже сигнальной линии
• Параметры: 12/26/9

📉 **Полосы Боллинджера**
• Касание верхней полосы: возможен откат вниз
• Касание нижней полосы: возможен отскок вверх
• Период: 20, отклонение: 2

📏 **Скользящие средние**
• SMA (простые): 5,10,20,50
• EMA (экспоненциальные): 5,10,20
"""
    kb = [[InlineKeyboardButton("◀️ Назад", callback_data='help')]]
    await bot.send_message(
        chat_id,
        text,
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode='Markdown'
    )


async def send_help_trading(bot, chat_id):
    """Помощь по торговле"""
    text = """
💰 *КАК ТОРГОВАТЬ*

1. Получите сигнал через кнопку 📊
2. Проанализируйте вероятности
3. Если уверенность >60% - можно входить
4. Ставка: не более 3% от депозита
5. Время экспирации: 3 минуты

*Правила риск-менеджмента:*
• Максимальная ставка: 3%
• Дневной лимит убытка: -15%
• Не удваивайте после проигрыша
• Лучше пропустить, чем ошибиться
"""
    kb = [[InlineKeyboardButton("◀️ Назад", callback_data='help')]]
    await bot.send_message(
        chat_id,
        text,
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode='Markdown'
    )


async def send_help_faq(bot, chat_id):
    """FAQ"""
    text = """
❓ *ЧАСТО ЗАДАВАЕМЫЕ ВОПРОСЫ*

❓ **Как часто обновляются сигналы?**
По запросу через команду /signal, автосигналы каждые 5 минут

❓ **Почему нет сигнала?**
Проверьте API ключ Twelve Data в настройках Render

❓ **Какой таймфрейм используется?**
1 минута (M1)

❓ **Можно ли доверять сигналам?**
Сигналы основаны на техническом анализе, но не гарантируют прибыль

❓ **Бот работает 24/7?**
Да, бот запущен на Render.com и работает круглосуточно
"""
    kb = [[InlineKeyboardButton("◀️ Назад", callback_data='help')]]
    await bot.send_message(
        chat_id,
        text,
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode='Markdown'
    )


# ========== ФОНОВАЯ ЗАДАЧА ДЛЯ АВТОСИГНАЛОВ ==========
async def auto_signal_worker():
    """Фоновая задача, отправляющая сигналы подписчикам каждые 5 минут"""
    logger.info("🚀 Фоновая задача автосигналов запущена (интервал 5 минут)")
    while True:
        await asyncio.sleep(300)  # 5 минут
        with subscribers_lock:
            current_subscribers = list(subscribers)
        if current_subscribers:
            logger.info(f"🔄 Отправка автоматических сигналов {len(current_subscribers)} подписчикам")
            for chat_id in current_subscribers:
                try:
                    logger.info(f"👉 Отправка автосигнала для {chat_id}")
                    bot = Bot(token=BOT_TOKEN)
                    await bot.send_message(chat_id, "🔄 Автоматический сигнал...")
                    indicators = await get_indicators()
                    if indicators:
                        msg = generate_message(indicators)
                        await bot.send_message(chat_id, msg, parse_mode='Markdown')
                        logger.info(f"✅ Автосигнал отправлен для {chat_id}")
                    else:
                        await bot.send_message(chat_id, "❌ Не удалось получить сигнал.")
                        logger.error(f"❌ Не удалось получить сигнал для {chat_id}")
                except Exception as e:
                    logger.error(f"❌ Ошибка автосигнала для {chat_id}: {e}")
        else:
            logger.debug("Нет подписчиков на автосигналы")


def start_background_loop():
    """Запуск фоновой задачи в отдельном потоке"""
    logger.info("🔄 Запуск фонового потока для автосигналов")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(auto_signal_worker())


# Запуск фонового потока
thread = threading.Thread(target=start_background_loop, daemon=True)
thread.start()
logger.info("✅ Фоновый поток для автосигналов запущен")

# ========== ЗАПУСК ПРИЛОЖЕНИЯ ==========
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)