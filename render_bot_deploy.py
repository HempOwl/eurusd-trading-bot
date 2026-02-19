import asyncio
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot
import os
import pandas as pd
import numpy as np
import pandas_ta as ta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создаём Flask приложение
app = Flask(__name__)

# Токен из переменных окружения
BOT_TOKEN = os.environ.get('BOT_TOKEN')
TWELVE_API_KEY = os.environ.get('TWELVE_API_KEY')

if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не задан!")
if not TWELVE_API_KEY:
    logger.error("❌ TWELVE_API_KEY не задан!")


class EURUSDProBot:
    """Профессиональный бот с индикаторами для Render"""

    def __init__(self):
        self.api_key = TWELVE_API_KEY
        self.df = None
        self.last_signal = None

        # Настройки индикаторов
        self.settings = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [5, 10, 20],
            'support_resistance_lookback': 50
        }

    async def fetch_historical_data(self, bars=100):
        """Получение исторических данных для расчета индикаторов"""
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': 'EUR/USD',
            'interval': '1min',
            'outputsize': bars,
            'apikey': self.api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'values' in data:
                            return data['values']
                    else:
                        logger.error(f"Ошибка API: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Ошибка получения истории: {e}")
            return None

    def update_dataframe(self, candles):
        """Обновление DataFrame с ценами"""
        if not candles:
            return

        # Создаем DataFrame
        df = pd.DataFrame(candles)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
        })

        # Конвертируем в числовые значения
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col])

        # Меняем порядок на хронологический (от старых к новым)
        df = df.iloc[::-1].reset_index(drop=True)

        self.df = df
        logger.info(f"✅ Загружено {len(df)} свечей")

    def calculate_indicators(self):
        """Расчет всех технических индикаторов"""
        if self.df is None or len(self.df) < 50:
            return None

        df = self.df.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values

        results = {}
        current_price = close[-1]

        # === 1. RSI (Relative Strength Index) ===
        try:
            rsi = talib.RSI(close, timeperiod=self.settings['rsi_period'])
            results['rsi'] = rsi[-1]
            if results['rsi'] > self.settings['rsi_overbought']:
                results['rsi_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (сигнал к продаже)'
            elif results['rsi'] < self.settings['rsi_oversold']:
                results['rsi_signal'] = 'ПЕРЕПРОДАННОСТЬ (сигнал к покупке)'
            elif results['rsi'] > 50:
                results['rsi_signal'] = 'ВОСХОДЯЩИЙ ТРЕНД'
            else:
                results['rsi_signal'] = 'НИСХОДЯЩИЙ ТРЕНД'
        except:
            results['rsi'] = 50
            results['rsi_signal'] = 'НЕЙТРАЛЬНО'

        # === 2. MACD ===
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=self.settings['macd_fast'],
                slowperiod=self.settings['macd_slow'],
                signalperiod=self.settings['macd_signal']
            )
            results['macd'] = macd[-1]
            results['macd_signal'] = macd_signal[-1]
            results['macd_hist'] = macd_hist[-1]

            if macd[-1] > macd_signal[-1] and macd_hist[-1] > 0:
                results['macd_trend'] = 'БЫЧИЙ СИГНАЛ'
            elif macd[-1] < macd_signal[-1] and macd_hist[-1] < 0:
                results['macd_trend'] = 'МЕДВЕЖИЙ СИГНАЛ'
            elif macd[-1] > macd_signal[-1]:
                results['macd_trend'] = 'ФОРМИРОВАНИЕ БЫЧЬЕГО ТРЕНДА'
            elif macd[-1] < macd_signal[-1]:
                results['macd_trend'] = 'ФОРМИРОВАНИЕ МЕДВЕЖЬЕГО ТРЕНДА'
            else:
                results['macd_trend'] = 'НЕЙТРАЛЬНО'
        except:
            results['macd'] = 0
            results['macd_signal'] = 0
            results['macd_hist'] = 0
            results['macd_trend'] = 'НЕЙТРАЛЬНО'

        # === 3. Полосы Боллинджера ===
        try:
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=self.settings['bb_period'],
                nbdevup=self.settings['bb_std'],
                nbdevdn=self.settings['bb_std']
            )
            results['bb_upper'] = upper[-1]
            results['bb_middle'] = middle[-1]
            results['bb_lower'] = lower[-1]
            results['bb_width'] = ((upper[-1] - lower[-1]) / middle[-1]) * 100

            # Позиция цены относительно полос
            if current_price >= upper[-1]:
                results['bb_position'] = 'ВЫШЕ ВЕРХНЕЙ ПОЛОСЫ'
                results['bb_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (возможен откат вниз)'
            elif current_price <= lower[-1]:
                results['bb_position'] = 'НИЖЕ НИЖНЕЙ ПОЛОСЫ'
                results['bb_signal'] = 'ПЕРЕПРОДАННОСТЬ (возможен отскок вверх)'
            elif current_price > middle[-1]:
                results['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И ВЕРХНЕЙ'
                results['bb_signal'] = 'НЕЙТРАЛЬНО'
            else:
                results['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И НИЖНЕЙ'
                results['bb_signal'] = 'НЕЙТРАЛЬНО'
        except:
            results['bb_upper'] = current_price * 1.02
            results['bb_middle'] = current_price
            results['bb_lower'] = current_price * 0.98
            results['bb_width'] = 2
            results['bb_position'] = 'СРЕДИНА'
            results['bb_signal'] = 'НЕЙТРАЛЬНО'

        # === 4. Скользящие средние ===
        results['sma'] = {}
        for period in self.settings['sma_periods']:
            try:
                sma = talib.SMA(close, timeperiod=period)
                results['sma'][period] = sma[-1]
                if current_price > sma[-1]:
                    results[f'sma_{period}_signal'] = '⬆️ ВЫШЕ'
                elif current_price < sma[-1]:
                    results[f'sma_{period}_signal'] = '⬇️ НИЖЕ'
                else:
                    results[f'sma_{period}_signal'] = '⏺️ ОКОЛО'
            except:
                results['sma'][period] = current_price
                results[f'sma_{period}_signal'] = '⏺️ ОКОЛО'

        results['ema'] = {}
        for period in self.settings['ema_periods']:
            try:
                ema = talib.EMA(close, timeperiod=period)
                results['ema'][period] = ema[-1]
                if current_price > ema[-1]:
                    results[f'ema_{period}_signal'] = '⬆️ ВЫШЕ'
                elif current_price < ema[-1]:
                    results[f'ema_{period}_signal'] = '⬇️ НИЖЕ'
                else:
                    results[f'ema_{period}_signal'] = '⏺️ ОКОЛО'
            except:
                results['ema'][period] = current_price
                results[f'ema_{period}_signal'] = '⏺️ ОКОЛО'

        # === 5. Поддержка и сопротивление ===
        support_resistance = self.find_support_resistance(high, low, close)
        results['support_levels'] = support_resistance['supports']
        results['resistance_levels'] = support_resistance['resistances']
        results['nearest_support'] = support_resistance['nearest_support']
        results['nearest_resistance'] = support_resistance['nearest_resistance']

        if results['nearest_support']:
            results['distance_to_support'] = (current_price - results['nearest_support']) * 10000
        else:
            results['distance_to_support'] = 0

        if results['nearest_resistance']:
            results['distance_to_resistance'] = (results['nearest_resistance'] - current_price) * 10000
        else:
            results['distance_to_resistance'] = 0

        # === 6. Общая вероятность ===
        results['prob_up'], results['prob_down'], results['final_confidence'] = self.calculate_probability(results)
        results['current_price'] = current_price
        results['timestamp'] = datetime.now()

        return results

    def find_support_resistance(self, high, low, close, window=5):
        """Поиск уровней поддержки и сопротивления"""
        supports = []
        resistances = []

        # Ищем локальные минимумы (поддержка)
        for i in range(window, len(close) - window):
            if all(low[i] <= low[i - j] for j in range(1, window + 1)) and \
                    all(low[i] <= low[i + j] for j in range(1, window + 1)):
                supports.append(low[i])

        # Ищем локальные максимумы (сопротивление)
        for i in range(window, len(close) - window):
            if all(high[i] >= high[i - j] for j in range(1, window + 1)) and \
                    all(high[i] >= high[i + j] for j in range(1, window + 1)):
                resistances.append(high[i])

        # Группируем близкие уровни
        supports = self.cluster_levels(supports)
        resistances = self.cluster_levels(resistances)

        # Сортируем
        supports.sort()
        resistances.sort()

        # Находим ближайшие уровни к текущей цене
        current = close[-1]
        nearest_support = None
        nearest_resistance = None

        for s in supports:
            if s < current:
                nearest_support = s

        for r in resistances:
            if r > current:
                nearest_resistance = r
                break

        return {
            'supports': supports[-3:],  # Последние 3 уровня поддержки
            'resistances': resistances[-3:],  # Последние 3 уровня сопротивления
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }

    def cluster_levels(self, levels, threshold=0.0005):
        """Группировка близких уровней (для EUR/USD threshold = 5 пипсов)"""
        if not levels:
            return []

        levels.sort()
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - sum(current_cluster) / len(current_cluster)) < threshold:
                current_cluster.append(level)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        clustered.append(sum(current_cluster) / len(current_cluster))
        return clustered

    def calculate_probability(self, indicators):
        """Расчет общей вероятности на основе всех индикаторов"""
        votes_up = 0
        votes_down = 0

        # RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                votes_up += 3
            elif rsi > 70:
                votes_down += 3
            elif rsi > 50:
                votes_up += 1
            else:
                votes_down += 1

        # MACD
        if 'macd_trend' in indicators:
            trend = indicators['macd_trend']
            if 'БЫЧИЙ' in trend:
                votes_up += 2
            elif 'МЕДВЕЖИЙ' in trend:
                votes_down += 2

        # Полосы Боллинджера
        if 'bb_signal' in indicators:
            bb = indicators['bb_signal']
            if 'ПЕРЕПРОДАННОСТЬ' in bb:
                votes_up += 2
            elif 'ПЕРЕКУПЛЕННОСТЬ' in bb:
                votes_down += 2

        # Скользящие средние
        for period in self.settings['sma_periods']:
            signal = indicators.get(f'sma_{period}_signal', '')
            if 'ВЫШЕ' in signal:
                votes_up += 1
            elif 'НИЖЕ' in signal:
                votes_down += 1

        for period in self.settings['ema_periods']:
            signal = indicators.get(f'ema_{period}_signal', '')
            if 'ВЫШЕ' in signal:
                votes_up += 1
            elif 'НИЖЕ' in signal:
                votes_down += 1

        # Поддержка/сопротивление
        if indicators.get('nearest_support') and indicators.get('nearest_resistance'):
            dist_to_support = indicators.get('distance_to_support', 100)
            dist_to_resistance = indicators.get('distance_to_resistance', 100)

            if dist_to_support < dist_to_resistance:
                votes_up += 1
            else:
                votes_down += 1

        total = votes_up + votes_down
        if total > 0:
            prob_up = round((votes_up / total) * 100, 1)
            prob_down = round((votes_down / total) * 100, 1)
            confidence = round(abs(votes_up - votes_down) / total * 100, 1)
        else:
            prob_up = prob_down = 50
            confidence = 0

        return prob_up, prob_down, confidence

    def generate_message(self, indicators):
        """Генерация подробного сообщения с индикаторами"""
        if not indicators:
            return "❌ Недостаточно данных для расчета индикаторов"

        price = indicators['current_price']
        prob_up = indicators['prob_up']
        prob_down = indicators['prob_down']
        confidence = indicators['final_confidence']

        # Определяем общую рекомендацию
        if prob_up > prob_down + 10 and confidence > 50:
            recommendation = "📈 СИЛЬНАЯ ПОКУПКА"
            emoji = "🟢"
        elif prob_up > prob_down:
            recommendation = "📈 ПОКУПКА"
            emoji = "🟢"
        elif prob_down > prob_up + 10 and confidence > 50:
            recommendation = "📉 СИЛЬНАЯ ПРОДАЖА"
            emoji = "🔴"
        elif prob_down > prob_up:
            recommendation = "📉 ПРОДАЖА"
            emoji = "🔴"
        else:
            recommendation = "⏸️ ОЖИДАНИЕ"
            emoji = "⚪"

        # Формируем сообщение
        message = f"""
{emoji} *ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ EUR/USD* {emoji}
⏰ {indicators['timestamp'].strftime('%H:%M:%S')}
💰 *Цена:* `{price:.5f}`

📊 *ОБЩАЯ ВЕРОЯТНОСТЬ*
┌─ ⬆️ ВВЕРХ: {prob_up}%
└─ ⬇️ ВНИЗ: {prob_down}%
🎯 Уверенность: {confidence}%
💡 Рекомендация: {recommendation}

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
        # Добавляем SMA
        for period in self.settings['sma_periods']:
            if period in indicators['sma']:
                signal = indicators.get(f'sma_{period}_signal', '')
                message += f"├─ SMA({period}): `{indicators['sma'][period]:.5f}` {signal}\n"

        message += "\n📊 *Экспоненциальные средние (EMA)*\n"
        for period in self.settings['ema_periods']:
            if period in indicators['ema']:
                signal = indicators.get(f'ema_{period}_signal', '')
                message += f"├─ EMA({period}): `{indicators['ema'][period]:.5f}` {signal}\n"

        # Уровни поддержки/сопротивления
        message += f"""
📊 *Уровни поддержки/сопротивления*
┌─ Ближайшая поддержка: `{indicators['nearest_support']:.5f}` (дист: {indicators['distance_to_support']:.0f} пипсов)
└─ Ближайшее сопротивление: `{indicators['nearest_resistance']:.5f}` (дист: {indicators['distance_to_resistance']:.0f} пипсов)
"""

        # Ключевые уровни
        if indicators['support_levels']:
            supports = [f"{s:.5f}" for s in indicators['support_levels']]
            message += f"├─ Уровни поддержки: {', '.join(supports)}\n"

        if indicators['resistance_levels']:
            resistances = [f"{r:.5f}" for r in indicators['resistance_levels']]
            message += f"└─ Уровни сопротивления: {', '.join(resistances)}\n"

        message += f"\n#{'BUY' if prob_up > prob_down else 'SELL'} #EURUSD #TECHNICAL #ANALYSIS"

        return message

    async def get_signal(self):
        """Получение сигнала с индикаторами"""
        try:
            # Получаем исторические данные
            candles = await self.fetch_historical_data(bars=100)
            if not candles:
                logger.error("❌ Нет данных для анализа")
                return None

            # Обновляем DataFrame
            self.update_dataframe(candles)

            # Рассчитываем индикаторы
            indicators = self.calculate_indicators()
            if not indicators:
                logger.error("❌ Не удалось рассчитать индикаторы")
                return None

            self.last_signal = indicators
            return indicators

        except Exception as e:
            logger.error(f"❌ Ошибка при получении сигнала: {e}")
            return None


# Создаём экземпляр бота
bot = EURUSDProBot()


@app.before_request
def before_request():
    """Создаём event loop для каждого запроса"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


@app.route('/')
def index():
    """Главная страница"""
    return """
    <h1>🤖 EUR/USD Pro Trading Bot</h1>
    <p>Бот с профессиональными индикаторами работает 24/7!</p>
    <p>Команды в Telegram:</p>
    <ul>
        <li><b>/signal</b> - получить сигнал со всеми индикаторами</li>
        <li><b>/status</b> - статус бота</li>
        <li><b>/help</b> - помощь</li>
    </ul>
    """


@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'indicators_available': bot.last_signal is not None
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка вебхуков от Telegram"""
    try:
        # Получаем обновление от Telegram
        update_data = request.get_json()
        logger.info(f"📨 Получено обновление от пользователя")

        # Создаём новый event loop для этого запроса
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Обрабатываем сообщение
            if 'message' in update_data and 'text' in update_data['message']:
                chat_id = update_data['message']['chat']['id']
                text = update_data['message']['text']

                # Запускаем обработку
                loop.run_until_complete(handle_message(chat_id, text))
            else:
                logger.info("Получено обновление без текстового сообщения")

            loop.close()
        except Exception as e:
            logger.error(f"Ошибка в цикле событий: {e}")
        finally:
            try:
                loop.close()
            except:
                pass

        return jsonify({'status': 'ok'})

    except Exception as e:
        logger.error(f"❌ Ошибка в webhook: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


async def handle_message(chat_id, text):
    """Обработка сообщения"""
    try:
        bot_instance = Bot(token=BOT_TOKEN)

        if text == '/start':
            await send_welcome(bot_instance, chat_id)
        elif text == '/signal':
            await send_signal(bot_instance, chat_id)
        elif text == '/status':
            await send_status(bot_instance, chat_id)
        elif text == '/help':
            await send_help(bot_instance, chat_id)
        else:
            await bot_instance.send_message(
                chat_id=chat_id,
                text="❌ Неизвестная команда. Используй /help"
            )
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке сообщения: {e}")


async def send_welcome(bot_instance, chat_id):
    """Отправка приветствия"""
    welcome_text = """
🤖 *EUR/USD ПРОФЕССИОНАЛЬНЫЙ БОТ*

Я анализирую валютную пару EUR/USD с использованием профессиональных индикаторов:

📊 RSI (индекс относительной силы)
📈 MACD
📉 Полосы Боллинджера
📏 Скользящие средние (SMA, EMA)
🎯 Уровни поддержки/сопротивления

*Доступные команды:*
/signal - получить полный анализ
/status - статус бота
/help - помощь

Бот работает 24/7 на Render.com!
    """
    await bot_instance.send_message(
        chat_id=chat_id,
        text=welcome_text,
        parse_mode='Markdown'
    )
    logger.info(f"✅ Приветствие отправлено пользователю {chat_id}")


async def send_signal(bot_instance, chat_id):
    """Отправка сигнала с индикаторами"""
    try:
        # Отправляем сообщение о начале анализа
        await bot_instance.send_message(
            chat_id=chat_id,
            text="🔄 Анализирую рынок с использованием всех индикаторов..."
        )

        # Получаем сигнал
        indicators = await bot.get_signal()

        if not indicators:
            await bot_instance.send_message(
                chat_id=chat_id,
                text="❌ Не удалось получить данные для анализа. Проверьте API ключ Twelve Data."
            )
            return

        # Генерируем сообщение
        message = bot.generate_message(indicators)

        # Отправляем результат
        await bot_instance.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode='Markdown'
        )
        logger.info(f"✅ Профессиональный сигнал отправлен пользователю {chat_id}")

    except Exception as e:
        logger.error(f"❌ Ошибка при отправке сигнала: {e}")
        await bot_instance.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при генерации сигнала"
        )


async def send_status(bot_instance, chat_id):
    """Статус бота"""
    status_text = f"""
📊 *СТАТУС ПРОФЕССИОНАЛЬНОГО БОТА*

✅ Бот работает 24/7 на Render.com
📈 Все индикаторы активны
💹 Последний сигнал: {'есть' if bot.last_signal else 'нет'}
⏰ Время сервера: {datetime.now().strftime('%H:%M:%S')}

*Настройки индикаторов:*
• RSI период: {bot.settings['rsi_period']}
• MACD: {bot.settings['macd_fast']}/{bot.settings['macd_slow']}/{bot.settings['macd_signal']}
• Полосы Боллинджера: {bot.settings['bb_period']} период, {bot.settings['bb_std']} std
• SMA периоды: {bot.settings['sma_periods']}
• EMA периоды: {bot.settings['ema_periods']}
    """
    await bot_instance.send_message(
        chat_id=chat_id,
        text=status_text,
        parse_mode='Markdown'
    )


async def send_help(bot_instance, chat_id):
    """Отправка помощи"""
    help_text = """
📖 *ПОМОЩЬ ПО ПРОФЕССИОНАЛЬНОМУ БОТУ*

*Команды:*
/signal - полный анализ со всеми индикаторами
/status - статус и настройки бота
/help - это сообщение

*Индикаторы:*
📊 **RSI** - определяет перекупленность/перепроданность
📈 **MACD** - показывает силу и направление тренда
📉 **Bollinger Bands** - определяет волатильность и экстремумы
📏 **SMA/EMA** - скользящие средние для определения тренда
🎯 **Support/Resistance** - ключевые уровни цены

*Как интерпретировать сигнал:*
• Если большинство индикаторов указывают вверх → ПОКУПКА
• Если большинство указывают вниз → ПРОДАЖА
• Уверенность >60% - можно входить
• Ставка не более 3% от депозита
• Экспирация 3 минуты

*Удачи в торговле!* 🚀
    """
    await bot_instance.send_message(
        chat_id=chat_id,
        text=help_text,
        parse_mode='Markdown'
    )


# Для локального тестирования
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)