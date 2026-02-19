import asyncio
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import os
import pandas as pd
import numpy as np

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
    """Профессиональный бот с упрощёнными индикаторами для Render"""

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

    # ---------- Вспомогательные функции для индикаторов ----------
    def _sma(self, data, period):
        """Простое скользящее среднее"""
        if len(data) < period:
            return data[-1]
        return sum(data[-period:]) / period

    def _ema(self, data, period):
        """Экспоненциальное скользящее среднее (упрощённо)"""
        if len(data) < period:
            return data[-1]
        # Начинаем с SMA
        ema = sum(data[-period:]) / period
        multiplier = 2 / (period + 1)
        for price in data[-period + 1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _rsi(self, data, period=14):
        """Расчёт RSI"""
        if len(data) < period + 1:
            return 50.0

        gains = []
        losses = []
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
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _bbands(self, data, period=20, std_dev=2):
        """Полосы Боллинджера (возвращает upper, middle, lower)"""
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

    def _macd(self, data, fast=12, slow=26, signal=9):
        """MACD линия, сигнальная линия, гистограмма (упрощённо)"""
        if len(data) < slow + signal:
            return 0.0, 0.0, 0.0

        # Рассчитываем EMA для fast и slow
        ema_fast = self._ema(data, fast)
        ema_slow = self._ema(data, slow)
        macd_line = ema_fast - ema_slow

        # Для сигнальной линии нужна EMA от MACD за signal период.
        # Упрощённо: вернём только линию
        return macd_line, 0.0, 0.0

    # ---------- Основной расчёт индикаторов ----------
    def calculate_indicators(self):
        """Расчёт упрощённых индикаторов без внешних библиотек"""
        if self.df is None or len(self.df) < 50:
            return None

        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values

        results = {}
        current_price = close[-1]

        # 1. RSI
        try:
            rsi_val = self._rsi(close, self.settings['rsi_period'])
            results['rsi'] = rsi_val
            if rsi_val > self.settings['rsi_overbought']:
                results['rsi_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (сигнал к продаже)'
            elif rsi_val < self.settings['rsi_oversold']:
                results['rsi_signal'] = 'ПЕРЕПРОДАННОСТЬ (сигнал к покупке)'
            elif rsi_val > 50:
                results['rsi_signal'] = 'ВОСХОДЯЩИЙ ТРЕНД'
            else:
                results['rsi_signal'] = 'НИСХОДЯЩИЙ ТРЕНД'
        except:
            results['rsi'] = 50.0
            results['rsi_signal'] = 'НЕЙТРАЛЬНО'

        # 2. MACD (упрощённо)
        try:
            macd_line, macd_signal_line, macd_hist = self._macd(
                close,
                self.settings['macd_fast'],
                self.settings['macd_slow'],
                self.settings['macd_signal']
            )
            results['macd'] = macd_line
            results['macd_signal'] = macd_signal_line
            results['macd_hist'] = macd_hist
            if macd_line > 0:
                results['macd_trend'] = 'БЫЧИЙ СИГНАЛ'
            elif macd_line < 0:
                results['macd_trend'] = 'МЕДВЕЖИЙ СИГНАЛ'
            else:
                results['macd_trend'] = 'НЕЙТРАЛЬНО'
        except:
            results['macd'] = 0.0
            results['macd_signal'] = 0.0
            results['macd_hist'] = 0.0
            results['macd_trend'] = 'НЕЙТРАЛЬНО'

        # 3. Полосы Боллинджера
        try:
            upper, middle, lower = self._bbands(
                close,
                self.settings['bb_period'],
                self.settings['bb_std']
            )
            results['bb_upper'] = upper
            results['bb_middle'] = middle
            results['bb_lower'] = lower
            results['bb_width'] = ((upper - lower) / middle) * 100

            if current_price >= upper:
                results['bb_position'] = 'ВЫШЕ ВЕРХНЕЙ ПОЛОСЫ'
                results['bb_signal'] = 'ПЕРЕКУПЛЕННОСТЬ (возможен откат вниз)'
            elif current_price <= lower:
                results['bb_position'] = 'НИЖЕ НИЖНЕЙ ПОЛОСЫ'
                results['bb_signal'] = 'ПЕРЕПРОДАННОСТЬ (возможен отскок вверх)'
            elif current_price > middle:
                results['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И ВЕРХНЕЙ'
                results['bb_signal'] = 'НЕЙТРАЛЬНО'
            else:
                results['bb_position'] = 'МЕЖДУ СРЕДНЕЙ И НИЖНЕЙ'
                results['bb_signal'] = 'НЕЙТРАЛЬНО'
        except:
            results['bb_upper'] = current_price * 1.02
            results['bb_middle'] = current_price
            results['bb_lower'] = current_price * 0.98
            results['bb_width'] = 2.0
            results['bb_position'] = 'СРЕДИНА'
            results['bb_signal'] = 'НЕЙТРАЛЬНО'

        # 4. Скользящие средние (SMA)
        results['sma'] = {}
        for period in self.settings['sma_periods']:
            try:
                sma_val = self._sma(close, period)
                results['sma'][period] = sma_val
                if current_price > sma_val:
                    results[f'sma_{period}_signal'] = '⬆️ ВЫШЕ'
                elif current_price < sma_val:
                    results[f'sma_{period}_signal'] = '⬇️ НИЖЕ'
                else:
                    results[f'sma_{period}_signal'] = '⏺️ ОКОЛО'
            except:
                results['sma'][period] = current_price
                results[f'sma_{period}_signal'] = '⏺️ ОКОЛО'

        # 5. Скользящие средние (EMA)
        results['ema'] = {}
        for period in self.settings['ema_periods']:
            try:
                ema_val = self._ema(close, period)
                results['ema'][period] = ema_val
                if current_price > ema_val:
                    results[f'ema_{period}_signal'] = '⬆️ ВЫШЕ'
                elif current_price < ema_val:
                    results[f'ema_{period}_signal'] = '⬇️ НИЖЕ'
                else:
                    results[f'ema_{period}_signal'] = '⏺️ ОКОЛО'
            except:
                results['ema'][period] = current_price
                results[f'ema_{period}_signal'] = '⏺️ ОКОЛО'

        # 6. Поддержка и сопротивление
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

        # 7. Общая вероятность
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

        supports.sort()
        resistances.sort()

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
            'supports': supports[-3:],
            'resistances': resistances[-3:],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance
        }

    def cluster_levels(self, levels, threshold=0.0005):
        """Группировка близких уровней (5 пипсов)"""
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
        """Расчёт вероятности на основе индикаторов"""
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
            dist_s = indicators.get('distance_to_support', 100)
            dist_r = indicators.get('distance_to_resistance', 100)
            if dist_s < dist_r:
                votes_up += 1
            else:
                votes_down += 1

        total = votes_up + votes_down
        if total > 0:
            prob_up = round((votes_up / total) * 100, 1)
            prob_down = round((votes_down / total) * 100, 1)
            confidence = round(abs(votes_up - votes_down) / total * 100, 1)
        else:
            prob_up = prob_down = 50.0
            confidence = 0.0

        return prob_up, prob_down, confidence

    def generate_message(self, indicators):
        """Генерация подробного сообщения с индикаторами"""
        if not indicators:
            return "❌ Недостаточно данных для расчета индикаторов"

        price = indicators['current_price']
        prob_up = indicators['prob_up']
        prob_down = indicators['prob_down']
        confidence = indicators['final_confidence']

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
        for period in self.settings['sma_periods']:
            if period in indicators['sma']:
                signal = indicators.get(f'sma_{period}_signal', '')
                message += f"├─ SMA({period}): `{indicators['sma'][period]:.5f}` {signal}\n"

        message += "\n📊 *Экспоненциальные средние (EMA)*\n"
        for period in self.settings['ema_periods']:
            if period in indicators['ema']:
                signal = indicators.get(f'ema_{period}_signal', '')
                message += f"├─ EMA({period}): `{indicators['ema'][period]:.5f}` {signal}\n"

        # Ближайшие уровни с проверкой на None
        support_str = f"`{indicators['nearest_support']:.5f}`" if indicators.get(
            'nearest_support') is not None else "`не определен`"
        resistance_str = f"`{indicators['nearest_resistance']:.5f}`" if indicators.get(
            'nearest_resistance') is not None else "`не определен`"
        dist_support_str = f"{indicators['distance_to_support']:.0f}" if indicators.get(
            'distance_to_support') is not None else "?"
        dist_resistance_str = f"{indicators['distance_to_resistance']:.0f}" if indicators.get(
            'distance_to_resistance') is not None else "?"

        message += f"""
📊 *Уровни поддержки/сопротивления*
┌─ Ближайшая поддержка: {support_str} (дист: {dist_support_str} пипсов)
└─ Ближайшее сопротивление: {resistance_str} (дист: {dist_resistance_str} пипсов)
"""
        if indicators.get('support_levels'):
            supports = [f"{s:.5f}" for s in indicators['support_levels'] if s is not None]
            if supports:
                message += f"├─ Уровни поддержки: {', '.join(supports)}\n"
        if indicators.get('resistance_levels'):
            resistances = [f"{r:.5f}" for r in indicators['resistance_levels'] if r is not None]
            if resistances:
                message += f"└─ Уровни сопротивления: {', '.join(resistances)}\n"

        message += f"\n#{'BUY' if prob_up > prob_down else 'SELL'} #EURUSD #TECHNICAL #ANALYSIS"
        return message

    async def get_signal(self):
        """Получение сигнала с индикаторами"""
        try:
            candles = await self.fetch_historical_data(bars=100)
            if not candles:
                logger.error("❌ Нет данных для анализа")
                return None
            self.update_dataframe(candles)
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


# ========== ФУНКЦИИ ДЛЯ СОЗДАНИЯ МЕНЮ ==========

def get_main_menu():
    """Создаёт главное меню с кнопками"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Получить сигнал", callback_data='signal'),
            InlineKeyboardButton("📈 Статус", callback_data='status')
        ],
        [
            InlineKeyboardButton("ℹ️ Помощь", callback_data='help'),
            InlineKeyboardButton("⚙️ Настройки", callback_data='settings')
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_help_menu():
    """Меню помощи"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Индикаторы", callback_data='help_indicators'),
            InlineKeyboardButton("💰 Торговля", callback_data='help_trading')
        ],
        [
            InlineKeyboardButton("❓ FAQ", callback_data='help_faq'),
            InlineKeyboardButton("◀️ Назад", callback_data='back_to_main')
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_settings_menu():
    """Меню настроек"""
    keyboard = [
        [
            InlineKeyboardButton("📏 RSI период", callback_data='set_rsi'),
            InlineKeyboardButton("📊 MACD", callback_data='set_macd')
        ],
        [
            InlineKeyboardButton("📉 Bollinger Bands", callback_data='set_bb'),
            InlineKeyboardButton("◀️ Назад", callback_data='back_to_main')
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ========== ФУНКЦИИ ДЛЯ ПОКАЗА МЕНЮ ==========

async def show_main_menu(bot_instance, chat_id):
    """Показывает главное меню"""
    await bot_instance.send_message(
        chat_id=chat_id,
        text="🤖 *Главное меню*\n\nВыберите действие:",
        reply_markup=get_main_menu(),
        parse_mode='Markdown'
    )


async def show_help_menu(bot_instance, chat_id):
    """Показывает меню помощи"""
    await bot_instance.send_message(
        chat_id=chat_id,
        text="📖 *Раздел помощи*\n\nВыберите тему:",
        reply_markup=get_help_menu(),
        parse_mode='Markdown'
    )


async def show_settings_menu(bot_instance, chat_id):
    """Показывает меню настроек"""
    settings_text = f"""
⚙️ *ТЕКУЩИЕ НАСТРОЙКИ*

📏 RSI период: {bot.settings['rsi_period']}
📊 MACD: {bot.settings['macd_fast']}/{bot.settings['macd_slow']}/{bot.settings['macd_signal']}
📉 Bollinger Bands: {bot.settings['bb_period']} период, {bot.settings['bb_std']} std
📏 SMA периоды: {bot.settings['sma_periods']}
📈 EMA периоды: {bot.settings['ema_periods']}

Выберите параметр для изменения:
    """
    await bot_instance.send_message(
        chat_id=chat_id,
        text=settings_text,
        reply_markup=get_settings_menu(),
        parse_mode='Markdown'
    )


# ========== ФУНКЦИИ ДЛЯ РАЗДЕЛОВ ПОМОЩИ ==========

async def send_help_indicators(bot_instance, chat_id):
    """Подробная помощь по индикаторам"""
    text = """
📊 *ИНДИКАТОРЫ*

*📈 RSI (Relative Strength Index)*
• Значение >70: перекупленность (сигнал к продаже)
• Значение <30: перепроданность (сигнал к покупке)
• Период: 14 свечей

*📊 MACD*
• Бычий сигнал: MACD выше сигнальной линии
• Медвежий сигнал: MACD ниже сигнальной линии
• Параметры: 12/26/9

*📉 Полосы Боллинджера*
• Касание верхней полосы: возможен откат вниз
• Касание нижней полосы: возможен отскок вверх
• Период: 20, отклонение: 2

*📏 Скользящие средние*
• SMA (простые): {bot.settings['sma_periods']}
• EMA (экспоненциальные): {bot.settings['ema_periods']}
    """

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='help')]]
    await bot_instance.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def send_help_trading(bot_instance, chat_id):
    """Помощь по торговле"""
    text = """
💰 *КАК ТОРГОВАТЬ*

1️⃣ Получите сигнал через кнопку 📊
2️⃣ Проанализируйте вероятности
3️⃣ Если уверенность >60% - можно входить
4️⃣ Ставка: не более 3% от депозита
5️⃣ Время экспирации: 3 минуты

*Правила риск-менеджмента:*
• Максимальная ставка: 3%
• Дневной лимит убытка: -15%
• Не удваивайте после проигрыша
• Лучше пропустить, чем ошибиться
    """

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='help')]]
    await bot_instance.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def send_help_faq(bot_instance, chat_id):
    """FAQ"""
    text = """
❓ *ЧАСТО ЗАДАВАЕМЫЕ ВОПРОСЫ*

*❓ Как часто обновляются сигналы?*
По запросу через команду /signal

*❓ Почему нет сигнала?*
Проверьте API ключ Twelve Data в настройках Render

*❓ Какой таймфрейм используется?*
1 минута (M1)

*❓ Можно ли доверять сигналам?*
Сигналы основаны на техническом анализе, но не гарантируют прибыль. Всегда используйте риск-менеджмент.

*❓ Бот работает 24/7?*
Да, бот запущен на Render.com и работает круглосуточно
    """

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='help')]]
    await bot_instance.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


# ========== ФУНКЦИИ ДЛЯ НАСТРОЕК ==========

async def edit_setting_rsi(bot_instance, chat_id):
    """Изменение настроек RSI"""
    text = f"""
📏 *НАСТРОЙКА RSI*

Текущее значение: {bot.settings['rsi_period']}

Для изменения параметров отредактируйте файл `render_bot_deploy.py` и перезапустите бота.

Доступные параметры:
• rsi_period: период расчёта RSI
• rsi_overbought: уровень перекупленности (70)
• rsi_oversold: уровень перепроданности (30)
    """

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='settings')]]
    await bot_instance.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def edit_setting_macd(bot_instance, chat_id):
    """Изменение настроек MACD"""
    text = f"""
📊 *НАСТРОЙКА MACD*

Текущие значения:
• Fast: {bot.settings['macd_fast']}
• Slow: {bot.settings['macd_slow']}
• Signal: {bot.settings['macd_signal']}

Для изменения параметров отредактируйте файл `render_bot_deploy.py`
    """

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='settings')]]
    await bot_instance.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


async def edit_setting_bb(bot_instance, chat_id):
    """Изменение настроек Bollinger Bands"""
    text = f"""
📉 *НАСТРОЙКА BOLLINGER BANDS*

Текущие значения:
• Период: {bot.settings['bb_period']}
• Стандартных отклонений: {bot.settings['bb_std']}

Для изменения параметров отредактируйте файл `render_bot_deploy.py`
    """

    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='settings')]]
    await bot_instance.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )


# ========== ОСНОВНЫЕ ФУНКЦИИ БОТА ==========

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
    return """
    <h1>🤖 EUR/USD Pro Trading Bot</h1>
    <p>Бот с упрощёнными индикаторами работает 24/7!</p>
    <p>Команды в Telegram:</p>
    <ul>
        <li><b>/start</b> - показать меню</li>
        <li><b>/signal</b> - получить сигнал</li>
        <li><b>/status</b> - статус бота</li>
        <li><b>/help</b> - помощь</li>
    </ul>
    """


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'indicators_available': bot.last_signal is not None
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        update_data = request.get_json()
        logger.info(f"📨 Получено обновление от пользователя")

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Создаём объект Update для обработки callback-запросов
            if 'callback_query' in update_data:
                # Для кнопок создаём специальный объект
                class CallbackQuery:
                    def __init__(self, data):
                        self.data = data
                        self.answer = lambda: None  # заглушка

                class Update:
                    def __init__(self, data):
                        self.callback_query = CallbackQuery(data['callback_query']['data'])

                update = Update(update_data)
                chat_id = update_data['callback_query']['from']['id']
                loop.run_until_complete(handle_message(chat_id, None, update))
            elif 'message' in update_data and 'text' in update_data['message']:
                chat_id = update_data['message']['chat']['id']
                text = update_data['message']['text']
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


async def handle_message(chat_id, text=None, update=None):
    """Обработка сообщений с поддержкой кнопок"""
    try:
        bot_instance = Bot(token=BOT_TOKEN)

        # Обработка callback-запросов от кнопок
        if update and hasattr(update, 'callback_query'):
            query = update.callback_query

            if query.data == 'signal':
                await send_signal(bot_instance, chat_id)
            elif query.data == 'status':
                await send_status(bot_instance, chat_id)
            elif query.data == 'help':
                await show_help_menu(bot_instance, chat_id)
            elif query.data == 'settings':
                await show_settings_menu(bot_instance, chat_id)
            elif query.data == 'back_to_main':
                await show_main_menu(bot_instance, chat_id)
            elif query.data == 'help_indicators':
                await send_help_indicators(bot_instance, chat_id)
            elif query.data == 'help_trading':
                await send_help_trading(bot_instance, chat_id)
            elif query.data == 'help_faq':
                await send_help_faq(bot_instance, chat_id)
            elif query.data == 'set_rsi':
                await edit_setting_rsi(bot_instance, chat_id)
            elif query.data == 'set_macd':
                await edit_setting_macd(bot_instance, chat_id)
            elif query.data == 'set_bb':
                await edit_setting_bb(bot_instance, chat_id)
            return

        # Обработка обычных текстовых команд
        if text == '/start':
            await send_welcome(bot_instance, chat_id)
        elif text == '/signal':
            await send_signal(bot_instance, chat_id)
        elif text == '/status':
            await send_status(bot_instance, chat_id)
        elif text == '/help':
            await show_help_menu(bot_instance, chat_id)
        elif text == '📊 Сигнал' or text == 'сигнал':
            await send_signal(bot_instance, chat_id)
        elif text == '📈 Статус' or text == 'статус':
            await send_status(bot_instance, chat_id)
        elif text == 'ℹ️ Помощь' or text == 'помощь':
            await show_help_menu(bot_instance, chat_id)
        else:
            await bot_instance.send_message(
                chat_id=chat_id,
                text="❌ Неизвестная команда. Используй кнопки меню или /help"
            )
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке сообщения: {e}")


async def send_welcome(bot_instance, chat_id):
    """Отправка приветствия с меню"""
    welcome_text = """
🤖 *EUR/USD ПРОФЕССИОНАЛЬНЫЙ БОТ*

Я анализирую валютную пару EUR/USD с использованием профессиональных индикаторов.

📊 *Доступные функции:*
• Получение сигналов с анализом
• Статус и настройки бота
• Подробная помощь по индикаторам

*Бот работает 24/7
    """

    await bot_instance.send_message(
        chat_id=chat_id,
        text=welcome_text,
        reply_markup=get_main_menu(),
        parse_mode='Markdown'
    )
    logger.info(f"✅ Приветствие с меню отправлено пользователю {chat_id}")


async def send_signal(bot_instance, chat_id):
    """Отправка сигнала с индикаторами"""
    try:
        await bot_instance.send_message(
            chat_id=chat_id,
            text="🔄 Анализирую рынок с использованием упрощённых индикаторов..."
        )
        indicators = await bot.get_signal()
        if not indicators:
            await bot_instance.send_message(
                chat_id=chat_id,
                text="❌ Не удалось получить данные для анализа. Проверьте API ключ Twelve Data."
            )
            return
        message = bot.generate_message(indicators)
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

✅ Бот работает 24/7
📈 Упрощённые индикаторы активны
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)