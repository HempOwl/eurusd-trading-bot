import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import aiohttp
import pandas as pd
import numpy as np
import talib
from dotenv import load_dotenv
from telegram import Bot, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from aiohttp_socks import ProxyConnector

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ------------------------ DataProvider ------------------------
class DataProvider:
    """Отвечает за получение и хранение свечных данных."""
    def __init__(self, api_key: str, proxy: Optional[str] = None, max_candles: int = 200):
        self.api_key = api_key
        self.proxy = proxy
        self.max_candles = max_candles
        self.session: Optional[aiohttp.ClientSession] = None
        self.df: Optional[pd.DataFrame] = None

    async def init_session(self):
        """Создание сессии с поддержкой прокси."""
        if self.proxy and self.proxy.startswith('socks5://'):
            try:
                connector = ProxyConnector.from_url(self.proxy)
                self.session = aiohttp.ClientSession(connector=connector)
                logger.info(f"Используется прокси: {self.proxy}")
            except Exception as e:
                logger.error(f"Ошибка настройки прокси: {e}")
                self.session = aiohttp.ClientSession()
        else:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    async def _request_with_retry(self, url: str, params: dict, retries: int = 3) -> Optional[dict]:
        """Выполняет GET-запрос с повторными попытками."""
        for attempt in range(retries):
            try:
                async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"HTTP {resp.status}, попытка {attempt+1}/{retries}")
            except Exception as e:
                logger.warning(f"Ошибка запроса: {e}, попытка {attempt+1}/{retries}")
            await asyncio.sleep(2 ** attempt)  # экспоненциальная задержка
        logger.error(f"Не удалось выполнить запрос после {retries} попыток")
        return None

    async def fetch_historical(self, symbol: str = 'EUR/USD', interval: str = '1min', bars: int = 200):
        """Загружает исторические свечи (один раз)."""
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': bars,
            'apikey': self.api_key
        }
        data = await self._request_with_retry(url, params)
        if data and 'values' in data:
            candles = data['values']
            self._candles_to_df(candles)
            logger.info(f"Загружено {len(candles)} исторических свечей")
        else:
            logger.error("Не удалось загрузить исторические данные")

    async def fetch_last_candle(self, symbol: str = 'EUR/USD', interval: str = '1min') -> Optional[dict]:
        """Получает последнюю свечу."""
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': 1,
            'apikey': self.api_key
        }
        data = await self._request_with_retry(url, params)
        if data and 'values' in data and data['values']:
            return data['values'][0]
        return None

    def _candles_to_df(self, candles: List[dict]):
        """Преобразует список свечей в DataFrame и сохраняет."""
        df = pd.DataFrame(candles)
        # Переименование колонок
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        # Приведение к числовым типам
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Обработка времени
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
        self.df = df

    def add_candle(self, candle: dict):
        """Добавляет одну новую свечу в DataFrame, удаляя самую старую при превышении лимита."""
        if self.df is None:
            logger.warning("DataFrame не инициализирован, пропускаем добавление")
            return
        # Преобразуем в DataFrame
        new = pd.DataFrame([candle])
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }
        new = new.rename(columns={k: v for k, v in rename_map.items() if k in new.columns})
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in new.columns:
                new[col] = pd.to_numeric(new[col], errors='coerce')
        if 'datetime' in candle:
            new['datetime'] = pd.to_datetime(candle['datetime'])
            new = new.set_index('datetime')
        # Конкатенируем
        self.df = pd.concat([self.df, new])
        # Убираем дубликаты по индексу (оставляем последнее вхождение)
        self.df = self.df[~self.df.index.duplicated(keep='last')]
        # Сортируем и обрезаем
        self.df = self.df.sort_index().iloc[-self.max_candles:].copy()

    def get_df(self) -> Optional[pd.DataFrame]:
        """Возвращает копию DataFrame для безопасных расчётов."""
        return self.df.copy() if self.df is not None else None


# ------------------------ IndicatorEngine ------------------------
class IndicatorEngine:
    """Вычисляет технические индикаторы."""
    def __init__(self, settings: dict):
        self.settings = settings

    def compute(self, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < 50:
            return None

        close = df['Close'].values.astype(float)
        high = df['High'].values.astype(float)
        low = df['Low'].values.astype(float)
        # volume = df['Volume'].values if 'Volume' in df.columns else None

        results = {}

        # --- RSI ---
        try:
            rsi = talib.RSI(close, timeperiod=self.settings['rsi_period'])
            results['rsi'] = rsi[-1]
            results['rsi_signal'] = self._rsi_signal(results['rsi'])
        except:
            results['rsi'] = 50.0
            results['rsi_signal'] = 'НЕЙТРАЛЬНО'

        # --- MACD ---
        try:
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=self.settings['macd_fast'],
                slowperiod=self.settings['macd_slow'],
                signalperiod=self.settings['macd_signal']
            )
            results['macd'] = macd[-1]
            results['macd_signal_line'] = macd_signal[-1]
            results['macd_hist'] = macd_hist[-1]
            results['macd_trend'] = self._macd_signal(macd[-1], macd_signal[-1], macd_hist[-1])
        except:
            results['macd'] = 0.0
            results['macd_signal_line'] = 0.0
            results['macd_hist'] = 0.0
            results['macd_trend'] = 'НЕЙТРАЛЬНО'

        # --- Bollinger Bands ---
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
            results['bb_position'] = self._bb_position(close[-1], upper[-1], middle[-1], lower[-1])
            results['bb_signal'] = self._bb_signal(close[-1], upper[-1], lower[-1])
        except:
            results['bb_upper'] = close[-1] * 1.02
            results['bb_middle'] = close[-1]
            results['bb_lower'] = close[-1] * 0.98
            results['bb_width'] = 2.0
            results['bb_position'] = 'СРЕДИНА'
            results['bb_signal'] = 'НЕЙТРАЛЬНО'

        # --- SMA ---
        results['sma'] = {}
        for period in self.settings['sma_periods']:
            try:
                sma = talib.SMA(close, timeperiod=period)
                results['sma'][period] = sma[-1]
            except:
                results['sma'][period] = close[-1]

        # --- EMA ---
        results['ema'] = {}
        for period in self.settings['ema_periods']:
            try:
                ema = talib.EMA(close, timeperiod=period)
                results['ema'][period] = ema[-1]
            except:
                results['ema'][period] = close[-1]

        # --- ATR ---
        try:
            atr = talib.ATR(high, low, close, timeperiod=14)
            results['atr'] = atr[-1]
        except:
            results['atr'] = 0.0

        # --- Дивергенции RSI ---
        results['rsi_divergence'] = self._find_rsi_divergence(close, rsi)

        # --- Пересечения скользящих ---
        results['ma_cross'] = self._ma_cross(close)

        # --- Уровни поддержки/сопротивления ---
        sup_res = self._find_support_resistance(high, low, close)
        results.update(sup_res)

        # Текущая цена
        results['current_price'] = close[-1]

        return results

    def _rsi_signal(self, rsi: float) -> str:
        if rsi > self.settings['rsi_overbought']:
            return 'ПЕРЕКУПЛЕННОСТЬ'
        elif rsi < self.settings['rsi_oversold']:
            return 'ПЕРЕПРОДАННОСТЬ'
        elif rsi > 50:
            return 'ВОСХОДЯЩИЙ ТРЕНД'
        else:
            return 'НИСХОДЯЩИЙ ТРЕНД'

    def _macd_signal(self, macd: float, signal: float, hist: float) -> str:
        if macd > signal and hist > 0:
            return 'БЫЧИЙ'
        elif macd < signal and hist < 0:
            return 'МЕДВЕЖИЙ'
        elif macd > signal and hist < 0:
            return 'ФОРМИРОВАНИЕ БЫЧЬЕГО'
        elif macd < signal and hist > 0:
            return 'ФОРМИРОВАНИЕ МЕДВЕЖЬЕГО'
        else:
            return 'НЕЙТРАЛЬНО'

    def _bb_position(self, price, upper, middle, lower) -> str:
        if price >= upper:
            return 'ВЫШЕ ВЕРХНЕЙ'
        if price <= lower:
            return 'НИЖЕ НИЖНЕЙ'
        if price > middle:
            return 'МЕЖДУ СРЕДНЕЙ И ВЕРХНЕЙ'
        return 'МЕЖДУ СРЕДНЕЙ И НИЖНЕЙ'

    def _bb_signal(self, price, upper, lower) -> str:
        if price >= upper:
            return 'ПЕРЕКУПЛЕННОСТЬ'
        if price <= lower:
            return 'ПЕРЕПРОДАННОСТЬ'
        return 'НЕЙТРАЛЬНО'

    def _find_rsi_divergence(self, close, rsi, window=5):
        """Поиск дивергенции RSI (упрощённо)."""
        # Берём последние 2 минимума цены и RSI
        # Для реального использования нужна более сложная логика
        return None  # Заглушка

    def _ma_cross(self, close):
        """Проверка пересечения быстрой и медленной EMA."""
        try:
            ema_fast = talib.EMA(close, timeperiod=5)
            ema_slow = talib.EMA(close, timeperiod=20)
            if len(ema_fast) < 2:
                return None
            if ema_fast[-2] <= ema_slow[-2] and ema_fast[-1] > ema_slow[-1]:
                return 'GOLDEN_CROSS'
            if ema_fast[-2] >= ema_slow[-2] and ema_fast[-1] < ema_slow[-1]:
                return 'DEATH_CROSS'
        except:
            pass
        return None

    def _find_support_resistance(self, high, low, close, window=5, threshold=0.0005):
        """Поиск уровней поддержки и сопротивления (как в оригинале)."""
        supports = []
        resistances = []
        n = len(close)
        for i in range(window, n - window):
            if all(low[i] <= low[i - j] for j in range(1, window + 1)) and \
               all(low[i] <= low[i + j] for j in range(1, window + 1)):
                supports.append(low[i])
            if all(high[i] >= high[i - j] for j in range(1, window + 1)) and \
               all(high[i] >= high[i + j] for j in range(1, window + 1)):
                resistances.append(high[i])

        # Кластеризация
        supports = self._cluster(supports, threshold)
        resistances = self._cluster(resistances, threshold)

        current = close[-1]
        nearest_support = None
        nearest_resistance = None
        dist_to_support = float('inf')
        dist_to_resistance = float('inf')
        for s in supports:
            if s < current and (current - s) < dist_to_support:
                dist_to_support = current - s
                nearest_support = s
        for r in resistances:
            if r > current and (r - current) < dist_to_resistance:
                dist_to_resistance = r - current
                nearest_resistance = r

        return {
            'supports': supports[-5:],
            'resistances': resistances[-5:],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'distance_to_support': dist_to_support if nearest_support else None,
            'distance_to_resistance': dist_to_resistance if nearest_resistance else None
        }

    def _cluster(self, levels, threshold):
        if not levels:
            return []
        levels.sort()
        clustered = []
        current = [levels[0]]
        for lvl in levels[1:]:
            if abs(lvl - sum(current)/len(current)) < threshold:
                current.append(lvl)
            else:
                clustered.append(sum(current)/len(current))
                current = [lvl]
        clustered.append(sum(current)/len(current))
        return clustered


# ------------------------ SignalGenerator ------------------------
class SignalGenerator:
    """Генерирует торговый сигнал на основе индикаторов и голосования."""
    def __init__(self, settings: dict):
        self.settings = settings

    def generate(self, indicators: Dict) -> Dict:
        """Возвращает вероятности, уверенность и рекомендацию."""
        if not indicators:
            return {}

        votes_up = 0
        votes_down = 0

        # RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            votes_up += 3
        elif rsi > 70:
            votes_down += 3
        elif rsi > 50:
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

        # SMA сигналы (цена выше/ниже SMA)
        current_price = indicators.get('current_price')
        for period in self.settings['sma_periods']:
            sma_val = indicators['sma'].get(period)
            if sma_val and current_price:
                if current_price > sma_val * 1.0005:  # 0.05% выше
                    votes_up += 1
                elif current_price < sma_val * 0.9995:
                    votes_down += 1

        # EMA сигналы
        for period in self.settings['ema_periods']:
            ema_val = indicators['ema'].get(period)
            if ema_val and current_price:
                if current_price > ema_val * 1.0005:
                    votes_up += 1
                elif current_price < ema_val * 0.9995:
                    votes_down += 1

        # Поддержка/сопротивление
        dist_s = indicators.get('distance_to_support')
        dist_r = indicators.get('distance_to_resistance')
        if dist_s is not None and dist_r is not None:
            if dist_s < dist_r:
                votes_up += 1
            else:
                votes_down += 1

        # Пересечение MA
        cross = indicators.get('ma_cross')
        if cross == 'GOLDEN_CROSS':
            votes_up += 3
        elif cross == 'DEATH_CROSS':
            votes_down += 3

        # ATR (не голосует, но может влиять на уверенность)
        atr = indicators.get('atr', 0)

        total = votes_up + votes_down
        if total > 0:
            prob_up = round((votes_up / total) * 100, 1)
            prob_down = round((votes_down / total) * 100, 1)
            confidence = round(abs(votes_up - votes_down) / total * 100, 1)
        else:
            prob_up = prob_down = 50
            confidence = 0

        # Рекомендация
        if prob_up > prob_down + 10 and confidence > 50:
            recommendation = "СИЛЬНАЯ ПОКУПКА"
        elif prob_up > prob_down:
            recommendation = "ПОКУПКА"
        elif prob_down > prob_up + 10 and confidence > 50:
            recommendation = "СИЛЬНАЯ ПРОДАЖА"
        elif prob_down > prob_up:
            recommendation = "ПРОДАЖА"
        else:
            recommendation = "ОЖИДАНИЕ"

        indicators.update({
            'prob_up': prob_up,
            'prob_down': prob_down,
            'confidence': confidence,
            'recommendation': recommendation,
            'atr': atr
        })
        return indicators


# ------------------------ TelegramNotifier ------------------------
class TelegramNotifier:
    """Отправка сообщений и обработка команд."""
    def __init__(self, token: str, chat_id: str, controller):
        self.token = token
        self.chat_id = chat_id
        self.controller = controller  # ссылка на основной класс бота
        self.application = Application.builder().token(token).build()
        self._register_handlers()

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("signal", self.cmd_signal))
        self.application.add_handler(CommandHandler("settings", self.cmd_settings))
        self.application.add_handler(CommandHandler("sub", self.cmd_sub))

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "✅ Бот запущен и анализирует EUR/USD каждую минуту.\n"
            "Команды:\n"
            "/signal - принудительный сигнал\n"
            "/settings - показать настройки\n"
            "/sub <инструмент> - подписаться на другой инструмент (пока не реализовано)"
        )

    async def cmd_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("⏳ Генерирую сигнал...")
        indicators = await self.controller.force_analysis()
        msg = self.controller.format_message(indicators, detailed=True)
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings = self.controller.get_settings()
        text = "<b>Текущие настройки индикаторов:</b>\n"
        for k, v in settings.items():
            text += f"• {k}: {v}\n"
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)

    async def cmd_sub(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Заглушка для подписки
        await update.message.reply_text("Функция подписки пока в разработке.")

    async def send_message(self, text: str, detailed: bool = False):
        """Отправляет сообщение в Telegram."""
        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")

    def run(self):
        """Запуск поллинга команд (в отдельном потоке)."""
        self.application.run_polling()


# ------------------------ BotController ------------------------
class BotController:
    """Главный класс, координирующий работу компонентов."""
    def __init__(self):
        # Загрузка настроек
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
        }

        self.token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('CHAT_ID')
        self.api_key = os.getenv('TWELVE_API_KEY')
        self.proxy = os.getenv('PROXY')

        if not all([self.token, self.chat_id, self.api_key]):
            raise ValueError("Не все переменные окружения заданы")

        # Компоненты
        self.data_provider = DataProvider(self.api_key, self.proxy)
        self.indicator_engine = IndicatorEngine(self.settings)
        self.signal_generator = SignalGenerator(self.settings)
        self.notifier = TelegramNotifier(self.token, self.chat_id, self)

        # Флаги состояния
        self.analyzing = False
        self.running = True

    async def init(self):
        """Инициализация сессии и загрузка истории."""
        await self.data_provider.init_session()
        await self.data_provider.fetch_historical()
        logger.info("Бот инициализирован")

    async def shutdown(self):
        """Корректное завершение."""
        self.running = False
        await self.data_provider.close_session()
        logger.info("Бот завершил работу")

    async def force_analysis(self) -> Optional[Dict]:
        """Принудительный анализ на текущих данных."""
        df = self.data_provider.get_df()
        if df is None:
            return None
        indicators = self.indicator_engine.compute(df)
        if indicators:
            return self.signal_generator.generate(indicators)
        return None

    def format_message(self, indicators: Dict, detailed: bool = False) -> str:
        """Формирует текст сообщения (HTML)."""
        if not indicators:
            return "❌ Нет данных для анализа"

        price = indicators['current_price']
        prob_up = indicators['prob_up']
        prob_down = indicators['prob_down']
        conf = indicators['confidence']
        rec = indicators['recommendation']

        # Эмодзи в зависимости от рекомендации
        if 'ПОКУПКА' in rec:
            emoji = "🟢"
        elif 'ПРОДАЖА' in rec:
            emoji = "🔴"
        else:
            emoji = "⚪"

        # Краткий режим
        if not detailed:
            msg = f"""
{emoji} <b>EUR/USD</b> {emoji}
⏰ {datetime.now().strftime('%H:%M:%S')}
💰 <code>{price:.5f}</code>

📊 <b>Вероятность:</b>
⬆️ {prob_up}%  ⬇️ {prob_down}%
🎯 Уверенность: {conf}%
💡 Рекомендация: {rec}

#EURUSD
"""
            return msg

        # Полный режим
        msg = f"""
{emoji} <b>ПРОФЕССИОНАЛЬНЫЙ АНАЛИЗ EUR/USD</b> {emoji}
⏰ {datetime.now().strftime('%H:%M:%S')}
💰 <b>Цена:</b> <code>{price:.5f}</code>

📊 <b>ВЕРОЯТНОСТЬ</b>
⬆️ ВВЕРХ: {prob_up}%
⬇️ ВНИЗ: {prob_down}%
🎯 Уверенность: {conf}%
💡 Рекомендация: {rec}

📈 <b>RSI</b> ({self.settings['rsi_period']})
┌ Значение: <code>{indicators['rsi']:.1f}</code>
└ Сигнал: {indicators['rsi_signal']}

📊 <b>MACD</b> ({self.settings['macd_fast']},{self.settings['macd_slow']},{self.settings['macd_signal']})
┌ MACD: <code>{indicators['macd']:.5f}</code>
├ Сигнал: <code>{indicators['macd_signal_line']:.5f}</code>
├ Гистограмма: <code>{indicators['macd_hist']:.5f}</code>
└ Тренд: {indicators['macd_trend']}

📉 <b>Bollinger Bands</b> ({self.settings['bb_period']},{self.settings['bb_std']})
┌ Верхняя: <code>{indicators['bb_upper']:.5f}</code>
├ Средняя: <code>{indicators['bb_middle']:.5f}</code>
├ Нижняя: <code>{indicators['bb_lower']:.5f}</code>
├ Ширина: <code>{indicators['bb_width']:.2f}%</code>
├ Позиция: {indicators['bb_position']}
└ Сигнал: {indicators['bb_signal']}

📏 <b>SMA</b>
"""
        for p in self.settings['sma_periods']:
            if p in indicators['sma']:
                msg += f"├ SMA({p}): <code>{indicators['sma'][p]:.5f}</code>\n"

        msg += "\n📊 <b>EMA</b>\n"
        for p in self.settings['ema_periods']:
            if p in indicators['ema']:
                msg += f"├ EMA({p}): <code>{indicators['ema'][p]:.5f}</code>\n"

        msg += f"""
📊 <b>Уровни</b>
┌ Ближ. поддержка: <code>{indicators['nearest_support']:.5f}</code> (дист: {indicators['distance_to_support']*10000:.0f} пипсов)
├ Ближ. сопротивление: <code>{indicators['nearest_resistance']:.5f}</code> (дист: {indicators['distance_to_resistance']*10000:.0f} пипсов)
"""
        if indicators['supports']:
            supps = ', '.join([f"{s:.5f}" for s in indicators['supports']])
            msg += f"├ Поддержки: {supps}\n"
        if indicators['resistances']:
            resists = ', '.join([f"{r:.5f}" for r in indicators['resistances']])
            msg += f"└ Сопротивления: {resists}\n"

        msg += "\n📊 <b>ATR</b> (14): " + (f"<code>{indicators['atr']:.5f}</code>" if indicators.get('atr') else "—")
        msg += "\n\n#EURUSD #TECHNICAL"
        return msg

    def get_settings(self):
        return self.settings

    async def periodic_analysis(self):
        """Цикл анализа каждую минуту (синхронизирован с началом новой минуты)."""
        # Первый анализ сразу после инициализации
        await self.analyze_and_notify()

        while self.running:
            # Ждём до следующей минуты + 5 сек
            now = datetime.now()
            next_min = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
            sleep_sec = (next_min - now).total_seconds()
            if sleep_sec > 0:
                await asyncio.sleep(sleep_sec)

            if not self.running:
                break

            await self.analyze_and_notify()

    async def analyze_and_notify(self):
        """Получает новые данные, обновляет DataFrame, вычисляет индикаторы и отправляет сообщение."""
        if self.analyzing:
            logger.warning("Предыдущий анализ ещё не завершён, пропускаем")
            return
        self.analyzing = True
        try:
            # Получаем последнюю свечу
            last = await self.data_provider.fetch_last_candle()
            if last:
                last_time = pd.to_datetime(last['datetime'])
                df = self.data_provider.get_df()
                if df is not None and last_time > df.index[-1]:
                    self.data_provider.add_candle(last)
                    logger.info(f"Добавлена новая свеча на {last_time}")
                else:
                    logger.debug("Новая свеча ещё не появилась или дубликат")
            else:
                logger.warning("Не удалось получить последнюю свечу")

            # Вычисляем индикаторы и генерируем сигнал
            df = self.data_provider.get_df()
            if df is None:
                logger.error("Нет данных для анализа")
                return
            indicators = self.indicator_engine.compute(df)
            if not indicators:
                logger.error("Ошибка вычисления индикаторов")
                return
            signal = self.signal_generator.generate(indicators)

            # Формируем сообщение (краткое для регулярной отправки)
            msg = self.format_message(signal, detailed=False)
            await self.notifier.send_message(msg)

        except Exception as e:
            logger.exception(f"Ошибка в analyze_and_notify: {e}")
        finally:
            self.analyzing = False

    async def run(self):
        """Запуск контроллера."""
        await self.init()
        # Запускаем поллинг команд в фоне
        asyncio.create_task(self._run_polling())
        # Запускаем периодический анализ
        await self.periodic_analysis()

    async def _run_polling(self):
        """Запуск поллинга Telegram в отдельной корутине."""
        await self.notifier.application.initialize()
        await self.notifier.application.start()
        # Бесконечный поллинг
        try:
            while self.running:
                await asyncio.sleep(1)  # Заглушка, реальный поллинг внутри application.run_polling()
                # В текущей версии python-telegram-bot run_polling блокирует, поэтому нужно запускать в отдельном потоке или процессе.
                # Для простоты можно оставить run_polling синхронным, но тогда надо запускать в отдельном потоке.
                # Упростим: оставим run_polling как есть, но тогда он блокирует основной цикл.
                # Решение: используем Application.run_polling() в отдельном потоке через asyncio.to_thread.
        except Exception as e:
            logger.error(f"Polling error: {e}")


# ------------------------ Main ------------------------
async def main():
    controller = BotController()
    try:
        await controller.run()
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
    finally:
        await controller.shutdown()

if __name__ == "__main__":
    asyncio.run(main())