import asyncio
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot
import os
import functools

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
if not BOT_TOKEN:
    logger.error("❌ BOT_TOKEN не задан!")

# Создаём одну сессию для всех запросов
session = None


class EURUSDAnalyzer:
    """Класс для анализа EUR/USD"""

    def __init__(self):
        self.api_key = os.environ.get('TWELVE_API_KEY', '')
        self.prices = []

    async def get_price(self):
        """Получение цены EUR/USD"""
        url = "https://api.twelvedata.com/price"
        params = {
            'symbol': 'EUR/USD',
            'apikey': self.api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data.get('price', 0))
        except Exception as e:
            logger.error(f"Ошибка получения цены: {e}")
            return None

    def analyze(self):
        """Простой анализ тренда"""
        if len(self.prices) < 2:
            return 50, 50, 0

        current = self.prices[-1]
        prev = self.prices[-2]

        if current > prev:
            prob_up = 65
            prob_down = 35
        else:
            prob_up = 35
            prob_down = 65

        confidence = 75
        return prob_up, prob_down, confidence

    async def generate_signal(self):
        """Генерация сигнала"""
        price = await self.get_price()
        if not price:
            return None

        self.prices.append(price)
        if len(self.prices) > 100:
            self.prices.pop(0)

        prob_up, prob_down, conf = self.analyze()

        signal = {
            'timestamp': datetime.now().isoformat(),
            'price': price,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'confidence': conf,
            'recommendation': 'ПОКУПКА' if prob_up > prob_down else 'ПРОДАЖА'
        }

        return signal


# Создаём экземпляр анализатора
analyzer = EURUSDAnalyzer()


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
    """Главная страница для проверки"""
    return """
    <h1>🤖 EUR/USD Trading Bot</h1>
    <p>Бот работает! Используй Telegram для управления.</p>
    <p>Эндпоинты:</p>
    <ul>
        <li><a href="/health">/health</a> - проверка здоровья</li>
    </ul>
    """


@app.route('/health', methods=['GET'])
def health():
    """Эндпоинт для проверки здоровья"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка вебхуков от Telegram"""
    try:
        # Получаем обновление от Telegram
        update_data = request.get_json()
        logger.info(f"Получено обновление: {update_data}")

        # Создаём новый event loop для этого запроса
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Обрабатываем обновление синхронно
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
        logger.error(f"Ошибка в webhook: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


async def handle_message(chat_id, text):
    """Обработка сообщения"""
    try:
        bot = Bot(token=BOT_TOKEN)

        if text == '/start':
            await send_welcome(bot, chat_id)
        elif text == '/signal':
            await send_signal(bot, chat_id)
        elif text == '/status':
            await send_status(bot, chat_id)
        elif text == '/help':
            await send_help(bot, chat_id)
        else:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Неизвестная команда. Используй /help"
            )
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")


async def send_welcome(bot, chat_id):
    """Отправка приветствия"""
    welcome_text = """
🤖 *EUR/USD Торговый бот*

Я помогаю анализировать валютную пару EUR/USD и отправляю торговые сигналы.

*Доступные команды:*
/signal - получить текущий сигнал
/status - статус бота
/help - помощь

Бот работает 24/7 на Render.com!
    """
    await bot.send_message(
        chat_id=chat_id,
        text=welcome_text,
        parse_mode='Markdown'
    )


async def send_signal(bot, chat_id):
    """Отправка сигнала"""
    try:
        signal = await analyzer.generate_signal()
        if not signal:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ Не удалось получить данные. Проверьте API ключ Twelve Data."
            )
            return

        direction = "⬆️ ВВЕРХ" if signal['prob_up'] > signal['prob_down'] else "⬇️ ВНИЗ"
        msg = f"""
🤖 *СИГНАЛ EUR/USD*

💰 Цена: {signal['price']:.5f}
⏰ {signal['timestamp']}

📊 *Вероятности:*
⬆️ ВВЕРХ: {signal['prob_up']}%
⬇️ ВНИЗ: {signal['prob_down']}%

⚡ Уверенность: {signal['confidence']}%
📈 Рекомендация: {signal['recommendation']}
    """

        await bot.send_message(
            chat_id=chat_id,
            text=msg,
            parse_mode='Markdown'
        )
        logger.info(f"✅ Сигнал отправлен пользователю {chat_id}")

    except Exception as e:
        logger.error(f"Ошибка при отправке сигнала: {e}")
        await bot.send_message(
            chat_id=chat_id,
            text="❌ Произошла ошибка при генерации сигнала"
        )


async def send_status(bot, chat_id):
    """Статус бота"""
    status = f"""
📊 *СТАТУС БОТА*

✅ Работает 24/7 на Render.com
💰 Последняя цена: {analyzer.prices[-1] if analyzer.prices else 'Н/Д'}
📈 Всего сигналов: {len(analyzer.prices)}
    """
    await bot.send_message(
        chat_id=chat_id,
        text=status,
        parse_mode='Markdown'
    )


async def send_help(bot, chat_id):
    """Отправка помощи"""
    help_text = """
📖 *ПОМОЩЬ*

/signal - получить текущий сигнал
/status - статус бота
/help - это сообщение

*Как торговать:*
1. Получи сигнал через /signal
2. Анализируй вероятность
3. Ставь на 3 минуты
4. Риск не более 3% от депозита
    """
    await bot.send_message(
        chat_id=chat_id,
        text=help_text,
        parse_mode='Markdown'
    )


# Для локального тестирования
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)