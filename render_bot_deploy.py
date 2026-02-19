import asyncio
import logging
import json
from datetime import datetime
from flask import Flask, request, jsonify
import aiohttp
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import os

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
    BOT_TOKEN = "ЗАМЕНИ_НА_ТВОЙ_ТОКЕН"  # Заглушка

# Создаём бота
bot = Bot(token=BOT_TOKEN)

# Хранилище последних сигналов (в продакшене лучше использовать БД)
last_signals = []

class EURUSDAnalyzer:
    """Класс для анализа EUR/USD (адаптированная версия из bot_pro.py)"""
    
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

@app.route('/')
def index():
    """Главная страница для проверки"""
    return """
    <h1>🤖 EUR/USD Trading Bot</h1>
    <p>Бот работает! Используй Telegram для управления.</p>
    <p>Эндпоинты:</p>
    <ul>
        <li><a href="/health">/health</a> - проверка здоровья</li>
        <li><a href="/last-signal">/last-signal</a> - последний сигнал</li>
    </ul>
    """

@app.route('/health', methods=['GET'])
def health():
    """Эндпоинт для проверки здоровья (нужен Render)"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/last-signal', methods=['GET'])
def get_last_signal():
    """Получить последний сигнал"""
    if last_signals:
        return jsonify(last_signals[-1])
    return jsonify({'error': 'No signals yet'})

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка вебхуков от Telegram"""
    try:
        # Получаем обновление от Telegram
        update_data = request.get_json()
        logger.info(f"Получено обновление: {update_data}")
        
        # Создаем объект Update
        update = Update.de_json(update_data, bot)
        
        # Запускаем обработку асинхронно
        asyncio.run(handle_update(update))
        
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Ошибка в webhook: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

async def handle_update(update):
    """Асинхронная обработка обновлений"""
    if not update.message or not update.message.text:
        return
    
    chat_id = update.message.chat.id
    text = update.message.text
    
    if text == '/start':
        await send_welcome(chat_id)
    elif text == '/signal':
        await send_signal(chat_id)
    elif text == '/status':
        await send_status(chat_id)
    elif text == '/help':
        await send_help(chat_id)
    else:
        await bot.send_message(
            chat_id=chat_id,
            text="❌ Неизвестная команда. Используй /help"
        )

async def send_welcome(chat_id):
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

async def send_signal(chat_id):
    """Отправка сигнала"""
    signal = await analyzer.generate_signal()
    if not signal:
        await bot.send_message(
            chat_id=chat_id,
            text="❌ Не удалось получить данные"
        )
        return
    
    # Сохраняем сигнал
    last_signals.append(signal)
    if len(last_signals) > 10:
        last_signals.pop(0)
    
    # Формируем сообщение
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

async def send_status(chat_id):
    """Статус бота"""
    status = f"""
📊 *СТАТУС БОТА*

✅ Работает 24/7 на Render.com
📈 Сигналов в памяти: {len(last_signals)}
💰 Последняя цена: {analyzer.prices[-1] if analyzer.prices else 'Н/Д'}
    """
    await bot.send_message(
        chat_id=chat_id,
        text=status,
        parse_mode='Markdown'
    )

async def send_help(chat_id):
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