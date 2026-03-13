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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация Flask
app = Flask(__name__)

# Токен бота из переменных окружения
BOT_TOKEN = "8330877438:AAGqvHO3_2UnvhjEeyoerPHl0nFXSJl4D5w"
bot = Bot(token=BOT_TOKEN)

# Хранилище данных (в production используй БД)
user_data = {}

@app.route('/')
def index():
    return 'Бот работает!'

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка вебхуков от Telegram"""
    try:
        # Получаем обновление от Telegram
        update_data = request.get_json()
        
        # Создаем объект Update
        update = Update.de_json(update_data, bot)
        
        # Обрабатываем команды
        if update.message and update.message.text:
            chat_id = update.message.chat.id
            text = update.message.text
            
            if text == '/start':
                # Отправляем приветствие
                asyncio.run(send_message_async(chat_id, "🚀 Бот запущен!"))
            elif text == '/status':
                # Показываем статус
                asyncio.run(send_message_async(chat_id, "✅ Бот работает на Render.com 24/7"))
        
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return jsonify({'status': 'error'}), 500

async def send_message_async(chat_id, text):
    """Асинхронная отправка сообщения"""
    await bot.send_message(chat_id=chat_id, text=text)

# Запуск Flask приложения
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))