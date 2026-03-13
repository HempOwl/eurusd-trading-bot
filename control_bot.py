import asyncio
import logging
import json
import subprocess
import os
import signal
import sys
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('control_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBotController:
    def __init__(self, config_file='config.json'):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.bot_token = config['telegram_token']
        self.chat_id = config['chat_id']
        self.trading_process = None
        self.trading_pid = None
        self.process_lock = asyncio.Lock()
        
    async def start_trading_bot(self):
        """Запуск основного торгового бота"""
        async with self.process_lock:
            if self.trading_process and self.trading_process.poll() is None:
                return False, "Бот уже запущен"
            
            try:
                # Запускаем торгового бота в отдельном процессе
                self.trading_process = subprocess.Popen(
                    [sys.executable, 'bot_pro.py'],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.trading_pid = self.trading_process.pid
                
                # Небольшая задержка для проверки запуска
                await asyncio.sleep(2)
                
                if self.trading_process.poll() is None:
                    logger.info(f"✅ Торговый бот запущен (PID: {self.trading_pid})")
                    return True, f"✅ Торговый бот запущен!\nPID: {self.trading_pid}"
                else:
                    # Проверяем ошибки
                    stdout, stderr = self.trading_process.communicate(timeout=1)
                    error_msg = stderr if stderr else "Неизвестная ошибка"
                    logger.error(f"❌ Ошибка запуска: {error_msg}")
                    return False, f"❌ Ошибка запуска:\n{error_msg}"
                    
            except Exception as e:
                logger.error(f"❌ Ошибка при запуске: {e}")
                return False, f"❌ Ошибка: {str(e)}"
    
    async def stop_trading_bot(self):
        """Остановка торгового бота"""
        async with self.process_lock:
            if not self.trading_process or self.trading_process.poll() is not None:
                return False, "Бот не запущен"
            
            try:
                # Пытаемся завершить процесс gracefully
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/F', '/PID', str(self.trading_pid)], 
                                 capture_output=True)
                else:  # Linux/Mac
                    os.kill(self.trading_pid, signal.SIGTERM)
                
                # Ждем завершения
                for _ in range(10):
                    if self.trading_process.poll() is not None:
                        break
                    await asyncio.sleep(0.5)
                
                # Если всё ещё работает - принудительно
                if self.trading_process.poll() is None:
                    self.trading_process.terminate()
                    await asyncio.sleep(1)
                
                self.trading_process = None
                self.trading_pid = None
                
                logger.info("✅ Торговый бот остановлен")
                return True, "✅ Торговый бот остановлен"
                
            except Exception as e:
                logger.error(f"❌ Ошибка при остановке: {e}")
                return False, f"❌ Ошибка при остановке: {str(e)}"
    
    async def get_status(self):
        """Получение статуса бота"""
        if self.trading_process and self.trading_process.poll() is None:
            # Бот работает, проверим логи
            uptime = "неизвестно"
            try:
                # Попробуем прочитать последнюю строку лога
                if os.path.exists('bot_pro.log'):
                    with open('bot_pro.log', 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1]
                            uptime = f"Последний сигнал: {last_line[:50]}..."
            except:
                pass
            
            return (
                "🟢 **СТАТУС: БОТ РАБОТАЕТ**\n\n"
                f"📊 PID: {self.trading_pid}\n"
                f"⏱️ {uptime}\n"
                f"💡 Используйте кнопки ниже для управления"
            )
        else:
            return (
                "🔴 **СТАТУС: БОТ ОСТАНОВЛЕН**\n\n"
                f"⏰ Время: {datetime.now().strftime('%H:%M:%S')}\n"
                f"💡 Нажмите '▶️ Запустить' для старта"
            )

# Создаем контроллер
controller = TradingBotController()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    keyboard = [
        [
            InlineKeyboardButton("▶️ Запустить бота", callback_data='start_bot'),
            InlineKeyboardButton("⏹️ Остановить", callback_data='stop_bot')
        ],
        [
            InlineKeyboardButton("🔄 Статус", callback_data='status'),
            InlineKeyboardButton("📊 Последний сигнал", callback_data='last_signal')
        ],
        [
            InlineKeyboardButton("⚙️ Настройки", callback_data='settings'),
            InlineKeyboardButton("📈 Помощь", callback_data='help')
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_text = (
        "🤖 **УПРАВЛЕНИЕ ТОРГОВЫМ БОТОМ**\n\n"
        "Привет! Я помогу тебе управлять ботом для торговли EUR/USD.\n\n"
        "📊 **Доступные команды:**\n"
        "▶️ Запустить - старт торгового бота\n"
        "⏹️ Остановить - остановка бота\n"
        "🔄 Статус - проверить работает ли бот\n"
        "📊 Последний сигнал - показать последний сигнал\n\n"
        "⚙️ Настройки - изменить параметры\n"
        "📈 Помощь - инструкция по торговле"
    )
    
    await update.message.reply_text(
        welcome_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'start_bot':
        success, message = await controller.start_trading_bot()
        await query.edit_message_text(
            text=message,
            parse_mode='Markdown'
        )
        # Возвращаем клавиатуру через 2 секунды
        await asyncio.sleep(2)
        await show_main_menu(query)
        
    elif query.data == 'stop_bot':
        success, message = await controller.stop_trading_bot()
        await query.edit_message_text(
            text=message,
            parse_mode='Markdown'
        )
        await asyncio.sleep(2)
        await show_main_menu(query)
        
    elif query.data == 'status':
        status_text = await controller.get_status()
        
        keyboard = [
            [
                InlineKeyboardButton("▶️ Запустить", callback_data='start_bot'),
                InlineKeyboardButton("⏹️ Остановить", callback_data='stop_bot')
            ],
            [InlineKeyboardButton("◀️ Назад", callback_data='back_to_menu')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=status_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    elif query.data == 'last_signal':
        await show_last_signal(query)
        
    elif query.data == 'settings':
        await show_settings(query)
        
    elif query.data == 'help':
        await show_help(query)
        
    elif query.data == 'back_to_menu':
        await show_main_menu(query)

async def show_main_menu(query):
    """Показать главное меню"""
    keyboard = [
        [
            InlineKeyboardButton("▶️ Запустить бота", callback_data='start_bot'),
            InlineKeyboardButton("⏹️ Остановить", callback_data='stop_bot')
        ],
        [
            InlineKeyboardButton("🔄 Статус", callback_data='status'),
            InlineKeyboardButton("📊 Последний сигнал", callback_data='last_signal')
        ],
        [
            InlineKeyboardButton("⚙️ Настройки", callback_data='settings'),
            InlineKeyboardButton("📈 Помощь", callback_data='help')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text="🤖 **ГЛАВНОЕ МЕНЮ**\n\nВыберите действие:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def show_last_signal(query):
    """Показать последний сигнал от бота"""
    try:
        if os.path.exists('bot_pro.log'):
            with open('bot_pro.log', 'r') as f:
                lines = f.readlines()
                # Ищем последние 10 строк с сигналами
                signals = []
                for line in reversed(lines):
                    if '✅ Сигнал с индикаторами отправлен' in line:
                        signals.append(line)
                        if len(signals) >= 3:
                            break
                
                if signals:
                    text = "📊 **ПОСЛЕДНИЕ СИГНАЛЫ**\n\n"
                    for i, signal in enumerate(signals[:3], 1):
                        text += f"{i}. {signal}\n"
                else:
                    text = "📊 Сигналов пока нет"
        else:
            text = "📊 Файл лога не найден"
    except:
        text = "📊 Не удалось прочитать сигналы"
    
    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='back_to_menu')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text=text,
        reply_markup=reply_markup
    )

async def show_settings(query):
    """Показать настройки"""
    keyboard = [
        [InlineKeyboardButton("⏱️ Интервал сигналов (1 мин)", callback_data='set_interval')],
        [InlineKeyboardButton("📊 Уровень уверенности (>60%)", callback_data='set_confidence')],
        [InlineKeyboardButton("◀️ Назад", callback_data='back_to_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text="⚙️ **НАСТРОЙКИ**\n\nЗдесь можно изменить параметры бота:\n\n"
             "⏱️ Интервал: 1 минута\n"
             "📊 Уверенность: >60%\n"
             "💰 Торговая пара: EUR/USD",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def show_help(query):
    """Показать помощь"""
    help_text = (
        "📈 **ИНСТРУКЦИЯ ПО ТОРГОВЛЕ**\n\n"
        "1️⃣ **Запусти бота** кнопкой '▶️ Запустить'\n"
        "2️⃣ **Получай сигналы** каждую минуту\n"
        "3️⃣ **Анализируй сигнал**:\n"
        "   • Уверенность >60% - можно входить\n"
        "   • RSI <30 - перепроданность (к покупке)\n"
        "   • RSI >70 - перекупленность (к продаже)\n"
        "   • MACD пересечение - смена тренда\n"
        "   • Цена у полос Боллинджера - отскок\n\n"
        "4️⃣ **Риск-менеджмент**:\n"
        "   • Ставка 2-3% от депозита\n"
        "   • Экспирация 3 минуты\n"
        "   • Стоп-лосс на день: -15%\n\n"
        "❓ **Кнопки управления**:\n"
        "▶️ Запустить - старт торгового бота\n"
        "⏹️ Остановить - остановка\n"
        "🔄 Статус - проверить работу\n"
        "📊 Последний сигнал - показать историю"
    )
    
    keyboard = [[InlineKeyboardButton("◀️ Назад", callback_data='back_to_menu')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        text=help_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /status"""
    status_text = await controller.get_status()
    
    keyboard = [
        [
            InlineKeyboardButton("▶️ Запустить", callback_data='start_bot'),
            InlineKeyboardButton("⏹️ Остановить", callback_data='stop_bot')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        status_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /stop"""
    success, message = await controller.stop_trading_bot()
    await update.message.reply_text(message)

def main():
    """Главная функция запуска управляющего бота"""
    # Создаем приложение
    application = Application.builder().token(controller.bot_token).build()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("stop", stop_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # Запускаем бота
    print("🤖 Управляющий бот запущен! Нажми Ctrl+C для остановки")
    print(f"📱 Открой Telegram и отправь /start боту @{controller.bot_token[:10]}...")
    
    # Запуск с обработкой сигналов
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Управляющий бот остановлен")
        # При остановке управляющего бота, останавливаем и торгового
        asyncio.run(controller.stop_trading_bot())