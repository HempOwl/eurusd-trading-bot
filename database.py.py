import asyncpg
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Set

logger = logging.getLogger(__name__)


class Database:
    def __init__(self):
        self.pool = None

    async def connect(self):
        """Создание пула соединений с базой данных"""
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL не задан!")

        # Render требует SSL [citation:1]
        if 'sslmode' not in db_url:
            db_url += '?sslmode=require'

        self.pool = await asyncpg.create_pool(db_url)
        logger.info("✅ Подключение к PostgreSQL установлено")

        # Создаём таблицы, если их нет
        await self.init_tables()

    async def init_tables(self):
        """Создание необходимых таблиц"""
        async with self.pool.acquire() as conn:
            # Таблица подписчиков
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS subscribers (
                    user_id BIGINT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Таблица сигналов
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    timestamp BIGINT NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    price DECIMAL NOT NULL,
                    direction VARCHAR(4) NOT NULL,
                    tp DECIMAL,
                    sl DECIMAL,
                    result VARCHAR(10),
                    exit_price DECIMAL,
                    exit_time BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Индексы для быстрого поиска
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result) WHERE result IS NULL')

        logger.info("✅ Таблицы созданы/проверены")

    # ========== РАБОТА С ПОДПИСЧИКАМИ ==========

    async def get_subscribers(self) -> Set[int]:
        """Получение всех подписчиков"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('SELECT user_id FROM subscribers')
            return {row['user_id'] for row in rows}

    async def add_subscriber(self, user_id: int):
        """Добавление подписчика"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO subscribers (user_id) VALUES ($1)
                ON CONFLICT (user_id) DO NOTHING
            ''', user_id)

    async def remove_subscriber(self, user_id: int):
        """Удаление подписчика"""
        async with self.pool.acquire() as conn:
            await conn.execute('DELETE FROM subscribers WHERE user_id = $1', user_id)

    # ========== РАБОТА СО СТАТИСТИКОЙ ==========

    async def add_signal(self, signal: Dict):
        """Добавление сигнала в статистику"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO signals (
                    timestamp, symbol, price, direction, tp, sl, result, exit_price, exit_time
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ''',
                               signal['timestamp'],
                               signal['symbol'],
                               signal['price'],
                               signal['direction'],
                               signal.get('tp'),
                               signal.get('sl'),
                               signal.get('result'),
                               signal.get('exit_price'),
                               signal.get('exit_time')
                               )

    async def update_signal_results(self, symbol: str, current_price: float):
        """Обновление результатов для открытых сигналов указанной пары"""
        async with self.pool.acquire() as conn:
            # Получаем все незакрытые сигналы для данной пары
            rows = await conn.fetch('''
                SELECT id, timestamp, price, direction, tp, sl
                FROM signals
                WHERE symbol = $1 AND result IS NULL
            ''', symbol)

            now = int(datetime.now().timestamp())
            updated = False

            for row in rows:
                signal_id = row['id']
                direction = row['direction']
                entry = row['price']
                tp = row['tp']
                sl = row['sl']

                # Проверка тайм-аута (1 час)
                if now - row['timestamp'] > 3600:
                    await conn.execute('''
                        UPDATE signals 
                        SET result = 'timeout', exit_price = $1, exit_time = $2
                        WHERE id = $3
                    ''', current_price, now, signal_id)
                    updated = True
                    continue

                if direction == 'buy':
                    if tp and current_price >= tp:
                        await conn.execute('''
                            UPDATE signals 
                            SET result = 'profit', exit_price = $1, exit_time = $2
                            WHERE id = $3
                        ''', tp, now, signal_id)
                        updated = True
                    elif sl and current_price <= sl:
                        await conn.execute('''
                            UPDATE signals 
                            SET result = 'loss', exit_price = $1, exit_time = $2
                            WHERE id = $3
                        ''', sl, now, signal_id)
                        updated = True
                else:  # sell
                    if tp and current_price <= tp:
                        await conn.execute('''
                            UPDATE signals 
                            SET result = 'profit', exit_price = $1, exit_time = $2
                            WHERE id = $3
                        ''', tp, now, signal_id)
                        updated = True
                    elif sl and current_price >= sl:
                        await conn.execute('''
                            UPDATE signals 
                            SET result = 'loss', exit_price = $1, exit_time = $2
                            WHERE id = $3
                        ''', sl, now, signal_id)
                        updated = True

            if updated:
                logger.info(f"✅ Обновлены результаты сигналов для {symbol}")

    async def get_summary(self, symbol: Optional[str] = None) -> Dict:
        """Получение сводки по статистике (для всех пар или для конкретной)"""
        async with self.pool.acquire() as conn:
            # Базовый запрос
            query = 'SELECT COUNT(*) as total FROM signals'
            params = []

            if symbol:
                query += ' WHERE symbol = $1'
                params.append(symbol)

            total = await conn.fetchval(query, *params)

            if total == 0:
                return {
                    'total': 0, 'profit': 0, 'loss': 0, 'timeout': 0, 'unknown': 0,
                    'win_rate': 0, 'avg_profit': 0, 'avg_loss': 0,
                    'total_profit_pips': 0, 'total_loss_pips': 0
                }

            # Подсчёт результатов
            result_query = '''
                SELECT 
                    COUNT(*) FILTER (WHERE result = 'profit') as profit,
                    COUNT(*) FILTER (WHERE result = 'loss') as loss,
                    COUNT(*) FILTER (WHERE result = 'timeout') as timeout,
                    COUNT(*) FILTER (WHERE result IS NULL) as unknown
            '''
            if symbol:
                result_query += ' WHERE symbol = $1'

            row = await conn.fetchrow(result_query, *params)
            profit = row['profit']
            loss = row['loss']
            timeout = row['timeout']
            unknown = row['unknown']

            # Расчёт пипсов
            pips_query = '''
                SELECT 
                    COALESCE(SUM(
                        CASE 
                            WHEN result = 'profit' AND direction = 'buy' THEN (exit_price - price) * 10000
                            WHEN result = 'profit' AND direction = 'sell' THEN (price - exit_price) * 10000
                            ELSE 0
                        END
                    ), 0) as total_profit_pips,
                    COALESCE(SUM(
                        CASE 
                            WHEN result = 'loss' AND direction = 'buy' THEN ABS((exit_price - price) * 10000)
                            WHEN result = 'loss' AND direction = 'sell' THEN ABS((price - exit_price) * 10000)
                            ELSE 0
                        END
                    ), 0) as total_loss_pips
                FROM signals
                WHERE result IN ('profit', 'loss')
            '''
            if symbol:
                pips_query += ' AND symbol = $1'

            pips_row = await conn.fetchrow(pips_query, *params)
            total_profit_pips = pips_row['total_profit_pips']
            total_loss_pips = pips_row['total_loss_pips']

            win_rate = (profit / (profit + loss) * 100) if (profit + loss) > 0 else 0
            avg_profit = total_profit_pips / profit if profit > 0 else 0
            avg_loss = total_loss_pips / loss if loss > 0 else 0

            return {
                'total': total,
                'profit': profit,
                'loss': loss,
                'timeout': timeout,
                'unknown': unknown,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_profit_pips': total_profit_pips,
                'total_loss_pips': total_loss_pips
            }


# Создаём глобальный экземпляр БД
db = Database()