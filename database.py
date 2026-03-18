import aiosqlite
import json
import time
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)

DB_PATH = "bot.db"


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        # Таблица подписчиков
        await db.execute('''
            CREATE TABLE IF NOT EXISTS subscribers (
                user_id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Таблица сигналов
        await db.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                direction TEXT NOT NULL,
                tp REAL,
                sl REAL,
                result TEXT,
                exit_price REAL,
                exit_time INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Таблица для кэширования свечей (для уменьшения запросов к Twelve Data)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                datetime TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, datetime)
            )
        ''')
        # Индексы
        await db.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result) WHERE result IS NULL')
        await db.commit()


# ========== РАБОТА С ПОДПИСЧИКАМИ ==========
async def get_subscribers() -> Set[int]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute('SELECT user_id FROM subscribers') as cursor:
            rows = await cursor.fetchall()
            return {row[0] for row in rows}


async def add_subscriber(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('INSERT OR IGNORE INTO subscribers (user_id) VALUES (?)', (user_id,))
        await db.commit()


async def remove_subscriber(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('DELETE FROM subscribers WHERE user_id = ?', (user_id,))
        await db.commit()


# ========== РАБОТА СО СТАТИСТИКОЙ ==========
async def add_signal(signal: Dict):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            INSERT INTO signals (timestamp, symbol, price, direction, tp, sl, result, exit_price, exit_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['timestamp'],
            signal['symbol'],
            signal['price'],
            signal['direction'],
            signal.get('tp'),
            signal.get('sl'),
            signal.get('result'),
            signal.get('exit_price'),
            signal.get('exit_time')
        ))
        await db.commit()


async def update_signal_results(symbol: str, current_price: float):
    async with aiosqlite.connect(DB_PATH) as db:
        # Получаем незакрытые сигналы для данной пары
        async with db.execute('''
            SELECT id, timestamp, price, direction, tp, sl
            FROM signals
            WHERE symbol = ? AND result IS NULL
        ''', (symbol,)) as cursor:
            rows = await cursor.fetchall()

        now = int(time.time())
        updated = False
        for row in rows:
            signal_id, ts, entry, direction, tp, sl = row
            # Проверка тайм-аута (1 час)
            if now - ts > 3600:
                await db.execute('''
                    UPDATE signals SET result = 'timeout', exit_price = ?, exit_time = ?
                    WHERE id = ?
                ''', (current_price, now, signal_id))
                updated = True
                continue

            if direction == 'buy':
                if tp and current_price >= tp:
                    await db.execute('''
                        UPDATE signals SET result = 'profit', exit_price = ?, exit_time = ?
                        WHERE id = ?
                    ''', (tp, now, signal_id))
                    updated = True
                elif sl and current_price <= sl:
                    await db.execute('''
                        UPDATE signals SET result = 'loss', exit_price = ?, exit_time = ?
                        WHERE id = ?
                    ''', (sl, now, signal_id))
                    updated = True
            else:  # sell
                if tp and current_price <= tp:
                    await db.execute('''
                        UPDATE signals SET result = 'profit', exit_price = ?, exit_time = ?
                        WHERE id = ?
                    ''', (tp, now, signal_id))
                    updated = True
                elif sl and current_price >= sl:
                    await db.execute('''
                        UPDATE signals SET result = 'loss', exit_price = ?, exit_time = ?
                        WHERE id = ?
                    ''', (sl, now, signal_id))
                    updated = True

        if updated:
            await db.commit()


async def get_summary(symbol: Optional[str] = None) -> Dict:
    async with aiosqlite.connect(DB_PATH) as db:
        base_query = 'SELECT COUNT(*) FROM signals'
        params = []
        if symbol:
            base_query += ' WHERE symbol = ?'
            params.append(symbol)

        async with db.execute(base_query, params) as cursor:
            total = (await cursor.fetchone())[0]

        if total == 0:
            return {
                'total': 0, 'profit': 0, 'loss': 0, 'timeout': 0, 'unknown': 0,
                'win_rate': 0, 'avg_profit': 0, 'avg_loss': 0,
                'total_profit_pips': 0, 'total_loss_pips': 0
            }

        # Считаем результаты
        result_query = '''
            SELECT 
                COUNT(*) FILTER (WHERE result = 'profit') as profit,
                COUNT(*) FILTER (WHERE result = 'loss') as loss,
                COUNT(*) FILTER (WHERE result = 'timeout') as timeout,
                COUNT(*) FILTER (WHERE result IS NULL) as unknown
            FROM signals
        '''
        if symbol:
            result_query += ' WHERE symbol = ?'
            params_res = [symbol]
        else:
            params_res = []

        async with db.execute(result_query, params_res) as cursor:
            row = await cursor.fetchone()
            profit, loss, timeout, unknown = row

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
            pips_query += ' AND symbol = ?'
            params_pips = [symbol]
        else:
            params_pips = []

        async with db.execute(pips_query, params_pips) as cursor:
            row = await cursor.fetchone()
            total_profit_pips, total_loss_pips = row or (0, 0)

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


# ========== КЭШИРОВАНИЕ СВЕЧЕЙ ==========
async def get_cached_candles(symbol: str, needed_bars: int = 200) -> Optional[List[Dict]]:
    """Возвращает кэшированные свечи, если их достаточно, иначе None."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute('''
            SELECT datetime, open, high, low, close, volume
            FROM candles
            WHERE symbol = ?
            ORDER BY datetime DESC
            LIMIT ?
        ''', (symbol, needed_bars)) as cursor:
            rows = await cursor.fetchall()
            if len(rows) < needed_bars:
                return None
            candles = []
            for row in reversed(rows):  # возвращаем в хронологическом порядке
                dt, o, h, l, c, v = row
                candles.append({
                    'datetime': dt,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v
                })
            return candles


async def save_candles_to_cache(symbol: str, candles: List[Dict]):
    async with aiosqlite.connect(DB_PATH) as db:
        for c in candles:
            await db.execute('''
                INSERT OR REPLACE INTO candles (symbol, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                c['datetime'],
                c['open'],
                c['high'],
                c['low'],
                c['close'],
                c.get('volume', 0)
            ))
        await db.commit()