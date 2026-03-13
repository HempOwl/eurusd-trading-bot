import sys
print(f"Python: {sys.version}")

print("\n=== Проверка библиотек ===\n")

try:
    import requests
    print("✅ requests установлен")
except:
    print("❌ requests НЕ установлен")

try:
    import aiohttp
    print("✅ aiohttp установлен")
except:
    print("❌ aiohttp НЕ установлен")

try:
    import telegram
    print("✅ python-telegram-bot установлен")
except:
    print("❌ python-telegram-bot НЕ установлен")

try:
    import socks
    print("✅ PySocks установлен")
except:
    print("❌ PySocks НЕ установлен")

try:
    from aiohttp_socks import ProxyConnector
    print("✅ aiohttp-socks установлен")
except:
    print("❌ aiohttp-socks НЕ установлен")

print("\n=== Проверка завершена ===")