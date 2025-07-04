# config.py

from dotenv import load_dotenv
import os

# Загружаем переменные окружения из .env файла ОДИН РАЗ при импорте этого модуля
load_dotenv()

# Читаем переменные и экспортируем их для использования в других частях приложения
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
SECRET_KEY = "MYSECRETFORSODA"

# Проверка, что переменная для БД установлена
if DATABASE_URL is None:
    raise ValueError("Необходимо установить переменную окружения DATABASE_URL в файле .env")
