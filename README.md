# Airflow + MLflow Pipeline для анализа Titanic

## Настройка

1. **Инициализируйте .kaggle директорию:**
   ```bash
   ./init-kaggle.sh
   ```

2. **Настройте Kaggle API credentials:**
   ```bash
   cp .env.example .env
   # Отредактируйте .env файл, добавив ваши Kaggle credentials
   ```

3. **Соберите и запустите сервисы:**
   ```bash
   docker-compose build
   docker-compose up -d
   ```

## Доступ к сервисам

- Airflow UI: http://localhost:8080 (admin/admin)
- MLflow UI: http://localhost:5001

