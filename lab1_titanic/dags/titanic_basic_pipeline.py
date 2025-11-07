from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import mlflow
import io
import logging
import sys
import os

# Добавляем пути для импорта скриптов lab1
sys.path.append('/opt/airflow/scripts/lab1')

try:
    from data_loader import download_titanic_data, save_data_locally, load_data_from_local
    from data_preprocessor import preprocess_titanic_data
except ImportError as e:
    logging.error(f"Ошибка импорта скриптов: {e}")
    # Fallback импорт
    import importlib.util
    
    # Импорт data_loader
    spec = importlib.util.spec_from_file_location("data_loader", "/opt/airflow/scripts/lab1/data_loader.py")
    data_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader)
    download_titanic_data = data_loader.download_titanic_data
    save_data_locally = data_loader.save_data_locally
    load_data_from_local = data_loader.load_data_from_local
    
    # Импорт data_preprocessor
    spec = importlib.util.spec_from_file_location("data_preprocessor", "/opt/airflow/scripts/lab1/data_preprocessor.py")
    data_preprocessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_preprocessor)
    preprocess_titanic_data = data_preprocessor.preprocess_titanic_data

# Настройка MLflow будет выполнена внутри функций задач

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def download_and_log_data():
    """Задача 1: Загрузка данных и логирование в MLflow"""
    # Настройка MLflow внутри функции с обработкой ошибок
    mlflow_enabled = False
    try:
        # Пытаемся подключиться к MLflow серверу
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("titanic_basic_pipeline")
        mlflow_enabled = True
        logging.info("MLflow подключен успешно")
    except Exception as e:
        logging.warning(f"Не удалось подключиться к MLflow серверу: {e}")
        try:
            # Fallback на локальное хранилище с доступными правами
            mlflow.set_tracking_uri("file:///tmp/mlruns")
            mlflow.set_experiment("titanic_basic_pipeline")
            mlflow_enabled = True
            logging.info("MLflow настроен с локальным хранилищем")
        except Exception as e2:
            logging.warning(f"Не удалось настроить локальное хранилище MLflow: {e2}. Продолжаем без логирования.")
    
    # Загружаем данные
    train_df, test_df = download_titanic_data()
    
    # Логируем в MLflow только если подключение успешно
    if mlflow_enabled:
        try:
            with mlflow.start_run(run_name="data_download"):
                # Логируем параметры и метрики
                mlflow.log_param("dataset", "titanic")
                mlflow.log_param("source", "github")
                mlflow.log_metric("train_samples", len(train_df))
                mlflow.log_metric("test_samples", len(test_df))
                mlflow.log_metric("total_features", train_df.shape[1])
                
                # Логируем информацию о данных (только в Airflow логи)
                info_buffer = io.StringIO()
                train_df.info(buf=info_buffer)
                logging.info(f"Информация о данных:\n{info_buffer.getvalue()}")
                
                logging.info("Данные успешно залогированы в MLflow")
        except Exception as e:
            logging.error(f"Ошибка при логировании в MLflow: {e}")
            logging.info("Продолжаем без логирования в MLflow")
    else:
        # Логируем информацию в Airflow логи
        logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        logging.info(f"Train columns: {list(train_df.columns)}")
    
    # Сохраняем данные для следующей задачи
    save_data_locally(train_df, test_df)
    logging.info("Данные успешно загружены и сохранены")

def preprocess_and_log_data():
    """Задача 2: Предобработка данных и логирование"""
    # Настройка MLflow внутри функции с обработкой ошибок
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("titanic_basic_pipeline")
        mlflow_enabled = True
        logging.info("MLflow подключен успешно")
    except Exception as e:
        logging.warning(f"Не удалось подключиться к MLflow серверу: {e}")
        try:
            mlflow.set_tracking_uri("file:///tmp/mlruns")
            mlflow.set_experiment("titanic_basic_pipeline")
            mlflow_enabled = True
            logging.info("MLflow настроен с локальным хранилищем")
        except Exception as e2:
            logging.warning(f"Не удалось настроить локальное хранилище MLflow: {e2}. Продолжаем без логирования.")
    
    # Загружаем данные из предыдущей задачи
    train_df, test_df = load_data_from_local()
    
    # Предобработка
    processed_train, processed_test, encoders = preprocess_titanic_data(train_df, test_df)
    
    # Логируем в MLflow только если подключение успешно
    if mlflow_enabled:
        try:
            with mlflow.start_run(run_name="data_preprocessing"):
                # Логируем параметры обработки
                mlflow.log_param("missing_value_strategy", "median/mode")
                mlflow.log_param("categorical_encoding", "label_encoding")
                
                # Логируем метрики после обработки
                mlflow.log_metric("processed_train_samples", len(processed_train))
                mlflow.log_metric("processed_test_samples", len(processed_test))
                mlflow.log_metric("processed_features", processed_train.shape[1])
                
                # Логируем sample обработанных данных в Airflow логи
                sample_data = processed_train.head(10).to_string()
                logging.info(f"Sample обработанных данных:\n{sample_data}")
                
                logging.info("Данные успешно залогированы в MLflow")
        except Exception as e:
            logging.error(f"Ошибка при логировании в MLflow: {e}")
            logging.info("Продолжаем без логирования в MLflow")
    else:
        # Логируем информацию в Airflow логи
        logging.info(f"После обработки - Train: {processed_train.shape}, Test: {processed_test.shape}")
        logging.info(f"Использованные энкодеры: {list(encoders.keys())}")
    
    # Сохраняем обработанные данные
    save_data_locally(processed_train, processed_test, "/tmp/titanic_processed_final")
    logging.info("Данные успешно обработаны и сохранены")

def log_dataset_summary():
    """Задача 3: Финальное логирование сводки"""
    # Настройка MLflow внутри функции с обработкой ошибок
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("titanic_basic_pipeline")
        mlflow_enabled = True
        logging.info("MLflow подключен успешно")
    except Exception as e:
        logging.warning(f"Не удалось подключиться к MLflow серверу: {e}")
        try:
            mlflow.set_tracking_uri("file:///tmp/mlruns")
            mlflow.set_experiment("titanic_basic_pipeline")
            mlflow_enabled = True
            logging.info("MLflow настроен с локальным хранилищем")
        except Exception as e2:
            logging.warning(f"Не удалось настроить локальное хранилище MLflow: {e2}. Продолжаем без логирования.")
    
    train_df, test_df = load_data_from_local("/tmp/titanic_processed_final")
    
    # Создаем сводку
    summary = {
        "total_samples": len(train_df) + len(test_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features_count": train_df.shape[1],
        "memory_usage_mb": train_df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Логируем в MLflow только если подключение успешно
    if mlflow_enabled:
        try:
            with mlflow.start_run(run_name="dataset_summary"):
                # Логируем сводку
                for key, value in summary.items():
                    mlflow.log_metric(key, value)
                
                # Логируем список колонок
                mlflow.log_param("features_list", str(list(train_df.columns)))
                
                logging.info("Сводка успешно залогирована в MLflow")
        except Exception as e:
            logging.error(f"Ошибка при логировании в MLflow: {e}")
            logging.info("Продолжаем без логирования в MLflow")
    
    # Всегда логируем сводку в Airflow логи
    logging.info(f"Финальная сводка: {summary}")
    logging.info(f"Список признаков: {list(train_df.columns)}")
    logging.info("Pipeline успешно завершен!")

# Определяем DAG
with DAG(
    'titanic_basic_pipeline',
    default_args=default_args,
    description='Базовый пайплайн для датасета Titanic',
    schedule_interval=None,  # Запуск только вручную
    catchup=False,
    tags=['titanic', 'mlflow', 'kaggle'],
) as dag:

    download_task = PythonOperator(
        task_id='download_titanic_data',
        python_callable=download_and_log_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_titanic_data',
        python_callable=preprocess_and_log_data,
    )

    summary_task = PythonOperator(
        task_id='log_dataset_summary',
        python_callable=log_dataset_summary,
    )

    download_task >> preprocess_task >> summary_task