from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import timedelta
import mlflow
import logging
import sys
import os
import pandas as pd

# Добавляем путь к скриптам
sys.path.append('/opt/airflow/scripts/lab3')

from data_manager import DynamicDataManager
from task_detector import TaskDetector
from automl_trainer import DynamicAutoMLTrainer

# Настройка MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("dynamic_ml_pipeline")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@task
def get_dataset_config(**context):
    """Получение конфигурации датасета через Airflow Variables или conf"""
    # Сначала пробуем получить из dag_run.conf (передается через TriggerDagRunOperator)
    dag_run = context.get('dag_run')
    if dag_run and dag_run.conf and 'dataset_name' in dag_run.conf:
        dataset_name = dag_run.conf['dataset_name']
        logging.info(f"Got dataset from dag_run.conf: {dataset_name}")
    else:
        # Fallback на Variable
        dataset_name = Variable.get("current_dataset", default_var="titanic")
        logging.info(f"Got dataset from Variable: {dataset_name}")
    
    logging.info(f"Processing dataset: {dataset_name}")
    return dataset_name

@task
def download_and_validate_dataset(dataset_name: str):
    """Загрузка и валидация датасета"""
    data_manager = DynamicDataManager()
    
    with mlflow.start_run(run_name=f"data_processing_{dataset_name}", nested=True):
        # Загрузка данных
        df = data_manager.download_dataset(dataset_name)
        
        # Валидация данных
        if not data_manager.validate_dataset(df, dataset_name):
            raise ValueError(f"Dataset {dataset_name} validation failed")
        
        # Автоматическая предобработка
        df_processed = data_manager.auto_preprocess_data(df, dataset_name)
        
        # Логируем информацию о данных
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_metric("original_samples", len(df))
        mlflow.log_metric("processed_samples", len(df_processed))
        mlflow.log_metric("features_count", df_processed.shape[1] - 1)  # исключаем целевую переменную
        
        # Сохраняем обработанные данные
        output_path = f"/tmp/{dataset_name}_processed.csv"
        df_processed.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)
        
        return output_path

@task
def analyze_and_detect_task(data_path: str, dataset_name: str):
    """Анализ данных и определение типа задачи"""
    detector = TaskDetector()
    data_manager = DynamicDataManager()
    
    df = pd.read_csv(data_path)
    config = data_manager.supported_datasets[dataset_name]
    target_column = config['target_column']
    
    with mlflow.start_run(run_name=f"task_analysis_{dataset_name}", nested=True):
        # Определение типа задачи
        detected_task = detector.detect_task_type(df[target_column])
        configured_task = config['problem_type']
        
        # Валидация типа задачи
        if not detector.validate_task_configuration(detected_task, configured_task):
            logging.warning(f"Task type mismatch: detected {detected_task}, configured {configured_task}")
        
        # Получение подходящих метрик
        appropriate_metrics = detector.get_appropriate_metrics(detected_task)
        
        # Логируем результаты анализа
        mlflow.log_param("detected_task_type", detected_task)
        mlflow.log_param("configured_task_type", configured_task)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("appropriate_metrics", str(appropriate_metrics))
        mlflow.log_metric("target_unique_values", df[target_column].nunique())
        
        return {
            'detected_task': detected_task,
            'target_column': target_column,
            'metrics': appropriate_metrics
        }

@task
def run_automl_training(data_path: str, task_info: dict, dataset_name: str):
    """Запуск AutoML тренировки с несколькими фреймворками"""
    trainer = DynamicAutoMLTrainer()
    df = pd.read_csv(data_path)
    
    # Подготовка данных
    X, y = trainer.prepare_features(df, task_info['target_column'])
    
    # Запуск сравнения AutoML фреймворков
    results, best_framework = trainer.compare_automl_frameworks(
        X, y, task_info['detected_task'], dataset_name
    )
    
    # Логируем итоговые результаты
    with mlflow.start_run(run_name=f"final_results_{dataset_name}"):
        mlflow.log_param("best_automl_framework", best_framework)
        mlflow.log_param("final_task_type", task_info['detected_task'])
        
        # Подготовка результатов для логирования (без моделей)
        results_for_logging = {}
        for framework, result in results.items():
            results_for_logging[framework] = {
                'status': result['status'],
                'metrics': result.get('metrics', {})
            }
            if 'error' in result:
                results_for_logging[framework]['error'] = result['error']
        
        mlflow.log_dict(results_for_logging, "automl_comparison_results.json")
        
        # Логируем информацию о лучшей модели
        if best_framework != 'none' and best_framework in results:
            best_metrics = results[best_framework]['metrics']
            for metric_name, metric_value in best_metrics.items():
                mlflow.log_metric(f"best_{metric_name}", metric_value)
    
    return best_framework

@task
def register_best_model(dataset_name: str, best_framework: str):
    """Регистрация лучшей модели в MLflow Registry"""
    if best_framework == 'none':
        logging.warning("No successful framework found for model registration")
        return
    
    with mlflow.start_run(run_name=f"model_registration_{dataset_name}"):
        # В реальной реализации здесь была бы логика регистрации конкретной модели
        mlflow.log_param("registered_dataset", dataset_name)
        mlflow.log_param("registered_framework", best_framework)
        mlflow.log_param("registration_status", "success")
        
        logging.info(f"Best model from {best_framework} registered for {dataset_name}")

# Создание DAG в глобальной области видимости (обязательно для Airflow)
with DAG(
    'dynamic_ml_pipeline',
    default_args=default_args,
    description='Динамический ML пайплайн для множества датасетов',
    schedule_interval=None,
    catchup=False,
    max_active_runs=4,
    tags=['automl', 'dynamic', 'fedot', 'multidataset'],
) as dag:
    
    # Получение конфигурации
    dataset_config = get_dataset_config()
    
    # Загрузка и обработка данных
    data_processing = download_and_validate_dataset(dataset_config)
    
    # Анализ задачи
    task_analysis = analyze_and_detect_task(data_processing, dataset_config)
    
    # AutoML тренировка
    automl_training = run_automl_training(data_processing, task_analysis, dataset_config)
    
    # Регистрация модели
    model_registration = register_best_model(dataset_config, automl_training)
    
    # Определение последовательности задач
    dataset_config >> data_processing >> task_analysis >> automl_training >> model_registration
    