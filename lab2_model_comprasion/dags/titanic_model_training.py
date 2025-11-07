from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import mlflow
from mlflow.tracking import MlflowClient
import logging
import sys
import os
import requests

# Добавляем пути к скриптам lab2
sys.path.append('/opt/airflow/scripts/lab2')
sys.path.append('/opt/airflow/dags/lab2')

# Импортируем из lab2 скриптов
try:
    from data_loader import download_titanic_data, save_data_locally, load_data_from_local
    from data_preprocessor import preprocess_titanic_data
    from model_trainer import ModelTrainer
    logging.info("Успешно импортированы модули из lab2")
except ImportError as e:
    logging.error(f"Ошибка импорта модулей из стандартного пути: {e}")
    # Fallback импорт
    try:
        script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lab2_model_comprasion', 'scripts')
        sys.path.append(script_path)
        from data_loader import download_titanic_data, save_data_locally, load_data_from_local
        from data_preprocessor import preprocess_titanic_data
        from model_trainer import ModelTrainer
        logging.info("Успешно импортированы модули через fallback путь")
    except ImportError as fallback_error:
        logging.error(f"Критическая ошибка импорта: {fallback_error}")
        raise

# Настройка MLflow - перенесем в функции для выполнения в runtime
def setup_mlflow():
    """Настройка MLflow во время выполнения задач"""
    try:
        # Пытаемся подключиться к MLflow
        mlflow_uri = "http://mlflow:5000"
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Проверяем подключение к MLflow с таймаутом
        response = requests.get(f"{mlflow_uri}/health", timeout=10)
        if response.status_code == 200:
            mlflow_client = MlflowClient()
            try:
                experiments = mlflow_client.search_experiments()
                logging.info(f"MLflow подключен успешно. Найдено экспериментов: {len(experiments)}")
            except AttributeError:
                # Fallback для старых версий MLflow
                experiments = mlflow_client.list_experiments()
                logging.info(f"MLflow подключен успешно. Найдено экспериментов: {len(experiments)}")
            
            # Создаем или получаем эксперимент
            experiment_name = "titanic_model_comparison"
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                logging.info(f"Создан новый эксперимент: {experiment_name}")
            except Exception:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    experiment_id = experiment.experiment_id
                    logging.info(f"Используем существующий эксперимент: {experiment_name}")
                else:
                    logging.warning("Не удалось найти или создать эксперимент")
            
            mlflow.set_experiment(experiment_name)
            return mlflow_client
        else:
            raise Exception(f"MLflow недоступен, статус: {response.status_code}")
            
    except Exception as e:
        logging.error(f"Ошибка настройки MLflow: {e}")
        logging.warning("Продолжаем выполнение без MLflow")
        return None

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'email_on_failure': False,
}

def download_data():
    """Задача 1: Загрузка данных"""
    try:
        mlflow_client = setup_mlflow()
        
        if mlflow_client is not None:
            with mlflow.start_run(run_name="data_download"):
                train_df, test_df = download_titanic_data()
                save_data_locally(train_df, test_df)
                mlflow.log_param("dataset", "titanic")
                mlflow.log_metric("train_samples", len(train_df))
                logging.info(f"Данные загружены с MLflow. Train: {train_df.shape}, Test: {test_df.shape}")
        else:
            train_df, test_df = download_titanic_data()
            save_data_locally(train_df, test_df)
            logging.info(f"Данные загружены без MLflow. Train: {train_df.shape}, Test: {test_df.shape}")
    except Exception as e:
        logging.error(f"Ошибка в загрузке данных: {e}")
        raise

def preprocess_data():
    """Задача 2: Предобработка данных"""
    try:
        mlflow_client = setup_mlflow()
        
        if mlflow_client is not None:
            with mlflow.start_run(run_name="data_preprocessing"):
                train_df, test_df = load_data_from_local()
                processed_train, processed_test, encoders = preprocess_titanic_data(train_df, test_df)
                save_data_locally(processed_train, processed_test, "/tmp/titanic_processed")
                mlflow.log_metric("processed_features", processed_train.shape[1])
                logging.info(f"Предобработка завершена с MLflow. Train shape: {processed_train.shape}, Test shape: {processed_test.shape}")
        else:
            train_df, test_df = load_data_from_local()
            processed_train, processed_test, encoders = preprocess_titanic_data(train_df, test_df)
            save_data_locally(processed_train, processed_test, "/tmp/titanic_processed")
            logging.info(f"Предобработка завершена без MLflow. Train shape: {processed_train.shape}, Test shape: {processed_test.shape}")
    except Exception as e:
        logging.error(f"Ошибка в предобработке данных: {e}")
        raise

def train_and_compare_models():
    """Задача 3: Тренировка и сравнение моделей"""
    mlflow_client = setup_mlflow()
    
    if mlflow_client is not None:
        with mlflow.start_run(run_name="model_training_comparison"):
            return _train_models_logic(mlflow_client)
    else:
        return _train_models_logic(None)

def _train_models_logic(mlflow_client):
    try:
        # Загружаем обработанные данные
        train_df, test_df = load_data_from_local("/tmp/titanic_processed")
        logging.info(f"Загружены данные: train {train_df.shape}, test {test_df.shape}")
        
        # Проверяем наличие целевой переменной
        if 'Survived' not in train_df.columns:
            raise ValueError("Отсутствует целевая переменная 'Survived' в тренировочных данных")
        
        # Для валидации используем часть тренировочных данных
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=train_df['Survived']
        )
        
        logging.info(f"Разделение данных: train {train_df.shape}, validation {val_df.shape}")
        
        # Тренируем модели - передаем val_df как test_df для валидации
        trainer = ModelTrainer()
        results = trainer.train_models(train_df, val_df, experiment_name="titanic_model_comparison")
        
        if not results:
            raise ValueError("Не удалось обучить ни одной модели")
        
        # Находим лучшую модель
        best_model_name, best_model_info = trainer.find_best_model(results)
        
        # Логируем информацию о лучшей модели только если MLflow доступен
        if mlflow_client is not None:
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_model_f1", best_model_info['metrics']['f1_score'])
            mlflow.log_metric("best_model_accuracy", best_model_info['metrics']['accuracy'])
            
            # Сохраняем информацию о всех моделях
            comparison_data = {
                model_name: {
                    'metrics': model_info['metrics'],
                    'run_id': model_info.get('run_id', 'no_mlflow')
                }
                for model_name, model_info in results.items()
            }
            
            mlflow.log_dict(comparison_data, "model_comparison.json")
        
        logging.info(f"Best model: {best_model_name} with F1: {best_model_info['metrics']['f1_score']:.4f}")
        
        return best_model_name
        
    except Exception as e:
        logging.error(f"Ошибка в тренировке моделей: {e}")
        raise

def register_best_model(**context):
    """Задача 4: Регистрация лучшей модели в MLflow Registry"""
    try:
        best_model_name = context['task_instance'].xcom_pull(task_ids='train_and_compare_models')
        
        if not best_model_name:
            raise ValueError("Не удалось получить имя лучшей модели из предыдущей задачи")
        
        mlflow_client = setup_mlflow()
        
        if mlflow_client is None:
            logging.warning("MLflow недоступен, пропускаем регистрацию модели")
            return
        
        with mlflow.start_run(run_name="model_registration"):
            # Находим run_id лучшей модели
            experiment = mlflow.get_experiment_by_name("titanic_model_comparison")
            if not experiment:
                raise ValueError("Эксперимент 'titanic_model_comparison' не найден")
                
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                raise ValueError("Не найдено ни одного run в эксперименте")
            
            # Ищем run с нужным именем
            best_runs = runs[runs['tags.mlflow.runName'] == best_model_name]
            if best_runs.empty:
                logging.warning(f"Run с именем {best_model_name} не найден, используем последний run")
                best_run = runs.iloc[0]
            else:
                best_run = best_runs.iloc[0]
                
            best_run_id = best_run.run_id
            
            # Регистрируем модель
            model_uri = f"runs:/{best_run_id}/{best_model_name}"
            model_name = "titanic-survival-predictor"
            
            if mlflow_client is None:
                logging.warning("MLflow клиент недоступен, пропускаем регистрацию модели")
                return
                
            try:
                # Пытаемся создать зарегистрированную модель, если она не существует
                mlflow_client.create_registered_model(model_name)
                logging.info(f"Создана новая зарегистрированная модель: {model_name}")
            except Exception as e:
                logging.info(f"Model {model_name} already exists: {e}")
            
            # Создаем новую версию модели
            model_version = mlflow_client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=best_run_id
            )
            
            # Переводим версию в Production
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            
            # Логируем информацию о регистрации
            mlflow.log_param("registered_model", model_name)
            mlflow.log_param("model_version", model_version.version)
            mlflow.log_param("best_model_type", best_model_name)
            mlflow.log_metric("model_version_number", int(model_version.version))
            
            logging.info(f"Model {model_name} version {model_version.version} registered successfully!")
            
    except Exception as e:
        logging.error(f"Ошибка при регистрации модели: {e}")
        raise

def create_comparison_report(**context):
    """Задача 5: Создание финального отчета сравнения моделей"""
    best_model_name = context['task_instance'].xcom_pull(task_ids='train_and_compare_models')
    mlflow_client = setup_mlflow()
    
    if mlflow_client is not None:
        with mlflow.start_run(run_name="comparison_report"):
            _create_report_logic(context, best_model_name, mlflow_client)
    else:
        _create_report_logic(context, best_model_name, None)

def _create_report_logic(context, best_model_name, mlflow_client):
    # Создаем базовый отчет
    report = {
        'best_model': best_model_name,
        'comparison_date': str(context['execution_date']),
        'model_metrics': {}
    }
    
    if mlflow_client is not None:
        try:
            # Получаем данные всех runs для создания отчета
            experiment = mlflow.get_experiment_by_name("titanic_model_comparison")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                
                # Фильтруем только runs моделей (исключаем служебные)
                model_runs = runs[runs['tags.mlflow.runName'].isin(['logistic_regression', 'random_forest', 'svm', 'xgboost'])]
                
                report['total_models_trained'] = len(model_runs)
                
                for _, run in model_runs.iterrows():
                    model_name = run['tags.mlflow.runName']
                    report['model_metrics'][model_name] = {
                        'accuracy': run.get('metrics.accuracy', 0),
                        'precision': run.get('metrics.precision', 0),
                        'recall': run.get('metrics.recall', 0),
                        'f1_score': run.get('metrics.f1_score', 0),
                        'run_id': run.run_id
                    }
                
                # Логируем отчет в MLflow
                mlflow.log_dict(report, "final_comparison_report.json")
        except Exception as e:
            logging.warning(f"Не удалось получить данные из MLflow для отчета: {e}")
            report['total_models_trained'] = 4  # Предполагаем 4 модели
    else:
        report['total_models_trained'] = 4  # Предполагаем 4 модели
        logging.info("Создаем отчет без MLflow")
    
    # Создаем простой текстовый отчет
    text_report = f"""
    TITANIC MODEL COMPARISON REPORT
    ===============================
    Execution Date: {context['execution_date']}
    Total Models Trained: {report.get('total_models_trained', 'Unknown')}
    
    BEST MODEL: {best_model_name}
    """
    
    if report['model_metrics']:
        text_report += "\n    MODEL PERFORMANCE:\n"
        for model_name, metrics in report['model_metrics'].items():
            text_report += f"""
        {model_name.upper()}:
          Accuracy:  {metrics['accuracy']:.4f}
          F1-Score:  {metrics['f1_score']:.4f}
          Precision: {metrics['precision']:.4f}
          Recall:    {metrics['recall']:.4f}
        """
    
    # Сохраняем отчет в файл
    report_path = "/tmp/model_comparison_summary.txt"
    with open(report_path, 'w') as f:
        f.write(text_report)
    
    if mlflow_client is not None:
        mlflow.log_text(text_report, "model_comparison_summary.txt")
    
    logging.info("Comparison report generated successfully!")
    logging.info(f"Report saved to: {report_path}")

# Определяем DAG
with DAG(
    'titanic_model_comparison',
    default_args=default_args,
    description='Сравнительный анализ ML моделей для датасета Titanic',
    schedule_interval=None,
    catchup=False,
    tags=['titanic', 'mlflow', 'model-comparison', 'airflow'],
) as dag:

    download_task = PythonOperator(
        task_id='download_titanic_data',
        python_callable=download_data,
    )

    preprocess_task = PythonOperator(
        task_id='preprocess_titanic_data',
        python_callable=preprocess_data,
    )

    train_models_task = PythonOperator(
        task_id='train_and_compare_models',
        python_callable=train_and_compare_models,
    )

    register_model_task = PythonOperator(
        task_id='register_best_model',
        python_callable=register_best_model,
        provide_context=True,
    )

    create_report_task = PythonOperator(
        task_id='create_comparison_report',
        python_callable=create_comparison_report,
        provide_context=True,
    )

    download_task >> preprocess_task >> train_models_task >> [register_model_task, create_report_task]