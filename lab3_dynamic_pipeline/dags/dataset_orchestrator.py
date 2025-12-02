from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models import Variable, DagModel
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging
import mlflow

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@task
def get_all_datasets():
    """Получение списка всех датасетов для обработки"""
    datasets = ["titanic", "house_prices", "iris", "wine_quality"]
    return datasets

@task
def ensure_child_dag_unpaused():
    """Убедиться, что дочерний DAG не на паузе"""
    from airflow.settings import Session
    session = Session()
    try:
        dag_model = session.query(DagModel).filter(DagModel.dag_id == 'dynamic_ml_pipeline').first()
        if dag_model and dag_model.is_paused:
            dag_model.is_paused = False
            session.commit()
            logging.info("Unpaused dynamic_ml_pipeline DAG")
        else:
            logging.info("dynamic_ml_pipeline DAG is already unpaused or doesn't exist yet")
    finally:
        session.close()

def process_single_dataset(dataset_name: str):
    """Задача для обработки одного датасета"""
    # Устанавливаем переменную для основного DAG
    Variable.set("current_dataset", dataset_name)
    logging.info(f"Set current_dataset to: {dataset_name}")
    return dataset_name

@task
def create_summary_report(processed_datasets: list):
    """Создание сводного отчета по всем датасетам"""
    logging.info(f"Created summary report for {len(processed_datasets)} datasets")
    
    with mlflow.start_run(run_name="multi_dataset_summary"):
        mlflow.log_param("total_datasets_processed", len(processed_datasets))
        mlflow.log_param("processed_dataset_names", str(processed_datasets))
        
    return "summary_completed"

with DAG(
    'dataset_orchestrator',
    default_args=default_args,
    description='Оркестратор обработки множества датасетов',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=['orchestrator', 'multi-dataset'],
) as dag:
    
    datasets = get_all_datasets()
    unpause_task = ensure_child_dag_unpaused()
    
    # Убедимся, что дочерний DAG не на паузе перед запуском
    datasets >> unpause_task
    
    # Последовательное выполнение для избежания deadlock
    trigger_tasks = []
    prev_task = unpause_task
    
    for dataset in ["titanic", "house_prices", "iris", "wine_quality"]:
        # Задача установки переменной для датасета
        set_var_task = PythonOperator(
            task_id=f'set_dataset_{dataset}',
            python_callable=process_single_dataset,
            op_kwargs={'dataset_name': dataset}
        )
        
        # Задача запуска дочернего DAG
        trigger_task = TriggerDagRunOperator(
            task_id=f'trigger_pipeline_{dataset}',
            trigger_dag_id='dynamic_ml_pipeline',
            wait_for_completion=False,  # Не ждем завершения, чтобы не блокировать executor
            conf={'dataset_name': dataset},
            reset_dag_run=True,
        )
        
        # Последовательная цепочка: prev -> set_var -> trigger
        prev_task >> set_var_task >> trigger_task
        prev_task = trigger_task
        
        trigger_tasks.append(trigger_task)
    
    # Summary запускается после завершения последнего триггера
    summary = create_summary_report(trigger_tasks)
    prev_task >> summary
