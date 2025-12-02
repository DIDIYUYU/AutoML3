import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
import logging

class TaskDetector:
    def __init__(self):
        self.supported_tasks = {
            'binary_classification': ['binary'],
            'multiclass_classification': ['multiclass'],
            'regression': ['continuous']
        }
    
    def detect_task_type(self, target_series: pd.Series) -> str:
        """Автоматическое определение типа задачи на основе целевой переменной"""
        try:
            target_type = type_of_target(target_series)
            logging.info(f"Detected target type: {target_type}")
            
            if target_type in self.supported_tasks['binary_classification']:
                return 'binary_classification'
            elif target_type in self.supported_tasks['multiclass_classification']:
                return 'multiclass_classification'
            elif target_type in self.supported_tasks['regression']:
                return 'regression'
            else:
                logging.warning(f"Unknown target type: {target_type}. Defaulting to regression")
                return 'regression'
                
        except Exception as e:
            logging.error(f"Error detecting task type: {e}")
            return 'regression'
    
    def get_appropriate_metrics(self, task_type: str) -> list:
        """Получение подходящих метрик для типа задачи"""
        metrics_map = {
            'binary_classification': ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
            'multiclass_classification': ['accuracy', 'f1_macro', 'precision_macro'],
            'regression': ['rmse', 'mae', 'r2', 'mse']
        }
        return metrics_map.get(task_type, ['accuracy'])
    
    def validate_task_configuration(self, detected_task: str, configured_task: str) -> bool:
        """Валидация соответствия обнаруженного и сконфигурированного типа задачи"""
        task_compatibility = {
            'classification': ['binary_classification', 'multiclass_classification'],
            'regression': ['regression']
        }
        
        if configured_task in task_compatibility:
            return detected_task in task_compatibility[configured_task]
        
        return detected_task == configured_task