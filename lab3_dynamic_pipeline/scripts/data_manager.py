import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
import json
import os
from typing import Tuple, Dict, Any

class DynamicDataManager:
    def __init__(self):
        self.supported_datasets = self._load_dataset_configs()
    
    def _load_dataset_configs(self) -> Dict:
        """Загрузка конфигурации датасетов"""
        config_path = "/opt/airflow/config/lab3/datasets_config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def download_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Динамическая загрузка датасета по имени"""
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        config = self.supported_datasets[dataset_name]
        
        try:
            api = KaggleApi()
            api.authenticate()
            
            download_path = f"/tmp/{dataset_name}"
            api.dataset_download_files(
                config['kaggle_dataset'], 
                path=download_path, 
                unzip=True
            )
            
            # Поиск CSV файла в скачанной директории
            for file in os.listdir(download_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(f"{download_path}/{file}")
                    logging.info(f"Dataset {dataset_name} loaded. Shape: {df.shape}")
                    return df
            
            raise FileNotFoundError(f"No CSV file found for {dataset_name}")
            
        except Exception as e:
            logging.error(f"Error downloading {dataset_name}: {e}")
            raise
    
    def auto_preprocess_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Автоматическая предобработка данных"""
        config = self.supported_datasets[dataset_name]
        df_processed = df.copy()
        target_column = config['target_column']
        
        # Удаление столбцов с уникальными значениями (например, ID)
        for col in df_processed.columns:
            if df_processed[col].nunique() == len(df_processed):
                df_processed = df_processed.drop(col, axis=1)
        
        # Обработка пропущенных значений
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['object', 'category']:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                else:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Кодирование категориальных переменных (кроме целевой)
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != target_column:
                df_processed[col] = df_processed[col].astype('category').cat.codes
        
        # Обработка целевой переменной для классификации
        if config['problem_type'] == 'classification':
            if df_processed[target_column].dtype == 'object':
                df_processed[target_column] = df_processed[target_column].astype('category').cat.codes
        
        return df_processed
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Валидация датасета"""
        config = self.supported_datasets[dataset_name]
        
        if config['target_column'] not in df.columns:
            logging.error(f"Target column {config['target_column']} not found")
            return False
        
        if len(df) < 10:
            logging.error("Dataset too small")
            return False
        
        return True