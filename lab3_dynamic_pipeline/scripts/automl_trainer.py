import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import logging
import mlflow
import mlflow.sklearn
import optuna
import json
from typing import Dict, Any

class DynamicAutoMLTrainer:
    def __init__(self):
        self.optuna_config = {
            'n_trials': 10,  # Количество попыток оптимизации
            'timeout': 180,  # 5 минут на подбор
        }
    
    def prepare_features(self, df: pd.DataFrame, target_column: str):
        """Подготовка фич для обучения"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Дополнительная обработка для FEDOT
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            X[col] = X[col].astype(str)
        
        return X, y
    
    def train_with_optuna(self, X: pd.DataFrame, y: pd.Series, problem: str):
        """Тренировка модели с использованием Optuna для гиперпараметров"""
        try:
            def objective(trial):
                if problem in ['classification', 'binary_classification', 'multiclass_classification']:
                    n_estimators = trial.suggest_int('n_estimators', 50, 150)
                    max_depth = trial.suggest_int('max_depth', 3, 15)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                    
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                    score = cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()
                else:
                    n_estimators = trial.suggest_int('n_estimators', 50, 150)
                    max_depth = trial.suggest_int('max_depth', 3, 15)
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                    
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                    score = cross_val_score(model, X, y, cv=3, scoring='r2').mean()
                
                return score
            
            # Оптимизация гиперпараметров
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.optuna_config['n_trials'], timeout=self.optuna_config['timeout'])
            
            # Обучение лучшей модели
            best_params = study.best_params
            if problem in ['classification', 'binary_classification', 'multiclass_classification']:
                model = RandomForestClassifier(**best_params, random_state=42)
            else:
                model = RandomForestRegressor(**best_params, random_state=42)
            
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Метрики
            if problem in ['classification', 'binary_classification', 'multiclass_classification']:
                metrics = {'accuracy': accuracy_score(y, predictions), 'f1': study.best_value}
            else:
                metrics = {'r2': r2_score(y, predictions), 'rmse': mean_squared_error(y, predictions, squared=False)}
            
            return model, metrics, predictions
            
        except Exception as e:
            logging.error(f"Optuna training failed: {e}")
            raise
    
    def train_with_baseline(self, X: pd.DataFrame, y: pd.Series, problem: str):
        """Тренировка базовой модели sklearn для сравнения"""
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.metrics import accuracy_score, r2_score
            
            if problem in ['classification', 'binary_classification', 'multiclass_classification']:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Метрики
            if problem in ['classification', 'binary_classification', 'multiclass_classification']:
                score = accuracy_score(y, predictions)
                metrics = {'accuracy': score}
            else:
                score = r2_score(y, predictions)
                metrics = {'r2': score}
            
            return model, metrics, predictions
            
        except Exception as e:
            logging.error(f"Baseline training failed: {e}")
            raise
    
    def compare_automl_frameworks(self, X: pd.DataFrame, y: pd.Series, problem: str, dataset_name: str):
        """Сравнение различных AutoML фреймворков"""
        results = {}
        
        with mlflow.start_run(run_name=f"automl_comparison_{dataset_name}", nested=True):
            # Логируем параметры
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("problem_type", problem)
            mlflow.log_param("samples", len(X))
            mlflow.log_param("features", X.shape[1])
            
            # Тестируем Optuna AutoML
            try:
                optuna_model, optuna_metrics, optuna_predictions = self.train_with_optuna(X, y, problem)
                results['optuna'] = {
                    'metrics': optuna_metrics,
                    'model': optuna_model,
                    'status': 'success'
                }
                
                # Логируем метрики Optuna
                for metric_name, metric_value in optuna_metrics.items():
                    mlflow.log_metric(f"optuna_{metric_name}", metric_value)
                    
            except Exception as e:
                logging.warning(f"Optuna failed: {e}")
                results['optuna'] = {'status': 'failed', 'error': str(e)}
            
            # Тестируем базовую модель sklearn
            try:
                baseline_model, baseline_metrics, _ = self.train_with_baseline(X, y, problem)
                results['baseline'] = {
                    'metrics': baseline_metrics,
                    'model': baseline_model,
                    'status': 'success'
                }
                
                # Логируем метрики базовой модели
                for metric_name, metric_value in baseline_metrics.items():
                    mlflow.log_metric(f"baseline_{metric_name}", metric_value)
                    
            except Exception as e:
                logging.warning(f"Baseline training failed: {e}")
                results['baseline'] = {'status': 'failed', 'error': str(e)}
            
            # Определяем лучший фреймворк
            best_framework = self._select_best_framework(results, problem)
            mlflow.log_param("best_automl_framework", best_framework)
            
            return results, best_framework
    
    def _select_best_framework(self, results: Dict, problem: str) -> str:
        """Выбор лучшего AutoML фреймворка на основе метрик"""
        best_framework = None
        best_score = -float('inf')
        
        # Определяем основную метрику в зависимости от типа задачи
        if problem in ['classification', 'binary_classification', 'multiclass_classification']:
            primary_metric = 'accuracy'  # используем accuracy вместо f1 для универсальности
        else:
            primary_metric = 'r2'
        
        for framework, result in results.items():
            if result['status'] == 'success' and primary_metric in result['metrics']:
                score = result['metrics'][primary_metric]
                if score > best_score:
                    best_score = score
                    best_framework = framework
        
        return best_framework or 'none'