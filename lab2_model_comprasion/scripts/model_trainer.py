import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import logging
import json
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }
    
    def prepare_features(self, train_df, test_df):
        """Подготовка фич для обучения"""
        # Выбираем фичи для модели, проверяя их наличие
        all_feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
        
        # Фильтруем только доступные колонки
        available_train_cols = [col for col in all_feature_columns if col in train_df.columns]
        available_test_cols = [col for col in all_feature_columns if col in test_df.columns]
        
        # Используем пересечение доступных колонок
        feature_columns = list(set(available_train_cols) & set(available_test_cols))
        
        if not feature_columns:
            raise ValueError("Нет доступных фич для обучения модели")
        
        logging.info(f"Используемые фичи: {feature_columns}")
        
        X_train = train_df[feature_columns]
        y_train = train_df['Survived']
        X_test = test_df[feature_columns]
        
        # Для тестовых данных (где нет Survived) возвращаем только фичи
        if 'Survived' in test_df.columns:
            y_test = test_df['Survived']
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, None
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Оценка модели и создание визуализаций"""
        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        
        # Создание визуализаций
        self._create_confusion_matrix_plot(cm, model_name)
        self._create_feature_importance_plot(model, X_test.columns, model_name)
        
        return metrics, cm
    
    def _create_confusion_matrix_plot(self, cm, model_name):
        """Создание визуализации матрицы ошибок"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Сохраняем в временный файл
        plt.savefig(f'/tmp/confusion_matrix_{model_name}.png')
        plt.close()
    
    def _create_feature_importance_plot(self, model, feature_names, model_name):
        """Создание визуализации важности фич (для моделей, которые поддерживают)"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f'Feature Importance - {model_name}')
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
                plt.tight_layout()
                
                # Сохраняем в временный файл
                plt.savefig(f'/tmp/feature_importance_{model_name}.png')
                plt.close()
        except Exception as e:
            logging.warning(f"Could not create feature importance plot for {model_name}: {e}")
    
    def train_models(self, train_df, val_df, experiment_name="titanic_model_comparison"):
        """Тренировка всех моделей и логирование в MLflow"""
        
        try:
            # Подготовка данных
            X_train, X_val, y_train, y_val = self.prepare_features(train_df, val_df)
            
            logging.info(f"Подготовлены данные для обучения: X_train {X_train.shape}, X_val {X_val.shape}")
            
            results = {}
            
            # Проверяем доступность MLflow
            mlflow_available = self._check_mlflow_availability()
            
            for model_name, model in self.models.items():
                try:
                    if mlflow_available:
                        # Используем nested run только если мы внутри активного run
                        active_run = mlflow.active_run()
                        if active_run:
                            with mlflow.start_run(run_name=model_name, nested=True):
                                results[model_name] = self._train_single_model(model, model_name, X_train, X_val, y_train, y_val, True)
                        else:
                            with mlflow.start_run(run_name=model_name):
                                results[model_name] = self._train_single_model(model, model_name, X_train, X_val, y_train, y_val, True)
                    else:
                        results[model_name] = self._train_single_model(model, model_name, X_train, X_val, y_train, y_val, False)
                except Exception as e:
                    logging.error(f"Ошибка при обучении модели {model_name}: {e}")
                    continue  # Продолжаем с другими моделями
            
            if not results:
                raise ValueError("Не удалось обучить ни одной модели")
                
            return results
            
        except Exception as e:
            logging.error(f"Критическая ошибка в train_models: {e}")
            raise
    
    def find_best_model(self, results, metric='f1_score'):
        """Нахождение лучшей модели по выбранной метрике"""
        best_model_name = None
        best_metric = -1
        
        for model_name, result in results.items():
            if result['metrics'][metric] > best_metric:
                best_metric = result['metrics'][metric]
                best_model_name = model_name
        
        return best_model_name, results[best_model_name]
    
    def _check_mlflow_availability(self):
        """Проверка доступности MLflow"""
        try:
            # Проверяем, что tracking URI установлен
            tracking_uri = mlflow.get_tracking_uri()
            if not tracking_uri or tracking_uri == "file:///tmp/mlruns":
                return False
            
            # Проверяем, что можем получить список экспериментов
            mlflow_client = mlflow.tracking.MlflowClient()
            try:
                experiments = mlflow_client.search_experiments()
                logging.info(f"MLflow доступен, найдено экспериментов: {len(experiments)}")
            except AttributeError:
                # Fallback для старых версий MLflow
                experiments = mlflow_client.list_experiments()
                logging.info(f"MLflow доступен, найдено экспериментов: {len(experiments)}")
            return True
        except Exception as e:
            logging.warning(f"MLflow недоступен: {e}")
            return False
    
    def _train_single_model(self, model, model_name, X_train, X_val, y_train, y_val, use_mlflow=True):
        """Тренировка одной модели"""
        logging.info(f"Training {model_name}...")
        
        # Логируем параметры модели
        if use_mlflow and hasattr(model, 'get_params'):
            try:
                mlflow.log_params(model.get_params())
            except Exception as e:
                logging.warning(f"Could not log params for {model_name}: {e}")
        
        # Тренировка модели
        model.fit(X_train, y_train)
        
        # Оценка модели
        metrics, cm = self.evaluate_model(model, X_val, y_val, model_name)
        
        if use_mlflow:
            # Логируем метрики
            for metric_name, metric_value in metrics.items():
                try:
                    mlflow.log_metric(metric_name, metric_value)
                except Exception as e:
                    logging.warning(f"Could not log metric {metric_name}: {e}")
            
            # Логируем модель
            try:
                mlflow.sklearn.log_model(model, model_name)
            except Exception as e:
                logging.warning(f"Could not log model {model_name}: {e}")
            
            # Логируем визуализации
            try:
                mlflow.log_artifact(f'/tmp/confusion_matrix_{model_name}.png', "plots")
            except Exception as e:
                logging.warning(f"Could not log confusion matrix for {model_name}: {e}")
            
            try:
                mlflow.log_artifact(f'/tmp/feature_importance_{model_name}.png', "plots")
            except Exception as e:
                logging.warning(f"Could not log feature importance for {model_name}: {e}")
            
            # Логируем дополнительные параметры
            try:
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("features_used", list(X_train.columns))
                mlflow.log_metric("training_samples", len(X_train))
                mlflow.log_metric("validation_samples", len(X_val))
            except Exception as e:
                logging.warning(f"Could not log additional params for {model_name}: {e}")
        
        result = {
            'model': model,
            'metrics': metrics
        }
        
        if use_mlflow:
            try:
                result['run_id'] = mlflow.active_run().info.run_id
            except:
                result['run_id'] = 'no_mlflow'
        else:
            result['run_id'] = 'no_mlflow'
        
        logging.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        return result

        