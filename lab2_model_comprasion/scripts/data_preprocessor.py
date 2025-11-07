import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

def validate_titanic_data(df, expected_columns=None):
    """Валидация данных Titanic"""
    if expected_columns is None:
        # Основные колонки, которые должны быть в данных Titanic
        expected_columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    # Проверяем, что DataFrame не пустой
    if df.empty:
        logging.error("DataFrame пустой")
        return False
    
    # Проверяем наличие критически важных колонок
    critical_cols = ['PassengerId', 'Pclass', 'Sex']
    missing_critical = [col for col in critical_cols if col not in df.columns]
    if missing_critical:
        logging.error(f"Отсутствуют критически важные колонки: {missing_critical}")
        return False
    
    # Предупреждаем о недостающих колонках, но не блокируем выполнение
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Отсутствуют колонки (будут обработаны): {missing_cols}")
        
    logging.info(f"Валидация прошла успешно. Размер данных: {df.shape}")
    logging.info(f"Доступные колонки: {list(df.columns)}")
    return True

def preprocess_titanic_data(train_df, test_df):
    """Базовая предобработка данных Titanic"""
    
    # Валидируем входные данные
    if not validate_titanic_data(train_df):
        raise ValueError("Ошибка валидации тренировочных данных")
    
    # Для тестовых данных не требуем колонку Survived
    test_expected_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    if not validate_titanic_data(test_df, test_expected_cols):
        raise ValueError("Ошибка валидации тестовых данных")
    
    # Создаем копии для безопасной обработки
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Объединяем для согласованной обработки
    combined = pd.concat([train_copy, test_copy], ignore_index=True, sort=False)
    
    # Логируем исходные размеры
    logging.info(f"Исходные данные - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Обработка пропущенных значений с проверкой
    try:
        if 'Age' in combined.columns:
            age_median = combined['Age'].median()
            if pd.isna(age_median):
                age_median = 30  # Значение по умолчанию
            combined['Age'].fillna(age_median, inplace=True)
            logging.info(f"Заполнено {combined['Age'].isna().sum()} пропущенных значений в Age медианой: {age_median}")
        
        if 'Embarked' in combined.columns:
            embarked_mode = combined['Embarked'].mode()
            if len(embarked_mode) > 0:
                combined['Embarked'].fillna(embarked_mode[0], inplace=True)
            else:
                combined['Embarked'].fillna('S', inplace=True)  # Значение по умолчанию
            logging.info(f"Заполнено {combined['Embarked'].isna().sum()} пропущенных значений в Embarked")
        
        if 'Fare' in combined.columns:
            fare_median = combined['Fare'].median()
            if pd.isna(fare_median):
                fare_median = 15.0  # Значение по умолчанию
            combined['Fare'].fillna(fare_median, inplace=True)
            logging.info(f"Заполнено {combined['Fare'].isna().sum()} пропущенных значений в Fare медианой: {fare_median}")
            
    except Exception as e:
        logging.error(f"Ошибка при обработке пропущенных значений: {e}")
        raise
    
    # Создание новых признаков (только если есть необходимые колонки)
    if 'SibSp' in combined.columns and 'Parch' in combined.columns:
        combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
        combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
        logging.info("Созданы признаки FamilySize и IsAlone")
    else:
        logging.warning("Не удалось создать признаки FamilySize и IsAlone - отсутствуют колонки SibSp или Parch")
        # Создаем фиктивные признаки, если исходные отсутствуют
        if 'SibSp' not in combined.columns:
            combined['SibSp'] = 0
        if 'Parch' not in combined.columns:
            combined['Parch'] = 0
        combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
        combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # Кодирование категориальных переменных
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked']
    
    for col in categorical_cols:
        if col in combined.columns:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
            label_encoders[col] = le
            logging.info(f"Закодирована колонка {col}")
        else:
            logging.warning(f"Колонка {col} не найдена, пропускаем кодирование")
    
    # Разделяем обратно
    processed_train = combined.iloc[:len(train_df)].copy()
    processed_test = combined.iloc[len(train_df):].copy()
    
    # Удаляем целевой признак из тестовых данных
    if 'Survived' in processed_test.columns:
        processed_test = processed_test.drop('Survived', axis=1)
    
    logging.info(f"После обработки - Train: {processed_train.shape}, Test: {processed_test.shape}")
    
    return processed_train, processed_test, label_encoders
