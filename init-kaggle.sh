#!/bin/bash

# Создаем .kaggle директорию если она не существует
if [ ! -d ".kaggle" ]; then
    echo "Создаем .kaggle директорию..."
    mkdir -p .kaggle
    
    # Создаем пустой kaggle.json если он не существует
    if [ ! -f ".kaggle/kaggle.json" ]; then
        echo "Создаем пустой kaggle.json..."
        echo '{"username":"","key":""}' > .kaggle/kaggle.json
        chmod 600 .kaggle/kaggle.json
    fi
fi

echo ".kaggle директория готова"