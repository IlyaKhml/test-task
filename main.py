from model_loader import load_model, load_config, update_pipeline_with_config
from preprocessing import preprocess_df

import os
import warnings
import pandas as pd
from typing import Optional

def main(data_path: str, 
         model_path: str = "models/model_mtp_group.pkl", 
         config_path: Optional[str] = None) -> list[str]:
    """
    Загружает данные, выполняет предобработку, загружает модель и делает предсказание.

    :param data_path: Путь к файлу данных (Excel, CSV)
    :param model_path: Путь к файлу модели (по умолчанию: models/model_mtp_group.pkl)
    :param config_path: Необязательный путь к файлу конфигурации модели. Если не указан, конфигурация не загружается.
    :return: Список с предсказаниями модели
    """
    try:
        if data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        else:
            raise ValueError("Неподдерживаемый формат файла. Ожидается .xlsx или .csv")
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке данных: {e}")
    
    # Предобработка данных
    data = preprocess_df(data)
    
    # Загрузка модели
    model = load_model(model_path)
    
    # Загрузка и применение конфигурации, если она указана
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден по пути: {config_path}")
        
        config = load_config(config_path)
        model = update_pipeline_with_config(model, config)
    
    try:
        result = model.predict(data)
    except Exception as e:
        raise RuntimeError(f"Ошибка при предсказании: {e}")
    
    print(result)
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск предсказания модели")
    parser.add_argument("data_path", type=str, help="Путь к файлу данных (Excel, CSV)")
    parser.add_argument("--model_path", type=str, default="models/model_mtp_group.pkl", help="Путь к файлу модели")
    parser.add_argument("--config_path", type=str, help="Путь к файлу конфигурации модели (необязательно)")
    
    args = parser.parse_args()
    main(args.data_path, args.model_path, args.config_path)