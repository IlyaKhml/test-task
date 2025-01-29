import pandas as pd
from model_loader import load_model
from preprocessing import preprocess_df

def main(data_path: str, model_path: str = "models/model_mtp_group.pkl") -> list[str]:
    """
    Загружает данные, выполняет предобработку, загружает модель и делает предсказание.

    :param data_path: Путь к файлу данных (Excel, csv)
    :param model_path: Путь к файлу модели (по умолчанию: models/model_mtp_group.pkl)
    :return: Список с предсказаниями модели
    """
    try:
        if data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)

        elif data_path.endswith('.csv'):
            data = pd.read_csv(data_path)

    except Exception as e:
        raise ValueError(f"Ошибка при загрузке данных: {e}")
    
    data = preprocess_df(data)
    model = load_model(model_path)
    
    try:
        result = model.predict(data)
    except Exception as e:
        raise RuntimeError(f"Ошибка при предсказании: {e}")
    
    print(result)
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Запуск предсказания модели")
    parser.add_argument("data_path", type=str, help="Путь к файлу данных (Excel)")
    parser.add_argument("--model_path", type=str, default="models/model_mtp_group.pkl", help="Путь к файлу модели")
    
    args = parser.parse_args()
    main(args.data_path, args.model_path)