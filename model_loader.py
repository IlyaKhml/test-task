import dill
import json


def load_model(model_path: str):
    """
    Загружает модель по пути model_path.

    :model_path: (str) Название модели
    :return: Загруженная модель
    """
    with open(model_path, "rb") as f:
        model = dill.load(f)

    return model


def load_config(config_path: str):
    """
    Загружает конфиг по пути config_path. 

    :param config_path: (str) Путь до конфига
    :return: (dict) Загруженный конфиг
    """
    with open(config_path, "r", encoding="utf-8") as f:
        loaded_config = json.load(f)

    return loaded_config


def update_pipeline_with_config(pipeline, config):
    """
    Обновляет параметры пайплайна в соответствии с файлом конфигурации.

    Аргументы:
        pipeline: Пайплайн машинного обучения, подлежащий обновлению.
        config: Словарь, содержащий новые настройки конфигурации. 
                Ожидаемые ключи - 'column_mapping', 'categorical_cols', 
                'text_cols'.

    Возвращает:
        Обновленный пайплайн с новым отображением столбцов и настройками.
    """
    # Получаем обновлённые значения
    column_mapping = config["column_mapping"]
    categorical_cols = config["categorical_cols"]
    text_cols = config["text_cols"]

    # Обновляем параметры уже существующего SafeColumnMapper в пайплайне
    safe_column_mapper = pipeline.named_steps["safe_column_mapper"]
    safe_column_mapper.required_cols = categorical_cols + text_cols
    safe_column_mapper.mapping = column_mapping

    # Обновляем `preprocessor` в загруженном пайплайне
    preprocessor = pipeline.named_steps['preprocessing']
    pipeline.named_steps["preprocessing"].transformers = [
        ("cat", preprocessor.transformers[0][1], categorical_cols),
        ("text", preprocessor.transformers[1][1], text_cols)
    ]

    return pipeline