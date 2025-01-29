import dill

def load_model(model_path: str):
    """
    Загружает модель по пути model_path.

    :model_path: Название модели
    :return: Загруженная модель
    """
    # Загрузка
    with open(model_path, "rb") as f:
        model = dill.load(f)

    return model