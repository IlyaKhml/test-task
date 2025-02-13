import pandas as pd

def preprocess_df(df: pd.DataFrame, threshold_percent: float = 99, del_first_row: bool = True) -> pd.DataFrame:
    """
    Очищает DataFrame, удаляя столбцы с пропусками выше заданного порога и (опционально) первую строку.
    Функция выполняет следующие действия:
    1. Удаляет первую строку, если параметр `del_first_row` равен `True` И все значения в первой строке являются целыми числами.
    2. Удаляет столбцы, в которых процент пропущенных значений (`NaN`) превышает заданный порог.
    3. Заполняет оставшиеся пропуски пустыми строками.
    Args:
        df (pd.DataFrame): Входной DataFrame для обработки.
        threshold_percent (float, optional): Порог для удаления столбцов (в процентах). 
            Столбцы с процентом пропусков выше этого значения будут удалены. 
            Должен быть в диапазоне от 0 до 100. По умолчанию 99.
        del_first_row (bool, optional): Флаг, указывающий, нужно ли удалять первую строку, 
            если все значения в ней являются целыми числами. По умолчанию `True`.
    Returns:
        pd.DataFrame: Очищенный DataFrame.
    """
    if not (0 <= threshold_percent <= 100):
        raise ValueError("Порог должен быть в диапазоне от 0 до 100.")
    
    # Проверяем, нужно ли удалять первую строку
    if del_first_row and not df.empty:
        first_row = df.iloc[0]
        # Проверяем, можно ли преобразовать все значения первой строки в int
        if all(isinstance(value, (int, float)) and float(value).is_integer() for value in first_row if pd.notna(value)):
            df = df.iloc[1:].reset_index(drop=True)
    
    # Вычисляем порог для удаления столбцов
    threshold = threshold_percent / 100
    columns_to_keep = df.columns[df.isnull().mean() <= threshold]
    
    # Удаляем столбцы с пропусками выше порога и заполняем оставшиеся NaN пустыми строками
    return df[columns_to_keep].fillna("")