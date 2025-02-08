import pandas as pd

def preprocess_df(df: pd.DataFrame, threshold_percent: float = 99, del_first_row: bool = True) -> pd.DataFrame:
    """
    Очищает DataFrame, удаляя столбцы с пропусками выше заданного порога и (опционально) первую строку.

    Функция выполняет следующие действия:
    1. Удаляет первую строку, если параметр `del_first_row` равен `True`.
    2. Удаляет столбцы, в которых процент пропущенных значений (`NaN`) превышает заданный порог.
    3. Заполняет оставшиеся пропуски пустыми строками.

    Args:
        df (pd.DataFrame): Входной DataFrame для обработки.
        threshold_percent (float, optional): Порог для удаления столбцов (в процентах). 
            Столбцы с процентом пропусков выше этого значения будут удалены. 
            Должен быть в диапазоне от 0 до 100. По умолчанию 99.
        del_first_row (bool, optional): Флаг, указывающий, нужно ли удалять первую строку. 
            По умолчанию `True`.

    Returns:
        pd.DataFrame: Очищенный DataFrame.
    """
    if not (0 <= threshold_percent <= 100):
        raise ValueError("Порог должен быть в диапазоне от 0 до 100.")

    df = df.iloc[1:].reset_index(drop=True) if del_first_row else df.copy()

    threshold = threshold_percent / 100
    columns_to_keep = df.columns[df.isnull().mean() <= threshold]

    return df[columns_to_keep].fillna("")