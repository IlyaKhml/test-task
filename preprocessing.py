import pandas as pd

def preprocess_df(df: pd.DataFrame, threshold_percent: float = 99, del_first_row: bool = True) -> pd.DataFrame:
    """
    Очищает DataFrame, удаляя столбцы с пропусками выше порога и (опционально) первую строку.

    :df: Входной DataFrame
    :threshold_percent: Порог удаления столбцов (в процентах пропусков)
    :del_first_row: Флаг удаления первой строки
    :return: Очищенный DataFrame
    """
    if not (0 <= threshold_percent <= 100):
        raise ValueError("Порог должен быть в диапазоне от 0 до 100.")
    
    df = df.iloc[1:].reset_index(drop=True) if del_first_row else df.copy()
    
    threshold = threshold_percent / 100
    columns_to_keep = df.columns[df.isnull().mean() <= threshold]
    
    return df[columns_to_keep].fillna("")





