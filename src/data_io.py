from pathlib import Path

import pandas as pd


def load_data(path) -> pd.DataFrame:
    """
    读取 CSV 文件并返回 DataFrame。
    不进行任何预处理，不删除缺失值。
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"数据文件不存在，请检查文件路径：{file_path}")

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"数据文件为空或不包含有效的 CSV 数据：{file_path}") from exc


def split_data(df:pd.DataFrame):
    """
    根据 split 列将数据拆分为训练集、验证集和测试集。    保留原始列和缺失值，不修改原始 DataFrame。
    """
    if "split" not in df.columns:
        raise ValueError("数据中缺少必需的 'split' 列，无法拆分数据集。")

    # 检查 split 列是否包含规定的数据集标识
    required_splits = {"train", "valid", "test"}
    actual_splits = set(df["split"].dropna().unique())
    missing_splits = required_splits - actual_splits

    if missing_splits:
        missing = ", ".join(sorted(missing_splits))
        raise ValueError(f"split 列缺少以下数据集标识：{missing}")

    # 分别筛选训练集、验证集和测试集，并保留原始索引
    train_df = df.loc[df["split"] == "train"].copy()
    valid_df = df.loc[df["split"] == "valid"].copy()
    test_df = df.loc[df["split"] == "test"].copy()

    return train_df, valid_df, test_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    获取模型特征列名。

    排除：
    - record_id：记录标识符
    - split：数据集划分标识
    - default_next_month：预测目标，避免目标泄漏
    """
    excluded_columns = {
        "record_id",
        "split",
        "default_next_month",
    }

    # 保留原始列顺序，只排除不应作为模型特征的列
    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    return feature_columns
