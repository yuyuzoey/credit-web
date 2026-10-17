import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager

def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按 split 列统计各数据集的基本信息。"""
    required_columns = ["split", "default_next_month", "record_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要的列: {', '.join(missing_columns)}")

    summary = []
    for split_name, group in df.groupby("split", dropna=False):
        sample_count = len(group)
        default_rate = group["default_next_month"].mean()
        missing_count = int(group.isna().sum().sum())
        duplicate_count = int(group["record_id"].duplicated().sum())
        summary.append({
            "数据集": split_name,
            "样本量": sample_count,
            "违约率": default_rate,
            "缺失值数": missing_count,
            "record_id重复数": duplicate_count,
        })

    result = pd.DataFrame(summary)
    return result

def plot_distributions(df: pd.DataFrame) -> list:
    """绘制特征分布图（全英文标签，避免云端乱码）。"""
    figures = []

    # 图一：信用额度分布
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(df["credit_limit_ntd"].dropna(), bins=30, edgecolor="black", alpha=0.75, label="Count")
    ax1.set_xlabel("Credit Limit (NTD)")
    ax1.set_ylabel("Count")
    ax1.set_title("Credit Limit Distribution")
    ax1.legend()
    fig1.tight_layout()
    figures.append(fig1)

    # 图二：年龄分布
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(df["age_years"].dropna(), bins=30, edgecolor="black", alpha=0.75, label="Count")
    ax2.set_xlabel("Age (Years)")
    ax2.set_ylabel("Count")
    ax2.set_title("Age Distribution")
    ax2.legend()
    fig2.tight_layout()
    figures.append(fig2)

    # 图三：还款状态柱状图
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    status = df["pay_status_2005_09"].dropna()
    counts = status.value_counts().sort_index()
    ax3.bar(counts.index.astype(str), counts.values, edgecolor="black", alpha=0.75, label="Count")
    ax3.set_xlabel("Pay Status Code")
    ax3.set_ylabel("Count")
    ax3.set_title("Pay Status Distribution (2005-09)")
    ax3.legend()
    fig3.tight_layout()
    figures.append(fig3)

    return figures

def check_negative_pay_status(df: pd.DataFrame) -> dict:
    """统计所有 pay_status 列中的负值数量。"""
    pay_status_cols = [col for col in df.columns if "pay_status" in col]
    result = {}
    for col in pay_status_cols:
        neg_count = int((df[col] < 0).sum())
        result[col] = neg_count
    return result
