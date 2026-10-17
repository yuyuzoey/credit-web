# -*- coding: utf-8 -*-
"""信用卡违约预测项目的数据检查工具。"""

import re
import pandas as pd
import matplotlib.pyplot as plt


def split_summary(df):
    """按 split 分组统计数据集概况。"""
    required = [
        "split",
        "default_next_month",
        "record_id",
    ]
    missing = [
        col for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "缺少必要的列：" + ", ".join(missing)
        )

    rows = []
    groups = df.groupby(
        "split",
        dropna=False,
        sort=False,
    )

    for name, group in groups:
        # 统计当前分组的全部缺失值
        missing_count = int(
            group.isnull().sum().sum()
        )

        # 统计重复的 record_id，不计首次出现
        duplicate_count = int(
            group["record_id"].duplicated().sum()
        )

        # 违约率为违约标签的均值
        default_rate = group[
            "default_next_month"
        ].mean()

        rows.append({
            "数据集": name,
            "样本量": len(group),
            "违约率": default_rate,
            "缺失值数": missing_count,
            "record_id重复数": duplicate_count,
        })

    columns = [
        "数据集",
        "样本量",
        "违约率",
        "缺失值数",
        "record_id重复数",
    ]

    return pd.DataFrame(rows, columns=columns)


def plot_distributions(df):
    """绘制三个特征的分布图并返回图形列表。"""
    required = [
        "credit_limit_ntd",
        "age_years",
        "pay_status_2005_09",
    ]
    missing = [
        col for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "缺少必要的列：" + ", ".join(missing)
        )

    figures = []

    # 图一：信用额度直方图
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(
        df["credit_limit_ntd"].dropna(),
        bins=30,
        edgecolor="black",
        alpha=0.75,
        label="信用额度",
    )
    ax1.set_xlabel("Credit Limit")
    ax1.set_ylabel("Count")
    ax1.set_title("Credit Limit Distribution")
    ax1.legend()
    fig1.tight_layout()
    figures.append(fig1)

    # 图二：年龄直方图
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(
        df["age_years"].dropna(),
        bins=30,
        edgecolor="black",
        alpha=0.75,
        label="Age",
    )
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Count")
    ax2.set_title("Age Distribution")
    ax2.legend()
    fig2.tight_layout()
    figures.append(fig2)

    # 图三：2005 年 9 月还款状态柱状图
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    status = df[
        "pay_status_2005_09"
    ].dropna()
    counts = status.value_counts().sort_index()

    ax3.bar(
        counts.index.astype(str),
        counts.values,
        edgecolor="black",
        alpha=0.75,
        label="Count",
    )
    ax3.set_xlabel("Pay Status Distribution")
    ax3.set_ylabel("Count")
    ax3.set_title("2005年9月还款状态分布")
    ax3.legend()
    fig3.tight_layout()
    figures.append(fig3)

    # 不调用 plt.show()，由调用方决定如何显示
    return figures


def check_negative_pay_status(df):
    """统计所有还款状态列中的负值数量。"""
    pattern = re.compile(
        r"^pay_status_2005_\d{2}$"
    )
    columns = [
        col for col in df.columns
        if pattern.match(str(col))
    ]

    result = {}

    for col in columns:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )
        result[col] = int((values < 0).sum())

    return result
