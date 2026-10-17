# -*- coding: utf-8 -*-
"""信用卡违约预测项目：数据概览页面。"""

import streamlit as st
import pandas as pd

from src.data_io import (
    load_data,
    split_data,
    get_feature_columns,
)
from src.data_check import (
    split_summary,
    plot_distributions,
    check_negative_pay_status,
)


# 设置页面标题和布局
st.set_page_config(
    page_title="数据概览",
    layout="wide",
)

st.title("数据概览")

# 加载信用卡违约数据
try:
    df = load_data("data/01_credit_default.csv")
except FileNotFoundError:
    st.error(
        "数据文件未找到，请检查 data/01_credit_default.csv"
    )
    st.stop()

if df is None:
    st.error(
        "数据文件未找到，请检查 data/01_credit_default.csv"
    )
    st.stop()

# 检查必要的数据列
required_columns = [
    "split",
    "default_next_month",
    "record_id",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "数据缺少必要的列：" + ", ".join(missing_columns)
    )
    st.stop()

# 生成分组统计表
summary = split_summary(df)

# 统计整体缺失值数量
missing_total = int(df.isnull().sum().sum())

# 统计 record_id 重复数量
duplicate_total = int(
    df["record_id"].duplicated().sum()
)

# 获取三组数据的统计指标
split_names = ["train", "valid", "test"]
split_stats = {}

for name in split_names:
    group = df[df["split"] == name]
    split_stats[name] = {
        "count": len(group),
        "rate": group["default_next_month"].mean(),
    }

# 第一行：三组样本量和缺失值总数
st.subheader("数据集统计指标")

row1 = st.columns(4)

row1[0].metric(
    "训练集样本量",
    "{:,}".format(split_stats["train"]["count"]),
)
row1[1].metric(
    "验证集样本量",
    "{:,}".format(split_stats["valid"]["count"]),
)
row1[2].metric(
    "测试集样本量",
    "{:,}".format(split_stats["test"]["count"]),
)
row1[3].metric(
    "缺失值总数",
    "{:,}".format(missing_total),
)

# 第二行：三组违约率和 record_id 重复数
row2 = st.columns(4)

for index, name in enumerate(split_names):
    rate = split_stats[name]["rate"]
    value = "N/A" if pd.isna(rate) else "{:.2%}".format(
        rate
    )
    labels = {
        "train": "训练集违约率",
        "valid": "验证集违约率",
        "test": "测试集违约率",
    }
    row2[index].metric(labels[name], value)

row2[3].metric(
    "record_id 重复数",
    "{:,}".format(duplicate_total),
)

# 展示完整的数据集分组统计表
st.subheader("各数据集详细统计")
st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)

# 展示特征分布图
st.subheader("特征分布")

try:
    figures = plot_distributions(df)

    for fig in figures:
        st.pyplot(fig)
        # 展示完成后关闭图形，释放内存
        import matplotlib.pyplot as plt
        plt.close(fig)

except (ValueError, KeyError) as exc:
    st.error("绘制分布图失败：{}".format(exc))

# 统计还款状态中的负值
st.subheader("还款状态编码检查")

negative_counts = check_negative_pay_status(df)

if negative_counts:
    negative_df = pd.DataFrame(
        list(negative_counts.items()),
        columns=["还款状态列", "负值数量"],
    )
    st.dataframe(
        negative_df,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("未找到符合规则的还款状态列。")

# 说明还款状态编码的含义
st.info(
    "还款状态中的 -2、-1、0 是编码，不是缺失值，"
    "本项目保留原始编码不填充。"
)

# 说明项目的使用范围
st.warning(
    "相关性不等于因果关系，本项目仅作教学分析，"
    "不作为真实授信依据。"
)
