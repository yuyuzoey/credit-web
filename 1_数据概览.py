# -*- coding: utf-8 -*-
"""信用卡违约预测项目：数据概览页面。"""

import streamlit as st
import pandas as pd

from src.data_io import load_data
from src.data_check import split_summary, check_negative_pay_status

# 设置页面标题和布局
st.set_page_config(page_title="数据概览", layout="wide")
st.title("数据概览")

# 加载数据
try:
    df = load_data("data/01_credit_default.csv")
except Exception as e:
    st.error(f"数据文件未找到或读取失败: {e}")
    st.stop()

# 生成分组统计表
summary = split_summary(df)
missing_total = int(df.isnull().sum().sum())
duplicate_total = int(df["record_id"].duplicated().sum())

# 第一行：样本量和缺失值
st.subheader("数据集统计指标")
row1 = st.columns(4)
row1[0].metric("训练集样本量", f"{len(df[df['split']=='train']):,}")
row1[1].metric("验证集样本量", f"{len(df[df['split']=='valid']):,}")
row1[2].metric("测试集样本量", f"{len(df[df['split']=='test']):,}")
row1[3].metric("缺失值总数", f"{missing_total:,}")

# 第二行：违约率和重复数
row2 = st.columns(4)
train_rate = df[df['split']=='train']["default_next_month"].mean()
valid_rate = df[df['split']=='valid']["default_next_month"].mean()
test_rate = df[df['split']=='test']["default_next_month"].mean()

row2[0].metric("训练集违约率", f"{train_rate:.2%}")
row2[1].metric("验证集违约率", f"{valid_rate:.2%}")
row2[2].metric("测试集违约率", f"{test_rate:.2%}")
row2[3].metric("record_id 重复数", f"{duplicate_total:,}")

# 表格
st.subheader("各数据集详细统计")
st.dataframe(summary, use_container_width=True, hide_index=True)

# ==========================================
# 重点在这里：彻底抛弃 matplotlib，使用 Streamlit 原生图表！
# ==========================================
st.subheader("特征分布")
tab1, tab2, tab3 = st.tabs(["授信额度分布", "年龄分布", "还款状态分布"])

with tab1:
    st.bar_chart(df["credit_limit_ntd"].value_counts().sort_index(), x_label="Credit Limit", y_label="Count")
with tab2:
    st.bar_chart(df["age_years"].value_counts().sort_index(), x_label="Age", y_label="Count")
with tab3:
    st.bar_chart(df["pay_status_2005_09"].value_counts().sort_index(), x_label="Pay Status Code", y_label="Count")

# 说明文字
st.info("说明：还款状态中的 -2、-1、0 是编码，不是缺失值。")
st.warning("相关性不等于因果关系，本项目仅作教学分析，不作为真实授信依据。")
