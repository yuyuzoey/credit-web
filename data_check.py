import streamlit as st
import pandas as pd
from src.data_io import load_data
from src.data_check import split_summary, check_negative_pay_status

st.set_page_config(page_title="数据概览", layout="wide")
st.title("数据概览")

try:
    df = load_data("data/01_credit_default.csv")
    
    # 1. 统计卡片
    st.subheader("数据集统计指标")
    summary_df = split_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("训练集样本量", f"{len(df[df['split']=='train']):,}")
    with col2:
        st.metric("验证集样本量", f"{len(df[df['split']=='valid']):,}")
    with col3:
        st.metric("测试集样本量", f"{len(df[df['split']=='test']):,}")
    with col4:
        st.metric("缺失值总数", f"{int(df.isna().sum().sum())}")

    # 2. 详细表格
    st.subheader("各数据集详细统计")
    st.dataframe(summary_df, use_container_width=True)

    # 3. 特征分布（直接使用 Streamlit 原生图表，彻底告别方块乱码！）
    st.subheader("特征分布")
    
    tab1, tab2, tab3 = st.tabs(["授信额度分布", "年龄分布", "还款状态分布"])
    
    with tab1:
        st.bar_chart(df["credit_limit_ntd"].value_counts().sort_index(), x_label="Credit Limit", y_label="Count")
        
    with tab2:
        st.bar_chart(df["age_years"].value_counts().sort_index(), x_label="Age", y_label="Count")
        
    with tab3:
        st.bar_chart(df["pay_status_2005_09"].value_counts().sort_index(), x_label="Pay Status Code", y_label="Count")

    st.info("说明：还款状态中的 -2、-1、0 是编码，不是缺失值，本项目保留原始编码不填充。")
    st.warning("相关性不等于因果关系，本项目仅作教学分析。")

except Exception as e:
    st.error(f"数据加载或处理失败: {e}")
