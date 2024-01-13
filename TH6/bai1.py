import opendatasets as od
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

st.balloons()

st.title("Bai 1 xây dựng Web App đơn giản với Streamlit")
uploaded_file = st.file_uploader("/content/petrol_consumption.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    st.subheader("Thống kê mô tả dữ liệu")
    st.write(df.describe())

    st.subheader("Mô tả các thuộc tính")
    st.write(df.info())

    st.subheader("Biểu đồ histogram")
    selected_column = st.selectbox("Chọn thuộc tính", df.columns)
    plt.figure(figsize=(8, 6))
    sns.histplot(df[selected_column], kde=True)
    st.pyplot()

    st.subheader("Hệ số tương quan")
    correlation_matrix = df.corr()
    st.write(correlation_matrix)

    st.subheader("Biểu đồ phân tán")
    dependent_variable = 'Petrol_Consumption'
    independent_variable = st.selectbox("Chọn biến độc lập", df.columns)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[independent_variable], y=df[dependent_variable])
    st.pyplot()

    st.subheader("Boxplot cho mỗi thuộc tính")
    selected_boxplot_column = st.selectbox("Chọn thuộc tính cho Boxplot", df.columns)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[selected_boxplot_column])
    st.pyplot()

    st.subheader("Đồ thị đường biểu diễn sự biến thiên của biến phụ thuộc")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=df.index, y=df[dependent_variable])
    st.pyplot()

    st.subheader("Biểu đồ pairplot cho sự tương quan giữa các cặp biến")
    sns.pairplot(df)
    st.pyplot()
