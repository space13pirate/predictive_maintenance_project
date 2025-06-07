import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from clearml import Dataset
from dotenv import load_dotenv

# Загрузка переменных среды из .env
load_dotenv()

# Кэшируемая функция загрузки только из ClearML
@st.cache_data
def load_from_clearml():
    dataset = Dataset.get(
        dataset_project="Predictive Maintenance",
        dataset_name="Predictive Maintenance Dataset"
    )
    local_path = dataset.get_local_copy()
    file_path = os.path.join(local_path, "predictive_maintenance.csv")
    data = pd.read_csv(file_path)
    return data

# Главная функция страницы анализа данных
def data_analysis_page():
    st.title("Детальный анализ исходного датасета")

    st.subheader("🔍 Загрузка датасета")
    source = st.radio("Выберите источник данных:", ("Локальный файл", "ClearML"))

    data = None
    if source == "Локальный файл":
        uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("✅ Датасет успешно загружен из локального файла")
    elif source == "ClearML":
        try:
            st.info("⏳ Загружается последняя версия из ClearML...")
            data = load_from_clearml()
            st.success("✅ Датасет успешно загружен из ClearML")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки из ClearML: {e}")
            return

    if data is None:
        st.warning("⚠ Пожалуйста, загрузите датасет для продолжения")
        return

    st.write("📌 Первые строки датасета:")
    st.dataframe(data.head())

    # Детальный анализ исходного датасета
    st.header("Детальный анализ исходного датасета")
    
    st.subheader("1. Визуализация распределений и связей")

    numeric_features = data.select_dtypes(include='number').columns
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    axes = axes.flatten()

    for i, feature in enumerate(numeric_features):
        sns.histplot(data[feature], kde=True, bins=30, color='steelblue', ax=axes[i])
        axes[i].set_title(f"Распределение: {feature}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    plt.suptitle("Распределение количественных признаков с KDE", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    st.pyplot(fig)

    st.subheader("2. Корреляционная матрица признаков")

    corr_matrix = data.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Корреляционная матрица признаков", fontsize=16)
    st.pyplot(fig)

    st.subheader("3. Парный анализ взаимосвязей признаков")

    feature_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    if 'Machine failure' in data.columns:
        fig = sns.pairplot(data[feature_cols + ['Machine failure']], hue='Machine failure', palette='Set1')
        st.pyplot(fig)

    failure_labels = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    for label in failure_labels:
        if label in data.columns:
            fig = sns.pairplot(data[feature_cols + [label]], hue=label, palette='tab10', diag_kind='kde')
            fig.fig.suptitle(f"Парные взаимосвязи признаков по метке {label}", y=1.02, fontsize=14)
            st.pyplot(fig)
