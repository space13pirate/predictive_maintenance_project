import streamlit as st
from streamlit_option_menu import option_menu
from analysis_and_model import analysis_and_model_page
from data_analysis import data_analysis_page
from presentation import presentation_page

# Настройка страницы
st.set_page_config(page_title="Прогноз отказов", layout="wide")

# Боковая панель навигации
with st.sidebar:
    selected = option_menu(
        "Навигация",  # Заголовок меню
        ["Анализ и модель", "Анализ данных", "Презентация"],    # Список страниц
        icons=["activity", "bar-chart", "file-earmark-slides"], # Иконки
        default_index=2 # Индекс стартовой страницы
    )

# Отображение выбранной страницы
if selected == "Анализ и модель":
    analysis_and_model_page()
elif selected == "Анализ данных":
    data_analysis_page()
elif selected == "Презентация":
    presentation_page()
