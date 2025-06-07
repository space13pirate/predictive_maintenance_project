import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("📊 Презентация проекта")

    # Markdown-презентация с улучшенным стилем
    presentation_markdown = """
<style>
.reveal {
    font-size: 1.5em;
    padding: 25px;
    line-height: 1.6;
    font-family: 'Segoe UI', 'Noto Sans', 'Helvetica Neue', Arial, sans-serif;
}

.reveal h1 {
    color: #007acc;
    border-bottom: 2px solid #007acc;
    padding-bottom: 10px;
    text-align: center;
}

.reveal h2 {
    color: #333333;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 15px;
}

.reveal ul {
    list-style-type: square;
    padding-left: 30px;
}

.reveal li {
    margin-bottom: 12px;
}

.reveal p {
    margin-bottom: 15px;
}
</style>

# Бинарная и мультиклассовая классификации для предиктивного обслуживания оборудования
---

## Введение
- Задача: предсказание отказов оборудования.
- Датасет: AI4I 2020 Predictive Maintenance Dataset.
- Цель проекта:
    - Разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0).
    - В рамках продвинутой версии проекта реализованы дополнительные функции:  
        - мультиклассовая классификация  
        - детальный анализ данных  
        - более качественная работа с предобработкой данных  
        - работа с более мощными моделями  
        - оптимизация обучения модели  
        - интеграция с инструментами MLOps (ClearML)

---

## Этапы работы
1. Загрузка и анализ данных  
2. Предобработка и балансировка классов  
3. Обучение моделей (CatBoost, XGBoost)  
4. Оптимизация с помощью Optuna  
5. Визуализация метрик  
6. Развёртывание через ClearML Serving  
7. Интеграция в Streamlit

---

## Streamlit-приложение
- Страница 1 – анализ, обучение, предсказание
- Страница 2 – детальный анализ исходного датасета
- Страница 3 – презентация проекта 
- Использован reveal.js + streamlit-reveal-slides

---

## Заключение
- Добились точности > 0.98 на валидации  
- Реализовали Streamlit-интерфейс и docker-развёртывание  
- Возможные улучшения:  
    - Добавить explainability (SHAP)  
    - Использовать временные признаки  
    - Поддержка REST API и CI/CD
"""

    # Боковая панель для настройки презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("🎨 Тема", [
            "black", "white", "league", "beige", "sky",
            "night", "serif", "simple", "solarized"
        ], index=1)  # default: white
        height = st.number_input("Высота слайдов", value=700)
        transition = st.selectbox("Переход", [
            "slide", "convex", "concave", "zoom", "none"
        ])
        plugins = st.multiselect("Плагины", [
            "highlight", "katex", "mathjax2", "mathjax3",
            "notes", "search", "zoom"
        ], default=["zoom"])

    # Рендер презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )
