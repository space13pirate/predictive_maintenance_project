import os
import streamlit as st
import pandas as pd
from clearml import Dataset, Task, Model
import joblib
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
    df = pd.read_csv(file_path)
    return df

# Функция для встроенной страницы анализа и инференса
def analysis_and_model_page():
    st.title("🔧 Анализ и модель")

    # 1. Загрузка датасета из ClearML
    st.subheader("📂 Загрузка датасета")
    try:
        with st.spinner("Загрузка датасета из ClearML..."):
            df = load_from_clearml()
            st.success("✅ Датасет успешно загружен из ClearML")
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке датасета: {e}")
        return
    
    st.write("📌 Первые строки датасета:")
    st.dataframe(df.head())

    # 2. Загрузка бинарной модели CatBoost
    st.subheader(":gear: Бинарная классификация (CatBoost)")
    try:
        # Подключение к задаче (если нет активной — создаем временную)
        task = Task.current_task() or Task.init(
            project_name="Predictive Maintenance",
            task_name="Load Models for Inference",
            task_type=Task.TaskTypes.inference
        )

        try:
            model_cat = Model(model_id="27871909cd884d18abead8cce9ba8e43")  # Получение объекта модели по ID           
            local_cat_path = model_cat.get_local_copy()                     # Загрузка локальной копии            
            cat_model = joblib.load(local_cat_path)                         # Загрузка модели с помощью joblib

            st.success("✅ CatBoost модель загружена")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки CatBoost модели: {e}")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки CatBoost модели: {e}")
        return
    
    st.markdown("""
    **Accuracy**: 0.9883  
    **F1-score (class 1)**: 0.79  
    **AUC**: 0.9768
    """)

    # 3. Мультиклассовая классификация (XGBoost)
    st.subheader(":gear: Мультиклассовая классификация (XGBoost)")
    try:
        model_xgb = Model(model_id="6efcaeca84a046ffbb2449f074a24b62")
        local_xgb_path = model_xgb.get_local_copy()
        xgb_model = joblib.load(local_xgb_path)

        st.success("✅ XGBoost модель загружена")
    except Exception as e:
        st.error(f"❌ Ошибка загрузки XGBoost модели: {e}")
        return

    st.markdown("""
    **Accuracy**: 0.9837  
    **Macro F1**: 0.51  
    **ROC-AUC**: 0.9321
    """)

    # 4. Ввод значений и предсказание
    st.subheader(":crystal_ball: Предсказание")

    # Начальные значения (можно менять)
    default_values = {
        "Type": 1,
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1200.0,
        "Torque [Nm]": 1.0,
        "Tool wear [min]": 1.0
    }

    with st.form("prediction_form"):
        # Ввод пользователем
        type_ = st.number_input("Type", value=default_values["Type"])
        air_temp = st.number_input("Air temperature [K]", value=default_values["Air temperature [K]"])
        process_temp = st.number_input("Process temperature [K]", value=default_values["Process temperature [K]"])
        rot_speed = st.number_input("Rotational speed [rpm]", value=default_values["Rotational speed [rpm]"])
        torque = st.number_input("Torque [Nm]", value=default_values["Torque [Nm]"])
        tool_wear = st.number_input("Tool wear [min]", value=default_values["Tool wear [min]"])

        # Кнопка запуска предсказания
        submitted = st.form_submit_button("🔮 Предсказать")

    if submitted:
        # Подготовка входных данных
        full_input = pd.DataFrame([{
            "Type": type_,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear
        }])

        # Подготовка данных для моделей
        input_cat = full_input.rename(columns={
            "Air temperature [K]": "Air_temperature_K",
            "Process temperature [K]": "Process_temperature_K",
            "Rotational speed [rpm]": "Rotational_speed_rpm",
            "Torque [Nm]": "Torque_Nm",
            "Tool wear [min]": "Tool_wear_min"
        })

        input_xgb = full_input.drop(columns=["Type"]).rename(columns={
            "Air temperature [K]": "Air_temperature_K",
            "Process temperature [K]": "Process_temperature_K",
            "Rotational speed [rpm]": "Rotational_speed_rpm",
            "Torque [Nm]": "Torque_Nm",
            "Tool wear [min]": "Tool_wear_min"
        })

        # Предсказания
        prediction_bin = cat_model.predict(input_cat)[0]
        prediction_multi = xgb_model.predict(input_xgb)[0]

        # Словарь меток
        failure_labels = {
            0: "HDF (недостаточный теплоотвод)",
            1: "Нет отказа",
            2: "OSF (перегрузка)",
            3: "PWF (мощностной сбой)",
            4: "RNF (случайный отказ)",
            5: "TWF (износ инструмента)"
        }

        # Вывод результатов
        st.success(f"Предсказание (бинарный): {'Отказ' if prediction_bin == 1 else 'Нет отказа'}")
        st.info(f"Предсказание (тип отказа): {failure_labels.get(prediction_multi, 'Неизвестно')}")
