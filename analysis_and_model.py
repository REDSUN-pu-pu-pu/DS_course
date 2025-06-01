import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np


def analysis_and_model_page():
    st.title("🔧 Анализ и прогнозирование отказов оборудования", anchor=False)
    st.markdown("Загрузите CSV-файл для анализа данных и обучения модели Random Forest.")

    # Контейнер для загрузки файла
    with st.container(border=True):
        uploaded_file = st.file_uploader("📂 Выберите CSV-файл", type="csv", help="Файл должен содержать необходимые столбцы.")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Предобработка
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
        data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})

        numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                         'Torque [Nm]', 'Tool wear [min]']

        scaler = StandardScaler()
        data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Метрики
        st.subheader("📊 Результаты модели", anchor=False)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("ROC-AUC", f"{auc:.2%}")

        # Интерактивная матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        fig = ff.create_annotated_heatmap(
            z=cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'],
            colorscale='Blues', showscale=True
        )
        fig.update_layout(title="Матрица ошибок", width=400, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_fig = px.line(roc_df, x='FPR', y='TPR', title='ROC-кривая', labels={'FPR': 'False Positive Rate', 'TPR': 'True Positive Rate'})
        roc_fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(roc_fig, use_container_width=True)

        # Ввод данных для прогноза
        st.subheader("🔍 Прогноз для новых данных", anchor=False)
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_type = st.selectbox("Тип оборудования", options=['L', 'M', 'H'], help="Выберите тип: L (Low), M (Medium), H (High)")
                air_temp = st.number_input("Температура воздуха [K]", min_value=0.0, value=300.0, step=0.1)
                process_temp = st.number_input("Температура процесса [K]", min_value=0.0, value=310.0, step=0.1)
            with col2:
                rotation_speed = st.number_input("Скорость вращения [rpm]", min_value=0.0, value=1500.0, step=10.0)
                torque = st.number_input("Крутящий момент [Nm]", min_value=0.0, value=40.0, step=0.1)
                tool_wear = st.number_input("Износ инструмента [min]", min_value=0.0, value=100.0, step=1.0)
            submitted = st.form_submit_button("Сделать прогноз 🚀")

        if submitted:
            input_dict = {
                'Type': [input_type],
                'Air temperature [K]': [air_temp],
                'Process temperature [K]': [process_temp],
                'Rotational speed [rpm]': [rotation_speed],
                'Torque [Nm]': [torque],
                'Tool wear [min]': [tool_wear],
            }

            input_df = pd.DataFrame(input_dict)
            input_df['Type'] = input_df['Type'].map({'L': 0, 'M': 1, 'H': 2})
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.markdown("### 🎯 Результат предсказания")
            st.info(
                "**Отказ оборудования**" if prediction == 1 else "**Оборудование работает нормально**",
                icon="⚙️"
            )
            st.progress(probability)
            st.write(f"**Вероятность отказа:** {probability:.2%}")