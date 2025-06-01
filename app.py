import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

# Настройка темы
st.set_page_config(page_title="Прогнозирование отказов", page_icon="⚙️", layout="wide")

# Стилизация
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stMetric {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    selected_page = st.selectbox("📋 Навигация", ["Анализ", "Презентация"], help="Переключение между страницами")

pages = {
    "Анализ": analysis_and_model_page,
    "Презентация": presentation_page
}

pages[selected_page]()