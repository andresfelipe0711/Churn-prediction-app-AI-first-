import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# CONFIGURACION DE PAGINA RESPONSIVE
st.set_page_config(
    page_title="Churn Risk Simulator",
    page_icon="📡",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# CSS para asegurar que los contenedores sean flexibles en móviles
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    .stApp { background: linear-gradient(180deg, #0E1117 0%, #1d4d59 100%); }
    .metric-container {
        background-color: #1f333b;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        text-align: center;
    }
    /* Forzar que las columnas se apilen en pantallas pequeñas */
    @media (max-width: 640px) {
        .stHorizontalBlock {
            flex-direction: column !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Diccionario de textos
texts = {
    "es": {
        "app_title": "Simulador de riesgo de fuga de clientes",
        "sidebar_title": "Parámetros del Cliente",
        "tenure": "Antigüedad (meses)",
        "monthly_charges": "Cargos Mensuales",
        "total_charges": "Cargos Totales",
        "contract_type": "Tipo de Contrato",
        "prediction_results": "Resultados de la Predicción",
        "churn_prob": "Probabilidad de Fuga",
        "high_risk_alert": "ALTO RIESGO: Aplicar Estrategia de Océano Azul inmediatamente",
        "feature_importance": "Importancia de Factores",
        "feature_description": "Análisis de impacto:"
    },
    "en": {
        "app_title": "Customer Churn Risk Simulator",
        "sidebar_title": "Client Parameters",
        "tenure": "Tenure (months)",
        "monthly_charges": "Monthly Charges",
        "total_charges": "Total Charges",
        "contract_type": "Contract Type",
        "prediction_results": "Prediction Results",
        "churn_prob": "Churn Probability",
        "high_risk_alert": "HIGH RISK: Apply Blue Ocean Strategy immediately",
        "feature_importance": "Feature Importance",
        "feature_description": "Impact analysis:"
    }
}

# CARGA DE MODELO (RUTA CORREGIDA)
@st.cache_resource
def load_model_and_scaler():
    # Eliminamos 'Portfolio_project_2/' porque los archivos están en la raíz del repo
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo (.pkl) en la raíz del repositorio.")
    st.stop()

# --- INTERFAZ ---
with st.sidebar:
    language = st.radio("Language / Idioma", ("English", "Español"))
    lang_code = "en" if language == "English" else "es"
    current_texts = texts[lang_code]
    
    st.header(current_texts["sidebar_title"])
    tenure = st.slider(current_texts["tenure"], 0, 72, 30)
    monthly_charges = st.slider(current_texts["monthly_charges"], 18.0, 118.0, 50.0)
    total_charges = st.slider(current_texts["total_texts"], 0.0, 8685.0, 1500.0)
    contract = st.selectbox(current_texts["contract_type"], ["Month-to-month", "One year", "Two year"])

# Procesamiento y Predicción
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, 1 if contract == "Month-to-month" else 0, 0, 0, 1, 0, 0, 1]], 
                         columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Month-to-month', 'Contract_One year', 
                                  'Contract_Two year', 'InternetService_Fiber optic', 'InternetService_DSL', 'InternetService_No', 
                                  'OnlineSecurity_No'])

scaled_input = scaler.transform(input_data)
prob = model.predict_proba(scaled_input)[0][1]

# --- DASHBOARD RESPONSIVE ---
st.title(current_texts["app_title"])

# Las columnas en Streamlit son responsive por defecto si se usa use_container_width
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(current_texts["churn_prob"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1d4d59"}}
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown(f"<div class='metric-container'><h3>{current_texts['prediction_results']}</h3></div>", unsafe_allow_html=True)
    if prob > 0.7:
        st.error(current_texts["high_risk_alert"])
    else:
        st.success("Riesgo controlado / Controlled risk")

# Gráfico de barras inferior
st.subheader(current_texts["feature_importance"])
importance = pd.DataFrame({'Feature': input_data.columns, 'Value': model.coef_[0]})
fig_bar = px.bar(importance, x='Value', y='Feature', orientation='h', template="plotly_dark")
st.plotly_chart(fig_bar, use_container_width=True)
