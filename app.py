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

# CONFIGURACION DE PAGINA Y TEMA OSCURO
st.set_page_config(
    page_title="Churn Risk Simulator",
    page_icon="📡",
    layout="wide", # Mantener layout wide es bueno para pantallas grandes
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema oscuro profesional
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0E1117 0%, #1d4d59 100%);
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .metric-container {
        background-color: #1f333b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2a3d43;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 20px; /* Espacio para mobile */
    }
    .stSlider > div > div > div > div {
        color: #FFFFFF;
    }
    .stSlider > label {
        color: #FFFFFF;
    }
    .stSelectbox > label {
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }
    .gauge-label {
        font-size: 20px;
        font-weight: bold;
        color: #e0e0e0;
        text-align: center;
        margin-top: 10px;
    }
    /* Estilos para hacer Plotly responsive */
    .plotly-container {
        width: 100% !important;
        height: auto !important;
    }
    .stPlotlyChart {
        width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Diccionario de textos en diferentes idiomas
texts = {
    "es": {
        "app_title": "Simulador de riesgo de fuga de clientes",
        "sidebar_title": "Parámetros del Cliente",
        "tenure": "Antigüedad (meses)",
        "monthly_charges": "Cargos Mensuales",
        "total_charges": "Cargos Totales",
        "contract_type": "Tipo de Contrato",
        "contract_month_to_month": "Mes a Mes",
        "contract_one_year": "Un año",
        "contract_two_year": "Dos años",
        "prediction_results": "Resultados de la Predicción",
        "churn_prob": "Probabilidad de Fuga",
        "high_risk_alert": "ALTO RIESGO: Aplicar Estrategia de Océano Azul inmediatamente",
        "feature_importance": "Importancia de Factores en esta Predicción",
        "feature_description": "Descripción de por qué una variable tiene impacto basado en el análisis de coeficientes:",
        "tenure_desc": "**Antigüedad (meses):** El análisis reveló que la mayoría de los clientes con <12 meses de antigüedad son más propensos a la fuga.",
        "monthly_charges_desc": "**Cargos Mensuales:** Clientes con <$50 o >$80 tienden a mostrar diferentes patrones. El modelo sugiere que cargos muy bajos o muy altos pueden ser un indicador.",
        "total_charges_desc": "**Cargos Totales:** El valor acumulado. El análisis muestra que el negocio está fallando en retener clientes que acumulan pocos cargos, posiblemente por problemas iniciales.",
        "contract_type_desc": "**Tipo de Contrato:** Los contratos mes a mes son el mayor indicador de fuga. Los contratos de un año ofrecen un punto medio y los de dos años muestran la menor fuga.",
        "internet_service_fiber_optic_desc": "**Servicio de Internet Fibra Óptica:** Alto riesgo por alto valor. A pesar de ser el servicio más rápido, puede estar asociado con una mayor insatisfacción del cliente si la calidad no es consistente.",
        "online_security_desc": "**Seguridad en línea:** Junto con 'Backup' y 'Device Protection', se encontró que estos servicios son críticos para la retención. Los clientes que carecen de estas características son un grupo de alto riesgo."
    },
    "en": {
        "app_title": "Customer Churn Risk Simulator",
        "sidebar_title": "Client Parameters",
        "tenure": "Tenure (months)",
        "monthly_charges": "Monthly Charges",
        "total_charges": "Total Charges",
        "contract_type": "Contract Type",
        "contract_month_to_month": "Month-to-Month",
        "contract_one_year": "One Year",
        "contract_two_year": "Two Year",
        "prediction_results": "Prediction Results",
        "churn_prob": "Churn Probability",
        "high_risk_alert": "HIGH RISK: Apply Blue Ocean Strategy immediately",
        "feature_importance": "Feature Importance in this Prediction",
        "feature_description": "Description of why a variable has an impact based on coefficient analysis:",
        "tenure_desc": "**Tenure (months):** Analysis revealed that most customers with <12 months of tenure are more prone to churn.",
        "monthly_charges_desc": "**Monthly Charges:** Customers with <$50 or >$80 tend to show different patterns. The model suggests that very low or very high charges can be an indicator.",
        "total_charges_desc": "**Total Charges:** The accumulated value. Analysis shows that the business is failing to retain customers who accumulate low charges, possibly due to initial issues.",
        "contract_type_desc": "**Contract Type:** Month-to-month contracts are the biggest churn indicator. One-year contracts offer a middle ground, and two-year contracts show the lowest churn.",
        "internet_service_fiber_optic_desc": "**Fiber Optic Internet Service:** High risk for high value. Despite being the fastest service, it may be associated with higher customer dissatisfaction if quality is inconsistent.",
        "online_security_desc": "**Online Security:** Along with 'Backup' and 'Device Protection', these services were found to be critical for retention. Customers lacking these features are a high-risk group."
    }
}

# Cargar el modelo y scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('Portfolio_project_2/logistic_regression_model.pkl')
    scaler = joblib.load('Portfolio_project_2/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Función para obtener la descripción de la importancia de la variable
def get_feature_description(feature_name, lang):
    descriptions = {
        "tenure": texts[lang]["tenure_desc"],
        "MonthlyCharges": texts[lang]["monthly_charges_desc"],
        "TotalCharges": texts[lang]["total_charges_desc"],
        "Contract_Month-to-month": texts[lang]["contract_type_desc"],
        "InternetService_Fiber optic": texts[lang]["internet_service_fiber_optic_desc"],
        "OnlineSecurity_No": texts[lang]["online_security_desc"]
    }
    return descriptions.get(feature_name, "Descripción no disponible.")

# Función para obtener la importancia de las características usando los coeficientes del modelo
@st.cache_data
def get_feature_importance(_model, df_scaled_input_data, original_feature_names):
    coefficients = _model.coef_[0]
    
    # Asegúrate de que las características originales coincidan con el orden del scaler
    # Esto es crucial para la correcta interpretación de los coeficientes
    feature_importance = pd.DataFrame({
        'Feature': original_feature_names,
        'Coefficient': coefficients
    })
    
    # Usar el valor absoluto de los coeficientes para la magnitud, pero el signo para la dirección.
    # Para el gráfico, nos interesa más la magnitud.
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    
    # Ordenar por el valor absoluto del coeficiente para mostrar los más influyentes
    # Y luego ordenar por la dirección del coeficiente para la visualización
    feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False).head(10)
    
    return feature_importance

# Cargar el dataset para obtener los nombres de las columnas originales
@st.cache_data
def load_data():
    df = pd.read_csv('churn_dataset.csv')
    return df

df_original = load_data()

# Preprocesamiento de datos (similar al notebook)
# Esto necesita ser consistente con cómo se entrenó el modelo
def preprocess_input(input_data):
    processed_data = pd.DataFrame([input_data])
    
    # Crear columnas dummy para Contract
    processed_data['Contract_Month-to-month'] = (processed_data['Contract'] == 'Month-to-month').astype(int)
    processed_data['Contract_One year'] = (processed_data['Contract'] == 'One year').astype(int)
    processed_data['Contract_Two year'] = (processed_data['Contract'] == 'Two year').astype(int)
    
    # Crear dummies para InternetService (solo si es necesario, basado en el modelo)
    processed_data['InternetService_Fiber optic'] = (processed_data['InternetService'] == 'Fiber optic').astype(int)
    processed_data['InternetService_DSL'] = (processed_data['InternetService'] == 'DSL').astype(int)
    processed_data['InternetService_No'] = (processed_data['InternetService'] == 'No').astype(int)

    # Crear dummies para OnlineSecurity (solo si es necesario, basado en el modelo)
    processed_data['OnlineSecurity_No'] = (processed_data['OnlineSecurity'] == 'No').astype(int)
    
    # Eliminar columnas originales no necesarias para el modelo
    processed_data = processed_data.drop(columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                                  'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Contract', 
                                                  'InternetService', 'OnlineSecurity'], errors='ignore')
    
    # Asegurarse de que el orden y las columnas coincidan con el dataset de entrenamiento
    # Esto es crucial. Las columnas que faltan deben llenarse con 0.
    # Obtener las columnas usadas para entrenar el modelo (excepto 'Churn')
    model_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_Month-to-month', 'Contract_One year', 
                      'Contract_Two year', 'InternetService_Fiber optic', 'InternetService_DSL', 'InternetService_No', 
                      'OnlineSecurity_No'] # Añade aquí todas las features que tu modelo espera
    
    # Añadir columnas que podrían faltar y rellenar con 0
    for feature in model_features:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
            
    # Reordenar las columnas para que coincidan con el orden del entrenamiento
    processed_data = processed_data[model_features]
    
    return processed_data

# Sidebar para la selección de idioma
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Flag_of_Colombia.svg/1200px-Flag_of_Colombia.svg.png", width=50) # Pequeña bandera para identificar
    language = st.radio("Select Language / Seleccionar Idioma", ("English", "Español"))
    lang_code = "en" if language == "English" else "es"

current_texts = texts[lang_code]
st.title(current_texts["app_title"])

# SIDEBAR - Controles para la entrada del usuario
with st.sidebar:
    st.header(current_texts["sidebar_title"])

    tenure = st.slider(current_texts["tenure"], 0, 72, 30)
    monthly_charges = st.slider(current_texts["monthly_charges"], 18.0, 118.0, 50.0)
    total_charges = st.slider(current_texts["total_charges"], 0.0, 8685.0, 1500.0)
    
    contract_options = {
        current_texts["contract_month_to_month"]: "Month-to-month",
        current_texts["contract_one_year"]: "One year",
        current_texts["contract_two_year"]: "Two year"
    }
    contract_display = st.selectbox(current_texts["contract_type"], list(contract_options.keys()))
    contract_value = contract_options[contract_display]
    
    # Otros servicios para imputación
    internet_service_fiber_optic = st.checkbox("Internet Service: Fiber optic?", value=True)
    online_security_no = st.checkbox("Online Security: No?", value=True)

# Preparar los datos de entrada para la predicción
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract_value,
    'InternetService': 'Fiber optic' if internet_service_fiber_optic else 'DSL', # Asumir DSL si no es Fiber Optic
    'OnlineSecurity': 'No' if online_security_no else 'Yes'
    # Añadir aquí otras características importantes si tu modelo las usa
}

processed_input = preprocess_input(input_data)
scaled_input = scaler.transform(processed_input)

# Realizar la predicción
prediction_proba = model.predict_proba(scaled_input)[0][1]

st.header(current_texts["prediction_results"])

# Usar contenedores para agrupar elementos y mejorar responsividad
with st.container():
    col_prob, col_alert = st.columns([1, 2]) # Ajustar proporciones si es necesario

    with col_prob:
        st.subheader(current_texts["churn_prob"])
        
        # Gauge Chart para la probabilidad de Churn
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "lightblue"},
                    {'range': [70, 100], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))
        fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10)) # Ajustar margen para responsive
        st.plotly_chart(fig_gauge, use_container_width=True) # use_container_width es clave

    with col_alert:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        if prediction_proba > 0.5:
            st.error(current_texts["high_risk_alert"])
        else:
            st.success("Bajo riesgo de fuga. ¡Bien hecho!")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---") # Separador visual

# Importancia de las características
st.subheader(current_texts["feature_importance"])
original_feature_names = processed_input.columns.tolist() # Obtener nombres de columnas procesadas
feature_importance_df = get_feature_importance(model, scaled_input, original_feature_names)


# Crear un gráfico de barras interactivo con Plotly Express
fig_bar = px.bar(feature_importance_df, 
                 y='Feature', 
                 x='Coefficient', # Usar el coeficiente directo para mostrar impacto positivo/negativo
                 orientation='h', 
                 title='Impacto de las variables en la predicción de Churn',
                 labels={'Coefficient': 'Impacto del Coeficiente', 'Feature': 'Variable'},
                 color='Coefficient', # Color basado en el coeficiente
                 color_continuous_scale=px.colors.sequential.RdBu, # Escala rojo/azul para positivo/negativo
                 template="plotly_dark")

fig_bar.update_layout(
    height=400, # Altura fija, el ancho se adaptará
    margin=dict(l=50, r=50, t=50, b=50), # Ajustar márgenes
    xaxis_title="Impacto en la Predicción (Coeficiente)",
    yaxis_title="Variable",
    hovermode="y unified"
)
st.plotly_chart(fig_bar, use_container_width=True) # use_container_width es crucial para responsive

# Descripción de la importancia de las características
st.subheader(current_texts["feature_description"])
for index, row in feature_importance_df.iterrows():
    feature = row['Feature']
    description = get_feature_description(feature, lang_code)
    st.markdown(f"**{feature}:** {description}")
