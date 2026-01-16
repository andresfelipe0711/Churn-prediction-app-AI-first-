"""
Customer Churn Risk Simulator - Streamlit App
Aplicación web interactiva para simular el riesgo de fuga de clientes
Desarrollado para el equipo de Customer Success
"""

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

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y TEMA OSCURO
# ============================================================================

st.set_page_config(
    page_title="Churn Risk Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema oscuro profesional
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #1a1d29 100%);
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .metric-container {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2a2d3a;
    }
    .stAlert {
        background-color: #2a1f1f;
        border-left: 4px solid #ff4444;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0;
    }
    .sidebar .sidebar-content {
        background-color: #1a1d29;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO Y MODELO
# ============================================================================

@st.cache_data
def load_data():
    """Carga el dataset de churn"""
    try:
        # Intentar cargar desde el directorio local
        data_path = os.path.join(os.path.dirname(__file__), 'churn_dataset.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Si no existe, intentar desde URL
            url = "https://raw.githubusercontent.com/andresfelipe0711/churn-prediction/refs/heads/main/churn_dataset.csv"
            df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

def preprocess_data(churn_df):
    """Preprocesa los datos siguiendo la lógica del notebook"""
    # Copiar para no modificar el original
    churn = churn_df.copy()
    
    # Convertir nombres de columnas a minúsculas
    churn.columns = churn.columns.str.lower()
    
    # Renombrar columnas
    churn.rename(columns={
        'customerid': 'customer_id',
        'seniorcitizen': 'senior_citizen',
        'phoneservice': 'phone_service',
        'multiplelines': 'multiple_lines',
        'internetservice': 'internet_service',
        'onlinesecurity': 'online_security',
        'onlinebackup': 'online_backup',
        'deviceprotection': 'device_protection',
        'techsupport': 'tech_support',
        'streamingtv': 'streaming_tv',
        'streamingmovies': 'streaming_movies',
        'paperlessbilling': 'paperless_billing',
        'totalcharges': 'total_charges',
        'paymentmethod': 'payment_method',
        'monthlycharges': 'monthly_charges'
    }, inplace=True)
    
    # Convertir total_charges a numérico
    churn['total_charges'] = pd.to_numeric(churn['total_charges'], errors='coerce')
    
    # Convertir churn a binario
    churn['churn'] = churn['churn'].map({'Yes': 1, 'No': 0})
    
    # Label encoding
    condition = {'Yes': 1, 'No': 0}
    label_cols = ['partner', 'dependents', 'phone_service', 'paperless_billing']
    for col in label_cols:
        churn[col] = churn[col].map(condition)
    
    # One-hot encoding
    cols_to_encode = ['contract', 'payment_method', 'multiple_lines', 'internet_service',
                      'online_security', 'online_backup', 'device_protection', 'tech_support',
                      'streaming_tv', 'streaming_movies']
    
    churn_dummies = pd.get_dummies(churn[cols_to_encode], drop_first=True)
    for col in churn_dummies.columns:
        churn_dummies[col] = churn_dummies[col].map({True: 1, False: 0})
    
    # Combinar datos
    churn_labels = churn[['partner', 'dependents', 'phone_service', 'paperless_billing', 'churn']]
    churn_dummies = pd.concat([churn_dummies, churn_labels], axis=1)
    
    cols_num = churn[['tenure', 'monthly_charges', 'total_charges', 'senior_citizen']]
    churn_dummies = pd.concat([cols_num, churn_dummies], axis=1)
    
    # Eliminar valores faltantes
    churn_dummies = churn_dummies.dropna()
    
    return churn_dummies

def prepare_features(churn_dummies):
    """Prepara X e y para el modelo"""
    X = churn_dummies.drop('churn', axis=1)
    y = churn_dummies['churn']
    return X, y

@st.cache_resource
def train_model(X, y, random_state=42):
    """Entrena el modelo de Regresión Logística"""
    # Escalar features numéricas
    scaler = StandardScaler()
    scaled_features = ['tenure', 'monthly_charges', 'total_charges']
    
    # Crear copia para no modificar X original
    X_scaled = X.copy()
    X_scaled[scaled_features] = scaler.fit_transform(X[scaled_features])
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Entrenar modelo
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model, scaler, X_train.columns

def prepare_customer_data(input_data, feature_columns, scaler):
    """Prepara los datos del cliente para la predicción usando el mismo pipeline de preprocesamiento"""
    # Crear DataFrame con todas las columnas esperadas, inicializadas en 0
    customer_df = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Asignar valores numéricos directos (antes de escalar) usando .loc para evitar problemas de tipo
    customer_df.loc[0, 'tenure'] = float(input_data['tenure'])
    customer_df.loc[0, 'monthly_charges'] = float(input_data['monthly_charges'])
    customer_df.loc[0, 'total_charges'] = float(input_data['total_charges'])
    customer_df.loc[0, 'senior_citizen'] = float(input_data.get('senior_citizen', 0))
    
    # Label encoded features (ya vienen como 0 o 1)
    customer_df.loc[0, 'partner'] = float(input_data.get('partner', 0))
    customer_df.loc[0, 'dependents'] = float(input_data.get('dependents', 0))
    customer_df.loc[0, 'phone_service'] = float(input_data.get('phone_service', 1))
    customer_df.loc[0, 'paperless_billing'] = float(input_data.get('paperless_billing', 0))
    
    # Contract (one-hot encoding, drop_first=True significa que Month-to-month es la referencia)
    contract = input_data.get('contract', 'Month-to-month')
    if contract == 'One year':
        # Buscar columna que contenga contract y one year (flexible en mayúsculas/minúsculas)
        for col in customer_df.columns:
            if 'contract' in col and 'one year' in col.lower():
                customer_df.loc[0, col] = 1.0
                break
    elif contract == 'Two year':
        for col in customer_df.columns:
            if 'contract' in col and 'two year' in col.lower():
                customer_df.loc[0, col] = 1.0
                break
    
    # Payment method (one-hot encoding, drop_first=True significa que Electronic check es la referencia)
    payment = input_data.get('payment_method', 'Electronic check')
    for col in customer_df.columns:
        if 'payment_method' in col:
            if 'Bank transfer' in payment and 'Bank transfer' in col:
                customer_df.loc[0, col] = 1.0
                break
            elif 'Credit card' in payment and 'Credit card' in col:
                customer_df.loc[0, col] = 1.0
                break
            elif 'Mailed check' in payment and 'Mailed check' in col:
                customer_df.loc[0, col] = 1.0
                break
    
    # Internet service (one-hot encoding, drop_first=True significa que Fiber optic es la referencia)
    internet = input_data.get('internet_service', 'Fiber optic')
    if internet == 'DSL':
        for col in customer_df.columns:
            if 'internet_service' in col and 'DSL' in col:
                customer_df.loc[0, col] = 1.0
                break
    elif internet == 'No':
        for col in customer_df.columns:
            if 'internet_service' in col and '_No' in col:
                customer_df.loc[0, col] = 1.0
                break
    
    # Servicios adicionales (one-hot encoding, drop_first=True significa que No es la referencia)
    services_map = {
        'multiple_lines': input_data.get('multiple_lines', 'No'),
        'online_security': input_data.get('online_security', 'No'),
        'online_backup': input_data.get('online_backup', 'No'),
        'device_protection': input_data.get('device_protection', 'No'),
        'tech_support': input_data.get('tech_support', 'No'),
        'streaming_tv': input_data.get('streaming_tv', 'No'),
        'streaming_movies': input_data.get('streaming_movies', 'No')
    }
    
    for service, value in services_map.items():
        if value == 'Yes':
            # Buscar columna que contenga el nombre del servicio y "_Yes"
            for col in customer_df.columns:
                if service in col and '_Yes' in col:
                    customer_df.loc[0, col] = 1.0
                    break
    
    # Convertir todas las columnas a float64 explícitamente y llenar cualquier NaN con 0
    customer_df = customer_df.astype(float).fillna(0.0)
    
    # Escalar características numéricas
    scaled_features = ['tenure', 'monthly_charges', 'total_charges']
    customer_df_scaled = customer_df.copy()
    customer_df_scaled[scaled_features] = scaler.transform(customer_df[scaled_features])
    
    # Asegurar que no haya NaN después del escalado
    customer_df_scaled = customer_df_scaled.fillna(0.0)
    
    return customer_df_scaled

def get_feature_importance(model, feature_names):
    """Obtiene la importancia de las características usando los coeficientes del modelo"""
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False).head(10)
    
    return feature_importance

def get_feature_description(feature_name):
    """Obtiene la descripción de por qué una variable tiene impacto basado en el análisis del notebook"""
    descriptions = {
        'tenure': "**Factor de Lealtad por Tiempo:** El análisis reveló que la mayoría del churn ocurre dentro de los primeros 10-12 meses de servicio. Esto sugiere que la fase de 'Onboarding' y el primer año de experiencia están fallando en asegurar el valor del cliente. Por otro lado, los clientes que permanecen más allá de los 24 meses muestran una probabilidad significativamente mayor de retención a largo plazo. Su mediana de tenure es casi 4 veces mayor que aquellos que churnearon.",
        
        'contract_Two year': "**Estabilidad Contractual:** Los contratos de dos años proporcionan el mayor nivel de compromiso y reducen significativamente el riesgo de churn. Este tipo de contrato actúa como un 'ancla' que hace más difícil para los clientes cambiar de proveedor, especialmente cuando se combina con otros servicios.",
        
        'contract_One year': "**Compromiso Intermedio:** Los contratos de un año ofrecen un balance entre flexibilidad y estabilidad. Mientras que proporcionan más seguridad que los contratos mes-a-mes, aún permiten cierta movilidad al cliente después del primer año.",
        
        'internet_service_Fiber optic': "**Alto Riesgo por Alto Valor:** A pesar de ser el servicio de mayor velocidad, los usuarios de Fiber Optic son los más propensos a dejar la empresa. El volumen de churn en este segmento es significativamente mayor que en cualquier otra categoría. Esto forma parte del 'Premium Friction Triangle': se vende un producto complejo y costoso ($80-$100+ rango) pero el churn se concentra casi exclusivamente entre clientes que carecen de Soporte Técnico.",
        
        'monthly_charges': "**Sensibilidad al Precio:** Los clientes que se van son consistentemente cobrados $20 a $30 más por mes que aquellos que se quedan, independientemente de cuánto tiempo hayan estado con la empresa. El churn no es solo sobre mal servicio; es una reacción económica racional a ser sobrecobrados. La empresa está efectivamente 'excluyendo por precio' a una cuarta parte de su base de clientes. La lealtad no se puede comprar si la brecha de precio es demasiado amplia.",
        
        'total_charges': "**Valor Acumulado:** El análisis muestra que el negocio está fallando en convertir clientes de alto ingreso mensual en valor total a largo plazo. La empresa está atrayendo clientes de alto gasto pero perdiéndolos antes de que alcancen su segundo año, resultando en pérdida de valor de vida del cliente.",
        
        'payment_method_Electronic check': "**Punto de Fricción en Pagos:** La mayoría de los clientes que churnearon pagaron mediante cheque electrónico. Esto ocurre porque los cheques electrónicos toman 3-5 días hábiles para liquidar, lo que lleva a problemas tanto para el cliente como para el vendedor. Si el cliente envía el pago justo antes del tiempo límite, cuando el vendedor recibe el dinero, el cliente ya estará en mora, resultando en posibles interrupciones del servicio.",
        
        'phone_service': "**Servicio Base:** Aunque es un servicio fundamental, la falta de servicios adicionales combinada con otros factores de riesgo puede aumentar la probabilidad de churn.",
        
        'online_security_Yes': "**Seguridad como Ancla:** La seguridad en línea es un factor crítico de retención. Cuando un cliente opta por la seguridad en línea, está señalando un compromiso a largo plazo con la empresa. Los clientes sin protección enfrentan el costo completo de reparación o reemplazo cuando experimentan fallos de hardware, lo que puede convertirse en el detonante final para cancelar el servicio.",
        
        'streaming_tv_Yes': "**Servicios Adicionales:** Los servicios de streaming pueden actuar como factores de retención adicionales, creando más 'anclas' que mantienen a los clientes comprometidos con el servicio completo.",
        
        'tech_support_Yes': "**Gap de Soporte Crítico:** El análisis reveló que la falta de soporte es un punto de inflexión para un cliente. Es importante tener servicio post-venta y soporte cuando el cliente lo necesita. El 'Premium Friction Triangle' muestra que los clientes de Fiber Optic que carecen de Soporte Técnico tienen el mayor riesgo de churn."
    }
    
    # Buscar descripción coincidente (case insensitive, parcial)
    for key, desc in descriptions.items():
        if key.lower() in feature_name.lower():
            return desc
    
    # Descripción genérica si no hay match
    return "Esta variable contribuye a la predicción del modelo de churn según su coeficiente de regresión logística."

# ============================================================================
# SISTEMA DE IDIOMAS
# ============================================================================

def get_translations(lang='es'):
    """Retorna diccionario de traducciones según el idioma seleccionado"""
    translations = {
        'es': {
            'title': '📡 Customer Churn Risk Simulator',
            'subtitle': 'Simulador de Riesgo de Fuga para Customer Success',
            'description': """
            **¿Qué hace esta aplicación?**
            
            Esta herramienta utiliza un modelo de Machine Learning (Regresión Logística) entrenado con datos históricos de clientes 
            de telecomunicaciones para predecir la probabilidad de que un cliente abandone el servicio (churn). 
            
            **¿Cómo funciona?**
            
            Simplemente ajusta los parámetros del cliente en el panel lateral izquierdo (tenure, cargos mensuales, tipo de contrato, etc.) 
            y la aplicación calculará en tiempo real la probabilidad de churn. El modelo analiza patrones como el tiempo de permanencia, 
            cargos mensuales, tipo de contrato y otros factores identificados en el análisis de datos.
            
            **¿Para qué sirve?**
            
            Esta herramienta está diseñada para el equipo de Customer Success, permitiéndoles:
            - Identificar clientes en riesgo antes de que churnen
            - Simular escenarios para entender qué factores afectan más el riesgo
            - Tomar decisiones informadas sobre estrategias de retención
            - Aplicar la "Estrategia de Océano Azul" a clientes de alto riesgo
            """,
            'lang_button': '🌐 Idioma / Language',
            'load_model': 'Cargando modelo...',
            'error_load': 'No se pudo cargar el dataset. Por favor verifica que el archivo existe.',
            'sidebar_title': '⚙️ Parámetros del Cliente',
            'variables_main': '📊 Variables Principales',
            'variables_additional': '🔧 Configuración Adicional',
            'tenure_label': 'Tenure (meses)',
            'tenure_help': 'Tiempo que el cliente ha estado con la empresa',
            'monthly_charges_label': 'Monthly Charges ($)',
            'monthly_charges_help': 'Cargo mensual del cliente',
            'total_charges_label': 'Total Charges ($)',
            'total_charges_help': 'Cargo total acumulado del cliente',
            'contract_label': 'Tipo de Contrato',
            'contract_help': 'Tipo de contrato del cliente',
            'payment_label': 'Método de Pago',
            'internet_label': 'Servicio de Internet',
            'tech_support_label': 'Soporte Técnico',
            'security_label': 'Seguridad en Línea',
            'partner_label': 'Tiene Pareja',
            'dependents_label': 'Tiene Dependientes',
            'churn_prob': '📈 Probabilidad de Churn',
            'high_risk_alert': '🚨 **ALTO RIESGO: Aplicar Estrategia de Océano Azul inmediatamente**',
            'high_risk_rec': """**Recomendaciones Inmediatas:**
            - Contactar al cliente para ofrecer soporte técnico prioritario
            - Evaluar un paquete de valor agregado (Gamer, Teleworker, etc.)
            - Revisar el método de pago y ofrecer alternativas
            - Considerar incentivos de fidelización""",
            'moderate_risk': '✅ Riesgo moderado. Probabilidad de churn: {:.1f}%',
            'metrics': '📊 Métricas',
            'prob_label': 'Probabilidad de Churn',
            'prediction_label': 'Predicción',
            'pred_high': '⚠️ Riesgo Alto',
            'pred_low': '✅ Bajo Riesgo',
            'explainability': '🔍 Explicabilidad del Modelo',
            'explainability_desc': '**Factores que más influyen en esta predicción:**',
            'explainability_impact': '📝 Explicación del Impacto de las Variables Principales',
            'view_coeff': '📋 Ver todos los coeficientes',
            'var_col': 'Variable',
            'coeff_col': 'Coeficiente',
            'abs_coeff_col': '|Coeficiente|',
            'insights': '🎯 Insights Estratégicos para Decision Makers',
            'findings': '📊 Hallazgos Críticos del Análisis',
            'recommendations': '💡 Recomendaciones Estratégicas (Blue Ocean)',
            'risk_factors': '⚠️ Factores de Riesgo Identificados',
            'model_metrics': '📈 Métricas del Modelo',
            'footer_title': '📚 Fuente de Datos y Modelo',
            'footer_text': 'Esta aplicación está basada en el análisis y modelo desarrollado en el repositorio:',
            'github_repo': 'https://github.com/andresfelipe0711/churn-prediction'
        },
        'en': {
            'title': '📡 Customer Churn Risk Simulator',
            'subtitle': 'Churn Risk Simulator for Customer Success',
            'description': """
            **What does this application do?**
            
            This tool uses a Machine Learning model (Logistic Regression) trained with historical telecommunications customer data 
            to predict the probability that a customer will abandon the service (churn).
            
            **How does it work?**
            
            Simply adjust the customer parameters in the left sidebar panel (tenure, monthly charges, contract type, etc.) 
            and the application will calculate the churn probability in real-time. The model analyzes patterns such as tenure, 
            monthly charges, contract type, and other factors identified in the data analysis.
            
            **What is it for?**
            
            This tool is designed for the Customer Success team, allowing them to:
            - Identify at-risk customers before they churn
            - Simulate scenarios to understand which factors affect risk the most
            - Make informed decisions about retention strategies
            - Apply "Blue Ocean Strategy" to high-risk customers
            """,
            'lang_button': '🌐 Language / Idioma',
            'load_model': 'Loading model...',
            'error_load': 'Could not load dataset. Please verify that the file exists.',
            'sidebar_title': '⚙️ Customer Parameters',
            'variables_main': '📊 Main Variables',
            'variables_additional': '🔧 Additional Configuration',
            'tenure_label': 'Tenure (months)',
            'tenure_help': 'Time the customer has been with the company',
            'monthly_charges_label': 'Monthly Charges ($)',
            'monthly_charges_help': 'Customer monthly charge',
            'total_charges_label': 'Total Charges ($)',
            'total_charges_help': 'Customer total accumulated charge',
            'contract_label': 'Contract Type',
            'contract_help': 'Customer contract type',
            'payment_label': 'Payment Method',
            'internet_label': 'Internet Service',
            'tech_support_label': 'Tech Support',
            'security_label': 'Online Security',
            'partner_label': 'Has Partner',
            'dependents_label': 'Has Dependents',
            'churn_prob': '📈 Churn Probability',
            'high_risk_alert': '🚨 **HIGH RISK: Apply Blue Ocean Strategy immediately**',
            'high_risk_rec': """**Immediate Recommendations:**
            - Contact the customer to offer priority technical support
            - Evaluate a value-added package (Gamer, Teleworker, etc.)
            - Review payment method and offer alternatives
            - Consider loyalty incentives""",
            'moderate_risk': '✅ Moderate risk. Churn probability: {:.1f}%',
            'metrics': '📊 Metrics',
            'prob_label': 'Churn Probability',
            'prediction_label': 'Prediction',
            'pred_high': '⚠️ High Risk',
            'pred_low': '✅ Low Risk',
            'explainability': '🔍 Model Explainability',
            'explainability_desc': '**Factors that most influence this prediction:**',
            'explainability_impact': '📝 Explanation of Main Variables Impact',
            'view_coeff': '📋 View all coefficients',
            'var_col': 'Variable',
            'coeff_col': 'Coefficient',
            'abs_coeff_col': '|Coefficient|',
            'insights': '🎯 Strategic Insights for Decision Makers',
            'findings': '📊 Critical Findings from Analysis',
            'recommendations': '💡 Strategic Recommendations (Blue Ocean)',
            'risk_factors': '⚠️ Identified Risk Factors',
            'model_metrics': '📈 Model Metrics',
            'footer_title': '📚 Data and Model Source',
            'footer_text': 'This application is based on the analysis and model developed in the repository:',
            'github_repo': 'https://github.com/andresfelipe0711/churn-prediction'
        }
    }
    return translations.get(lang, translations['es'])

# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    # Inicializar idioma en session_state
    if 'language' not in st.session_state:
        st.session_state.language = 'es'
    
    # Botón de cambio de idioma en la parte superior izquierda
    col_lang, col_title = st.columns([1, 10])
    with col_lang:
        if st.button('🌐 ES/EN', help='Cambiar idioma / Change language'):
            st.session_state.language = 'en' if st.session_state.language == 'es' else 'es'
            st.rerun()
    
    with col_title:
        t = get_translations(st.session_state.language)
        st.title(t['title'])
        st.markdown(f"### {t['subtitle']}")
    
    # Descripción de la aplicación
    st.markdown("---")
    st.markdown(t['description'])
    st.markdown("---")
    
    # Cargar datos y entrenar modelo
    t = get_translations(st.session_state.language)
    with st.spinner(t['load_model']):
        df = load_data()
        if df is None:
            st.error(t['error_load'])
            return
        
        # Preprocesar datos
        churn_dummies = preprocess_data(df)
        X, y = prepare_features(churn_dummies)
        
        # Entrenar modelo
        model, scaler, feature_columns = train_model(X, y)
    
    # ========================================================================
    # SIDEBAR - CONTROLES DE ENTRADA
    # ========================================================================
    
    st.sidebar.header(t['sidebar_title'])
    st.sidebar.markdown("---")
    
    # Variables principales (mencionadas por el usuario)
    st.sidebar.subheader(t['variables_main'])
    
    tenure = st.sidebar.slider(
        t['tenure_label'],
        min_value=0,
        max_value=72,
        value=12,
        step=1,
        help=t['tenure_help']
    )
    
    monthly_charges = st.sidebar.slider(
        t['monthly_charges_label'],
        min_value=18.0,
        max_value=120.0,
        value=70.0,
        step=1.0,
        help=t['monthly_charges_help']
    )
    
    total_charges = st.sidebar.slider(
        t['total_charges_label'],
        min_value=0.0,
        max_value=9000.0,
        value=840.0,
        step=50.0,
        help=t['total_charges_help']
    )
    
    contract = st.sidebar.selectbox(
        t['contract_label'],
        options=['Month-to-month', 'One year', 'Two year'],
        index=0,
        help=t['contract_help']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(t['variables_additional'])
    
    # Variables adicionales importantes
    payment_method = st.sidebar.selectbox(
        t['payment_label'],
        options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        index=0
    )
    
    internet_service = st.sidebar.selectbox(
        t['internet_label'],
        options=['Fiber optic', 'DSL', 'No'],
        index=0
    )
    
    tech_support = st.sidebar.selectbox(
        t['tech_support_label'],
        options=['Yes', 'No'],
        index=1
    )
    
    online_security = st.sidebar.selectbox(
        t['security_label'],
        options=['Yes', 'No'],
        index=1
    )
    
    partner = st.sidebar.selectbox(
        t['partner_label'],
        options=['Yes', 'No'],
        index=1
    )
    
    dependents = st.sidebar.selectbox(
        t['dependents_label'],
        options=['Yes', 'No'],
        index=1
    )
    
    # Preparar datos de entrada
    input_data = {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'tech_support': tech_support,
        'online_security': online_security,
        'partner': 1 if partner == 'Yes' else 0,
        'dependents': 1 if dependents == 'Yes' else 0,
        'phone_service': 1,
        'paperless_billing': 0,
        'multiple_lines': 'No',
        'online_backup': 'No',
        'device_protection': 'No',
        'streaming_tv': 'No',
        'streaming_movies': 'No',
        'senior_citizen': 0
    }
    
    # ========================================================================
    # PREDICCIÓN
    # ========================================================================
    
    try:
        customer_df = prepare_customer_data(input_data, feature_columns, scaler)
        
        # Validación final: asegurar que no hay NaN
        if customer_df.isnull().any().any():
            st.warning("Advertencia: Se detectaron valores NaN. Rellenando con 0.")
            customer_df = customer_df.fillna(0.0)
        
        # Asegurar que el orden de columnas coincide con el modelo
        customer_df = customer_df[feature_columns]
        
        # Verificar que todos los valores sean numéricos finitos
        if not np.isfinite(customer_df.values).all():
            st.warning("Advertencia: Se detectaron valores infinitos. Rellenando con 0.")
            customer_df = customer_df.replace([np.inf, -np.inf], 0.0)
        
        churn_probability = model.predict_proba(customer_df)[0][1]
        churn_prediction = model.predict(customer_df)[0]
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.code(str(e))
        # Mostrar información de debug
        with st.expander("🔍 Información de Debug"):
            st.write(f"Columnas esperadas: {len(feature_columns)}")
            st.write(f"Primeras 10 columnas: {list(feature_columns[:10])}")
            try:
                st.write(f"Columnas en customer_df: {list(customer_df.columns[:10])}")
                st.write(f"Shape del DataFrame: {customer_df.shape}")
                st.write(f"Valores NaN: {customer_df.isnull().sum().sum()}")
            except:
                st.write("No se pudo crear customer_df")
        return
    
    # ========================================================================
    # VISUALIZACIÓN DE RESULTADOS
    # ========================================================================
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(t['churn_prob'])
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = churn_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de Churn (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=300
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Alerta de alto riesgo
        if churn_probability > 0.5:
            st.error(t['high_risk_alert'])
            st.warning(t['high_risk_rec'])
        else:
            st.success(t['moderate_risk'].format(churn_probability*100))
    
    with col2:
        st.subheader(t['metrics'])
        st.metric(
            label=t['prob_label'],
            value=f"{churn_probability*100:.1f}%",
            delta=f"{churn_probability*100 - 50:.1f}%" if churn_probability > 0.5 else f"{50 - churn_probability*100:.1f}%"
        )
        st.metric(
            label=t['prediction_label'],
            value=t['pred_high'] if churn_prediction == 1 else t['pred_low']
        )
        st.metric(
            label="Tenure",
            value=f"{tenure} meses" if st.session_state.language == 'es' else f"{tenure} months"
        )
    
    # ========================================================================
    # EXPLICABILIDAD - FEATURE IMPORTANCE
    # ========================================================================
    
    st.markdown("---")
    st.subheader(t['explainability'])
    st.markdown(t['explainability_desc'])
    
    # Obtener importancia de características
    feature_importance = get_feature_importance(model, feature_columns)
    
    # Crear gráfico de barras
    fig_bar = px.bar(
        feature_importance,
        x='abs_coefficient',
        y='feature',
        orientation='h',
        title="Top 10 Factores de Influencia (Coeficientes del Modelo)",
        labels={'abs_coefficient': 'Impacto (|Coeficiente|)', 'feature': 'Variable'},
        color='abs_coefficient',
        color_continuous_scale='Reds'
    )
    
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
        height=400
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Descripciones de las variables más importantes
    st.markdown(f"#### {t['explainability_impact']}")
    
    # Mostrar descripciones para las top 5 variables
    top_features = feature_importance.head(5)
    for idx, row in top_features.iterrows():
        feature_name = row['feature']
        description = get_feature_description(feature_name)
        
        with st.expander(f"🔍 {feature_name} (Impacto: {row['abs_coefficient']:.3f})"):
            st.markdown(description)
    
    # Tabla de coeficientes (sin styling que requiere matplotlib)
    with st.expander(t['view_coeff']):
        # Crear copia para formatear
        display_df = feature_importance[['feature', 'coefficient', 'abs_coefficient']].copy()
        display_df['coefficient'] = display_df['coefficient'].apply(lambda x: f"{x:.4f}")
        display_df['abs_coefficient'] = display_df['abs_coefficient'].apply(lambda x: f"{x:.4f}")
        display_df.columns = [t['var_col'], t['coeff_col'], t['abs_coeff_col']]
        st.dataframe(display_df, use_container_width=True)
    
    # ========================================================================
    # INFORMACIÓN PARA DECISION MAKERS
    # ========================================================================
    
    st.markdown("---")
    st.markdown(f"### {t['insights']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {t['findings']}")
        st.markdown("""
        **1. El Triángulo de Fricción Premium**
        - Los clientes de Fiber Optic ($80-$100+/mes) tienen el mayor riesgo de churn
        - El churn se concentra en clientes sin Soporte Técnico
        - **Acción:** Implementar soporte técnico obligatorio para segmentos premium
        
        **2. El Problema del Primer Año**
        - El 27% de los clientes churnan en el último mes
        - La mayoría del churn ocurre en los primeros 10-12 meses
        - **Acción:** Mejorar el proceso de onboarding y crear incentivos de retención temprana
        
        **3. Brecha de Precio**
        - Los clientes que churnearon pagaban $20-$30 más/mes que los que se quedan
        - La lealtad no se puede comprar si la brecha de precio es demasiado amplia
        - **Acción:** Implementar estrategia de "Océano Azul" - agregar valor en lugar de competir en precio
        """)
    
    with col2:
        st.markdown(f"#### {t['recommendations']}")
        st.markdown("""
        **1. Soporte Técnico 24/7 con IA**
        - Implementar chatbot de WhatsApp entrenado con guías internas
        - Priorizar clientes de Fiber Optic sin soporte técnico
        - **ROI Esperado:** Reducción del 15% en churn del segmento premium
        
        **2. Paquetes de Valor Especializados**
        - **Gamer Package:** Priorización de tráfico para servidores de juegos
        - **Teleworker Package:** Línea de respaldo 4G/5G automática
        - **Residential Cybersecurity:** Protección integrada contra malware a nivel router
        - Justifica precios premium sin reducirlos
        
        **3. Programas de Fidelización**
        - Actualizaciones gratuitas de router para clientes 2+ años
        - "Health check" anual de conectividad profesional
        - Créditos para plataformas de streaming y antivirus
        - Bonos por referidos
        """)
    
    st.markdown("---")
    st.markdown(f"#### {t['risk_factors']}")
    
    risk_factors = st.container()
    with risk_factors:
        st.markdown("""
        **Alta Prioridad de Intervención:**
        - Clientes con contratos mes-a-mes + Fiber Optic + Sin Tech Support
        - Clientes con más de $80/mes en cargos mensuales
        - Clientes en su primer año de servicio con cargos altos
        
        **Método de Pago de Riesgo:**
        - Cheque electrónico tiene mayor correlación con churn
        - **Recomendación:** Ofrecer incentivos para migrar a pagos automáticos
        
        **Impacto Financiero:**
        - El modelo identifica correctamente el 57% de los clientes que churnearán (Recall: 0.57)
        - Aplicar estrategias preventivas puede reducir significativamente la pérdida de ingresos
        """)
    
    st.markdown("---")
    st.markdown(f"#### {t['model_metrics']}")
    if st.session_state.language == 'es':
        st.info("""
        - **Modelo:** Regresión Logística (seleccionado sobre Random Forest)
        - **Recall:** 0.57 (identifica correctamente 57% de clientes que churnearán)
        - **Precisión:** 0.65 para la clase de churn
        - **Variables Escaladas:** Tenure, Monthly Charges, Total Charges
        - **Interpretación:** Los coeficientes positivos aumentan la probabilidad de churn, los negativos la disminuyen
        """)
    else:
        st.info("""
        - **Model:** Logistic Regression (selected over Random Forest)
        - **Recall:** 0.57 (correctly identifies 57% of customers who will churn)
        - **Precision:** 0.65 for the churn class
        - **Scaled Variables:** Tenure, Monthly Charges, Total Charges
        - **Interpretation:** Positive coefficients increase churn probability, negative ones decrease it
        """)
    
    # ========================================================================
    # FOOTER - CITACIÓN DEL REPOSITORIO
    # ========================================================================
    
    st.markdown("---")
    st.markdown(f"### {t['footer_title']}")
    st.markdown(f"{t['footer_text']}")
    st.markdown(f"[🔗 {t['github_repo']}]({t['github_repo']})")
    st.markdown("")

if __name__ == "__main__":
    main()
