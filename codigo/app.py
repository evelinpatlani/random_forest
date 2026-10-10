import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configuración de la página web
st.title("🚗 Predictor de Precios de Autos (XGBoost)")
st.write("Esta aplicación utiliza un modelo de Machine Learning (XGBoost) entrenado con datos de múltiples marcas de vehículos para estimar su precio.")

# Carga de datos optimizada con caché
@st.cache_resource
def cargar_y_entrenar_modelo():
    archivos = [
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/audi.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/bmw.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/cclass.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/focus.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/ford.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/hyundi.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/merc.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/skoda.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/toyota.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/vauxhall.csv",
        "https://raw.githubusercontent.com/evelinpatlani/random_forest/refs/heads/main/vw.csv"
    ]

    dataframes = []
    for url in archivos:
        df_temp = pd.read_csv(url)
        if 'tax(£)' in df_temp.columns:
            df_temp.rename(columns={'tax(£)': 'tax'}, inplace=True)
        marca = url.split('/')[-1].replace('.csv', '').upper()
        df_temp['brand'] = marca
        dataframes.append(df_temp)

    dataset_limpio = pd.concat(dataframes, ignore_index=True).dropna(axis=0)

    y = dataset_limpio['price']
    X = dataset_limpio.drop('price', axis=1)

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

    # Identificar de forma robusta las columnas de texto
    columnas_texto = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    procesador_texto = ColumnTransformer(
        transformers=[
            ('texto_a_numeros', OneHotEncoder(handle_unknown='ignore'), columnas_texto)
        ],
        remainder='passthrough'
    )

    mi_pipeline = Pipeline(steps=[
        ('preprocesamiento', procesador_texto),
        ('modelo', XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=-1, random_state=1))
    ])

    mi_pipeline.fit(train_X, train_y)
    return mi_pipeline, X, columnas_texto

with st.spinner("Entrenando el modelo en segundo plano, por favor espera un momento..."):
    pipeline, X_referencia, columnas_texto = cargar_y_entrenar_modelo()

st.success("¡Modelo listo!")

# Interfaz de usuario para ingresar datos de prueba
st.subheader("Ingresa las características del vehículo:")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Año del auto", min_value=1990, max_value=2026, value=2018)
    mileage = st.number_input("Millaje (Mileage)", min_value=0, max_value=300000, value=25000)
    tax = st.number_input("Impuesto (Tax)", min_value=0, max_value=600, value=145)

with col2:
    mpg = st.number_input("Rendimiento (MPG)", min_value=0.0, max_value=200.0, value=55.4)
    engineSize = st.number_input("Tamaño del motor (Engine Size)", min_value=0.0, max_value=6.0, value=2.0)
    brand = st.selectbox("Marca", options=X_referencia['brand'].unique())

# Botón para predecir
if st.button("Calcular Precio Estimado"):
    # Creamos un DataFrame básico con lo que ingresó el usuario
    nuevo_auto = pd.DataFrame({
        'year': [year],
        'mileage': [mileage],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engineSize],
        'brand': [brand]
    })
    
    # Rellenar inteligentemente las columnas faltantes según su tipo original
    for col in X_referencia.columns:
        if col not in nuevo_auto.columns:
            if col in columnas_texto:
                nuevo_auto[col] = "Desconocido"  # Relleno seguro para texto
            else:
                nuevo_auto[col] = 0              # Relleno seguro para números

    prediccion = pipeline.predict(nuevo_auto)[0]
    st.markdown(f"### Precio estimado: **£{prediccion:,.2f}**")
