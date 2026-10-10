import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. CARGA DE DATOS LOCALES
# ==========================================
archivos = [
    "../audi.csv",
    "../bmw.csv",
    "../cclass.csv",
    "../focus.csv",
    "../ford.csv",
    "../hyundi.csv",
    "../merc.csv",
    "../skoda.csv",
    "../toyota.csv",
    "../vauxhall.csv",
    "../vw.csv"
]

dataframes = []
for archivo in archivos:
    df_temp = pd.read_csv(archivo)
    if 'tax(£)' in df_temp.columns:
        df_temp.rename(columns={'tax(£)': 'tax'}, inplace=True)
    
    nombre_archivo = os.path.basename(archivo)
    marca = nombre_archivo.replace('.csv', '').upper()
    df_temp['brand'] = marca
    dataframes.append(df_temp)

dataset_limpio = pd.concat(dataframes, ignore_index=True).dropna(axis=0)

# ==========================================
# 2. PREPARACIÓN AUTOMÁTICA CON PIPELINE
# ==========================================
y = dataset_limpio['price']
X = dataset_limpio.drop('price', axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# CORRECCIÓN: Selección robusta de columnas de texto/categóricas
columnas_texto = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

# ==========================================
# 3. CREACIÓN DEL PIPELINE
# ==========================================
procesador_texto = ColumnTransformer(
    transformers=[
        ('texto_a_numeros', OneHotEncoder(handle_unknown='ignore'), columnas_texto)
    ],
    remainder='passthrough'
)

mi_pipeline = Pipeline(steps=[
    ('preprocesamiento', procesador_texto),
    ('modelo', XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=1))
])

# ==========================================
# 4. ENTRENAMIENTO Y PREDICCIÓN
# ==========================================
mi_pipeline.fit(train_X, train_y)

val_predictions = mi_pipeline.predict(val_X)

mae = mean_absolute_error(val_y, val_predictions)
print(f"Error Absoluto Medio (MAE) general: {mae:,.2f}")
print("-" * 60)

# ==========================================
# 5. MOSTRAR LOS 5 PRIMEROS REGISTROS
# ==========================================
resultados = pd.DataFrame({
    'Valor Real': val_y.head(5).values,
    'Predicción': val_predictions[:5],
    'Error Absoluto': abs(val_y.head(5).values - val_predictions[:5])
})

resultados = resultados.round(2)

print("Comparativa de los 5 primeros registros:")
print(resultados)
