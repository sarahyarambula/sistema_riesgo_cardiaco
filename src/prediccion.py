# Motor de predicción del sistema de riesgo cardíaco

import os
import joblib
import pandas as pd
import shap


# Obtener la ruta principal del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas de los archivos del modelo y escalador
RUTA_MODELO = os.path.join(BASE_DIR, "modelos", "modelo_final.pkl")
RUTA_SCALER = os.path.join(BASE_DIR, "modelos", "scaler.pkl")


# Cargar modelo y escalador
modelo = joblib.load(RUTA_MODELO)
scaler = joblib.load(RUTA_SCALER)

explainer = shap.TreeExplainer(modelo)


# Orden exacto de las variables utilizadas por el modelo
COLUMNAS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


def predecir_riesgo(datos_paciente):
    """
    Recibe los datos de un paciente y devuelve la predicción.
    
    0 = Sin enfermedad
    1 = Con enfermedad
    """

    # Convertir los datos en DataFrame
    paciente = pd.DataFrame([datos_paciente])

    # Asegurar el orden correcto de las variables
    paciente = paciente[COLUMNAS]

    # Aplicar el escalado
    paciente_scaled = scaler.transform(paciente)

    # Realizar predicción
    prediccion = modelo.predict(paciente_scaled)[0]

    # Obtener probabilidad
    probabilidades = modelo.predict_proba(paciente_scaled)[0]

    probabilidad_enfermedad = probabilidades[1]

    valores_shap = explainer.shap_values(paciente_scaled)

    # Interpretar resultado
    if prediccion == 1:
        resultado = "Con enfermedad"
    else:
        resultado = "Sin enfermedad"

    return resultado, probabilidad_enfermedad, valores_shap