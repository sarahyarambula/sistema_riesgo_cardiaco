# Sistema de Predicción de Riesgo Cardíaco

## Descripción

El Sistema de Predicción de Riesgo Cardíaco es un prototipo de aplicación web desarrollado con Python y Machine Learning, cuyo objetivo es realizar una clasificación estimada de riesgo cardíaco a partir de diferentes características de un paciente.

El sistema recibe datos relacionados con características demográficas, presión arterial, colesterol, frecuencia cardíaca, electrocardiograma y otros indicadores presentes en el conjunto de datos utilizado para el entrenamiento.

La aplicación utiliza un modelo de Machine Learning para generar una clasificación de:

- Sin enfermedad
- Con enfermedad

Además, muestra una probabilidad estimada por el modelo para la clase "Con enfermedad".

> **Aviso:** Este proyecto tiene fines educativos y académicos. Los resultados generados por el sistema no constituyen un diagnóstico médico ni sustituyen la valoración de un profesional de la salud.

---

## Objetivo del proyecto

Desarrollar un sistema basado en Machine Learning capaz de procesar datos de pacientes y generar una predicción estimada relacionada con la presencia o ausencia de enfermedad cardíaca.

El proyecto integra conceptos de:

- Inteligencia Artificial
- Machine Learning
- Procesamiento de datos
- Análisis exploratorio
- Modelos de clasificación
- Python
- Desarrollo de aplicaciones web
- Interfaz mediante Streamlit

---

## Dataset

Para el desarrollo del proyecto se utilizó el conjunto de datos **Heart Disease** de la **UCI Machine Learning Repository**.

El conjunto de datos original contiene información relacionada con enfermedades cardíacas. Para este proyecto se realizó un proceso de limpieza y preparación de los datos, obteniendo un conjunto de **297 registros utilizables**.

Las características utilizadas por el modelo son:

- `age` - Edad
- `sex` - Sexo
- `cp` - Tipo de dolor de pecho
- `trestbps` - Presión arterial en reposo
- `chol` - Colesterol
- `fbs` - Azúcar en sangre en ayunas
- `restecg` - Electrocardiograma en reposo
- `thalach` - Frecuencia cardíaca máxima
- `exang` - Angina inducida por ejercicio
- `oldpeak` - Depresión del segmento ST
- `slope` - Pendiente del segmento ST
- `ca` - Número de vasos principales
- `thal` - Variable relacionada con el resultado de la prueba de talio

La variable objetivo utilizada para la clasificación es `target`.

---

## Proceso de Machine Learning

El desarrollo del modelo siguió las siguientes etapas:

1. Carga del conjunto de datos.
2. Limpieza y preparación de los datos.
3. Análisis exploratorio.
4. Análisis de correlaciones.
5. Separación de variables predictoras y variable objetivo.
6. División de los datos en entrenamiento y prueba.
7. Estandarización de las características.
8. Entrenamiento de diferentes modelos de clasificación.
9. Evaluación mediante métricas de rendimiento.
10. Comparación de modelos.
11. Selección del modelo final.
12. Integración del modelo en una aplicación web.

---

## Modelos utilizados

Durante el desarrollo se evaluaron dos modelos principales:

### Regresión Logística

La Regresión Logística obtuvo:

- Accuracy: **83.33%**
- F1 de la clase 1: **0.815**
- ROC AUC: **0.950**

### Random Forest

El modelo Random Forest obtuvo:

- Accuracy: **86.67%**
- F1 de la clase 1: **0.852**
- ROC AUC: **0.941**

Después de comparar los resultados, se seleccionó **Random Forest** como modelo final para la aplicación, debido a que obtuvo un mejor accuracy y un mejor F1 para la clase positiva en la prueba realizada.

> La selección corresponde al conjunto de pruebas realizado durante este proyecto y no representa una validación clínica del modelo.

---

## Optimización del modelo

También se realizó una búsqueda de hiperparámetros utilizando `GridSearchCV` sobre el modelo Random Forest.

La configuración seleccionada durante la búsqueda fue:

- `n_estimators`: 300
- `max_depth`: 5
- `min_samples_split`: 2
- `min_samples_leaf`: 2

Sin embargo, el modelo optimizado obtuvo un accuracy de **81.67%** en el conjunto de prueba, inferior al Random Forest original, cuyo accuracy fue de **86.67%**.

Por este motivo, se conservó el Random Forest original como modelo final.

---

## Pruebas funcionales

La aplicación fue probada utilizando siete casos diferentes.

| Prueba | Resultado | Probabilidad estimada |
|---:|---|---:|
| 1 | Sin enfermedad | 3.0% |
| 2 | Sin enfermedad | 7.0% |
| 3 | Con enfermedad | 60.0% |
| 4 | Sin enfermedad | 44.0% |
| 5 | Con enfermedad | 57.0% |
| 6 | Sin enfermedad | 16.0% |
| 7 | Con enfermedad | 62.0% |

Las pruebas permitieron comprobar el funcionamiento de la interfaz, el procesamiento de los datos de entrada y la integración del modelo de Machine Learning con la aplicación web.

---

## Tecnologías utilizadas

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Jupyter Notebook**
- **Visual Studio Code**

---

## Estructura del proyecto

```text
sistema_riesgo_cardiaco/
│
├── .gitignore
├── README.md
├── app.py
│
├── datos/
│   ├── heart_clean.csv
│   ├── heart_model.csv
│   └── processed.cleveland.data
│
├── modelos/
│   ├── modelo_final.pkl
│   ├── modelo_logistico.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
│
└── src/
    └── prediccion.py

## Componentes principales

### `app.py`

Contiene la interfaz web desarrollada con Streamlit y permite ingresar los datos del paciente y visualizar el resultado de la predicción.

### `src/prediccion.py`

Contiene la lógica encargada de cargar el modelo y el escalador, preparar los datos de entrada y generar la predicción.

### `modelos/modelo_final.pkl`

Contiene el modelo Random Forest seleccionado como modelo final.

### `modelos/modelo_logistico.pkl`

Contiene el modelo de Regresión Logística utilizado para la comparación.

### `modelos/scaler.pkl`

Contiene el objeto utilizado para estandarizar los datos antes de enviarlos al modelo.

### `notebooks/01_analisis_exploratorio.ipynb`

Contiene el análisis exploratorio, preparación de datos, entrenamiento, evaluación y comparación de los modelos.

---

## Ejecución del proyecto

Para ejecutar el proyecto localmente se debe activar el entorno virtual y ejecutar Streamlit.

### Activar el entorno virtual

En PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1

### Ejecutar la aplicación

Desde la carpeta principal del proyecto:

```powershell
streamlit run app.py

Después de ejecutar el comando, Streamlit proporciona una dirección local para acceder a la aplicación desde el navegador.

---

## Resultados

El modelo final seleccionado fue Random Forest, con un accuracy de **86.67%** sobre el conjunto de prueba utilizado durante el desarrollo.

La aplicación permite introducir los datos de un paciente y obtener una clasificación estimada junto con la probabilidad estimada por el modelo.

---

## Limitaciones

Entre las principales limitaciones del proyecto se encuentran:

- El conjunto de datos utilizado es relativamente pequeño.
- El modelo fue desarrollado con fines académicos.
- Los resultados dependen de los datos utilizados durante el entrenamiento.
- La probabilidad mostrada corresponde a la estimación producida por el modelo y no a una probabilidad clínica validada.
- El sistema no sustituye la evaluación de un profesional de la salud.
- No se ha realizado una validación clínica ni una implementación para uso médico real.

---

## Conclusión

El proyecto permitió integrar técnicas de análisis de datos, Machine Learning y desarrollo web para construir un prototipo funcional de predicción de riesgo cardíaco.

La implementación demuestra el proceso completo desde la preparación de los datos hasta la integración de un modelo de clasificación en una aplicación web interactiva.