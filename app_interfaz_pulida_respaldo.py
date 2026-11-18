import os
import sys
from io import BytesIO

import pandas as pd
import streamlit as st


# --------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------------------------------

st.set_page_config(
    page_title="CardioPredict",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# RUTAS DEL PROYECTO
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from prediccion import predecir_riesgo, modelo


# --------------------------------------------------
# DATASET DE REFERENCIA
# --------------------------------------------------

RUTA_DATOS = os.path.join(
    BASE_DIR,
    "datos",
    "heart_clean.csv"
)


@st.cache_data
def cargar_dataset_referencia(ruta):
    return pd.read_csv(ruta)


df_referencia = cargar_dataset_referencia(RUTA_DATOS)


# --------------------------------------------------
# VARIABLES Y DESCRIPCIONES
# --------------------------------------------------

VARIABLES = [
    "Edad",
    "Sexo",
    "Tipo de dolor de pecho",
    "Presión arterial",
    "Colesterol",
    "Azúcar en ayunas",
    "Electrocardiograma",
    "Frecuencia cardíaca máxima",
    "Angina por ejercicio",
    "Depresión ST",
    "Pendiente ST",
    "Vasos principales",
    "Thal"
]

DESCRIPCIONES_VARIABLES = {
    "Edad": "Representa la edad del paciente expresada en años.",

    "Sexo": (
        "Representa el sexo registrado del paciente dentro del "
        "conjunto de datos utilizado por el modelo."
    ),

    "Tipo de dolor de pecho": (
        "Representa la categoría de dolor de pecho registrada "
        "para el paciente."
    ),

    "Presión arterial": (
        "Representa la presión arterial registrada en reposo, "
        "expresada en mmHg."
    ),

    "Colesterol": (
        "Representa el nivel de colesterol sérico registrado, "
        "expresado en mg/dl."
    ),

    "Azúcar en ayunas": (
        "Indica si el nivel de azúcar en sangre en ayunas supera "
        "el umbral utilizado en el conjunto de datos."
    ),

    "Electrocardiograma": (
        "Representa el resultado del electrocardiograma realizado "
        "en reposo."
    ),

    "Frecuencia cardíaca máxima": (
        "Representa la frecuencia cardíaca máxima registrada "
        "durante la prueba."
    ),

    "Angina por ejercicio": (
        "Indica si se presentó angina durante el ejercicio."
    ),

    "Depresión ST": (
        "Representa una medida relacionada con la depresión del "
        "segmento ST registrada durante la prueba de ejercicio."
    ),

    "Pendiente ST": (
        "Representa la pendiente del segmento ST durante "
        "la prueba de ejercicio."
    ),

    "Vasos principales": (
        "Representa el número de vasos principales registrados "
        "mediante fluoroscopia."
    ),

    "Thal": (
        "Representa una variable del conjunto de datos relacionada "
        "con el resultado de una prueba incluida en el dataset."
    ),
}


# --------------------------------------------------
# ESTADÍSTICAS DE REFERENCIA
# --------------------------------------------------

estadisticas_referencia = {
    "Edad": {
        "mediana": df_referencia["age"].median()
    },
    "Presión arterial": {
        "mediana": df_referencia["trestbps"].median()
    },
    "Colesterol": {
        "mediana": df_referencia["chol"].median()
    },
    "Frecuencia cardíaca máxima": {
        "mediana": df_referencia["thalach"].median()
    },
    "Depresión ST": {
        "mediana": df_referencia["oldpeak"].median()
    },
    "Vasos principales": {
        "mediana": df_referencia["ca"].median()
    }
}


def interpretar_valor_numerico(nombre_variable, valor):
    """
    Compara un valor numérico con la mediana observada en el dataset
    de referencia. Esta comparación es estadística, no clínica.
    """
    mediana = estadisticas_referencia[nombre_variable]["mediana"]

    # Margen del 10 % alrededor de la mediana
    margen = abs(mediana) * 0.10

    if margen == 0:
        margen = 0.1

    if valor < mediana - margen:
        return (
            f"Este valor se encuentra por debajo de la mediana observada "
            f"en el conjunto de datos de referencia ({mediana:.1f})."
        )

    if valor > mediana + margen:
        return (
            f"Este valor se encuentra por encima de la mediana observada "
            f"en el conjunto de datos de referencia ({mediana:.1f})."
        )

    return (
        f"Este valor se encuentra cercano a la mediana observada "
        f"en el conjunto de datos de referencia ({mediana:.1f})."
    )


# --------------------------------------------------
# ESTADO DE LA SESIÓN
# --------------------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []


# --------------------------------------------------
# ENCABEZADO GENERAL
# --------------------------------------------------

st.title("❤️ CardioPredict")

st.caption(
    "Sistema de Predicción de Riesgo Cardíaco mediante Machine Learning"
)

st.info(
    "ℹ️ Herramienta académica de apoyo. Los resultados generados por "
    "el modelo no constituyen un diagnóstico médico ni sustituyen una "
    "valoración profesional."
)


# --------------------------------------------------
# NAVEGACIÓN PRINCIPAL
# --------------------------------------------------

tab_inicio, tab_evaluacion, tab_historial = st.tabs([
    "🏠 Inicio",
    "❤️ Evaluación",
    "📋 Historial"
])


# ==================================================
# PESTAÑA: INICIO
# ==================================================

with tab_inicio:

    st.header("🏠 Bienvenido a CardioPredict")

    st.write(
        "CardioPredict analiza 13 características clínicas mediante un "
        "modelo Random Forest para generar una estimación orientativa "
        "sobre la clase 'Con enfermedad'."
    )

    st.divider()

    st.subheader("🤖 Modelo utilizado")

    st.write(
        "El modelo principal es **Random Forest**, un algoritmo de "
        "Machine Learning basado en múltiples árboles de decisión. "
        "Durante el desarrollo se compararon distintos modelos y se "
        "seleccionó esta versión por su desempeño en las pruebas realizadas."
    )

    st.subheader("📊 Métricas del modelo")

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric("Accuracy", "86.67%")

    with col_m2:
        st.metric("F1", "0.852")

    with col_m3:
        st.metric("AUC", "0.941")

    st.caption(
        "Métricas obtenidas en las pruebas académicas realizadas durante "
        "el desarrollo del proyecto."
    )

    st.divider()

    with st.expander("🧾 Ver las 13 variables utilizadas"):

        st.write(
            "El modelo recibe las siguientes variables clínicas como entrada:"
        )

        st.markdown(
            """
            - Edad
            - Sexo
            - Tipo de dolor de pecho
            - Presión arterial en reposo
            - Colesterol
            - Azúcar en ayunas
            - Electrocardiograma en reposo
            - Frecuencia cardíaca máxima
            - Angina inducida por ejercicio
            - Depresión ST
            - Pendiente ST
            - Número de vasos principales
            - Thal
            """
        )

    st.divider()

    st.header("📈 Importancia global de las variables")

    st.write(
        "La siguiente gráfica muestra qué variables tienen mayor peso "
        "en el comportamiento general del modelo Random Forest."
    )

    importancia = modelo.feature_importances_

    df_importancia = pd.DataFrame({
        "Variable": VARIABLES,
        "Importancia": importancia
    }).sort_values(
        "Importancia",
        ascending=True
    )

    st.bar_chart(
        df_importancia.set_index("Variable")["Importancia"]
    )

    st.caption(
        "Esta gráfica describe la importancia global del modelo. "
        "No representa la explicación de un paciente específico."
    )


# ==================================================
# PESTAÑA: EVALUACIÓN
# ==================================================

with tab_evaluacion:

    st.header("❤️ Evaluación del paciente")

    st.write(
        "Ingrese los datos solicitados. Las variables principales son "
        "las mismas utilizadas durante el entrenamiento del modelo."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # COLUMNA 1
    # --------------------------------------------------

    with col1:

        st.subheader("Información general")

        age = st.number_input(
            "Edad",
            min_value=1,
            max_value=120,
            value=55,
            help="Edad del paciente expresada en años."
        )

        sex = st.selectbox(
            "Sexo",
            options=[0, 1],
            format_func=lambda x: "Femenino" if x == 0 else "Masculino",
            help="Sexo registrado en el conjunto de datos."
        )

        cp = st.selectbox(
            "Tipo de dolor de pecho",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 - Dolor típico de angina",
                1: "1 - Dolor atípico de angina",
                2: "2 - Dolor no anginoso",
                3: "3 - Sin dolor de pecho"
            }[x],
            help="Categoría del dolor de pecho utilizada por el dataset."
        )

        trestbps = st.number_input(
            "Presión arterial en reposo",
            min_value=50,
            max_value=250,
            value=140,
            help="Presión arterial registrada en reposo, en mmHg."
        )

        chol = st.number_input(
            "Colesterol",
            min_value=50,
            max_value=600,
            value=250,
            help="Nivel de colesterol sérico registrado, en mg/dl."
        )

        fbs = st.selectbox(
            "Azúcar en sangre en ayunas > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Sí",
            help=(
                "Indica si el nivel registrado supera el umbral "
                "de 120 mg/dl utilizado en el dataset."
            )
        )

    # --------------------------------------------------
    # COLUMNA 2
    # --------------------------------------------------

    with col2:

        st.subheader("Datos cardiovasculares")

        restecg = st.selectbox(
            "Electrocardiograma en reposo",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 - Normal",
                1: "1 - Anormalidad de onda ST-T",
                2: "2 - Hipertrofia ventricular izquierda"
            }[x],
            help="Resultado categórico del electrocardiograma en reposo."
        )

        thalach = st.number_input(
            "Frecuencia cardíaca máxima alcanzada",
            min_value=50,
            max_value=250,
            value=150,
            help="Frecuencia cardíaca máxima registrada durante la prueba."
        )

        exang = st.selectbox(
            "Angina inducida por ejercicio",
            options=[0, 1],
            format_func=lambda x: {
                0: "0 - No se presenta",
                1: "1 - Se presenta"
            }[x],
            help="Indica si se presentó angina durante el ejercicio."
        )

        oldpeak = st.number_input(
            "Depresión del segmento ST",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help=(
                "Valor de depresión del segmento ST registrado "
                "durante la prueba de ejercicio."
            )
        )

        slope = st.selectbox(
            "Pendiente del segmento ST",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 - Ascendente",
                1: "1 - Plana",
                2: "2 - Descendente"
            }[x],
            help="Categoría de la pendiente del segmento ST."
        )

        ca = st.number_input(
            "Número de vasos principales",
            min_value=0,
            max_value=4,
            value=0,
            help=(
                "Número de vasos principales registrados mediante "
                "fluoroscopia."
            )
        )

        thal = st.number_input(
            "Resultado Thal",
            min_value=0,
            max_value=3,
            value=2,
            help=(
                "Variable categórica Thal utilizada por el dataset "
                "y por el modelo."
            )
        )

    # --------------------------------------------------
    # INFORMACIÓN COMPLEMENTARIA
    # --------------------------------------------------

    st.divider()

    with st.expander("📝 Información complementaria del paciente"):

        st.write(
            "Estos datos son opcionales y se guardan únicamente como contexto. "
            "No modifican la predicción del Random Forest."
        )

        col3, col4 = st.columns(2)

        with col3:

            antecedentes_cardiacos = st.selectbox(
                "¿Cuenta con antecedentes cardíacos?",
                options=["No", "Sí"]
            )

            tratamiento_previo = st.selectbox(
                "¿Recibe actualmente algún tratamiento?",
                options=["No", "Sí"]
            )

        with col4:

            medicamentos = st.text_input(
                "Medicamentos actuales",
                placeholder="Opcional"
            )

            observaciones = st.text_area(
                "Observaciones",
                placeholder="Información adicional del paciente..."
            )

        st.caption(
            "Esta información no forma parte de las 13 variables "
            "utilizadas para calcular la predicción."
        )

    # --------------------------------------------------
    # BOTÓN DE PREDICCIÓN
    # --------------------------------------------------

    st.divider()

    if st.button(
        "🔍 Realizar predicción",
        use_container_width=True
    ):

        datos = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        resultado, probabilidad, valores_shap = predecir_riesgo(datos)

        # --------------------------------------------------
        # GUARDAR EN HISTORIAL
        # --------------------------------------------------

        st.session_state.historial.append({
            "Edad": age,
            "Sexo": "Femenino" if sex == 0 else "Masculino",
            "Antecedentes cardíacos": antecedentes_cardiacos,
            "Tratamiento previo": tratamiento_previo,
            "Medicamentos": medicamentos if medicamentos else "No especificado",
            "Observaciones": observaciones if observaciones else "Sin observaciones",
            "Resultado": resultado,
            "Probabilidad": f"{probabilidad * 100:.1f}%"
        })

        # --------------------------------------------------
        # RESULTADO Y PROBABILIDAD
        # --------------------------------------------------

        st.divider()
        st.header("📊 Resultado de la evaluación")

        col_resultado, col_probabilidad = st.columns([1.2, 1])

        with col_resultado:

            st.subheader("Clasificación estimada")

            if resultado == "Con enfermedad":
                st.error(
                    "⚠️ **Con enfermedad**"
                )
            else:
                st.success(
                    "✅ **Sin enfermedad**"
                )

        with col_probabilidad:

            st.metric(
                label="Probabilidad para 'Con enfermedad'",
                value=f"{probabilidad * 100:.1f}%"
            )

        st.progress(
            min(max(probabilidad, 0.0), 1.0)
        )

        st.caption(
            "La probabilidad corresponde a la estimación del modelo "
            "y no representa una probabilidad clínica validada."
        )

        # --------------------------------------------------
        # EXPLICACIÓN SHAP
        # --------------------------------------------------

        st.divider()

        st.header("🔎 ¿Qué influyó en esta predicción?")

        st.write(
            "La gráfica muestra las cinco variables con mayor influencia "
            "en esta evaluación individual según SHAP."
        )

        valores_paciente = {
            "Edad": age,
            "Sexo": "Femenino" if sex == 0 else "Masculino",
            "Tipo de dolor de pecho": {
                0: "Dolor típico de angina",
                1: "Dolor atípico de angina",
                2: "Dolor no anginoso",
                3: "Sin dolor de pecho"
            }[cp],
            "Presión arterial": f"{trestbps} mmHg",
            "Colesterol": f"{chol} mg/dl",
            "Azúcar en ayunas": "Sí" if fbs == 1 else "No",
            "Electrocardiograma": {
                0: "Normal",
                1: "Anormalidad de onda ST-T",
                2: "Hipertrofia ventricular izquierda"
            }[restecg],
            "Frecuencia cardíaca máxima": thalach,
            "Angina por ejercicio": "Sí" if exang == 1 else "No",
            "Depresión ST": oldpeak,
            "Pendiente ST": {
                0: "Ascendente",
                1: "Plana",
                2: "Descendente"
            }[slope],
            "Vasos principales": ca,
            "Thal": thal
        }

        # Obtener valores SHAP para la clase "Con enfermedad"
        if isinstance(valores_shap, list):
            valores = valores_shap[1][0]
        elif len(valores_shap.shape) == 3:
            valores = valores_shap[0, :, 1]
        else:
            valores = valores_shap[0]

        df_shap = pd.DataFrame({
            "Variable": VARIABLES,
            "Influencia": valores
        })

        df_shap["Influencia absoluta"] = df_shap["Influencia"].abs()

        df_shap = df_shap.sort_values(
            "Influencia absoluta",
            ascending=False
        )

        df_shap_top5 = df_shap.head(5)

        st.bar_chart(
            df_shap_top5.set_index("Variable")["Influencia"]
        )

        st.caption(
            "Valores positivos indican una influencia hacia 'Con enfermedad'; "
            "valores negativos indican una influencia hacia 'Sin enfermedad'."
        )

        # --------------------------------------------------
        # INTERPRETACIÓN TEXTUAL PERSONALIZADA
        # --------------------------------------------------

        st.divider()

        st.header("🧠 Interpretación personalizada")

        st.write(
            f"El modelo clasificó este caso como **{resultado}**. "
            f"La probabilidad estimada para la clase **'Con enfermedad'** "
            f"fue de **{probabilidad * 100:.1f}%**."
        )

        st.write(
            "Puedes desplegar cada variable para conocer qué representa, "
            "cómo se comportó frente a los datos de referencia y cómo "
            "influyó en esta predicción."
        )

        variables_principales = df_shap.head(3)

        for posicion, (_, fila) in enumerate(
            variables_principales.iterrows(),
            start=1
        ):

            nombre_variable = fila["Variable"]
            influencia = float(fila["Influencia"])
            valor_paciente = valores_paciente[nombre_variable]
            descripcion = DESCRIPCIONES_VARIABLES[nombre_variable]

            titulo_expander = (
                f"🔎 {posicion}. {nombre_variable} — "
                f"valor: {valor_paciente}"
            )

            with st.expander(
                titulo_expander,
                expanded=(posicion == 1)
            ):

                st.markdown("**¿Qué representa este dato?**")
                st.write(descripcion)

                if nombre_variable in estadisticas_referencia:

                    valor_numerico = {
                        "Edad": age,
                        "Presión arterial": trestbps,
                        "Colesterol": chol,
                        "Frecuencia cardíaca máxima": thalach,
                        "Depresión ST": oldpeak,
                        "Vasos principales": ca
                    }[nombre_variable]

                    st.markdown(
                        "**¿Cómo se compara con los datos de referencia?**"
                    )

                    st.write(
                        interpretar_valor_numerico(
                            nombre_variable,
                            valor_numerico
                        )
                    )

                    st.caption(
                        "Comparación estadística contra la mediana del dataset; "
                        "no representa un rango clínico de normalidad."
                    )

                st.markdown(
                    "**¿Cómo influyó en esta predicción?**"
                )

                if influencia > 0:
                    st.write(
                        "El valor registrado desplazó la predicción del "
                        "modelo hacia **'Con enfermedad'**."
                    )

                elif influencia < 0:
                    st.write(
                        "El valor registrado desplazó la predicción del "
                        "modelo hacia **'Sin enfermedad'**."
                    )

                else:
                    st.write(
                        "La variable tuvo una influencia prácticamente "
                        "neutra en la decisión del modelo."
                    )

                st.markdown(
                    "**¿Por qué aparece entre las variables más importantes?**"
                )

                st.write(
                    f"Su valor SHAP fue **{influencia:.4f}** y la magnitud "
                    f"de su contribución fue **{abs(influencia):.4f}**. "
                    "Esa magnitud se encuentra entre las tres mayores de "
                    "las 13 variables analizadas para este paciente."
                )

                st.write(
                    "Esto indica que la variable tuvo un peso relevante "
                    "en la construcción de esta predicción individual, "
                    "pero no significa que por sí sola determine el resultado."
                )

        st.info(
            "ℹ️ Esta explicación describe el comportamiento del modelo. "
            "Las comparaciones con el dataset son estadísticas y SHAP no "
            "establece causalidad médica."
        )


# ==================================================
# PESTAÑA: HISTORIAL
# ==================================================

with tab_historial:

    st.header("📋 Historial de predicciones")

    if st.session_state.historial:

        st.caption(
            f"Evaluaciones registradas en esta sesión: "
            f"{len(st.session_state.historial)}"
        )

        st.dataframe(
            st.session_state.historial,
            use_container_width=True,
            hide_index=True
        )

        output = BytesIO()

        df_historial = pd.DataFrame(
            st.session_state.historial
        )

        with pd.ExcelWriter(
            output,
            engine="openpyxl"
        ) as writer:

            df_historial.to_excel(
                writer,
                index=False,
                sheet_name="Historial"
            )

        output.seek(0)

        col_descarga, col_limpiar = st.columns(2)

        with col_descarga:

            st.download_button(
                label="📥 Descargar historial en Excel",
                data=output,
                file_name="historial_predicciones.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True
            )

        with col_limpiar:

            if st.button(
                "🗑️ Limpiar historial",
                use_container_width=True
            ):
                st.session_state.historial = []
                st.rerun()

    else:

        st.info(
            "Aún no hay evaluaciones registradas en esta sesión. "
            "Realiza una predicción para comenzar el historial."
        )


# --------------------------------------------------
# PIE DE PÁGINA
# --------------------------------------------------

st.divider()

st.caption(
    "CardioPredict · Proyecto académico de Machine Learning"
)