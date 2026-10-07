import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from libreria_funciones_tf import (
    validar_columnas,
    calcular_tasa_renovacion,
    formatear_numero,
    formatear_porcentaje)
from libreria_clases_tf import DataAnalyzer, PALETA_RENOVACION, COLOR_BARRAS

st.set_page_config(page_title="Insurance Company - EDA", layout="wide")
sns.set_theme(style="whitegrid")


COLUMNAS_ESPERADAS = [
    "id", "perc_premium_paid_by_cash_credit", "age_in_days", "Income",
    "Count_3-6_months_late", "Count_6-12_months_late", "Count_more_than_12_months_late",
    "application_underwriting_score", "no_of_premiums_paid", "sourcing_channel",
    "residence_area_type", "premium", "renewal"]

CORTES_ATRASOS = [-1, 0, 1, 2, 100]
ETIQUETAS_ATRASOS = ["0 atrasos", "1 atraso", "2 atrasos", "3+ atrasos"]
CORTES_PAGO_EFECTIVO = [-0.01, 0.25, 0.50, 0.75, 1.0]
ETIQUETAS_PAGO_EFECTIVO = ["0-25%", "25-50%", "50-75%", "75-100%"]
CORTES_EDAD = [0, 30, 40, 50, 60, 120]
ETIQUETAS_EDAD = ["<30", "30-40", "40-50", "50-60", "60+"]

if "df" not in st.session_state:
    st.session_state.df = None



# sidebar
# =========================================================
st.sidebar.title("Insurance Company")
st.sidebar.caption("Análisis de renovación de pólizas")

seccion = st.sidebar.radio(
    "Navegación",
    ["Home", "Carga del dataset", "Análisis Exploratorio (EDA)", "Conclusiones"])

st.sidebar.markdown("---")
if st.session_state.df is not None:
    st.sidebar.success(f"Dataset cargado: {formatear_numero(len(st.session_state.df))} filas")
else:
    st.sidebar.warning("Dataset no cargado")



# HOME
# =========================================================
if seccion == "Home":
    st.title("¿Qué hace que un cliente renueve su póliza?")
    st.subheader("Análisis Exploratorio de Datos (EDA) - Insurance Company")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Objetivo del análisis")
        st.write(
            "Identificar los factores asociados a la **renovación de pólizas de seguro** "
            "(variable `renewal`) a partir del historial de clientes: perfil demográfico, "
            "ingresos, comportamiento de pago, canal de captación y valor de la prima. "
            "El enfoque es **descriptivo y orientado a la toma de decisiones** "
            "(retención de clientes), no a la predicción.")

        st.markdown("### Sobre el dataset")
        st.write(
            "`InsuranceCompany.csv` contiene cerca de 80 mil clientes con 13 variables: "
            "porcentaje de la prima pagada en efectivo/crédito, edad (en días), ingreso, "
            "número de pagos atrasados (3-6, 6-12 y más de 12 meses), puntaje de evaluación "
            "(underwriting score), número de primas pagadas, canal de captación (A-E), "
            "tipo de residencia (Urban/Rural), valor de la prima y si renovó (1) o no (0).")

    with col2:
        st.markdown("### Autora")
        st.markdown("**Nombre:** Diana Patricia Ferreccio Rodriguez")
        st.markdown("**Curso:** Especialización en Python for Analytics - DMC Institute")
        st.markdown("**Año:** 2026")

        st.markdown("### Tecnologías utilizadas")
        st.markdown("- Python")
        st.markdown("- Pandas y NumPy")
        st.markdown("- Matplotlib y Seaborn")
        st.markdown("- Streamlit")
        st.markdown("- Programación Orientada a Objetos (clase `DataAnalyzer`)")

    st.info("Para comenzar, ve al módulo **Carga del dataset** en el menú lateral.")



# CARGA DEL DATASET
# =========================================================
elif seccion == "Carga del dataset":
    st.title("Carga del dataset")
    st.write("Sube el archivo **InsuranceCompany.csv** para habilitar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            df = pd.read_csv(archivo)
            faltantes = validar_columnas(df, COLUMNAS_ESPERADAS)

            if len(faltantes) > 0:
                st.error(f"El archivo no tiene las columnas esperadas. Faltan: {', '.join(faltantes)}")
            else:
                st.session_state.df = df
                st.success(f"Archivo '{archivo.name}' cargado correctamente")
        except Exception as error:
            st.error(f"No se pudo leer el archivo: {error}")

    if st.session_state.df is not None:
        df = st.session_state.df

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", formatear_numero(df.shape[0]))
        col2.metric("Columnas", df.shape[1])
        col3.metric("Tasa de renovación", formatear_porcentaje(calcular_tasa_renovacion(df)))

        n_filas = st.slider("Número de filas a mostrar (head)", min_value=5, max_value=50, value=5)
        st.dataframe(df.head(n_filas), width="stretch")
    else:
        st.warning("Aún no se ha cargado ningún archivo. El análisis está deshabilitado.")



# ANÁLISIS EXPLORATORIO (EDA)
# =========================================================
elif seccion == "Análisis Exploratorio (EDA)":
    st.title("Análisis Exploratorio de Datos")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **Carga del dataset**.")
        st.stop()

    analizador = DataAnalyzer(st.session_state.df)
    df = analizador.df

    tabs = st.tabs([
        "1. Info general", "2. Tipos de variables", "3. Estadísticas", "4. Faltantes",
        "5. Numéricas", "6. Categóricas", "7. Num vs Cat", "8. Cat vs Cat",
        "9. Análisis dinámico", "10. Hallazgos"])

    # 1_INFORMACIÓN GENERAL
    with tabs[0]:
        st.header("Ítem 1: Información general del dataset")
        st.write("Revisamos estructura, tipos de datos y valores nulos de cada columna.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Resultado de `df.info()`**")
            st.code(analizador.texto_info())
        with col2:
            st.markdown("**Tipos de datos, nulos y valores únicos**")
            st.dataframe(analizador.resumen_columnas(), width="stretch")

        st.write(
            f"El dataset tiene **{formatear_numero(df.shape[0])} clientes**. La variable `renewal` "
            "es numérica (0/1) pero representa una categoría, por eso en la app la tratamos como "
            "categórica y creamos la etiqueta `renovacion` (Renovó / No renovó).")

  
    # 2_CLASIFICACIÓN DE VARIABLES

    with tabs[1]:
        st.header("Ítem 2: Clasificación de variables")
        st.write(
            "Usamos la función personalizada `clasificar_variables()` (archivo "
            "`libreria_funciones_tf.py`). Se excluye `id` y `age_in_days` (se reemplaza por "
            "`edad_anios`). También se agregan dos variables derivadas: `edad_anios` y `total_atrasos`.")

        conteo = analizador.conteo_tipos()
        col1, col2 = st.columns(2)
        col1.metric("Variables numéricas", conteo["Numéricas"])
        col2.metric("Variables categóricas", conteo["Categóricas"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numéricas**")
            st.dataframe(pd.DataFrame({"Variable": analizador.columnas_numericas}), width="stretch")
        with col2:
            st.markdown("**Categóricas**")
            st.dataframe(pd.DataFrame({"Variable": analizador.columnas_categoricas}), width="stretch")

   
    # 3_ESTADÍSTICAS DESCRIPTIVAS
    with tabs[2]:
        st.header("Ítem 3: Estadísticas descriptivas")
        st.write("Tabla de `.describe()` con mediana, moda y coeficiente de variación agregados.")

        columnas_elegidas = st.multiselect(
            "Variables a describir",
            analizador.columnas_numericas,
            default=analizador.columnas_numericas,
            key="multi_describe")
        if len(columnas_elegidas) > 0:
            st.dataframe(analizador.estadisticas_descriptivas(columnas_elegidas), width="stretch")

        st.markdown("**Moda de las variables categóricas**")
        st.dataframe(analizador.moda_categoricas(), width="stretch")

        media_ingreso = df["Income"].mean()
        mediana_ingreso = df["Income"].median()
        st.markdown("**Interpretación**")
        st.write(f"- **Income:** media de {formatear_numero(media_ingreso)} vs mediana de "
            f"{formatear_numero(mediana_ingreso)}. La media es mayor porque hay ingresos "
            f"extremadamente altos (máximo {formatear_numero(df['Income'].max())}): la distribución "
            "tiene sesgo a la derecha, por lo que la **mediana** representa mejor al cliente típico.")
        st.write(
            f"- **Edad:** el cliente típico tiene {df['edad_anios'].median():.0f} años "
            f"(media {df['edad_anios'].mean():.1f}); la media y la mediana son parecidas, distribución bastante simétrica.")
        st.write(
            f"- **Atrasos:** la mediana de `total_atrasos` es {df['total_atrasos'].median():.0f}; la mayoría "
            "de clientes no tiene atrasos, pero la desviación estándar alta indica un grupo con muchos atrasos.")
        st.write(
            f"- **Underwriting score:** muy concentrado (media {df['application_underwriting_score'].mean():.2f}, "
            f"desv. {df['application_underwriting_score'].std():.2f}): poca dispersión entre clientes.")


    # 4_VALORES FALTANTES
    with tabs[3]:
        st.header("Ítem 4: Análisis de valores faltantes")
        tabla_nulos = analizador.valores_faltantes()

        if len(tabla_nulos) == 0:
            st.success("El dataset no tiene valores faltantes.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(tabla_nulos, width="stretch")
            with col2:
                st.pyplot(analizador.grafico_faltantes())

            st.markdown("**Discusión**")
            st.write(
                f"- `application_underwriting_score` es la variable con más nulos "
                f"({tabla_nulos['% Nulos'].max():.2f}% de los registros). Al ser un porcentaje bajo, "
                "se puede trabajar ignorando los nulos o imputando con la **mediana**.")
            st.write(
                "- Las tres columnas de atrasos tienen los mismos 97 nulos (los mismos clientes). "
                "Son muy pocos y no afectan las conclusiones. Pandas los ignora al calcular medias.")

  
    # 5_DISTRIBUCIÓN DE VARIABLES NUMÉRICAS
    with tabs[4]:
        st.header("Ítem 5: Distribución de variables numéricas")

        col1, col2, col3 = st.columns(3)
        with col1:
            variable_hist = st.selectbox("Variable numérica", analizador.columnas_numericas, key="sel_hist")
        with col2:
            bins = st.slider("Número de bins", min_value=10, max_value=100, value=30, step=5)
        with col3:
            percentil = st.slider("Recortar valores por encima del percentil", 90, 100, 99,
                                  help="Útil para Income y premium, que tienen valores extremos.")

        separar = st.checkbox("Separar por renovación (Renovó / No renovó)", value=False)

        st.pyplot(analizador.grafico_histograma(variable_hist, bins, percentil, separar))

        serie = df[variable_hist]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Media", formatear_numero(serie.mean(), 2))
        col2.metric("Mediana", formatear_numero(serie.median(), 2))
        col3.metric("Moda", formatear_numero(serie.mode().iloc[0], 2))
        col4.metric("Asimetría (skew)", f"{serie.skew():.2f}")

        if serie.skew() > 1:
            st.info("Distribución con **sesgo positivo** (cola a la derecha): la media queda por encima de la mediana.")
        elif serie.skew() < -1:
            st.info("Distribución con **sesgo negativo** (cola a la izquierda): la media queda por debajo de la mediana.")
        else:
            st.info("Distribución **aproximadamente simétrica**: media y mediana son parecidas.")


    # 6_VARIABLES CATEGÓRICAS
    with tabs[5]:
        st.header("Ítem 6: Análisis de variables categóricas")
        variable_cat = st.selectbox("Variable categórica", analizador.columnas_categoricas, key="sel_cat")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(analizador.tabla_frecuencias(variable_cat), width="stretch")
        with col2:
            st.pyplot(analizador.grafico_barras(variable_cat))

        frecuencias = analizador.tabla_frecuencias(variable_cat)
        st.write(
            f"La categoría más frecuente de **{variable_cat}** es **{frecuencias.index[0]}** con "
            f"{frecuencias['Proporción %'].iloc[0]:.1f}% de los clientes.")
        st.caption(
            "Nota: `renewal` está muy desbalanceada (~94% renueva). Por eso en los siguientes ítems "
            "comparamos **tasas de renovación** por grupo y no solo conteos.")

    # 7_IVARIADO: NUMÉRICO VS CATEGÓRICO
    with tabs[6]:
        st.header("Ítem 7: Análisis bivariado (numérico vs categórico)")
        st.write("Comparamos cómo cambia una variable numérica entre los clientes que renuevan y los que no.")

        col1, col2 = st.columns(2)
        with col1:
            variable_num = st.selectbox(
                "Variable numérica", analizador.columnas_numericas,
                index=analizador.columnas_numericas.index("Income"), key="sel_biv_num")
        with col2:
            opciones_grupo = ["renovacion", "sourcing_channel", "residence_area_type"]
            variable_grupo = st.selectbox("Agrupar por", opciones_grupo, key="sel_biv_cat")

        percentil_box = st.slider("Recortar outliers por encima del percentil", 90, 100, 99, key="perc_box")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(analizador.grafico_boxplot(variable_num, variable_grupo, percentil_box))
        with col2:
            st.dataframe(analizador.comparar_grupos(variable_num, variable_grupo), width="stretch")

        st.markdown("**Comparación de medianas: Renovó vs No renovó**")
        medianas = df.groupby("renovacion")[
            ["Income", "edad_anios", "perc_premium_paid_by_cash_credit", "total_atrasos", "premium"]
        ].median().T
        medianas["Diferencia %"] = ((medianas["Renovó"] / medianas["No renovó"] - 1) * 100).round(1)
        st.dataframe(medianas.round(3), width="stretch")
        st.write(
            "Los clientes que renuevan tienen **mayor ingreso**, son **mayores** y pagan una "
            "**menor proporción de la prima en efectivo/crédito**. El valor de la prima casi no cambia.")


    # 8_BIVARIADO: CATEGÓRICO VS CATEGÓRICO
    with tabs[7]:
        st.header("Ítem 8: Análisis bivariado (categórico vs categórico)")

        variable_fila = st.selectbox(
            "Variable a comparar con la renovación", ["sourcing_channel", "residence_area_type"], key="sel_cat_cat")
        ver_conteos = st.checkbox("Ver conteos absolutos en lugar de porcentajes")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(analizador.grafico_barras_apiladas(variable_fila, "renovacion"))
        with col2:
            st.dataframe(analizador.tabla_cruzada(variable_fila, "renovacion", normalizar=not ver_conteos),
                         width="stretch")

        tasas = analizador.tasa_renovacion_por(variable_fila)
        mejor = tasas["Tasa renovación %"].idxmax()
        peor = tasas["Tasa renovación %"].idxmin()
        st.write(
            f"La mayor tasa de renovación está en **{mejor}** ({tasas.loc[mejor, 'Tasa renovación %']}%) y la menor "
            f"en **{peor}** ({tasas.loc[peor, 'Tasa renovación %']}%). Diferencia: "
            f"{tasas.loc[mejor, 'Tasa renovación %'] - tasas.loc[peor, 'Tasa renovación %']:.1f} puntos porcentuales.")

 
    # 9_ANÁLISIS DINÁMICO
    with tabs[8]:
        st.header("Ítem 9: Análisis basado en parámetros seleccionados")
        st.write("Filtra un segmento de clientes y elige qué variables analizar.")

        col1, col2, col3 = st.columns(3)
        with col1:
            canales = st.multiselect(
                "Canal de captación", sorted(df["sourcing_channel"].unique()),
                default=sorted(df["sourcing_channel"].unique()))
        with col2:
            areas = st.multiselect(
                "Tipo de residencia", sorted(df["residence_area_type"].unique()),
                default=sorted(df["residence_area_type"].unique()))
        with col3:
            edad_min, edad_max = st.slider(
                "Rango de edad (años)", int(df["edad_anios"].min()), int(df["edad_anios"].max()),
                (int(df["edad_anios"].min()), int(df["edad_anios"].max())))

        solo_con_atrasos = st.checkbox("Solo clientes con al menos un pago atrasado")

        # Filtro con condiciones booleanas de Pandas
        filtro = (
            df["sourcing_channel"].isin(canales)
            & df["residence_area_type"].isin(areas)
            & df["edad_anios"].between(edad_min, edad_max))
        if solo_con_atrasos:
            filtro = filtro & (df["total_atrasos"] > 0)
        df_segmento = df[filtro]

        if len(df_segmento) == 0:
            st.error("No hay clientes con esos filtros. Cambia la selección.")
        else:
            tasa_segmento = calcular_tasa_renovacion(df_segmento)
            tasa_total = calcular_tasa_renovacion(df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Clientes en el segmento", formatear_numero(len(df_segmento)))
            col2.metric("Tasa de renovación", formatear_porcentaje(tasa_segmento),
                        delta=f"{tasa_segmento - tasa_total:.1f} pp vs total")
            col3.metric("Ingreso mediano", formatear_numero(df_segmento["Income"].median()))

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                variable_dinamica = st.selectbox("Variable numérica a analizar", analizador.columnas_numericas,
                                                 key="sel_dinamica")
            with col2:
                tipo_grafico = st.selectbox("Tipo de gráfico", ["Boxplot por renovación", "Histograma por renovación"])

            datos = df_segmento[df_segmento[variable_dinamica] <= df_segmento[variable_dinamica].quantile(0.99)]
            fig, ax = plt.subplots(figsize=(8, 4))
            if tipo_grafico == "Boxplot por renovación":
                sns.boxplot(data=datos, x="renovacion", y=variable_dinamica, hue="renovacion",
                            palette=PALETA_RENOVACION, legend=False, ax=ax)
            else:
                sns.histplot(data=datos, x=variable_dinamica, hue="renovacion", stat="density",
                             common_norm=False, bins=30, palette=PALETA_RENOVACION, ax=ax)
            ax.set_title(f"{variable_dinamica} en el segmento seleccionado (sin el 1% superior)")
            fig.tight_layout()
            st.pyplot(fig)

            st.markdown("**Matriz de correlación de las variables elegidas**")
            columnas_corr = st.multiselect(
                "Variables para la correlación", analizador.columnas_numericas + ["renewal"],
                default=["perc_premium_paid_by_cash_credit", "total_atrasos", "edad_anios", "Income", "renewal"])
            if len(columnas_corr) >= 2:
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(df_segmento[columnas_corr].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                            vmin=-1, vmax=1, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.caption("Elige al menos 2 variables.")


    # 10_HALLAZGOS CLAVE
    with tabs[9]:
        st.header("Ítem 10: Hallazgos clave")
        st.write("Resumen visual de la **tasa de renovación** según los factores más relevantes.")

        tasa_atrasos = analizador.tasa_renovacion_por("total_atrasos", CORTES_ATRASOS, ETIQUETAS_ATRASOS)
        tasa_pago = analizador.tasa_renovacion_por("perc_premium_paid_by_cash_credit",
                                                   CORTES_PAGO_EFECTIVO, ETIQUETAS_PAGO_EFECTIVO)
        tasa_edad = analizador.tasa_renovacion_por("edad_anios", CORTES_EDAD, ETIQUETAS_EDAD)
        tasa_canal = analizador.tasa_renovacion_por("sourcing_channel")
        tasa_total = calcular_tasa_renovacion(df)

        # Gráfico resumen con subplots 2x2
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        graficos = [
            (axes[0, 0], tasa_atrasos, "Por número de pagos atrasados"),
            (axes[0, 1], tasa_pago, "Por % de prima pagada en efectivo/crédito"),
            (axes[1, 0], tasa_edad, "Por rango de edad"),
            (axes[1, 1], tasa_canal, "Por canal de captación"),]
        for ax, tabla, titulo in graficos:
            ax.bar(tabla.index.astype(str), tabla["Tasa renovación %"], color=COLOR_BARRAS)
            ax.axhline(tasa_total, color="#E4572E", linestyle="--", label=f"Promedio {tasa_total:.1f}%")
            for i, valor in enumerate(tabla["Tasa renovación %"]):
                ax.text(i, valor + 0.5, f"{valor:.1f}%", ha="center")
            ax.set_ylim(50, 102)
            ax.set_title(titulo)
            ax.set_ylabel("Tasa de renovación %")
            ax.legend(loc="lower left")
        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("### Insights principales")
        st.write(
            f"1. **Los atrasos son la señal más fuerte:** sin atrasos renueva el "
            f"{tasa_atrasos.iloc[0, 0]}% y con 3+ atrasos solo el {tasa_atrasos.iloc[-1, 0]}%.")
        st.write(
            f"2. **Forma de pago:** quienes pagan 75-100% de la prima en efectivo/crédito renuevan "
            f"{tasa_pago.iloc[-1, 0]}% vs {tasa_pago.iloc[0, 0]}% de quienes pagan 0-25%.")
        st.write(
            f"3. **Edad:** la renovación sube con la edad, de {tasa_edad.iloc[0, 0]}% (<30 años) "
            f"a {tasa_edad.iloc[-1, 0]}% (60+).")
        st.write(
            f"4. **Canal:** el canal A (el más grande) renueva {tasa_canal.loc['A', 'Tasa renovación %']}% y "
            f"el D {tasa_canal.loc['D', 'Tasa renovación %']}%.")
        tasa_area = analizador.tasa_renovacion_por("residence_area_type")
        st.write(
            f"5. **Residencia no diferencia:** Urban {tasa_area.loc['Urban', 'Tasa renovación %']}% vs "
            f"Rural {tasa_area.loc['Rural', 'Tasa renovación %']}%.")



# CONCLUSIONES
# =========================================================
elif seccion == "Conclusiones":
    st.title("Conclusiones finales")

    if st.session_state.df is None:
        st.warning("Primero carga el dataset en el módulo **Carga del dataset**.")
        st.stop()

    analizador = DataAnalyzer(st.session_state.df)
    df = analizador.df

    tasa_total = calcular_tasa_renovacion(df)
    tasa_atrasos = analizador.tasa_renovacion_por("total_atrasos", CORTES_ATRASOS, ETIQUETAS_ATRASOS)
    tasa_pago = analizador.tasa_renovacion_por("perc_premium_paid_by_cash_credit",
                                               CORTES_PAGO_EFECTIVO, ETIQUETAS_PAGO_EFECTIVO)
    tasa_edad = analizador.tasa_renovacion_por("edad_anios", CORTES_EDAD, ETIQUETAS_EDAD)
    tasa_canal = analizador.tasa_renovacion_por("sourcing_channel")
    pct_con_atrasos = (df["total_atrasos"] > 0).mean() * 100
    medianas_ingreso = df.groupby("renovacion")["Income"].median()

    col1, col2, col3 = st.columns(3)
    col1.metric("Tasa de renovación global", formatear_porcentaje(tasa_total))
    col2.metric("Clientes con algún atraso", formatear_porcentaje(pct_con_atrasos))
    col3.metric("Renovación con 3+ atrasos", formatear_porcentaje(tasa_atrasos.iloc[-1, 0]))

    st.markdown("---")

    st.markdown("#### 1. El historial de atrasos es la principal alerta de no renovación")
    st.write(
        f"La renovación cae de {tasa_atrasos.iloc[0, 0]}% (sin atrasos) a {tasa_atrasos.iloc[-1, 0]}% "
        f"(3 o más atrasos). Como el {pct_con_atrasos:.1f}% de la cartera ya tiene al menos un atraso, "
        "conviene activar una **gestión de retención temprana** desde el primer pago demorado "
        "(recordatorios, reprogramación de cuotas, contacto del agente).")

    st.markdown("#### 2. La forma de pago de la prima anticipa el riesgo")
    st.write(
        f"Los clientes que pagan más del 75% de su prima en efectivo/crédito renuevan "
        f"{tasa_pago.iloc[-1, 0]}%, frente a {tasa_pago.iloc[0, 0]}% de los que pagan menos del 25%. "
        "Promover **débito automático** u otros medios de pago recurrentes puede mejorar la retención.")

    st.markdown("#### 3. Los clientes jóvenes requieren una estrategia diferenciada")
    st.write(
        f"La renovación crece con la edad: {tasa_edad.iloc[0, 0]}% en menores de 30 años vs "
        f"{tasa_edad.iloc[-1, 0]}% en mayores de 60. Para el segmento joven se recomiendan "
        "productos más flexibles y comunicación digital.")

    st.markdown("#### 4. El canal de captación influye en la calidad del cliente")
    st.write(
        f"El canal A concentra la mayor parte de la cartera y tiene la mejor renovación "
        f"({tasa_canal.loc['A', 'Tasa renovación %']}%), mientras que el canal D tiene la más baja "
        f"({tasa_canal.loc['D', 'Tasa renovación %']}%). Se sugiere revisar los criterios de captación "
        "y el seguimiento post-venta de los canales C, D y E.")

    st.markdown("#### 5. El ingreso ayuda, pero la zona de residencia y la prima no diferencian")
    st.write(
        f"El ingreso mediano de quienes renuevan ({formatear_numero(medianas_ingreso['Renovó'])}) es mayor "
        f"que el de quienes no renuevan ({formatear_numero(medianas_ingreso['No renovó'])}). En cambio, "
        "urbano y rural renuevan prácticamente igual y la prima mediana es la misma en ambos grupos. "
        "Por lo tanto, los recursos de retención deben enfocarse en **comportamiento de pago, edad y canal**, "
        "no en la zona geográfica.")
