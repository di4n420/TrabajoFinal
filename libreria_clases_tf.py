import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from libreria_funciones_tf import clasificar_variables, crear_variables_derivadas

# Colores para "No renovó" (rojo) y "Renovó" (azul) en todos los gráficos
PALETA_RENOVACION = {"No renovó": "#E4572E", "Renovó": "#2E86AB"}
COLOR_BARRAS = "#2E86AB"


# =========================================================
# PASO 2 - CLASE DataAnalyzer
# Encapsula: clasificación de variables, estadísticas y visualizaciones
# =========================================================
class DataAnalyzer:

    def __init__(self, df):
        # Atributos del objeto
        self.df_original = df
        self.df = crear_variables_derivadas(df)
        clasificacion = clasificar_variables(
            self.df,
            columnas_excluir=["id", "age_in_days", "renovacion"],
            columnas_categoricas_extra=["renewal"]
        )
        self.columnas_numericas = clasificacion["numericas"]
        self.columnas_categoricas = clasificacion["categoricas"]

    # -----------------------------------------------------
    # ÍTEM 1 - Información general
    # -----------------------------------------------------
    def texto_info(self):
        """Captura el resultado de df.info() como texto para mostrarlo en Streamlit."""
        buffer = io.StringIO()
        self.df_original.info(buf=buffer)
        return buffer.getvalue()

    def resumen_columnas(self):
        """Tabla con tipo de dato, nulos y valores únicos por columna."""
        resumen = pd.DataFrame({
            "Tipo de dato": self.df_original.dtypes.astype(str),
            "Valores nulos": self.df_original.isnull().sum(),
            "Valores únicos": self.df_original.nunique()
        })
        return resumen

    # -----------------------------------------------------
    # ÍTEM 2 - Clasificación de variables
    # -----------------------------------------------------
    def conteo_tipos(self):
        return {
            "Numéricas": len(self.columnas_numericas),
            "Categóricas": len(self.columnas_categoricas)
        }

    # -----------------------------------------------------
    # ÍTEM 3 - Estadísticas descriptivas
    # -----------------------------------------------------
    def estadisticas_descriptivas(self, columnas=None):
        if columnas is None:
            columnas = self.columnas_numericas
        tabla = self.df[columnas].describe().T
        tabla["mediana"] = self.df[columnas].median()
        tabla["moda"] = self.df[columnas].mode().iloc[0]
        # Coeficiente de variación: dispersión relativa a la media
        tabla["coef_variacion_%"] = (tabla["std"] / tabla["mean"]) * 100
        return tabla.round(2)

    def moda_categoricas(self):
        filas = []
        for columna in self.columnas_categoricas:
            moda = self.df[columna].mode().iloc[0]
            frecuencia = (self.df[columna] == moda).mean() * 100
            filas.append({"Variable": columna, "Moda": str(moda), "% de registros": round(frecuencia, 1)})
        return pd.DataFrame(filas)

    # -----------------------------------------------------
    # ÍTEM 4 - Valores faltantes
    # -----------------------------------------------------
    def valores_faltantes(self):
        nulos = self.df_original.isnull().sum()
        tabla = pd.DataFrame({
            "Nulos": nulos,
            "% Nulos": (nulos / len(self.df_original) * 100).round(2)
        })
        return tabla[tabla["Nulos"] > 0].sort_values("Nulos", ascending=False)

    def grafico_faltantes(self):
        tabla = self.valores_faltantes()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.barh(tabla.index, tabla["% Nulos"], color="#E4572E")
        for i, valor in enumerate(tabla["% Nulos"]):
            ax.text(valor, i, f" {valor:.2f}%", va="center")
        ax.set_xlabel("% de registros nulos")
        ax.set_title("Porcentaje de valores faltantes por variable")
        fig.tight_layout()
        return fig

    # -----------------------------------------------------
    # ÍTEM 5 - Distribución de variables numéricas
    # -----------------------------------------------------
    def filtrar_outliers(self, columna, percentil):
        """Quita los valores por encima del percentil elegido (útil para Income)."""
        limite = np.percentile(self.df[columna].dropna(), percentil)
        return self.df[self.df[columna] <= limite]

    def grafico_histograma(self, columna, bins=30, percentil=100, separar_por_renovacion=False):
        datos = self.filtrar_outliers(columna, percentil)
        fig, ax = plt.subplots(figsize=(8, 4))
        if separar_por_renovacion:
            sns.histplot(data=datos, x=columna, hue="renovacion", bins=bins, stat="density",
                         common_norm=False, palette=PALETA_RENOVACION, ax=ax)
        else:
            sns.histplot(data=datos, x=columna, bins=bins, color=COLOR_BARRAS, ax=ax)
        ax.axvline(datos[columna].mean(), color="black", linestyle="--", label="Media")
        ax.axvline(datos[columna].median(), color="orange", linestyle="-", label="Mediana")
        ax.set_title(f"Distribución de {columna}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    # -----------------------------------------------------
    # ÍTEM 6 - Variables categóricas
    # -----------------------------------------------------
    def tabla_frecuencias(self, columna):
        conteo = self.df[columna].value_counts()
        tabla = pd.DataFrame({
            "Conteo": conteo,
            "Proporción %": (conteo / conteo.sum() * 100).round(2)
        })
        return tabla

    def grafico_barras(self, columna):
        tabla = self.tabla_frecuencias(columna)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(tabla.index.astype(str), tabla["Conteo"], color=COLOR_BARRAS)
        for i, valor in enumerate(tabla["Proporción %"]):
            ax.text(i, tabla["Conteo"].iloc[i], f"{valor:.1f}%", ha="center", va="bottom")
        ax.set_title(f"Frecuencia de {columna}")
        ax.set_ylabel("Número de clientes")
        fig.tight_layout()
        return fig

    # -----------------------------------------------------
    # ÍTEM 7 - Bivariado: numérico vs categórico
    # -----------------------------------------------------
    def comparar_grupos(self, columna_numerica, columna_categorica):
        tabla = self.df.groupby(columna_categorica)[columna_numerica].agg(
            ["count", "mean", "median", "std"]
        )
        tabla.columns = ["Clientes", "Media", "Mediana", "Desv. estándar"]
        return tabla.round(2)

    def grafico_boxplot(self, columna_numerica, columna_categorica, percentil=100):
        datos = self.filtrar_outliers(columna_numerica, percentil)
        fig, ax = plt.subplots(figsize=(7, 4))
        if columna_categorica == "renovacion":
            sns.boxplot(data=datos, x=columna_categorica, y=columna_numerica,
                        hue=columna_categorica, palette=PALETA_RENOVACION, legend=False, ax=ax)
        else:
            sns.boxplot(data=datos, x=columna_categorica, y=columna_numerica, color=COLOR_BARRAS, ax=ax)
        ax.set_title(f"{columna_numerica} según {columna_categorica}")
        fig.tight_layout()
        return fig

    # -----------------------------------------------------
    # ÍTEM 8 - Bivariado: categórico vs categórico
    # -----------------------------------------------------
    def tabla_cruzada(self, columna_1, columna_2, normalizar=True):
        if normalizar:
            return (pd.crosstab(self.df[columna_1], self.df[columna_2], normalize="index") * 100).round(2)
        return pd.crosstab(self.df[columna_1], self.df[columna_2])

    def grafico_barras_apiladas(self, columna_1, columna_2):
        tabla = self.tabla_cruzada(columna_1, columna_2)
        fig, ax = plt.subplots(figsize=(7, 4))
        colores = None
        if columna_2 == "renovacion":
            colores = [PALETA_RENOVACION[c] for c in tabla.columns]
        tabla.plot(kind="bar", stacked=True, color=colores, ax=ax)
        ax.set_ylabel("% de clientes")
        ax.set_title(f"{columna_2} según {columna_1} (%)")
        ax.legend(title=columna_2, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------
    # ÍTEM 10 - Hallazgos: tasa de renovación por grupos
    # -----------------------------------------------------
    def tasa_renovacion_por(self, columna, cortes=None, etiquetas=None):
        """Tasa de renovación (%) por categoría o por tramos de una variable numérica."""
        if cortes is not None:
            grupos = pd.cut(self.df[columna], bins=cortes, labels=etiquetas)
        else:
            grupos = self.df[columna]
        tabla = self.df.groupby(grupos, observed=True)["renewal"].agg(["mean", "count"])
        tabla.columns = ["Tasa renovación %", "Clientes"]
        tabla["Tasa renovación %"] = (tabla["Tasa renovación %"] * 100).round(1)
        return tabla
