import numpy as np
import pandas as pd


# =========================================================
# PASO 1 - FUNCIONES PERSONALIZADAS
# Funciones simples y reutilizables que usa la clase DataAnalyzer y el app.py
# =========================================================

# Columnas de atrasos en el pago (las usamos varias veces)
COLUMNAS_ATRASOS = [
    "Count_3-6_months_late",
    "Count_6-12_months_late",
    "Count_more_than_12_months_late"
]


def validar_columnas(df, columnas_esperadas):
    """Verifica que el DataFrame tenga las columnas del caso InsuranceCompany.
    Retorna la lista de columnas que faltan (lista vacía si está todo bien)."""
    faltantes = []
    for columna in columnas_esperadas:
        if columna not in df.columns:
            faltantes.append(columna)
    return faltantes


def clasificar_variables(df, columnas_excluir=None, columnas_categoricas_extra=None):
    """Clasifica las columnas del DataFrame en numéricas y categóricas.

    - columnas_excluir: columnas que no se analizan (por ejemplo el id).
    - columnas_categoricas_extra: columnas numéricas que en realidad son categorías
      (por ejemplo renewal, que vale 0 o 1).
    """
    if columnas_excluir is None:
        columnas_excluir = []
    if columnas_categoricas_extra is None:
        columnas_categoricas_extra = []

    numericas = []
    categoricas = []

    for columna in df.columns:
        if columna in columnas_excluir:
            continue
        if columna in columnas_categoricas_extra:
            categoricas.append(columna)
        elif pd.api.types.is_numeric_dtype(df[columna]):
            numericas.append(columna)
        else:
            categoricas.append(columna)

    return {"numericas": numericas, "categoricas": categoricas}


def dias_a_anios(dias):
    """Convierte la edad en días a edad en años (con NumPy)."""
    return np.round(np.array(dias) / 365.25, 1)


def crear_variables_derivadas(df):
    """Agrega columnas nuevas que facilitan el análisis:
    - edad_anios: edad del cliente en años
    - total_atrasos: suma de todos los pagos atrasados
    - renovacion: etiqueta de texto para renewal (Renovó / No renovó)
    """
    df_nuevo = df.copy()
    df_nuevo["edad_anios"] = dias_a_anios(df_nuevo["age_in_days"])
    df_nuevo["total_atrasos"] = df_nuevo[COLUMNAS_ATRASOS].sum(axis=1)
    df_nuevo["renovacion"] = df_nuevo["renewal"].map({1: "Renovó", 0: "No renovó"})
    return df_nuevo


def calcular_tasa_renovacion(df):
    """Retorna la tasa de renovación en % (promedio de renewal * 100)."""
    if len(df) == 0:
        raise ValueError("No hay registros para calcular la tasa de renovación.")
    return df["renewal"].mean() * 100


def formatear_numero(valor, decimales=0):
    """Da formato con separador de miles usando f-strings."""
    return f"{valor:,.{decimales}f}"


def formatear_porcentaje(valor, decimales=1):
    """Da formato de porcentaje usando f-strings."""
    return f"{valor:.{decimales}f}%"
