import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def data_cleaning(df):
    """
    Realiza el proceso completo de limpieza de datos.
    
    Parameters:
    df (pd.DataFrame): El DataFrame original.
    
    Returns:
    pd.DataFrame: El DataFrame limpio.
    """
    print("🧾 [INFO] Iniciando limpieza de datos...")

    df = standardize_column_names(df)
    df = eliminar_duplicados(df)
    df = drop_irrelevant_columns(df)
    df = imputar_valores_nulos(df)
    save_cleaned_data(df)
    
    return df

def standardize_column_names(df):
    """
    Estandariza los nombres de columnas a minúsculas, sin espacios ni símbolos.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )

    print("🧾 [COLUMNS] Nombres de columnas estandarizados.")
    print("🧾 [INFO] Columnas actuales en el dataset:\n", df.columns.tolist())
    
    return df

def eliminar_duplicados(df):
    """
    Elimina filas duplicadas y reinicia el índice.
    """
    original_shape = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"✅ Duplicados eliminados. Filas eliminadas: {original_shape - df.shape[0]}")
    return df

def drop_irrelevant_columns(df):
    """
    Elimina columnas vacías, constantes y seleccionadas manualmente.
    """
    protected_cols = []
    manual_drop = ["region_1", "region_2", "taster_twitter_handle", "designation"]

    empty_cols = df.columns[df.isnull().all()].tolist()
    constant_cols = df.columns[df.nunique() <= 1].tolist()
    
    all_to_drop = list(set(empty_cols + constant_cols + manual_drop))
    final_to_drop = [col for col in all_to_drop if col in df.columns and col not in protected_cols]

    print("📌 Columnas a eliminar:", final_to_drop)
    
    df = df.drop(columns=final_to_drop, errors='ignore')
    print("✅ Limpieza de columnas irrelevantes completada.")
    
    return df

def imputar_valores_nulos(df):
    """
    Imputa valores nulos en columnas clave y elimina filas con nulos restantes.
    """
    print("🔍 Imputando valores nulos...")

    df["taster_name"] = df["taster_name"].fillna("Unknown")

    # Imputar precio por grupo
    df['price'] = df['price'].fillna(
        df.groupby(['variety', 'province'])['price'].transform('median')
    )
    df['price'] = df['price'].fillna(df['price'].median())

    # Visualizar nulos
    df.isnull().sum().sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
    plt.title("Valores nulos por columna")
    plt.xlabel("Columnas")
    plt.ylabel("Cantidad de valores nulos")
    plt.show()

    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Mapa de calor de valores nulos")
    plt.show()

    # Correlación solo con columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()


    # Eliminar filas con nulos restantes
    before = df.shape[0]
    df.dropna(inplace=True)
    after = df.shape[0]
    print(f"⚠️ Filas eliminadas por nulos restantes: {before - after}")

    return df

import os

def save_cleaned_data(df):
    """
    Guarda el DataFrame limpio en un archivo CSV, creando el directorio si no existe.
    """


    df.to_csv("data/cleaned_wine_data.csv")
    print("✅ Datos limpios guardados en: cleaned_wine_data.csv")

