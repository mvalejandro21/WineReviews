

def drop_irrelevant_columns(df):
    """
    Elimina columnas vacías, constantes o innecesarias para el análisis.
    """

    print("🧾 [INFO] Columnas actuales en el dataset:")

    print(df.columns.tolist())  # Muestra todas las columnas actuales

    # Detectar columnas vacías
    empty_cols = df.columns[df.isnull().all()].tolist()

    # Detectar columnas constantes
    constant_cols = df.columns[df.nunique() <= 1].tolist()

    # Lista manual de columnas que no queremos usar
    manual_drop = ["taster_twitter_handle"]

    cols_to_drop = list(set(empty_cols + constant_cols + manual_drop))

    print("🧾 [INFO] Columnas a eliminar:")
    print(cols_to_drop)

    return df