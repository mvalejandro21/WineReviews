def standardize_column_names(df):
    """
    Limpia los nombres de columnas: minúsculas, sin espacios ni caracteres raros.
    """
    df.columns = (
        df.columns
        .str.strip()               # elimina espacios al inicio/fin
        .str.lower()               # convierte a minúsculas
        .str.replace(" ", "_")     # reemplaza espacios por guiones bajos
        .str.replace(r"[^\w_]", "", regex=True)  # quita símbolos raros
    )

    print("🧾 [COLUMNS] Nombres de columnas estandarizados.")
    return df