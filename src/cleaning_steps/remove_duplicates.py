

def remove_duplicates(df):
   
    # Eliminar duplicados
    rows_before = df.shape[0]
    df = df.drop_duplicates()
    rows_after = df.shape[0]
    removed = rows_before - rows_after

    print(f"🧹 [CLEANING] Se eliminaron {removed} filas duplicadas. Quedan {rows_after} filas.")

    return df


