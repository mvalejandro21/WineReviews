from src.data_load import load_data
from src.cleaning_steps.remove_duplicates import remove_duplicates
from src.cleaning_steps.standarize_column_names import standardize_column_names
from src.cleaning_steps.drop_irrelevant_columns import drop_irrelevant_columns

def data_cleaning():

    df = load_data()

    if df is None:
        print("⛔ [STOP] Cancelando limpieza porque los datos no se cargaron correctamente.")
        return

    print("🔄 [INFO] Iniciando proceso de limpieza de datos...")
   

    # Eliminar duplicados
    df = remove_duplicates(df)

    # Estandarizar columnas
    df = standardize_column_names(df)

    # Eliminar columnas irrelevantes
    df = drop_irrelevant_columns(df)





