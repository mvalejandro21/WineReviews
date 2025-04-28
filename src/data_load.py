import pandas as pd

def load_data():
    try:
        df = pd.read_csv("data/raw-winemag-data-130k-v2.csv")

        if df.empty or df.shape[1] == 0:
            print("⚠️ [WARNING] CSV cargado pero está vacío o sin columnas.")
            return None

        print(f"✅ [SUCCESS] Datos cargados correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df

    except FileNotFoundError:
        print("❌ [ERROR] Archivo CSV no encontrado en la ruta especificada.")
        return None

    except pd.errors.ParserError:
        print("❌ [ERROR] Fallo al parsear el archivo CSV. ¿Está bien formado?")
        return None

    except Exception as e:
        print(f"❌ [ERROR] Error inesperado al cargar el CSV: {e}")
        return None

