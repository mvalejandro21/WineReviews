import spacy
import pandas as pd
from tqdm import tqdm

# Activar tqdm para pandas
tqdm.pandas(desc="🧠 Procesando descripciones")

# Cargar modelo spaCy
print("🔄 Cargando modelo de lenguaje spaCy...")
nlp = spacy.load("en_core_web_sm")
print("✅ Modelo cargado.\n")

# Cargar dataset asegurando que wine_id sea el índice original
print("📂 Cargando dataset...")
df = pd.read_csv("data/cleaned_wine_data.csv")
df.rename(columns={"Unnamed: 0": "wine_id"}, inplace=True)
df['description'] = df['description'].astype(str)
print(f"✅ Dataset cargado con {len(df)} filas.\n")

# Función para extraer sabores (adjetivos y sustantivos lematizados)
def extraer_sabores(texto):
    doc = nlp(texto.lower())
    return [token.lemma_ for token in doc 
            if token.pos_ in ["ADJ", "NOUN"] and len(token.lemma_) > 3]

# Aplicar extracción de sabores
print("🔍 Extrayendo sabores de las descripciones...")
df['sabores_detectados'] = df['description'].progress_apply(extraer_sabores)
print("✅ Extracción completada.\n")

# Lista de sabores que sí te interesan
safe_words = [
    "fruit", "tannin", "cherry", "ripe", "spice", "fresh", "berry", "plum", "apple", "soft", "blackberry", 
    "sweet", "light", "crisp", "citrus", "vanilla", "herb", "raspberry", "pepper", "juicy", "lemon", 
    "fruity", "firm", "chocolate", "currant"
]

# Crear columnas dummy por cada sabor
print("🎯 Generando columnas dummy por sabor...\n")
for sabor in safe_words:
    df[sabor] = df['sabores_detectados'].apply(lambda lista: 1 if sabor in lista else 0)

# Crear nuevo DataFrame con wine_id, title y sabores
print("📦 Creando DataFrame final con 'wine_id', 'title' y sabores...")
df_sabores = df[['wine_id', 'title'] + safe_words]

# Guardar como CSV sin index
output_path = "data/vinos_con_sabores_dummies.csv"
df_sabores.to_csv(output_path, index=False)
df.to_csv("data/cleaned_wine_data_with_sabores.csv", index=False)
import os

# Ruta del archivo que quieres eliminar
archivo = 'data/cleaned_wine_data.csv'
# Eliminar el archivo
os.remove(archivo)
print(f"✅ Dataset final guardado en: {output_path}")
