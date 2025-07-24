import spacy
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Activar tqdm para pandas
tqdm.pandas(desc="🧠 Procesando descripciones")

# Cargar modelo spaCy
print("🔄 Cargando modelo de lenguaje spaCy...")
nlp = spacy.load("en_core_web_sm")
print("✅ Modelo cargado.\n")

# Cargar dataset
print("📂 Cargando dataset...")
df = pd.read_csv("data/cleaned_wine_data.csv")
df['description'] = df['description'].astype(str)
print(f"✅ Dataset cargado con {len(df)} filas.\n")

# Función para extraer sabores (sustantivos y adjetivos con más de 3 letras)
def extraer_sabores(texto):
    doc = nlp(texto.lower())
    return [token.lemma_ for token in doc 
            if token.pos_ in ["ADJ", "NOUN"] and len(token.lemma_) > 3]

# Aplicar extracción de sabores con barra de progreso
print("🔍 Extrayendo sabores de las descripciones...")
df['sabores_detectados'] = df['description'].progress_apply(extraer_sabores)
print("✅ Extracción completada.\n")

# Sumar todas las listas de sabores en una sola (con barra manual)
print("📊 Unificando todas las palabras detectadas...")
todas_palabras = []
for lista in tqdm(df['sabores_detectados'], desc="📦 Unificando listas"):
    todas_palabras.extend(lista)
print(f"✅ Unificación completada. Total de palabras: {len(todas_palabras)}\n")

# Contar ocurrencias (puede tomar algo de tiempo si hay muchas palabras)
print("📈 Contando frecuencia de sabores...")
conteo = Counter()
for palabra in tqdm(todas_palabras, desc="🔢 Contando palabras"):
    conteo[palabra] += 1
print("✅ Conteo completado.\n")

# Mostrar top 50 sabores más frecuentes
sabores_comunes = conteo.most_common(50)
print("🍷 Sabores más comunes:")
print(pd.DataFrame(sabores_comunes, columns=["Sabor", "Frecuencia"]))