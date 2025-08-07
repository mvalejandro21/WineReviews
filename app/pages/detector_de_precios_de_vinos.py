import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# SOLUCIÓN AL ERROR: Aumentar el límite de celdas renderizables
pd.set_option("styler.render.max_elements", 1000000)

# Configuración de página
st.set_page_config(
    page_title="🍷 Wine Value Analyzer",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar datos con caché
@st.cache_data
def load_data():
    df = pd.read_csv("../data/wine_final_dataset.csv")
    return df

df = load_data()

# Inicializar estados de sesión
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = df['country'].unique().tolist()
    
if 'selected_varieties' not in st.session_state:
    st.session_state.selected_varieties = df['variety'].unique().tolist()

# Sidebar para controles
with st.sidebar:
    st.header("⚙️ Parámetros de Análisis")
    st.subheader("Umbrales de Categorización")
    price_threshold = st.slider("Desviación de Precio", 1, 20, 5)
    points_threshold = st.slider("Diferencia de Puntos", 1, 10, 2)
    
    st.subheader("Filtros de Datos")
    min_price, max_price = st.slider("Rango de Precio", 
                                     float(df['price'].min()), 
                                     float(df['price'].max()), 
                                     (float(df['price'].min()), float(df['price'].max())))
    min_points, max_points = st.slider("Rango de Puntos", 
                                       int(df['points'].min()), 
                                       int(df['points'].max()), 
                                       (int(df['points'].min()), int(df['points'].max())))
    
    # MEJORA: Selector de países con opción "Todos" y paginación
    st.markdown("**Países**")
    all_countries = st.checkbox("Seleccionar todos los países", 
                               value=True,
                               key='all_countries')
    
    if all_countries:
        selected_countries = df['country'].unique().tolist()
    else:
        # MEJORA: Búsqueda y paginación para muchos países
        search_country = st.text_input("Buscar país...", key='search_country')
        country_options = df['country'].unique()
        if search_country:
            country_options = [c for c in country_options if search_country.lower() in c.lower()]
        
        page_size = 20
        page_num = st.number_input("Página", min_value=1, max_value=len(country_options)//page_size + 1, value=1)
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        
        selected_countries = st.multiselect(
            "Selecciona países:",
            options=country_options[start_idx:end_idx],
            default=st.session_state.selected_countries if st.session_state.get('country_select') else None,
            key='country_select'
        )
        st.caption(f"Mostrando {start_idx+1}-{min(end_idx, len(country_options))} de {len(country_options)} países")

    # MEJORA: Selector de variedades con opción "Todos" y paginación
    st.markdown("**Variedades**")
    all_varieties = st.checkbox("Seleccionar todas las variedades", 
                               value=True,
                               key='all_varieties')
    
    if all_varieties:
        selected_varieties = df['variety'].unique().tolist()
    else:
        # MEJORA: Búsqueda y paginación para muchas variedades
        search_variety = st.text_input("Buscar variedad...", key='search_variety')
        variety_options = df['variety'].unique()
        if search_variety:
            variety_options = [v for v in variety_options if search_variety.lower() in v.lower()]
        
        page_size = 20
        page_num = st.number_input("Página", min_value=1, max_value=len(variety_options)//page_size + 1, value=1, key='page_variety')
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        
        selected_varieties = st.multiselect(
            "Selecciona variedades:",
            options=variety_options[start_idx:end_idx],
            default=st.session_state.selected_varieties if st.session_state.get('variety_select') else None,
            key='variety_select'
        )
        st.caption(f"Mostrando {start_idx+1}-{min(end_idx, len(variety_options))} de {len(variety_options)} variedades")
    
    # MEJORA: Control para limitar resultados
    max_display_rows = st.slider("Máximo de filas a mostrar", 100, 5000, 1000, step=100)
    
    # Actualizar estados de sesión
    st.session_state.selected_countries = selected_countries
    st.session_state.selected_varieties = selected_varieties

# Filtrar datos
filtered_df = df[
    (df['price'].between(min_price, max_price)) &
    (df['points'].between(min_points, max_points)) &
    (df['country'].isin(selected_countries)) &
    (df['variety'].isin(selected_varieties))
]

# MEJORA: Limitar tamaño para evitar problemas de rendimiento
if len(filtered_df) > max_display_rows:
    st.warning(f"⚠️ Se mostrarán solo las primeras {max_display_rows} filas de {len(filtered_df)} resultados")
    filtered_df = filtered_df.head(max_display_rows)

# Entrenar modelo solo con datos filtrados
@st.cache_resource
def train_model(data):
    categorical = ["country", "variety"]
    numerical = ["points"]
    
    features = ["points", "country", "variety"]
    target = "price"
    
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")
    
    model = Pipeline([
        ("pre", preprocessor),
        ("reg", LinearRegression())
    ])
    
    X = data[features]
    y = data[target]
    model.fit(X, y)
    return model

# Solo entrenar si hay datos suficientes
if len(filtered_df) > 10:
    model = train_model(filtered_df)
else:
    st.error("⚠️ No hay suficientes datos para generar recomendaciones. Amplía los filtros.")
    st.stop()

features = ["points", "country", "variety"]
target = "price"
# Predicción y análisis
filtered_df["predicted_price"] = model.predict(filtered_df[features])
filtered_df["deviation"] = filtered_df["price"] - filtered_df["predicted_price"]
filtered_df["score"] = filtered_df["points"] - filtered_df["points"].mean()

# Definir categorías con umbrales ajustables
filtered_df["categoria"] = filtered_df.apply(
    lambda row: "💎 Hidden Gem" if row["deviation"] < -price_threshold and row["score"] > points_threshold
    else "💸 Inflado" if row["deviation"] > price_threshold and row["score"] < -points_threshold
    else "⭐ Alto Valor" if row["deviation"] < 0 and row["score"] > 0
    else "⚖️ Precio Justo", axis=1
)

# Interfaz principal
st.title("🍷 Wine Value Analyzer")
st.markdown("### Descubre vinos con mejor relación calidad-precio")
st.divider()

# Mostrar selecciones actuales
st.info(f"""
**Filtros aplicados:**  
🗺️ Países: {len(selected_countries)} seleccionados | 
🍇 Variedades: {len(selected_varieties)} seleccionadas | 
💰 Rango precio: €{min_price:.2f}-€{max_price:.2f} | 
⭐ Rango puntos: {min_points}-{max_points} |
📊 Mostrando: {len(filtered_df)} vinos
""")

# ... (El resto del código permanece igual hasta la sección de la tabla) ...

# Tabla de resultados
st.divider()
st.subheader("🍇 Detalle de Vinos")

if len(filtered_df) > 0:
    categoria_seleccionada = st.selectbox(
        "Filtrar por categoría:", 
        options=filtered_df["categoria"].unique(),
        index=0
    )

    # Formatear valores para la tabla
    display_df = filtered_df[filtered_df["categoria"] == categoria_seleccionada][
        ["title", "country", "variety", "points", "price", "predicted_price", "deviation", "categoria"]
    ].sort_values("deviation" if categoria_seleccionada in ["💎 Hidden Gem", "💸 Inflado"] else "points", 
                ascending=categoria_seleccionada == "💎 Hidden Gem")

    display_df = display_df.rename(columns={
        "title": "Vino",
        "country": "País",
        "variety": "Variedad",
        "points": "Puntos",
        "price": "Precio Real",
        "predicted_price": "Precio Estimado",
        "deviation": "Desviación",
        "categoria": "Categoría"
    })

    # MEJORA: Función para aplicar estilos sin sobrecargar
    def highlight_row(row):
        if row['Categoría'] == '💎 Hidden Gem':
            return ['background-color: #e6f7ea'] * len(row)
        elif row['Categoría'] == '💸 Inflado':
            return ['background-color: #fde8e8'] * len(row)
        elif row['Categoría'] == '⭐ Alto Valor':
            return ['background-color: #e6f0ff'] * len(row)
        else:
            return ['background-color: #f4e6ff'] * len(row)
    
    # MEJORA: Formatear solo las columnas numéricas
    format_dict = {
        "Precio Real": "{:.2f}€",
        "Precio Estimado": "{:.2f}€",
        "Desviación": "{:.2f}€"
    }
    
    # Mostrar tabla con estilo optimizado
    st.dataframe(
        display_df.style
            .format(format_dict)
            .apply(highlight_row, axis=1),
        use_container_width=True,
        height=600
    )
else:
    st.warning("No hay vinos que coincidan con los filtros seleccionados")

# Footer
st.divider()
st.caption("🔍 Los precios estimados se calculan mediante un modelo de regresión que considera puntos, país y variedad")
st.caption("💎 'Hidden Gems': Vinos con precio bajo y calidad alta | 💸 'Inflados': Vinos con precio alto y calidad baja")
st.caption(f"📊 Total de vinos en análisis: {len(filtered_df)}")