import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Configuración inicial de la página
st.set_page_config(
    page_title="🍷 Sommelier AI - Recomendador de Vinos",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    # Lista de rutas posibles a probar
    possible_paths = [
        # Rutas relativas
        "data/wine_final_dataset.csv",
        "../data/wine_final_dataset.csv", 
        "./data/wine_final_dataset.csv",
        "wine_final_dataset.csv",
        
        # Rutas absolutas típicas
        "/mount/src/winereviews/data/wine_final_dataset.csv",
        "/app/data/wine_final_dataset.csv",
        
        # Rutas desde el directorio del script
        str(Path(__file__).parent / "data" / "wine_final_dataset.csv"),
        str(Path(__file__).parent.parent / "data" / "wine_final_dataset.csv"),
        
        # Otras rutas posibles
        "app/data/wine_final_dataset.csv",
    ]
    
    # Probar cada ruta
    for csv_path in possible_paths:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                print(f"✅ Archivo encontrado en: {csv_path}")
                return df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ Error con {csv_path}: {str(e)}")
            continue
    
    # Si ninguna ruta funciona, mostrar error
    raise FileNotFoundError("No se pudo encontrar wine_final_dataset.csv en ninguna ruta probada")

# Cargar datos
try:
    df = load_data()
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.info("""
    **Solución:**
    1. Asegúrate de que el archivo wine_final_dataset.csv existe
    2. Verifica que esté en la carpeta data/ del proyecto
    3. Si estás en Streamlit Cloud, sube el archivo a GitHub
    """)
    st.stop()

df = load_data()

# Preparación de datos
@st.cache_data
def preparar_datos(_df):
    feature_cols = _df.drop(columns=['wine_id', 'title']).select_dtypes(include=['number', 'bool']).columns
    return _df[feature_cols].astype(float), feature_cols

X, feature_cols = preparar_datos(df)

# Modelo de recomendación
@st.cache_resource(show_spinner=False)
def entrenar_modelo(_X):
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
    model.fit(_X)
    return model

model = entrenar_modelo(X)

# Función de recomendación mejorada
def recomendar_vinos(titulo, top_n=5):
    if titulo not in df['title'].values:
        return None

    idx = df[df['title'] == titulo].index[0]
    distances, indices = model.kneighbors([X.iloc[idx]], n_neighbors=top_n+1)
    
    # Excluir el vino de consulta y obtener recomendaciones
    rec_indices = indices[0][1:]
    recomendaciones = df.iloc[rec_indices].copy()
    recomendaciones['Similitud'] = (1 - distances[0][1:]).round(3)
    
    return recomendaciones

# Función para gráfico de sabores
def plot_flavor_profile(selected_wine, recommended_wine, features):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Valores para el vino seleccionado y recomendado
    selected_values = selected_wine[features].values
    recommended_values = recommended_wine[features].values
    
    # Crear gráfico de barras comparativo
    bar_width = 0.35
    index = np.arange(len(features))
    
    bars1 = ax.bar(index, selected_values, bar_width, 
                  label=f"Vino seleccionado: {selected_wine['title']}",
                  color='#9b59b6', alpha=0.8)
    
    bars2 = ax.bar(index + bar_width, recommended_values, bar_width, 
                  label=f"Recomendado: {recommended_wine['title']}",
                  color='#3498db', alpha=0.8)
    
    # Configuración del gráfico
    ax.set_xlabel('Características de Sabor')
    ax.set_ylabel('Intensidad')
    ax.set_title('Comparación de Perfiles de Sabor')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f.capitalize() for f in features], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

# --- INTERFAZ MEJORADA ---
st.title("🍷 Sommelier AI")
st.markdown("### Descubre vinos similares a tus preferencias")
st.divider()

# Sidebar para filtros y controles
with st.sidebar:
    st.header("⚙️ Configuración Avanzada")
    st.subheader("Filtros de Recomendación")
    num_recomendaciones = st.slider(
        "Número de recomendaciones", 
        min_value=3, 
        max_value=10, 
        value=5,
        help="Cantidad de vinos similares a mostrar"
    )
    
    umbral_similitud = st.slider(
        "Umbral mínimo de similitud",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Filtrar recomendaciones con similitud mínima"
    )
    
    st.markdown("---")
    st.subheader("Filtros por Atributos")
    paises = ['Todos'] + sorted(df['country'].dropna().unique().tolist())
    pais_seleccionado = st.selectbox("País", paises)
    
    st.markdown("---")
    st.caption("© 2023 Sommelier AI • Todos los derechos reservados")

# Selector de vino principal
with st.container():
    st.subheader("🍇 Selecciona un vino que te guste")
    vino_seleccionado = st.selectbox(
        "Busca en nuestra bodega:",
        options=sorted(df['title'].unique()),
        index=150,
        help="Comienza a escribir para buscar en nuestra base de vinos",
        label_visibility="collapsed"
    )

# Resultados con formato mejorado
if vino_seleccionado:
    st.divider()
    
    # Información del vino seleccionado
    selected_wine = df[df['title'] == vino_seleccionado].iloc[0]
    
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Placeholder para imagen del vino
            st.image("https://cdn.pixabay.com/photo/2015/11/07/12/00/alcohol-1031713_960_720.png", 
                     caption="Imagen ilustrativa", width=150)
        
        with col2:
            st.markdown(f"### {selected_wine['title']}")
            
            # Metadatos en columnas
            meta1, meta2, meta3 = st.columns(3)
            with meta1:
                st.metric("País", selected_wine.get('country', 'N/A'))
            with meta2:
                st.metric("Variedad", selected_wine.get('variety', 'N/A'))
            with meta3:
                st.metric("Puntuación", f"{selected_wine.get('points', 'N/A')}/100")
            
            # Precio y características
            st.metric("Precio Estimado", f"${selected_wine.get('price', 'N/A'):.2f}" 
                     if 'price' in df.columns and pd.notna(selected_wine.get('price')) else "N/A")
            
            # Top características de sabor
            if 'features' in locals():
                top_features = selected_wine[feature_cols].sort_values(ascending=False).head(3).index
                st.markdown("**Perfil de sabor dominante:**")
                st.markdown(", ".join([f.capitalize() for f in top_features]))
    
    # Espaciador
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    
    # Título de sección de recomendaciones
    st.subheader(f"✨ Vinos Recomendados")
    st.caption(f"Basado en similitud con: {vino_seleccionado}")
    
    with st.spinner('Buscando en nuestra bodega...'):
        recomendaciones = recomendar_vinos(vino_seleccionado, num_recomendaciones)
    
    if recomendaciones is not None:
        # Aplicar filtros
        if pais_seleccionado != 'Todos':
            recomendaciones = recomendaciones[recomendaciones['country'] == pais_seleccionado]
        recomendaciones = recomendaciones[recomendaciones['Similitud'] >= umbral_similitud]
        
        if recomendaciones.empty:
            st.warning("No se encontraron vinos que coincidan con tus criterios. Intenta ajustar los filtros.")
        else:
            # Mostrar recomendaciones en pestañas
            tabs = st.tabs([f"#{i+1}" for i in range(len(recomendaciones))])
            
            for i, tab in enumerate(tabs):
                with tab:
                    rec = recomendaciones.iloc[i]
                    
                    col_left, col_right = st.columns([1, 2])
                    
                    with col_left:
                        # Placeholder para imagen del vino recomendado
                        st.image("https://cdn.pixabay.com/photo/2015/11/07/12/00/alcohol-1031713_960_720.png", 
                                 caption="Imagen ilustrativa", width=150)
                        
                        # Barra de similitud
                        st.progress(rec['Similitud'], 
                                   text=f"**Similitud:** {rec['Similitud']:.2f}")
                        
                        # Metadatos
                        st.markdown(f"**País:** {rec.get('country', 'N/A')}")
                        st.markdown(f"**Variedad:** {rec.get('variety', 'N/A')}")
                        st.markdown(f"**Puntuación:** ⭐ {rec.get('points', 'N/A')}/100")
                        st.markdown(f"**Precio:** 💵 ${rec.get('price', 'N/A'):.2f}")
                    
                    with col_right:
                        # Gráfico comparativo de sabores
                        top_features = selected_wine[feature_cols].sort_values(ascending=False).head(5).index
                        fig = plot_flavor_profile(selected_wine, rec, top_features)
                        st.pyplot(fig)
                        
                        # Botón para más detalles (funcionalidad de ejemplo)
                        if st.button("Ver detalles completos", key=f"btn_{i}"):
                            st.session_state.selected_wine = rec['title']

    else:
        st.error("Vino no encontrado en nuestra base de datos", icon="🚨")

# Pie de página mejorado
st.divider()
st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
    <p style="color: #6c757d;">Sommelier AI utiliza algoritmos de aprendizaje automático para recomendarte vinos basados en tus preferencias</p>
    <p>¿Preguntas o sugerencias? <a href="mailto:contacto@sommelier-ai.com">contacto@sommelier-ai.com</a></p>
</div>
""", unsafe_allow_html=True)