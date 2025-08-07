import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Sommelier Virtual | Recomendador de Vinos",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARGAR DATOS ---
@st.cache_data
def load_data():
    return pd.read_csv("../data/wine_final_dataset.csv")

df = load_data()

# --- PREPARAR DATOS ---
columnas_excluir = ['wine_id', 'title', 'country', 'variety', 'points', 'price']
sabores_cols = [col for col in df.columns if col not in columnas_excluir and df[col].dtype in ['float64', 'int64', 'bool']]
df_sabores = df[sabores_cols].astype(float)

# --- DISEÑO DE INTERFAZ ---
st.title("🍷 Sommelier Virtual")
st.markdown("""
<div style="border-bottom: 2px solid #e0e0e0; padding-bottom: 15px; margin-bottom: 30px">
    Descubre vinos que se ajustan perfectamente a tu paladar. 
    Ajusta los perfiles de sabor según tus preferencias y encuentra tu próximo vino favorito.
</div>
""", unsafe_allow_html=True)

# --- BARRA LATERAL PARA FILTROS ---
with st.sidebar:
    st.header("⚙️ Configuración")
    top_n = st.slider("Número de recomendaciones", 3, 10, 5)
    umbral_similitud = st.slider("Umbral mínimo de similitud", 0.0, 1.0, 0.3, 0.05)
    
    st.markdown("---")
    st.subheader("Filtros Adicionales")
    paises = ['Todos'] + sorted(df['country'].dropna().unique().tolist())
    pais_seleccionado = st.selectbox("País", paises)
    
    st.markdown("---")
    st.caption("Desarrollado con ❤️ usando Streamlit")
    st.caption("Datos de vinos y aprendizaje automático")

# --- SECCIÓN DE PREFERENCIAS ---
st.subheader("🎚️ Personaliza tu Perfil de Sabor")
st.markdown("Ajusta la intensidad deseada para cada característica (0 = mínimo, 1 = máximo):")

# Organizar sliders en 3 columnas
cols = st.columns(3)
preferencias = {}
for i, sabor in enumerate(sabores_cols):
    with cols[i % 3]:
        preferencias[sabor] = st.slider(
            label=f"**{sabor.replace('_', ' ').title()}**",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help=f"Intensidad deseada de {sabor.replace('_', ' ')}"
        )

# --- BOTÓN DE RECOMENDACIÓN ---
st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    btn_recomendar = st.button("🍇 Descubrir Vinos Recomendados", use_container_width=True)

# --- PROCESAR RECOMENDACIONES ---
if btn_recomendar:
    with st.spinner("Buscando las mejores coincidencias..."):
        user_vector = pd.DataFrame([preferencias])[sabores_cols].astype(float)
        similitudes = cosine_similarity(user_vector, df_sabores)[0]
        df["similaridad"] = similitudes
        
        # Aplicar filtros
        resultados = df[df['similaridad'] >= umbral_similitud]
        if pais_seleccionado != 'Todos':
            resultados = resultados[resultados['country'] == pais_seleccionado]
        
        resultados = resultados.sort_values("similaridad", ascending=False).head(top_n)

    # --- MOSTRAR RESULTADOS ---
    if resultados.empty:
        st.warning("⚠️ No se encontraron vinos que coincidan con tus criterios. Intenta relajar los filtros.")
    else:
        st.markdown("---")
        st.subheader(f"✨ Top {len(resultados)} Recomendaciones")
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        
        for i, row in resultados.iterrows():
            with st.container():
                col_img, col_info = st.columns([1, 4])
                
                with col_img:
                    # Placeholder para imagen (implementar con URL real si está disponible)
                    st.image("https://cdn.pixabay.com/photo/2014/12/08/02/59/grape-560435_960_720.jpg", 
                             caption=row['country'] if pd.notna(row['country']) else "Vino")
                
                with col_info:
                    # Información principal
                    st.markdown(f"### 🍷 **{row['title']}**")
                    
                    # Metadatos
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.markdown(f"**País**: {row.get('country', 'N/A')}")
                    with col_meta2:
                        st.markdown(f"**Variedad**: {row.get('variety', 'N/A')}")
                    with col_meta3:
                        st.markdown(f"**Similitud**: `{row['similaridad']:.2f}`")
                    
                    # Puntuación y precio
                    col_stats1, col_stats2 = st.columns(2)
                    with col_stats1:
                        st.progress(int(row.get('points', 0)), text=f"⭐ **{row.get('points', 'N/A')}/100 Puntos**")
                    with col_stats2:
                        precio = row.get('price', 0)
                        st.metric("💰 Precio Estimado", f"${precio:.2f}" if precio else "N/A")
                    
                    # Gráfico de sabores relevantes
                    st.markdown("**Perfil de Sabores:**")
                    sabores_relevantes = {s: row[s] for s in sabores_cols if preferencias.get(s, 0) > 0.1}
                    if sabores_relevantes:
                        fig, ax = plt.subplots(figsize=(8, 2))
                        ax.barh(
                            list(sabores_relevantes.keys()), 
                            list(sabores_relevantes.values()),
                            color='#9b59b6'
                        )
                        ax.set_xlim(0, 1)
                        st.pyplot(fig)
                    else:
                        st.info("Activa más sabores en tus preferencias para ver el análisis detallado")
            
            st.markdown("---")

# --- MENSAJE INICIAL ---
else:
    st.markdown("<div style='text-align: center; padding: 40px; border: 2px dashed #e0e0e0; border-radius: 10px; margin-top: 30px'>"
                "<h3 style='color: #7f8c8d'>👆 Personaliza tus preferencias y haz clic en 'Descubrir Vinos'</h3>"
                "<p>Te recomendaremos vinos que se ajusten a tu perfil de sabor</p></div>", 
                unsafe_allow_html=True)

# --- PIE DE PÁGINA ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #7f8c8d">
    <p>Sommelier Virtual v1.0 | Proyecto de recomendación de vinos usando Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)