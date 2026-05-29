# src/visualizacion.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict
import os

def crear_grafo_topologico(resultados_topicos: Dict[str, dict], df: pd.DataFrame) -> go.Figure:
    """
    Construye una red semántica (grafo) de las palabras más frecuentes y sus tópicos.
    """
    G = nx.Graph()
    fig = go.Figure()
    
    # [Placeholder estructural: Aquí insertaremos la lógica de nodos y aristas de NetworkX]
    # Se añade un mock visual para garantizar que el renderizador de Plotly no falle
    fig.add_trace(go.Scatter(x=[1], y=[1], text=["Red Topológica en Construcción"], mode="text"))
    fig.update_layout(title="Topología de Conceptos (NetworkX)", xaxis=dict(visible=False), yaxis=dict(visible=False))
    
    return fig

def generar_reporte(
    df_clasificado: pd.DataFrame, 
    df_outliers: pd.DataFrame,
    resultados_topicos: Dict[str, dict], 
    resultados_similitud: pd.DataFrame, 
    titulo: str, 
    paleta: str
) -> None:
    
    # Genera un reporte interactivo en formato HTML mediante CSS Grid y asegura renderizado offline.

    print(f"\nGENERANDO REPORTE: {titulo}")
    
    # 1. Gráfico de Dispersión (Codificación Redundante Forma/Color)
    fig_topicos = go.Figure()
    if not df_clasificado.empty and 'sentimiento' in df_clasificado.columns:
        tfidf = TfidfVectorizer(max_features=100)
        vectores = tfidf.fit_transform(df_clasificado[df_clasificado.columns[0]]).toarray()
        
        pca = PCA(n_components=2, random_state=42)
        coordenadas = pca.fit_transform(vectores)
        df_clasificado['x'] = coordenadas[:, 0]
        df_clasificado['y'] = coordenadas[:, 1]
        
        fig_topicos = px.scatter(
            df_clasificado,
            x='x', y='y',
            color='sentimiento', symbol='sentimiento',
            hover_data=[df_clasificado.columns[0]],
            title="Distribución Semántica",
            color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
        )
        fig_topicos.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig_topicos.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 2. Análisis del Concepto Específico
    fig_precio = go.Figure()
    if not resultados_similitud.empty and 'similitud_precio_costo' in resultados_similitud.columns:
        resultados_similitud['indice'] = range(len(resultados_similitud))
        fig_precio = px.scatter(
            resultados_similitud,
            x='indice', y='similitud_precio_costo',
            color='sentimiento',
            hover_data=[resultados_similitud.columns[0]],
            title="Proximidad Vectorial: 'Precio/Valor'",
            color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
        )
        fig_precio.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 3. Grafo Semántico
    fig_grafo = crear_grafo_topologico(resultados_topicos, df_clasificado)

    # Top 5 Tabla
    html_top5 = "<h3>Insuficientes datos para ranking de conceptos.</h3>"
    if 'similitud_precio_costo' in resultados_similitud.columns:
        top5 = resultados_similitud.nlargest(5, 'similitud_precio_costo')
        html_top5 = "<table class='data-table'><tr><th>Comentario</th><th>Sentimiento</th><th>Proximidad Vectorial</th></tr>"
        for _, row in top5.iterrows():
            html_top5 += f"<tr><td>{row.iloc[0]}</td><td><span class='badge'>{row['sentimiento']}</span></td><td>{row['similitud_precio_costo']:.4f}</td></tr>"
        html_top5 += "</table>"

    # Inyección local de JS (include_plotlyjs=True) para ejecución Offline
    div_topicos = fig_topicos.to_html(full_html=False, include_plotlyjs=True)
    div_precio = fig_precio.to_html(full_html=False, include_plotlyjs=False)
    div_grafo = fig_grafo.to_html(full_html=False, include_plotlyjs=False)

    # Maquetación con CSS Grid
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{titulo}</title>
        <style>
            :root {{ --bg-color: #f4f7f6; --card-bg: #ffffff; --text-main: #2d3436; --accent: #0984e3; --border: #dfe6e9; }}
            body {{ font-family: 'Segoe UI', system-ui, sans-serif; background-color: var(--bg-color); color: var(--text-main); margin: 0; padding: 20px; }}
            .dashboard-header {{ text-align: center; margin-bottom: 30px; padding-bottom: 10px; border-bottom: 2px solid var(--border); }}
            
            /* CSS Grid Layout */
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-template-rows: auto auto auto;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            
            .card {{ background: var(--card-bg); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid var(--border); }}
            .card h2 {{ font-size: 1.2rem; color: var(--accent); margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
            
            /* Posicionamiento en el Grid */
            .grafica-principal {{ grid-column: span 2; }}
            .grafo-red {{ grid-column: span 1; }}
            .grafica-precio {{ grid-column: span 1; }}
            .tabla-resultados {{ grid-column: span 2; overflow-x: auto; }}
            
            .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
            .data-table th, .data-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border); }}
            .data-table th {{ background-color: var(--bg-color); font-weight: 600; }}
            .badge {{ padding: 4px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; background: #e0f7fa; color: #006064; }}
        </style>
    </head>
    <body>
        <div class="dashboard-header">
            <h1>{titulo}</h1>
            <p><strong>Motor de Análisis:</strong> Ejecución Local (Offline) | <strong>Modelo:</strong> SentenceTransformers (CUDA Acelerado)</p>
        </div>
        
        <div class="grid-container">
            <div class="card grafica-principal">
                <h2>Agrupamiento por Similitud Semántica (PCA)</h2>
                {div_topicos}
            </div>
            
            <div class="card grafo-red">
                <h2>Topología Semántica (NetworkX)</h2>
                {div_grafo}
            </div>
            
            <div class="card grafica-precio">
                <h2>Distancia Vectorial al Concepto "Precio/Costo"</h2>
                {div_precio}
            </div>
            
            <div class="card tabla-resultados">
                <h2>Extracción Contextual: Top 5 Comentarios (Precio/Valor)</h2>
                {html_top5}
            </div>
        </div>
    </body>
    </html>
    """

    nombre_archivo = f"{titulo.replace(' ', '_').lower()}_reporte.html"
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Reporte interactivo guardado exitosamente como: {nombre_archivo}")