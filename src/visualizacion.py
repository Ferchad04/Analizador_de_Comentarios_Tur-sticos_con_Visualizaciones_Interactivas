# src/visualizacion.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict

def crear_grafo_topologico(datos_red: dict) -> go.Figure:
    # Construye una red semántica interactiva utilizando NetworkX para el layout espacial
    # y Plotly para el renderizado vectorial.
    G = nx.Graph()
    fig = go.Figure()
    
    if not datos_red or not datos_red.get('nodos'):
        fig.add_trace(go.Scatter(x=[1], y=[1], text=["Datos insuficientes para la red"], mode="text"))
        return fig

    # Inyección de topología
    for nodo in datos_red['nodos']:
        G.add_node(nodo)
    for enlace in datos_red['enlaces']:
        G.add_edge(enlace['fuente'], enlace['destino'], weight=enlace['peso'])

    # Modelo de fuerzas (Fruchterman-Reingold)
    posiciones = nx.spring_layout(G, k=0.5, seed=42)

    # Aristas (Líneas)
    edge_x = []
    edge_y = []
    for arista in G.edges(data=True):
        x0, y0 = posiciones[arista[0]]
        x1, y1 = posiciones[arista[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    trace_aristas = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Nodos (Puntos) con factor de escala calibrado
    node_x = []
    node_y = []
    textos_nodos = []
    frecuencias = []
    for nodo in G.nodes():
        x, y = posiciones[nodo]
        node_x.append(x)
        node_y.append(y)
        textos_nodos.append(nodo)
        # Multiplicador reducido (1.2) para evitar desbordamiento visual
        frecuencias.append(len(list(G.neighbors(nodo))) * 1.2 + 8)

    trace_nodos = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=textos_nodos,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=frecuencias,
            color=frecuencias,
            line_width=2
        )
    )

    fig.add_trace(trace_aristas)
    fig.add_trace(trace_nodos)
    fig.update_layout(
        title="Red Semántica de Co-ocurrencia",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def generar_reporte(
    df_clasificado: pd.DataFrame, 
    df_outliers: pd.DataFrame,
    resultados_topicos: Dict[str, dict], 
    resultados_similitud: pd.DataFrame, 
    titulo: str, 
    paleta: str,
    columna_texto: str
) -> None:
    
    # Genera un reporte interactivo en formato HTML mediante CSS Grid y asegura renderizado offline.
    
    print(f"\n--- Generando Reporte: {titulo} ---")
    
    # 1. Gráfico de Dispersión (PCA)
    fig_topicos = go.Figure()
    if not df_clasificado.empty and 'sentimiento' in df_clasificado.columns:
        tfidf = TfidfVectorizer(max_features=100)
        # Inyección dinámica para evitar colapso de varianza
        vectores = tfidf.fit_transform(df_clasificado[columna_texto]).toarray()
        
        pca = PCA(n_components=2, random_state=42)
        coordenadas = pca.fit_transform(vectores)
        df_clasificado['x'] = coordenadas[:, 0]
        df_clasificado['y'] = coordenadas[:, 1]
        
        fig_topicos = px.scatter(
            df_clasificado,
            x='x', y='y',
            color='sentimiento', symbol='sentimiento',
            hover_data=[columna_texto],
            title="Distribución Semántica (PCA)",
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
            hover_data=[columna_texto],
            title="Proximidad Vectorial: 'Precio/Valor'",
            color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
        )
        fig_precio.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 3. Grafo Semántico
    fig_grafo = crear_grafo_topologico(resultados_topicos)

    # 4. Top 5 Tabla
    html_top5 = "<h3>Insuficientes datos para ranking de conceptos.</h3>"
    if 'similitud_precio_costo' in resultados_similitud.columns:
        top5 = resultados_similitud.nlargest(5, 'similitud_precio_costo')
        html_top5 = "<table class='data-table'><tr><th>Comentario</th><th>Sentimiento</th><th>Proximidad Vectorial</th></tr>"
        for _, row in top5.iterrows():
            html_top5 += f"<tr><td>{row[columna_texto]}</td><td><span class='badge'>{row['sentimiento']}</span></td><td>{row['similitud_precio_costo']:.4f}</td></tr>"
        html_top5 += "</table>"

    # Inyección local (Offline Rendering)
    div_topicos = fig_topicos.to_html(full_html=False, include_plotlyjs=True)
    div_precio = fig_precio.to_html(full_html=False, include_plotlyjs=False)
    div_grafo = fig_grafo.to_html(full_html=False, include_plotlyjs=False)

    # CSS Grid Layout
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
            
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-template-rows: auto auto auto;
                gap: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }}
            
            .card {{ background: var(--card-bg); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid var(--border); overflow: hidden; }}
            .card h2 {{ font-size: 1.2rem; color: var(--accent); margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 10px; }}
            
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
            <p><strong>Motor de Análisis:</strong> Ejecución Local (Offline) | <strong>Modelo:</strong> SentenceTransformers (CPU Acelerado)</p>
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